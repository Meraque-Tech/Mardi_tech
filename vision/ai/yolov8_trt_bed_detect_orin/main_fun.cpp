
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "model.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "cuda_utils.h"
#include "logging.h"


Logger gLogger;
// using namespace sl;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

int kInputH = 512;
int kInputW = 512;
int kNumClass = 80;

void serialize_engine(std::string &wts_name, std::string &engine_name, std::string &sub_type) {
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    IHostMemory *serialized_engine = nullptr;

    // Ultralytics' official YOLOv8 depth/width/max-channels scaling table --
    // buildEngineYolov8Det (block.cpp/model.cpp) is now the single generic
    // builder shared across n/s/m/l/x, replacing the old fixed
    // buildEngineYolov8n/s/m/l/x functions.
    float gd = 0.33f, gw = 0.25f;
    int max_channels = 1024;
    if (sub_type == "n") {
        gd = 0.33f; gw = 0.25f; max_channels = 1024;
    } else if (sub_type == "s") {
        gd = 0.33f; gw = 0.50f; max_channels = 1024;
    } else if (sub_type == "m") {
        gd = 0.67f; gw = 0.75f; max_channels = 768;
    } else if (sub_type == "l") {
        gd = 1.0f; gw = 1.0f; max_channels = 512;
    } else if (sub_type == "x") {
        gd = 1.0f; gw = 1.25f; max_channels = 512;
    }
    serialized_engine = buildEngineYolov8Det(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels);

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    // Destroy in reverse-creation order: config was created *from* builder
    // (builder->createBuilderConfig()), so it must be destroyed before
    // builder -- deleting builder first is exactly the "destroying a builder
    // object before destroying objects it created" API misuse TensorRT warns
    // about, and the undefined behavior that follows.
    delete serialized_engine;
    delete config;
    delete builder;
}


void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char *serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                    float **output_buffer_host, float **decode_ptr_host, float **decode_ptr_device, std::string cuda_post_process, int input_h, int input_w) {
    // getNbBindings()/getBindingIndex() were removed in TensorRT 10 (implicit-batch-era
    // API) -- the explicit-batch replacement identifies tensors by name via
    // getNbIOTensors()/getTensorIOMode() instead of a fixed binding index.
    assert(engine->getNbIOTensors() == 2);
    TensorIOMode input_mode = engine->getTensorIOMode(kInputTensorName);
    if (input_mode != TensorIOMode::kINPUT) {
        std::cerr << kInputTensorName << " should be input tensor" << std::endl;
        assert(false);
    }
    TensorIOMode output_mode = engine->getTensorIOMode(kOutputTensorName);
    if (output_mode != TensorIOMode::kOUTPUT) {
        std::cerr << kOutputTensorName << " should be output tensor" << std::endl;
        assert(false);
    }
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void **) input_buffer_device, kBatchSize * 3 * input_h * input_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    if (cuda_post_process == "c") {
        *output_buffer_host = new float[kBatchSize * kOutputSize];
    } else if (cuda_post_process == "g") {
        // Allocate memory for decode_ptr_host and copy to device
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void **)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, int batchsize, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::string cuda_post_process, float conf_thresh, float nms_thresh, int max_output_bbox, double *elapsed_ms = nullptr) {
    auto start = std::chrono::system_clock::now();
    // IExecutionContext::enqueue(batchSize, buffers, ...) was removed in
    // TensorRT 10 along with implicit-batch mode -- bind tensors by name and
    // use enqueueV3 instead.
    context.setInputTensorAddress(kInputTensorName, buffers[0]);
    context.setOutputTensorAddress(kOutputTensorName, buffers[1]);
    context.enqueueV3(stream);
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + max_output_bbox * bbox_element), stream));
        cuda_decode((float *)buffers[1], model_bboxes, conf_thresh, decode_ptr_device, max_output_bbox, stream);
        cuda_nms(decode_ptr_device, nms_thresh, max_output_bbox, stream);
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + max_output_bbox * bbox_element), cudaMemcpyDeviceToHost, stream));
    }

    // Sync before stopping the clock so the timing reflects actual GPU
    // completion, not just how long it took to submit the async work.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::system_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (elapsed_ms) *elapsed_ms = ms;
    if (cuda_post_process == "c") {
        std::cout << "inference time: " << ms << "ms" << std::endl;
    } else if (cuda_post_process == "g") {
        std::cout << "inference and gpu postprocess time: " << ms << "ms" << std::endl;
    }
}


// int getOCVtype(sl::MAT_TYPE type);

// cv::Mat slMat2cvMat(Mat& input);
// int getOCVtype(sl::MAT_TYPE type);

// cv::Mat slMat2cvMat(Mat& input) {
//     // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
//     // cv::Mat and sl::Mat will share a single memory structure
//     return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
// }

// // Mapping between MAT_TYPE and CV_TYPE
// int getOCVtype(sl::MAT_TYPE type) {
//     int cv_type = -1;
//     switch (type) {
//         case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
//         case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
//         case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
//         case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
//         case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
//         case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
//         case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
//         case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
//         default: break;
//     }
//     return cv_type;
// }

