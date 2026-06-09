#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                        IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
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

void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                    float** output_buffer_host, float** decode_ptr_host, float** decode_ptr_device,
                    std::string cuda_post_process) {
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
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    if (cuda_post_process == "c") {
        *output_buffer_host = new float[kBatchSize * kOutputSize];
    } else if (cuda_post_process == "g") {
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchsize,
           float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::string cuda_post_process) {
    auto start = std::chrono::system_clock::now();
    context.setInputTensorAddress(kInputTensorName, buffers[0]);
    context.setOutputTensorAddress(kOutputTensorName, buffers[1]);
    context.enqueueV3(stream);
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float*)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                   sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost,
                                   stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference and gpu postprocess time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Usage: ./yolov8_det_video [.engine] [cam_id: 0,1,...] [c/g] [show: true/false]
bool parse_args(int argc, char** argv, std::string& engine, int& cam_id, std::string& cuda_post_process,
                bool& show) {
    if (argc != 5)
        return false;
    engine = std::string(argv[1]);
    cam_id = std::stoi(argv[2]);
    cuda_post_process = std::string(argv[3]);
    std::string show_str = std::string(argv[4]);
    show = (show_str == "true" || show_str == "1");
    return true;
}

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    std::string engine_name;
    std::string cuda_post_process;
    int cam_id = 0;
    bool show = false;
    int model_bboxes;

    if (!parse_args(argc, argv, engine_name, cam_id, cuda_post_process, show)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolov8_det_video [.engine] [cam_id: 0/1/...] [c/g] [show: true/false]" << std::endl;
        return -1;
    }

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getTensorShape(kOutputTensorName);
    model_bboxes = out_dims.d[1];

    float* device_buffers[2];
    float* output_buffer_host = nullptr;
    float* decode_ptr_host = nullptr;
    float* decode_ptr_device = nullptr;
    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host,
                   &decode_ptr_device, cuda_post_process);

    cv::VideoCapture cap(cam_id);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera " << cam_id << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        std::vector<cv::Mat> img_batch = {frame};

        cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
        infer(*context, stream, (void**)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host,
              decode_ptr_device, model_bboxes, cuda_post_process);

        std::vector<std::vector<Detection>> res_batch;
        if (cuda_post_process == "c") {
            batch_nms(res_batch, output_buffer_host, 1, kOutputSize, kConfThresh, kNmsThresh);
        } else if (cuda_post_process == "g") {
            batch_process(res_batch, decode_ptr_host, 1, bbox_element, img_batch);
        }

        draw_bbox(img_batch, res_batch);

        if (show) {
            cv::imshow("YOLOv8 Detection", img_batch[0]);
            if (cv::waitKey(1) == 27)  // ESC to quit
                break;
        }
    }

    cap.release();
    if (show)
        cv::destroyAllWindows();

    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    CUDA_CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    delete[] output_buffer_host;
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
