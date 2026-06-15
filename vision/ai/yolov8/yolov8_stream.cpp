
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

// ─── shared MJPEG frame ──────────────────────────────────────────────────────
static std::mutex              g_frame_mutex;
static std::vector<uchar>      g_jpeg_frame;
static std::atomic<bool>       g_running{true};

static void signal_handler(int) { g_running = false; }

// ─── TensorRT helpers (same as yolov8_obb.cpp) ───────────────────────────────
void deserialize_engine(const std::string& engine_name, IRuntime** runtime,
                        ICudaEngine** engine, IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Cannot open engine: " << engine_name << std::endl;
        std::exit(1);
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* buf = new char[size];
    file.read(buf, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    *engine  = (*runtime)->deserializeCudaEngine(buf, size);
    *context = (*engine)->createExecutionContext();
    delete[] buf;
}

void prepare_buffer(ICudaEngine* engine, float** in_dev, float** out_dev,
                    float** out_host, float** dec_host, float** dec_dev,
                    const std::string& post) {
    const int inputIndex  = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0 && outputIndex == 1);

    CUDA_CHECK(cudaMalloc((void**)in_dev,  kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)out_dev, kBatchSize * kOutputSize * sizeof(float)));

    if (post == "c") {
        *out_host = new float[kBatchSize * kOutputSize];
    } else {
        *dec_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void**)dec_dev, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void run_infer(IExecutionContext& ctx, cudaStream_t& stream, void** bufs,
               float* out_host, float* dec_host, float* dec_dev,
               int model_bboxes, const std::string& post) {
    ctx.enqueue(kBatchSize, bufs, stream, nullptr);
    if (post == "c") {
        CUDA_CHECK(cudaMemcpyAsync(out_host, bufs[1],
                                   kBatchSize * kOutputSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemsetAsync(dec_dev, 0,
                                   sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode_obb((float*)bufs[1], model_bboxes, kConfThresh, dec_dev, kMaxNumOutputBbox, stream);
        cuda_nms_obb(dec_dev, kNmsThresh, kMaxNumOutputBbox, stream);
        CUDA_CHECK(cudaMemcpyAsync(dec_host, dec_dev,
                                   sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ─── MJPEG HTTP server ────────────────────────────────────────────────────────
// Each connecting client gets a dedicated thread that pushes frames.
void client_handler(int client_fd) {
    // Send HTTP headers for MJPEG stream
    const std::string header =
        "HTTP/1.0 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=--mjpegboundary\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n";
    send(client_fd, header.c_str(), header.size(), 0);

    while (g_running) {
        std::vector<uchar> jpeg;
        {
            std::lock_guard<std::mutex> lk(g_frame_mutex);
            jpeg = g_jpeg_frame;
        }
        if (jpeg.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        std::string part_header =
            "--mjpegboundary\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: " + std::to_string(jpeg.size()) + "\r\n\r\n";

        if (send(client_fd, part_header.c_str(), part_header.size(), MSG_NOSIGNAL) < 0) break;
        if (send(client_fd, reinterpret_cast<char*>(jpeg.data()), jpeg.size(), MSG_NOSIGNAL) < 0) break;
        const std::string tail = "\r\n";
        if (send(client_fd, tail.c_str(), tail.size(), MSG_NOSIGNAL) < 0) break;

        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // ~33 fps cap
    }
    close(client_fd);
}

void mjpeg_server(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "bind failed on port " << port << std::endl;
        return;
    }
    listen(server_fd, 5);
    std::cout << "MJPEG stream: http://localhost:" << port << "/  (open in browser)" << std::endl;

    while (g_running) {
        sockaddr_in client_addr{};
        socklen_t len = sizeof(client_addr);
        int client_fd = accept(server_fd, (sockaddr*)&client_addr, &len);
        if (client_fd < 0) continue;
        std::thread(client_handler, client_fd).detach();
    }
    close(server_fd);
}

// ─── argument parsing ─────────────────────────────────────────────────────────
// Usage: yolov8_stream -d <engine> <cam_id 0-3> <c|g> [port=8080] [flip: 0=vert 1=horiz 2=both]
bool parse_args(int argc, char** argv, std::string& engine, int& cam_id,
                std::string& post, int& port, int& flip_code) {
    if (argc < 5 || std::string(argv[1]) != "-d") return false;
    engine = argv[2];
    cam_id = std::stoi(argv[3]);
    post   = argv[4];
    if (post != "c" && post != "g") return false;
    if (cam_id < 0 || cam_id > 3) return false;
    if (argc >= 6) port      = std::stoi(argv[5]);
    if (argc >= 7) flip_code = std::stoi(argv[6]);
    return true;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string engine_name, post;
    int cam_id = 0, port = 8080, flip_code = -1;

    if (!parse_args(argc, argv, engine_name, cam_id, post, port, flip_code)) {
        std::cerr << "Usage: ./yolov8_stream -d <engine.engine> <cam 0-3> <c|g> [port=8080] [flip: 0=vert 1=horiz 2=both]\n";
        return -1;
    }

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    cudaSetDevice(kGpuId);

    // Load engine
    IRuntime*         runtime = nullptr;
    ICudaEngine*      engine  = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);

    auto out_dims   = engine->getBindingDimensions(1);
    int  model_bboxes = out_dims.d[0];

    float* device_buffers[2];
    float* output_buffer_host = nullptr;
    float* decode_ptr_host    = nullptr;
    float* decode_ptr_device  = nullptr;
    prepare_buffer(engine, &device_buffers[0], &device_buffers[1],
                   &output_buffer_host, &decode_ptr_host, &decode_ptr_device, post);

    // Open camera
    cv::VideoCapture cap(cam_id);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera " << cam_id << std::endl;
        return -1;
    }
    std::cout << "Opened camera " << cam_id << std::endl;

    // Start MJPEG server thread
    std::thread server_thread(mjpeg_server, port);

    while (g_running) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Camera read failed" << std::endl;
            break;
        }
        if (flip_code >= 0) cv::flip(frame, frame, flip_code);

        // Resize to model input size before GPU upload to reduce transfer cost
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);

        // Preprocess + infer (single-frame batch)
        std::vector<cv::Mat> batch = {resized};
        cuda_batch_preprocess(batch, device_buffers[0], kInputW, kInputH, stream);

        auto t0 = std::chrono::steady_clock::now();
        run_infer(*context, stream, (void**)device_buffers,
                  output_buffer_host, decode_ptr_host, decode_ptr_device,
                  model_bboxes, post);
        auto t1 = std::chrono::steady_clock::now();
        int infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        // Postprocess
        std::vector<std::vector<Detection>> res_batch;
        if (post == "c") {
            batch_nms_obb(res_batch, output_buffer_host, 1, kOutputSize, kConfThresh, kNmsThresh);
        } else {
            batch_process_obb(res_batch, decode_ptr_host, 1, bbox_element, batch);
        }

        // Draw detections
        draw_bbox_obb(batch, res_batch);

        // Overlay inference time
        std::string fps_text = "Infer: " + std::to_string(infer_ms) + " ms";
        cv::putText(batch[0], fps_text, cv::Point(8, 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(batch[0], fps_text, cv::Point(8, 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        // Encode to JPEG and publish
        std::vector<uchar> jpeg_buf;
        cv::imencode(".jpg", batch[0], jpeg_buf,
                     {cv::IMWRITE_JPEG_QUALITY, 80});
        {
            std::lock_guard<std::mutex> lk(g_frame_mutex);
            g_jpeg_frame = std::move(jpeg_buf);
        }
    }

    g_running = false;
    server_thread.join();

    // Cleanup
    cap.release();
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
