
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
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

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running = false; }

// ─── capture thread → inference thread ───────────────────────────────────────
// Capture thread always overwrites with the latest frame.
// Inference thread picks it up immediately after finishing the previous one.
static std::mutex              g_cap_mutex;
static cv::Mat                 g_latest_frame;
static std::condition_variable g_cap_cv;
static bool                    g_frame_ready = false;

// ─── inference thread → HTTP server ──────────────────────────────────────────
static std::mutex         g_jpeg_mutex;
static std::vector<uchar> g_jpeg_frame;

// ─── TensorRT helpers ────────────────────────────────────────────────────────
void deserialize_engine(const std::string& name, IRuntime** rt,
                        ICudaEngine** eng, IExecutionContext** ctx) {
    std::ifstream file(name, std::ios::binary);
    if (!file.good()) { std::cerr << "Cannot open engine: " << name << "\n"; std::exit(1); }
    file.seekg(0, file.end); size_t sz = file.tellg(); file.seekg(0, file.beg);
    char* buf = new char[sz]; file.read(buf, sz); file.close();
    *rt  = createInferRuntime(gLogger);
    *eng = (*rt)->deserializeCudaEngine(buf, sz);
    *ctx = (*eng)->createExecutionContext();
    delete[] buf;
}

void prepare_buffer(ICudaEngine* engine, float** in_dev, float** out_dev,
                    float** out_host, float** dec_host, float** dec_dev,
                    const std::string& post) {
    assert(engine->getBindingIndex(kInputTensorName)  == 0);
    assert(engine->getBindingIndex(kOutputTensorName) == 1);
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
            kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemsetAsync(dec_dev, 0,
            sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode_obb((float*)bufs[1], model_bboxes, kConfThresh, dec_dev, kMaxNumOutputBbox, stream);
        cuda_nms_obb(dec_dev, kNmsThresh, kMaxNumOutputBbox, stream);
        CUDA_CHECK(cudaMemcpyAsync(dec_host, dec_dev,
            sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ─── capture thread ───────────────────────────────────────────────────────────
// Runs independently; always keeps g_latest_frame = newest camera frame.
// Old frames are overwritten — inference always gets the freshest image.
void capture_thread(int cam_id, int flip_code) {
    cv::VideoCapture cap(cam_id);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera " << cam_id << "\n";
        g_running = false;
        return;
    }
    // Request native camera buffer size to reduce V4L2 latency
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    std::cout << "Camera " << cam_id << " opened ("
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
              << cap.get(cv::CAP_PROP_FPS) << " fps)\n";

    cv::Mat frame;
    while (g_running) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Camera read failed\n";
            g_running = false;
            break;
        }
        if (flip_code >= 0) cv::flip(frame, frame, flip_code);
        {
            std::lock_guard<std::mutex> lk(g_cap_mutex);
            g_latest_frame = frame.clone();
            g_frame_ready  = true;
        }
        g_cap_cv.notify_one();
    }
    cap.release();
}

// ─── MJPEG HTTP server ────────────────────────────────────────────────────────
void client_handler(int fd) {
    const std::string hdr =
        "HTTP/1.0 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=--b\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n";
    send(fd, hdr.c_str(), hdr.size(), 0);

    std::vector<uchar> last_sent;
    while (g_running) {
        std::vector<uchar> jpeg;
        {
            std::lock_guard<std::mutex> lk(g_jpeg_mutex);
            jpeg = g_jpeg_frame;
        }
        // Only push when we have a new frame
        if (jpeg.empty() || jpeg.data() == last_sent.data()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        last_sent = jpeg;

        std::string ph = "--b\r\nContent-Type: image/jpeg\r\nContent-Length: "
                         + std::to_string(jpeg.size()) + "\r\n\r\n";
        if (send(fd, ph.c_str(),   ph.size(),   MSG_NOSIGNAL) < 0) break;
        if (send(fd, (char*)jpeg.data(), jpeg.size(), MSG_NOSIGNAL) < 0) break;
        if (send(fd, "\r\n", 2,    MSG_NOSIGNAL) < 0) break;
    }
    close(fd);
}

void mjpeg_server(int port) {
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
    if (bind(sfd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "bind failed on port " << port << "\n"; return;
    }
    listen(sfd, 8);
    std::cout << "MJPEG stream → http://localhost:" << port << "/\n";
    while (g_running) {
        sockaddr_in ca{}; socklen_t cl = sizeof(ca);
        int cfd = accept(sfd, (sockaddr*)&ca, &cl);
        if (cfd < 0) continue;
        std::thread(client_handler, cfd).detach();
    }
    close(sfd);
}

// ─── args ─────────────────────────────────────────────────────────────────────
// ./yolov8_stream -d <engine> <cam 0-3> <c|g> [port=8080] [flip: 0=vert 1=horiz 2=both]
bool parse_args(int argc, char** argv, std::string& engine, int& cam_id,
                std::string& post, int& port, int& flip_code) {
    if (argc < 5 || std::string(argv[1]) != "-d") return false;
    engine    = argv[2];
    cam_id    = std::stoi(argv[3]);
    post      = argv[4];
    if (post != "c" && post != "g") return false;
    if (cam_id < 0 || cam_id > 3)  return false;
    if (argc >= 6) port      = std::stoi(argv[5]);
    if (argc >= 7) flip_code = std::stoi(argv[6]);
    return true;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string engine_name, post;
    int cam_id = 0, port = 8080, flip_code = -1;

    if (!parse_args(argc, argv, engine_name, cam_id, post, port, flip_code)) {
        std::cerr << "Usage: ./yolov8_stream -d <engine> <cam 0-3> <c|g> [port=8080] [flip: 0=vert 1=horiz 2=both]\n";
        return -1;
    }

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    cudaSetDevice(kGpuId);

    IRuntime*          runtime = nullptr;
    ICudaEngine*       engine  = nullptr;
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

    // JPEG encode params — quality 75 is visually near-lossless but ~15% faster encode
    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 75};

    // Start capture and server threads
    std::thread cap_thread(capture_thread, cam_id, flip_code);
    std::thread srv_thread(mjpeg_server, port);

    // ── inference loop ────────────────────────────────────────────────────────
    // Blocks only until the capture thread signals a new frame.
    // While GPU is inferring, the capture thread is already reading the next frame.
    while (g_running) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lk(g_cap_mutex);
            g_cap_cv.wait_for(lk, std::chrono::milliseconds(100),
                              [] { return g_frame_ready || !g_running; });
            if (!g_running) break;
            if (!g_frame_ready) continue;
            frame         = g_latest_frame;  // shallow copy (shared data)
            g_frame_ready = false;           // consume
        }

        // Resize to model input
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);

        // GPU preprocess + infer
        std::vector<cv::Mat> batch = {resized};
        cuda_batch_preprocess(batch, device_buffers[0], kInputW, kInputH, stream);

        auto t0 = std::chrono::steady_clock::now();
        run_infer(*context, stream, (void**)device_buffers,
                  output_buffer_host, decode_ptr_host, decode_ptr_device,
                  model_bboxes, post);
        int infer_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t0).count();

        // Postprocess
        std::vector<std::vector<Detection>> res_batch;
        if (post == "c")
            batch_nms_obb(res_batch, output_buffer_host, 1, kOutputSize, kConfThresh, kNmsThresh);
        else
            batch_process_obb(res_batch, decode_ptr_host, 1, bbox_element, batch);

        draw_bbox_obb(batch, res_batch);

        // Overlay inference time
        std::string label = "Infer: " + std::to_string(infer_ms) + " ms";
        cv::putText(batch[0], label, cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0,0,0), 3, cv::LINE_AA);
        cv::putText(batch[0], label, cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0,255,0), 2, cv::LINE_AA);

        // Encode and publish — no lock held during encoding (CPU-heavy)
        std::vector<uchar> jpeg;
        cv::imencode(".jpg", batch[0], jpeg, encode_params);
        {
            std::lock_guard<std::mutex> lk(g_jpeg_mutex);
            g_jpeg_frame = std::move(jpeg);
        }
    }

    g_running = false;
    g_cap_cv.notify_all();
    cap_thread.join();
    srv_thread.join();

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
