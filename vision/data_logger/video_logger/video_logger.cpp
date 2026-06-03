#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <csignal>
#include <atomic>
#include <cstdio>
#include <thread>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

// Letterbox to square YOLO size, padding with grey (114)
static cv::Mat letterbox(const cv::Mat& src, int target) {
    float scale = std::min(float(target) / src.cols, float(target) / src.rows);
    int nw = int(src.cols * scale);
    int nh = int(src.rows * scale);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh));
    cv::Mat out(target, target, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect((target - nw) / 2, (target - nh) / 2, nw, nh)));
    return out;
}

static std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
    return ss.str();
}

int main(int argc, char** argv) {
    int         device   = 0;
    std::string outdir   = "logs/videos";
    double      fps      = 20.0;   // conservative default — Pi camera stable at 20
    int         duration = -1;
    bool        show     = false;
    int         req_w    = 640;
    int         req_h    = 640;
    int         yolo_sz  = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--device"   && i+1 < argc) device   = std::stoi(argv[++i]);
        else if (a == "--output"   && i+1 < argc) outdir   = argv[++i];
        else if (a == "--fps"      && i+1 < argc) fps      = std::stod(argv[++i]);
        else if (a == "--duration" && i+1 < argc) duration = std::stoi(argv[++i]);
        else if (a == "--width"    && i+1 < argc) req_w    = std::stoi(argv[++i]);
        else if (a == "--height"   && i+1 < argc) req_h    = std::stoi(argv[++i]);
        else if (a == "--yolo"     && i+1 < argc) yolo_sz  = std::stoi(argv[++i]);
        else if (a == "--yolo")                   yolo_sz  = 640;
        else if (a == "--show")                   show     = true;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    fs::create_directories(outdir);

#if CV_VERSION_MAJOR > 3 || (CV_VERSION_MAJOR == 3 && CV_VERSION_MINOR >= 4)
    cv::VideoCapture cap(device, cv::CAP_V4L2);
#else
    cv::VideoCapture cap(device);
#endif
    if (!cap.isOpened()) {
        std::cerr << "Cannot open /dev/video" << device << "\n";
        return 1;
    }

    // Tell driver the target resolution and fps before first read
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  req_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, req_h);
    cap.set(cv::CAP_PROP_FPS,          fps);

    // Drain stale frames from the driver buffer
    cv::Mat dummy;
    for (int i = 0; i < 5; ++i) cap.read(dummy);

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int out_w  = (yolo_sz > 0) ? yolo_sz : width;
    int out_h  = (yolo_sz > 0) ? yolo_sz : height;

    std::string filename = outdir + "/video_" + timestamp() + ".mp4";

    // -r on both input and output enforces CFR — eliminates vibration
    std::string cmd =
        "ffmpeg -y"
        " -f rawvideo -pixel_format bgr24"
        " -video_size " + std::to_string(out_w) + "x" + std::to_string(out_h) +
        " -r " + std::to_string(fps) +
        " -i pipe:0"
        " -c:v libx264 -preset fast -crf 23"
        " -r " + std::to_string(fps) +   // enforce CFR output
        " -pix_fmt yuv420p"
        " -movflags +faststart"
        " " + filename +
        " 2>/dev/null";

    FILE* ffmpeg = popen(cmd.c_str(), "w");
    if (!ffmpeg) {
        std::cerr << "Failed to launch ffmpeg\n";
        return 1;
    }

    std::cout << "Video logger: /dev/video" << device
              << " -> " << filename
              << "  (capture " << width << "x" << height
              << " -> output " << out_w << "x" << out_h
              << " @ " << fps << " fps)\n";
    if (yolo_sz  > 0) std::cout << "YOLO letterbox: " << yolo_sz << "x" << yolo_sz << "\n";
    if (duration > 0) std::cout << "Duration: " << duration << "s\n";

    auto wall_start = clk::now();
    auto next_frame = clk::now();
    const auto frame_interval = std::chrono::duration_cast<clk::duration>(
        std::chrono::duration<double>(1.0 / fps));

    int frame_count = 0;
    cv::Mat frame;

    while (!g_stop) {
        // Block until it's time for the next frame
        std::this_thread::sleep_until(next_frame);
        next_frame += frame_interval;

        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Frame read failed, skipping.\n";
            continue;
        }

        cv::Mat out = (yolo_sz > 0) ? letterbox(frame, yolo_sz) : frame;
        fwrite(out.data, 1, out.total() * out.elemSize(), ffmpeg);
        ++frame_count;

        if (show) {
            cv::imshow("video_logger", out);
            if ((cv::waitKey(1) & 0xFF) == 'q') g_stop = true;
        }

        if (duration > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                clk::now() - wall_start).count();
            if (elapsed >= duration) g_stop = true;
        }
    }

    cap.release();
    pclose(ffmpeg);
    if (show) cv::destroyAllWindows();
    std::cout << "Saved " << frame_count << " frames to " << filename << "\n";
    return 0;
}
