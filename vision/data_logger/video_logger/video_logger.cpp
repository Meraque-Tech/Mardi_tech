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

namespace fs = std::filesystem;

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

// Resize + pad to square YOLO format (letterbox with grey borders)
static cv::Mat letterbox(const cv::Mat& src, int target) {
    float scale = std::min(float(target) / src.cols, float(target) / src.rows);
    int nw = int(src.cols * scale);
    int nh = int(src.rows * scale);
    int pad_x = (target - nw) / 2;
    int pad_y = (target - nh) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh));

    cv::Mat out(target, target, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_x, pad_y, nw, nh)));
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
    double      fps      = 30.0;
    int         duration = -1;
    bool        show     = false;
    int         req_w    = 640;
    int         req_h    = 480;
    int         yolo_sz  = 0;   // 0 = disabled, e.g. 640 for YOLOv8

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

    cv::VideoCapture cap(device);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open /dev/video" << device << "\n";
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH,  req_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, req_h);

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    int out_w = (yolo_sz > 0) ? yolo_sz : width;
    int out_h = (yolo_sz > 0) ? yolo_sz : height;

    std::string filename = outdir + "/video_" + timestamp() + ".mp4";

    // Pipe raw BGR24 frames into ffmpeg → H.264 mp4
    std::string cmd =
        "ffmpeg -y"
        " -f rawvideo -pixel_format bgr24"
        " -video_size " + std::to_string(out_w) + "x" + std::to_string(out_h) +
        " -framerate " + std::to_string(fps) +
        " -i pipe:0"
        " -c:v libx264 -preset fast -crf 23"
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
    if (yolo_sz > 0) std::cout << "YOLO letterbox: " << yolo_sz << "x" << yolo_sz << "\n";
    if (duration > 0) std::cout << "Duration: " << duration << "s\n";

    auto start = std::chrono::steady_clock::now();
    int  frame_count = 0;
    cv::Mat frame;

    while (!g_stop) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Frame read failed, skipping.\n";
            continue;
        }

        cv::Mat out = (yolo_sz > 0) ? letterbox(frame, yolo_sz)
                                     : frame;

        fwrite(out.data, 1, out.total() * out.elemSize(), ffmpeg);
        ++frame_count;

        if (show) {
            cv::imshow("video_logger", frame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) g_stop = true;
        }

        if (duration > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now() - start).count();
            if (elapsed >= duration) g_stop = true;
        }
    }

    cap.release();
    pclose(ffmpeg);   // flushes + finalizes the mp4 properly
    if (show) cv::destroyAllWindows();
    std::cout << "Saved " << frame_count << " frames to " << filename << "\n";
    return 0;
}
