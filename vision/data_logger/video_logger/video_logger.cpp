#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

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
    int         duration = -1;   // seconds, -1 = unlimited
    bool        show     = false;
    int         req_w    = 640;
    int         req_h    = 480;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--device"   && i+1 < argc) device   = std::stoi(argv[++i]);
        else if (a == "--output"   && i+1 < argc) outdir   = argv[++i];
        else if (a == "--fps"      && i+1 < argc) fps      = std::stod(argv[++i]);
        else if (a == "--duration" && i+1 < argc) duration = std::stoi(argv[++i]);
        else if (a == "--width"    && i+1 < argc) req_w    = std::stoi(argv[++i]);
        else if (a == "--height"   && i+1 < argc) req_h    = std::stoi(argv[++i]);
        else if (a == "--show")                   show     = true;
    }

    fs::create_directories(outdir);

    cv::VideoCapture cap(device);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open /dev/video" << device << "\n";
        return 1;
    }

    // Request resolution before first frame is read
    if (req_w > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH,  req_w);
    if (req_h > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, req_h);

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::string filename = outdir + "/video_" + timestamp() + ".mp4";
    cv::VideoWriter writer(
        filename,
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps,
        cv::Size(width, height)
    );
    if (!writer.isOpened()) {
        std::cerr << "Cannot open output video: " << filename << "\n";
        return 1;
    }

    std::cout << "Video logger: /dev/video" << device
              << " -> " << filename
              << "  (" << width << "x" << height << " @ " << fps << " fps)\n";
    if (duration > 0)
        std::cout << "Duration: " << duration << "s\n";

    auto start = std::chrono::steady_clock::now();
    int  frame_count = 0;
    cv::Mat frame;

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Frame read failed, skipping.\n";
            continue;
        }

        writer.write(frame);
        ++frame_count;

        if (show) {
            cv::imshow("video_logger", frame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
        }

        if (duration > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now() - start).count();
            if (elapsed >= duration) break;
        }
    }

    cap.release();
    writer.release();
    if (show) cv::destroyAllWindows();
    std::cout << "Saved " << frame_count << " frames to " << filename << "\n";
    return 0;
}
