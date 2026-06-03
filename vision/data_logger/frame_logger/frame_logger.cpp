#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

static std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto us  = std::chrono::duration_cast<std::chrono::microseconds>(
                   now.time_since_epoch()) % 1000000;
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S")
       << "_" << std::setw(6) << std::setfill('0') << us.count();
    return ss.str();
}

int main(int argc, char** argv) {
    int         device     = 0;
    std::string outdir     = "logs/frames";
    double      fps        = 1.0;
    int         max_frames = -1;
    bool        show       = false;
    int         req_w      = 640;
    int         req_h      = 480;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--device"     && i+1 < argc) device     = std::stoi(argv[++i]);
        else if (a == "--output"     && i+1 < argc) outdir     = argv[++i];
        else if (a == "--fps"        && i+1 < argc) fps        = std::stod(argv[++i]);
        else if (a == "--max-frames" && i+1 < argc) max_frames = std::stoi(argv[++i]);
        else if (a == "--width"      && i+1 < argc) req_w      = std::stoi(argv[++i]);
        else if (a == "--height"     && i+1 < argc) req_h      = std::stoi(argv[++i]);
        else if (a == "--show")                     show       = true;
    }

    fs::create_directories(outdir);

    cv::VideoCapture cap(device);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open /dev/video" << device << "\n";
        return 1;
    }

    if (req_w > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH,  req_w);
    if (req_h > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, req_h);

    int actual_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frame_count = 0;
    int interval_ms = static_cast<int>(1000.0 / fps);

    std::cout << "Frame logger: /dev/video" << device
              << " -> " << outdir
              << "  (" << actual_w << "x" << actual_h << " @ " << fps << " fps)\n";

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Frame read failed, skipping.\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
            continue;
        }

        std::string filename = outdir + "/frame_" + timestamp() + ".jpg";
        cv::imwrite(filename, frame);
        std::cout << "[" << ++frame_count << "] Saved " << filename << "\n";

        if (show) {
            cv::imshow("frame_logger", frame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
        }

        if (max_frames > 0 && frame_count >= max_frames) break;

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    cap.release();
    if (show) cv::destroyAllWindows();
    std::cout << "Total frames saved: " << frame_count << "\n";
    return 0;
}
