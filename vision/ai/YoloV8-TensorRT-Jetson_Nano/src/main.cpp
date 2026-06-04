//
// Created by  triple-Mm     on 24-1-2023.
// Modified by Q-engineering on  6-3-2024
//

#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <cctype>
#include <string>

using namespace std;
using namespace cv;

cv::Size       im_size(640, 640);
const int      num_labels  = 80;
const int      topk        = 100;
const float    score_thres = 0.25f;
const float    iou_thres   = 0.65f;

// Returns true if the source looks like a camera index ("0", "1") or device node ("/dev/video0").
static bool is_camera_source(const std::string& src)
{
    if (src.rfind("/dev/video", 0) == 0) return true;
    for (char c : src) if (!std::isdigit(c)) return false;
    return !src.empty();
}

int main(int argc, char** argv)
{
    float    f;
    float    FPS[16];
    int      i, Fcnt=0;
    cv::Mat  image;
    std::chrono::steady_clock::time_point Tbegin, Tend;

    if (argc < 3) {
        fprintf(stderr, "Usage: ./YoloV8rt <model.engine> <source> [--display]\n"
                        "  source:    0, 1, ...       camera index (webcam)\n"
                        "             /dev/video0     device node\n"
                        "             image.jpg       still image (loops)\n"
                        "             video.mp4       video file\n"
                        "  --display  show live window (requires X11; omit for headless)\n");
        return -1;
    }
    const string engine_file_path = argv[1];
    const string source           = argv[2];

    bool show_display = false;
    for (int a = 3; a < argc; a++) {
        if (string(argv[a]) == "--display") show_display = true;
    }

    for (i = 0; i < 16; i++) FPS[i] = 0.0;

    cout << "Set CUDA...\n" << endl;
    cudaSetDevice(0);

    cout << "Loading TensorRT model " << engine_file_path << endl;
    cout << "\nWait a second...." << std::flush;
    auto yolov8 = new YOLOv8(engine_file_path);

    cout << "\rLoading the pipe... " << string(10, ' ') << "\n\r" << endl;
    yolov8->MakePipe(true);

    const bool use_stream = is_camera_source(source);

    cv::VideoCapture cap;
    if (use_stream) {
        if (source.rfind("/dev/video", 0) == 0)
            cap.open(source);
        else
            cap.open(std::stoi(source));

        if (!cap.isOpened()) {
            cerr << "ERROR: Unable to open camera: " << source << endl;
            delete yolov8;
            return -1;
        }
        cout << "Streaming from: " << source << endl;
    }

    while (true) {
        if (use_stream) {
            cap >> image;
            if (image.empty()) {
                cerr << "ERROR: Unable to grab frame from camera" << endl;
                break;
            }
        } else {
            // Try as still image first; fall back to video file
            image = cv::imread(source);
            if (image.empty()) {
                if (!cap.isOpened()) {
                    cap.open(source);
                    if (!cap.isOpened()) {
                        cerr << "ERROR: Cannot open file: " << source << endl;
                        break;
                    }
                }
                cap >> image;
                if (image.empty()) break;
            }
        }

        yolov8->CopyFromMat(image, im_size);

        std::vector<Object> objs;

        Tbegin = std::chrono::steady_clock::now();
        yolov8->Infer();
        Tend = std::chrono::steady_clock::now();

        yolov8->PostProcess(objs, score_thres, iou_thres, topk, num_labels);
        yolov8->DrawObjects(image, objs);

        f = std::chrono::duration_cast<std::chrono::milliseconds>(Tend - Tbegin).count();
        if (f > 0.0) FPS[((Fcnt++) & 0x0F)] = 1000.0 / f;
        for (f = 0.0, i = 0; i < 16; i++) f += FPS[i];
        putText(image, cv::format("FPS %0.2f", f / 16), cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));

        if (show_display) {
            imshow("YOLOv8 TRT", image);
            char esc = cv::waitKey(1);
            if (esc == 27) break;
        }
    }

    if (show_display) cv::destroyAllWindows();
    delete yolov8;
    return 0;
}
