#include <iostream>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "yolo.hpp"

#include <rclcpp/rclcpp.hpp>

using namespace nvinfer1;

cv::Rect get_rect(BBox box) {
    return cv::Rect(round(box.x1), round(box.y1), round(box.x2 - box.x1), round(box.y2 - box.y1));
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("bed_detect");

    node->declare_parameter("engine_name", "bed_detect.engine");
    node->declare_parameter("camera_id", 0);
    node->declare_parameter("conf_thresh", 0.5);

    std::string engine_name = node->get_parameter("engine_name").as_string();
    int camera_id           = node->get_parameter("camera_id").as_int();
    float conf_thresh       = (float) node->get_parameter("conf_thresh").as_double();

    Yolo detector;
    if (detector.init(engine_name)) {
        RCLCPP_ERROR(node->get_logger(), "Detector init failed: %s", engine_name.c_str());
        return EXIT_FAILURE;
    }

    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        RCLCPP_ERROR(node->get_logger(), "Failed to open camera %d", camera_id);
        return EXIT_FAILURE;
    }

    RCLCPP_INFO(node->get_logger(), "Running on camera %d (press q to quit)", camera_id);

    cv::Mat frame;
    while (rclcpp::ok()) {
        if (!cap.read(frame) || frame.empty()) {
            RCLCPP_ERROR(node->get_logger(), "Failed to read frame");
            break;
        }

        auto detections = detector.run(frame, frame.rows, frame.cols, conf_thresh);

        for (auto &det : detections) {
            cv::Rect r = get_rect(det.box);
            cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(frame,
                        std::to_string(det.label) + " " + std::to_string(det.prob).substr(0, 4),
                        cv::Point(r.x, r.y - 4),
                        cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }

        cv::imshow("Bed Detection", frame);
        if (cv::waitKey(1) == 'q') break;

        rclcpp::spin_some(node);
    }

    cap.release();
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return EXIT_SUCCESS;
}
