#include "main_fun.cpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_msgs/msg/string.hpp>
#include "std_srvs/srv/trigger.hpp"
#include "mjpeg_server.h"

#include <signal.h>
#include <stdio.h>
#include <map>
#include "simple_tracker.h"

// Global state
rclcpp::Node::SharedPtr node;
int start_bed_detection_ = 0;
float conf_score_value = 0.8f;
bool bed_detection_fb_ = 0;

struct TrtParams {
    std::string engine_name;
    int         input_h;
    int         input_w;
    std::string precision;
    std::string cuda_post_process;
    float       conf_thresh;
    float       nms_thresh;
    int         max_output_bbox;
    int         mjpeg_port;
    int         camera_index;
    int         camera_width;
    int         camera_height;
    bool        is_track;
};

TrtParams declare_and_get_params(rclcpp::Node::SharedPtr n) {
    TrtParams p;
    p.engine_name       = n->declare_parameter<std::string>("engine_name",       "yolov8n.engine");
    p.input_h           = n->declare_parameter<int>        ("input_h",           416);
    p.input_w           = n->declare_parameter<int>        ("input_w",           416);
    p.precision         = n->declare_parameter<std::string>("precision",         "fp16");
    p.cuda_post_process = n->declare_parameter<std::string>("cuda_post_process", "g");
    p.conf_thresh       = n->declare_parameter<double>     ("conf_thresh",       0.5);
    p.nms_thresh        = n->declare_parameter<double>     ("nms_thresh",        0.45);
    p.max_output_bbox   = n->declare_parameter<int>        ("max_output_bbox",   100);
    p.mjpeg_port        = n->declare_parameter<int>        ("mjpeg_port",        8080);
    p.camera_index      = n->declare_parameter<int>        ("camera_index",      0);
    p.camera_width      = n->declare_parameter<int>        ("camera_width",      1280);
    p.camera_height     = n->declare_parameter<int>        ("camera_height",     720);
    conf_score_value    = n->declare_parameter<double>     ("conf_score_value",  0.8);
    p.is_track          = n->declare_parameter<bool>       ("is_track",          false);
    return p;
}


SimpleTracker tracker;

std::map<int, int> count_detections(const cv::Mat &frame, const std::vector<Detection> &res, bool is_track, SimpleTracker &trk) {
    if (is_track) {
        return trk.update(frame, res);  // cumulative unique counts per class (CSRT tracker)
    }
    std::map<int, int> counts;
    for (auto &it : res) counts[static_cast<int>(it.class_id)]++;
    return counts;  // per-frame counts per class
}

void sig_handler(int signal){
    std::cout << "\nCtrl+C pressed. Exiting..." << std::endl;
    exit(0);
}

void bed_detection_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
          std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    start_bed_detection_ = 1;
    response->success = 1;
    response->message = "bed detection started";
    std::cout << "\nbed detection started" << std::endl;
}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    signal(SIGINT, sig_handler);
    node = rclcpp::Node::make_shared("yolov8_trt");

    auto conf_pub = node->create_publisher<std_msgs::msg::Float32>("conf", 10);
    auto bed_status_pub = node->create_publisher<std_msgs::msg::UInt8>("bed_detection_status", 10);
    auto class_count_pub = node->create_publisher<std_msgs::msg::String>("class_counts", 10);

    auto bed_detection_service =
        node->create_service<std_srvs::srv::Trigger>("bed_detection", &bed_detection_cb);

    cv::Mat frame;

    TrtParams p = declare_and_get_params(node);
    RCLCPP_INFO(node->get_logger(), "engine: %s  res: %dx%d  precision: %s  post: %s",
        p.engine_name.c_str(), p.input_w, p.input_h, p.precision.c_str(), p.cuda_post_process.c_str());

    cudaSetDevice(kGpuId);
    int model_bboxes;

    // Serialize mode: pass -s <wts> <engine> <variant> as before
    {
        std::string wts_name, engine_name, img_dir, sub_type, cuda_pp;
        if (argc >= 4 && parse_args(argc, argv, wts_name, engine_name, img_dir, sub_type, cuda_pp)) {
            if (!wts_name.empty()) {
                serialize_engine(wts_name, engine_name, sub_type);
                return 0;
            }
        }
    }

    cv::VideoCapture cap(p.camera_index);
    if (!cap.isOpened()) {
        std::cout << "Failed to open webcam." << std::endl;
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  p.camera_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, p.camera_height);

    MjpegServer mjpeg_server;
    if (!mjpeg_server.start(p.mjpeg_port)) {
        std::cerr << "Failed to start MJPEG server on port " << p.mjpeg_port << std::endl;
    } else {
        std::cout << "MJPEG stream available at http://<host-ip>:" << p.mjpeg_port << "/" << std::endl;
    }

    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    deserialize_engine(p.engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getBindingDimensions(1);
    model_bboxes = out_dims.d[0];
    float *device_buffers[2];
    float *output_buffer_host = nullptr;
    float *decode_ptr_host = nullptr;
    float *decode_ptr_device = nullptr;

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host, &decode_ptr_device, p.cuda_post_process, p.input_h, p.input_w);

    while (rclcpp::ok()) {
        cap >> frame;
        if (frame.empty()) continue;

        if (start_bed_detection_ == 1) {
            std::vector<cv::Mat> img_batch{frame};

            cuda_batch_preprocess(img_batch, device_buffers[0], p.input_w, p.input_h, stream);
            infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize,
                  decode_ptr_host, decode_ptr_device, model_bboxes, p.cuda_post_process,
                  p.conf_thresh, p.nms_thresh, p.max_output_bbox);

            std::vector<std::vector<Detection>> res_batch;
            if (p.cuda_post_process == "c") {
                batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, p.conf_thresh, p.nms_thresh);
            } else if (p.cuda_post_process == "g") {
                batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
            }

            draw_bbox(img_batch, res_batch);
            frame = img_batch[0];  // use annotated frame for MJPEG stream

            auto &res = res_batch[0];
            auto bed_msg = std_msgs::msg::UInt8();

            // per-class counts
            std::map<int, int> class_counts = count_detections(frame, res, p.is_track, tracker);
            std::string counts_str;
            for (auto &kv : class_counts)
                counts_str += "class" + std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
            auto count_msg = std_msgs::msg::String();
            count_msg.data = counts_str;
            class_count_pub->publish(count_msg);
            if (!counts_str.empty()) std::cout << "counts: " << counts_str << std::endl;

            if (!res.empty()) {
                for (auto &it : res) {
                    std::cout << "it.conf ---> " << it.conf << std::endl;
                    if (it.conf > conf_score_value) {
                        auto conf_msg = std_msgs::msg::Float32();
                        conf_msg.data = it.conf;
                        conf_pub->publish(conf_msg);
                        bed_msg.data = 1;
                        std::cout << "Detected bed." << std::endl;
                        break;
                    }
                }
            } else {
                bed_msg.data = 0;
            }
            bed_status_pub->publish(bed_msg);
        }

        mjpeg_server.push_frame(frame);

        int key = cv::waitKey(1);
        if (key == 'q') {
            std::cout << "q pressed. Stopping." << std::endl;
            break;
        }

        rclcpp::spin_some(node);
    }

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
    cv::destroyAllWindows();

    return 0;
}
