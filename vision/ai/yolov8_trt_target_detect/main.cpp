#include "main_fun.cpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include "std_srvs/srv/trigger.hpp"
#include "mjpeg_server.h"

#include <signal.h>
#include <stdio.h>

// Global state
rclcpp::Node::SharedPtr node;
int start_target_detection_ = 0;
float conf_score_value = 0.8f;
bool target_detection_fb_ = 0;


void sig_handler(int signal){
    std::cout << "\nCtrl+C pressed. Exiting..." << std::endl;
    exit(0);
}

void target_detection_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
          std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    start_target_detection_ = 1;
    response->success = 1;
    response->message = "target detection started";
    std::cout << "\ntarget detection started" << std::endl;
}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    signal(SIGINT, sig_handler);
    node = rclcpp::Node::make_shared("yolov8_trt");

    auto conf_pub = node->create_publisher<std_msgs::msg::Float32>("conf", 10);
    auto target_status_pub = node->create_publisher<std_msgs::msg::UInt8>("target_detection_status", 10);

    auto target_detection_service =
        node->create_service<std_srvs::srv::Trigger>("target_detection", &target_detection_cb);

    cv::Mat frame;

    cudaSetDevice(kGpuId);
    std::string wts_name = "";
    std::string engine_name = "";
    std::string img_dir;
    std::string sub_type = "";
    std::string cuda_post_process = "";
    int model_bboxes;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, sub_type, cuda_post_process)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolov8_trt -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov8_trt -d [.engine] ../samples  [c/g]// deserialize plan file and run inference" << std::endl;
        return -1;
    }

    if (!wts_name.empty()) {
        serialize_engine(wts_name, engine_name, sub_type);
        return 0;
    }

    // Open webcam (device index 0 by default) — only needed for inference mode
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Failed to open webcam." << std::endl;
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    MjpegServer mjpeg_server;
    if (!mjpeg_server.start(8080)) {
        std::cerr << "Failed to start MJPEG server on port 8080." << std::endl;
    } else {
        std::cout << "MJPEG stream available at http://<host-ip>:8080/" << std::endl;
    }

    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getBindingDimensions(1);
    model_bboxes = out_dims.d[0];
    float *device_buffers[2];
    float *output_buffer_host = nullptr;
    float *decode_ptr_host = nullptr;
    float *decode_ptr_device = nullptr;

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host, &decode_ptr_device, cuda_post_process);

    try {
        if (argc > 5 && std::string(argv[5]) == "-conf") {
            conf_score_value = std::stof(argv[6]);
            std::cout << "conf_score_value -> " << conf_score_value << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return -1;
    }

    while (rclcpp::ok()) {
        cap >> frame;
        if (frame.empty()) continue;

        if (start_target_detection_ == 1) {
            std::vector<cv::Mat> img_batch{frame};

            cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
            infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize,
                  decode_ptr_host, decode_ptr_device, model_bboxes, cuda_post_process);

            std::vector<std::vector<Detection>> res_batch;
            if (cuda_post_process == "c") {
                batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
            } else if (cuda_post_process == "g") {
                batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
            }

            draw_bbox(img_batch, res_batch);

            auto &res = res_batch[0];
            auto target_msg = std_msgs::msg::UInt8();

            if (!res.empty()) {
                for (auto &it : res) {
                    std::cout << "it.conf ---> " << it.conf << std::endl;
                    if (it.conf > conf_score_value) {
                        auto conf_msg = std_msgs::msg::Float32();
                        conf_msg.data = it.conf;
                        conf_pub->publish(conf_msg);
                        target_msg.data = 1;
                        std::cout << "Detected target." << std::endl;
                        break;
                    }
                }
            } else {
                target_msg.data = 0;
            }
            target_status_pub->publish(target_msg);
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
