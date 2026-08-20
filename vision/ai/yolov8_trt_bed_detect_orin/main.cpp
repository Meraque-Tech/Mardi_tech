#include "main_fun.cpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "mjpeg_server.h"

#include <signal.h>
#include <stdio.h>
#include <atomic>
#include <chrono>
#include <map>
#include <unordered_map>
#include <thread>
#include "simple_tracker.h"

// Global state
rclcpp::Node::SharedPtr node;
float conf_score_value = 0.8f;
std::atomic<bool> detection_enabled{false};
std::atomic<bool> tracker_reset_requested{false};

struct TrtParams {
    std::string engine_name;
    std::string wts_name;
    std::string model_type;
    int         input_h;
    int         input_w;
    int         num_class;
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
    std::string video_source;
    std::string class_labels_json;
};

// Parses the flat {"0":"name","1":"name"} object the dashboard's "Class
// Labels" panel sends via /api/class_labels (see web_server.py
// call_set_class_labels). Deliberately minimal -- avoids pulling in a JSON
// library for a shape this constrained and fully producer-controlled.
std::unordered_map<int, std::string> parse_class_labels_json(const std::string &text) {
    std::unordered_map<int, std::string> labels;
    size_t pos = 0;
    while (true) {
        size_t key_start = text.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = text.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = text.substr(key_start + 1, key_end - key_start - 1);

        size_t colon = text.find(':', key_end + 1);
        if (colon == std::string::npos) break;
        size_t val_start = text.find('"', colon + 1);
        if (val_start == std::string::npos) break;
        size_t val_end = val_start + 1;
        std::string value;
        while (val_end < text.size() && text[val_end] != '"') {
            if (text[val_end] == '\\' && val_end + 1 < text.size()) val_end++;
            value += text[val_end];
            val_end++;
        }
        if (val_end >= text.size()) break;

        try {
            labels[std::stoi(key)] = value;
        } catch (const std::exception &) {
            // non-numeric key -- skip, keep scanning the rest of the object
        }
        pos = val_end + 1;
    }
    return labels;
}

TrtParams declare_and_get_params(rclcpp::Node::SharedPtr n) {
    TrtParams p;
    p.engine_name       = n->declare_parameter<std::string>("engine_name",       "yolov8n.engine");
    p.wts_name          = n->declare_parameter<std::string>("wts_name",          "yolov8n.wts");
    p.model_type        = n->declare_parameter<std::string>("model_type",        "n");
    p.input_h           = n->declare_parameter<int>        ("input_h",           416);
    p.input_w           = n->declare_parameter<int>        ("input_w",           416);
    p.num_class         = n->declare_parameter<int>        ("num_class",         80);
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
    // "camera" reads /dev/video<camera_index> as before; any other value is
    // treated as a video file path (see open_video_file below), letting the
    // dashboard point the same TensorRT pipeline at an uploaded video.
    p.video_source      = n->declare_parameter<std::string>("video_source",      "camera");
    // Flat {"<class_id>":"<name>"} map, pushed by the dashboard's "Class
    // Labels" panel via /api/class_labels -- lets the MJPEG overlay draw
    // names instead of raw indices (see parse_class_labels_json above).
    p.class_labels_json = n->declare_parameter<std::string>("class_labels_json", "{}");
    return p;
}


// Opens `preferred_index` if it works; otherwise scans /dev/video0.. for the
// first index that actually yields a frame (V4L2 exposes metadata-only nodes
// that succeed at isOpened() but never deliver a capture).
cv::VideoCapture open_camera(int preferred_index, int &opened_index, int max_scan = 10) {
    auto try_index = [](int index) {
        cv::VideoCapture cap(index, cv::CAP_V4L2);
        cv::Mat frame;
        if (cap.isOpened() && cap.read(frame) && !frame.empty()) {
            return cap;
        }
        cap.release();
        return cv::VideoCapture();
    };

    if (preferred_index >= 0) {
        cv::VideoCapture cap = try_index(preferred_index);
        if (cap.isOpened()) {
            RCLCPP_INFO(node->get_logger(), "camera_index %d opened", preferred_index);
            opened_index = preferred_index;
            return cap;
        }
        RCLCPP_WARN(node->get_logger(), "camera_index %d failed, scanning for a working camera", preferred_index);
    }

    for (int index = 0; index < max_scan; ++index) {
        if (index == preferred_index) continue;
        cv::VideoCapture cap = try_index(index);
        if (cap.isOpened()) {
            RCLCPP_INFO(node->get_logger(), "auto-detected camera at index %d", index);
            opened_index = index;
            return cap;
        }
    }
    opened_index = -1;
    return cv::VideoCapture();
}

// Opens an uploaded video file rather than a live camera. Kept separate from
// open_camera() because a file source never needs the multi-index scan and
// EOF means "loop back to the start", not "camera unplugged".
cv::VideoCapture open_video_file(const std::string &path) {
    cv::VideoCapture cap(path, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        RCLCPP_WARN(node->get_logger(), "failed to open video source '%s'", path.c_str());
        return cv::VideoCapture();
    }
    RCLCPP_INFO(node->get_logger(), "video source '%s' opened", path.c_str());
    return cap;
}


SimpleTracker tracker;

std::map<int, int> count_detections(const cv::Mat &frame, const std::vector<Detection> &res, bool is_track, SimpleTracker &trk) {
    if (is_track) {
        return trk.update(frame, res);  // cumulative unique counts per class (MOSSE tracker)
    }
    std::map<int, int> counts;
    for (auto &it : res) counts[static_cast<int>(it.class_id)]++;
    return counts;  // per-frame counts per class
}

void sig_handler(int signal){
    std::cout << "\nCtrl+C pressed. Exiting..." << std::endl;
    exit(0);
}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    signal(SIGINT, sig_handler);
    node = rclcpp::Node::make_shared("yolov8_trt");

    auto conf_pub = node->create_publisher<std_msgs::msg::Float32>("conf", 10);
    auto bed_status_pub = node->create_publisher<std_msgs::msg::UInt8>("bed_detection_status", 10);
    auto class_count_pub = node->create_publisher<std_msgs::msg::String>("class_counts", 10);
    auto state_qos = rclcpp::QoS(1).reliable().transient_local();
    auto detection_active_pub =
        node->create_publisher<std_msgs::msg::UInt8>("detection_active", state_qos);
    auto tracking_enabled_pub =
        node->create_publisher<std_msgs::msg::UInt8>("tracking_enabled", state_qos);
    auto camera_index_pub = node->create_publisher<std_msgs::msg::Int32>("camera_index", state_qos);
    auto infer_ms_pub = node->create_publisher<std_msgs::msg::Float32>("infer_ms", 10);

    auto publish_active = [&detection_active_pub](bool active) {
        std_msgs::msg::UInt8 msg;
        msg.data = active ? 1 : 0;
        detection_active_pub->publish(msg);
    };

    auto bed_detection_service = node->create_service<std_srvs::srv::Trigger>(
        "bed_detection",
        [&publish_active](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            const bool was_running = detection_enabled.exchange(true);
            if (!was_running) tracker_reset_requested = true;
            publish_active(true);
            response->success = true;
            response->message = was_running ? "bed detection already running" : "bed detection started";
            std::cout << "\n" << response->message << std::endl;
        });

    auto stop_detection_service = node->create_service<std_srvs::srv::Trigger>(
        "bed_detection_stop",
        [&publish_active, &bed_status_pub, &class_count_pub, &conf_pub](
            const std::shared_ptr<std_srvs::srv::Trigger::Request>,
            std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            const bool was_running = detection_enabled.exchange(false);
            publish_active(false);

            std_msgs::msg::UInt8 bed_msg;
            bed_msg.data = 0;
            bed_status_pub->publish(bed_msg);
            std_msgs::msg::Float32 conf_msg;
            conf_msg.data = 0.0f;
            conf_pub->publish(conf_msg);
            std_msgs::msg::String count_msg;
            count_msg.data = "";
            class_count_pub->publish(count_msg);

            response->success = true;
            response->message = was_running ? "bed detection stopped" : "bed detection already stopped";
            std::cout << "\n" << response->message << std::endl;
        });

    auto reset_tracker_service = node->create_service<std_srvs::srv::Trigger>(
        "reset_tracker",
        [](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
           std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            tracker_reset_requested = true;
            response->success = true;
            response->message = "tracker reset requested";
        });

    auto set_tracking_service = node->create_service<std_srvs::srv::SetBool>(
        "set_tracking",
        [&node, &tracking_enabled_pub](
            const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
            std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
            const auto result = node->set_parameter(rclcpp::Parameter("is_track", request->data));
            response->success = result.successful;
            response->message = result.successful
                ? (request->data ? "unique tracking enabled" : "per-frame counting enabled")
                : result.reason;
            if (result.successful) {
                tracker_reset_requested = true;
                std_msgs::msg::UInt8 tracking_msg;
                tracking_msg.data = request->data ? 1 : 0;
                tracking_enabled_pub->publish(tracking_msg);
            }
        });

    cv::Mat frame;

    cudaSetDevice(kGpuId);

    TrtParams p = declare_and_get_params(node);

    // Serialize mode: launch with a bare -s flag (see serialize_engine.launch.py).
    // All inputs -- wts_name, engine_name, model_type, input_h, input_w, num_class --
    // come from ROS params (trt_params.yaml), not positional CLI argv.
    bool serialize_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-s") { serialize_mode = true; break; }
    }
    if (serialize_mode) {
        kInputH = p.input_h;
        kInputW = p.input_w;
        kNumClass = p.num_class;
        RCLCPP_INFO(node->get_logger(), "serializing engine: %s -> %s  type: %s  res: %dx%d  classes: %d",
            p.wts_name.c_str(), p.engine_name.c_str(), p.model_type.c_str(), kInputW, kInputH, kNumClass);
        serialize_engine(p.wts_name, p.engine_name, p.model_type);
        node.reset();
        rclcpp::shutdown();
        return 0;
    }

    std_msgs::msg::UInt8 initial_tracking_msg;
    initial_tracking_msg.data = p.is_track ? 1 : 0;
    tracking_enabled_pub->publish(initial_tracking_msg);
    bool last_tracking_enabled = p.is_track;
    RCLCPP_INFO(node->get_logger(), "engine: %s  res: %dx%d  precision: %s  post: %s",
        p.engine_name.c_str(), p.input_w, p.input_h, p.precision.c_str(), p.cuda_post_process.c_str());

    int model_bboxes;

    // current_source tracks what's actually open right now, so the main loop
    // can detect a source change requested via the "video_source" ROS param
    // (see /api/video/source in web_server.py) and reopen accordingly.
    std::string current_source = p.video_source;
    bool using_camera = (current_source == "camera");
    int opened_camera_index = -1;
    cv::VideoCapture cap = using_camera
        ? open_camera(p.camera_index, opened_camera_index)
        : open_video_file(current_source);
    if (!cap.isOpened()) {
        std::cout << "Failed to open video source '" << current_source << "'." << std::endl;
        return 1;
    }
    if (using_camera) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  p.camera_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, p.camera_height);
    }

    std_msgs::msg::Int32 camera_index_msg;
    camera_index_msg.data = using_camera ? opened_camera_index : -1;
    camera_index_pub->publish(camera_index_msg);

    auto configure_camera = [&p](cv::VideoCapture &camera) {
        camera.set(cv::CAP_PROP_FRAME_WIDTH,  p.camera_width);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, p.camera_height);
    };
    auto last_good_frame_at = std::chrono::steady_clock::now();
    auto next_camera_retry_at = last_good_frame_at;

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
    // getBindingDimensions(index) was removed in TensorRT 10 along with
    // implicit-batch mode; the explicit-batch replacement looks up shape by
    // tensor name and its dims now include the batch dim at d[0], so the
    // bbox-count dim that used to be d[0] is now d[1].
    auto out_dims = engine->getTensorShape(kOutputTensorName);
    model_bboxes = out_dims.d[1];
    float *device_buffers[2];
    float *output_buffer_host = nullptr;
    float *decode_ptr_host = nullptr;
    float *decode_ptr_device = nullptr;

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host, &decode_ptr_device, p.cuda_post_process, p.input_h, p.input_w);

    publish_active(false);

    std::string current_class_labels_json = p.class_labels_json;
    std::unordered_map<int, std::string> class_labels = parse_class_labels_json(current_class_labels_json);

    while (rclcpp::ok()) {
        // Process web/ROS controls before starting the next inference cycle.
        rclcpp::spin_some(node);

        const std::string requested_class_labels_json = node->get_parameter("class_labels_json").as_string();
        if (requested_class_labels_json != current_class_labels_json) {
            current_class_labels_json = requested_class_labels_json;
            class_labels = parse_class_labels_json(current_class_labels_json);
        }

        const std::string requested_source = node->get_parameter("video_source").as_string();
        if (requested_source != current_source) {
            RCLCPP_INFO(node->get_logger(), "switching video source: '%s' -> '%s'",
                current_source.c_str(), requested_source.c_str());
            cap.release();
            current_source = requested_source;
            using_camera = (current_source == "camera");
            cap = using_camera
                ? open_camera(p.camera_index, opened_camera_index)
                : open_video_file(current_source);
            if (cap.isOpened()) {
                if (using_camera) configure_camera(cap);
                tracker_reset_requested = true;
                last_good_frame_at = std::chrono::steady_clock::now();
                camera_index_msg.data = using_camera ? opened_camera_index : -1;
                camera_index_pub->publish(camera_index_msg);
            } else {
                opened_camera_index = -1;
                next_camera_retry_at = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            }
        }

        if (!cap.isOpened()) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= next_camera_retry_at) {
                cap = using_camera ? open_camera(p.camera_index, opened_camera_index)
                                    : open_video_file(current_source);
                if (cap.isOpened()) {
                    if (using_camera) configure_camera(cap);
                    camera_index_msg.data = using_camera ? opened_camera_index : -1;
                    camera_index_pub->publish(camera_index_msg);
                    tracker_reset_requested = true;
                    last_good_frame_at = now;
                    RCLCPP_INFO(node->get_logger(), "video source recovered: '%s'", current_source.c_str());
                } else {
                    next_camera_retry_at = now + std::chrono::seconds(1);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }

        cap >> frame;
        const auto frame_received_at = std::chrono::steady_clock::now();
        if (frame.empty()) {
            if (!using_camera) {
                // Uploaded video reached EOF -- loop it rather than treating
                // this as a dropped source (that reopen path is camera-only).
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                last_good_frame_at = frame_received_at;
                continue;
            }
            if (frame_received_at - last_good_frame_at >= std::chrono::seconds(2)) {
                RCLCPP_WARN(
                    node->get_logger(),
                    "camera produced no frames for 2 seconds; reopening /dev/video%d",
                    opened_camera_index);
                cap.release();
                opened_camera_index = -1;
                camera_index_msg.data = -1;
                camera_index_pub->publish(camera_index_msg);
                next_camera_retry_at = frame_received_at + std::chrono::milliseconds(250);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            continue;
        }
        last_good_frame_at = frame_received_at;

        const bool is_track = node->get_parameter("is_track").as_bool();
        if (is_track != last_tracking_enabled) {
            tracker_reset_requested = true;
            last_tracking_enabled = is_track;
            std_msgs::msg::UInt8 tracking_msg;
            tracking_msg.data = is_track ? 1 : 0;
            tracking_enabled_pub->publish(tracking_msg);
        }
        if (tracker_reset_requested.exchange(false)) tracker.reset();

        if (detection_enabled.load()) {
            std::vector<cv::Mat> img_batch{frame};

            cuda_batch_preprocess(img_batch, device_buffers[0], p.input_w, p.input_h, stream);
            double infer_ms = 0.0;
            infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize,
                  decode_ptr_host, decode_ptr_device, model_bboxes, p.cuda_post_process,
                  p.conf_thresh, p.nms_thresh, p.max_output_bbox, &infer_ms);

            std_msgs::msg::Float32 infer_ms_msg;
            infer_ms_msg.data = static_cast<float>(infer_ms);
            infer_ms_pub->publish(infer_ms_msg);

            std::vector<std::vector<Detection>> res_batch;
            if (p.cuda_post_process == "c") {
                batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, p.conf_thresh, p.nms_thresh);
            } else if (p.cuda_post_process == "g") {
                batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
            }

            static const std::vector<Detection> no_detections;
            const auto &res = res_batch.empty() ? no_detections : res_batch[0];
            auto bed_msg = std_msgs::msg::UInt8();

            // Track against the clean camera frame, before annotations are drawn.
            std::map<int, int> class_counts = count_detections(frame, res, is_track, tracker);

            float max_conf = 0.0f;
            for (const auto &it : res) {
                max_conf = std::max(max_conf, it.conf);
                if (it.conf > conf_score_value) {
                    bed_msg.data = 1;
                }
            }

            auto conf_msg = std_msgs::msg::Float32();
            conf_msg.data = max_conf;
            conf_pub->publish(conf_msg);
            bed_status_pub->publish(bed_msg);

            std::string counts_str;
            for (auto &kv : class_counts)
                counts_str += "class" + std::to_string(kv.first) + ":" + std::to_string(kv.second) + " ";
            auto count_msg = std_msgs::msg::String();
            count_msg.data = counts_str;
            class_count_pub->publish(count_msg);

            draw_bbox(img_batch, res_batch, class_labels);
            frame = img_batch[0];  // annotated frame for the MJPEG stream
        }

        mjpeg_server.push_frame(frame);

        int key = cv::waitKey(1);
        if (key == 'q') {
            std::cout << "q pressed. Stopping." << std::endl;
            break;
        }

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
