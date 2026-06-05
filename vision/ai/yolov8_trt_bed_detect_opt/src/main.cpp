#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"

#include "GLViewer.hpp"
#include "yolo.hpp"
#include <string>

#include <sl/Camera.hpp>
#include <NvInfer.h>

// #include "ros_utils.cpp"


#include <signal.h>
#include <stdio.h>
#include <rclcpp/rclcpp.hpp>
#include "ros_utils.h"
// #include <geometry_msgs/msg/transform_stamped.hpp>
#include "std_srvs/srv/trigger.hpp"

using namespace nvinfer1;
bool bed_algo_loop_ = 1;
// bool start_bed_detection_ = 1;
// bool start_generate_flashpoint_ = 1;
bool start_bed_detection_;
bool start_generate_flashpoint_;
// bool svo_open = false;

void bed_detection_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
          std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    // int bed_detection_data = request->start_bed_detection;
    start_bed_detection_ = 1;
    response->success = 1;
    response->message = "bed detection started";
    bool bed_detection_fb_ = 1;
    bed_algo_loop_ = 1;
    start_generate_flashpoint_ = 0;
}

void generate_flashpoint_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
          std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    start_generate_flashpoint_ = 1;
    response->success = 1;
    response->message = "bed detection plane algo started";
}

void bed_algo_loop_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
          std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    bed_algo_loop_ = 0;
    response->success = 1;
    response->message = "bed_algo_loop algo stop";
}


void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    std::cout << "[Sample] ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

cv::Rect get_rect(BBox box) {
    return cv::Rect(round(box.x1), round(box.y1), round(box.x2 - box.x1), round(box.y2 - box.y1));
}

std::vector<sl::uint2> cvt(const BBox &bbox_in) {
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x1, bbox_in.y1);
    bbox_out[1] = sl::uint2(bbox_in.x2, bbox_in.y1);
    bbox_out[2] = sl::uint2(bbox_in.x2, bbox_in.y2);
    bbox_out[3] = sl::uint2(bbox_in.x1, bbox_in.y2);
    return bbox_out;
}

// void sig_handler(int signal){
//     std::cout << "\nCtrl+C pressed. Exiting..." << std::endl;
//     exit(0); 

// }

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("yolo_onnx_zed");
    rclcpp::QoS qos(rclcpp::KeepLast(10));
    auto static_target_tf = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
    auto dynamic_target_tf = std::make_shared<tf2_ros::TransformBroadcaster>(node);
    auto wp_navigation_markers_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("waypoints", rclcpp::QoS(1).transient_local());
    auto pose_array_pub = node->create_publisher<nav_msgs::msg::Path>("pose_array", 10);

    auto laser_scan_r_pub = node->create_publisher<sensor_msgs::msg::LaserScan>("laser_scan_r", 10);
    auto laser_scan_f_pub = node->create_publisher<sensor_msgs::msg::LaserScan>("laser_scan_f", 10);
    auto laser_scan_l_pub = node->create_publisher<sensor_msgs::msg::LaserScan>("laser_scan_l", 10);
    auto laser_scan_f_lr_pub = node->create_publisher<sensor_msgs::msg::LaserScan>("laser_scan_f_lr", 10);

    std::unique_ptr<tf2_ros::Buffer> tf_buffer = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    std::shared_ptr<tf2_ros::TransformListener> transform_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    auto bed_detection_status_pub = node->create_publisher<std_msgs::msg::UInt8>("bed_detection_status", 10);
    auto path_generation_status_pub = node->create_publisher<std_msgs::msg::UInt8>("path_generation_status", 10);

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr bed_detection_service =
    node->create_service<std_srvs::srv::Trigger>("start_bed_detection", &bed_detection_cb);

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr generate_flashpoint_service =
    node->create_service<std_srvs::srv::Trigger>("generate_flashpoint", &generate_flashpoint_cb);

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr bed_algo_loop_service =
    node->create_service<std_srvs::srv::Trigger>("bed_algo_loop", &bed_algo_loop_cb);

    // node->declare_parameter("w", 0.472);
    node->declare_parameter("NMS_THRESH", 0.4);
    node->declare_parameter("CONF_THRESH", 0.8);
    node->declare_parameter("dis_thresh", 0.9);
    node->declare_parameter("stride", 10);
    node->declare_parameter("k_dis", 1.0);
    node->declare_parameter("engine_name", "can_ggn_st_johns_os_hospital_bed_512_yolov8s_no_earlystop.engine");
    node->declare_parameter("engine_type", " ");
    node->declare_parameter("onnx_path", " ");
    node->declare_parameter("engine_path", " ");
    node->declare_parameter("svo_path", "HD2K_SN22518123_18-29-57.svo");
    node->declare_parameter("svo_open", false);
    node->declare_parameter("x_plane", 1.0);
    node->declare_parameter("y_plane", 1.0);
    node->declare_parameter("h_active", false);
    node->declare_parameter("v_active", false);

    double NMS_THRESH = node->get_parameter("NMS_THRESH").as_double();
    double CONF_THRESH = node->get_parameter("CONF_THRESH").as_double();
    double dis_thresh = node->get_parameter("dis_thresh").as_double();
    double stride = node->get_parameter("stride").as_int();
    double k_dis = node->get_parameter("k_dis").as_double();
    std::string engine_name = node->get_parameter("engine_name").as_string();
    std::string engine_type = node->get_parameter("engine_type").as_string();
    std::string onnx_path = node->get_parameter("onnx_path").as_string();
    std::string engine_path = node->get_parameter("engine_path").as_string();
    std::string svo_path = node->get_parameter("svo_path").as_string();
    bool svo_open = node->get_parameter("svo_open").as_bool();
    double x_plane = node->get_parameter("x_plane").as_double();
    double y_plane = node->get_parameter("y_plane").as_double();
    bool h_active = node->get_parameter("h_active").as_bool();
    bool v_active = node->get_parameter("v_active").as_bool();

    AlgoUtils algo_utils(static_target_tf, dynamic_target_tf,
    wp_navigation_markers_pub, std::move(tf_buffer), k_dis, stride, 
    laser_scan_r_pub, laser_scan_f_pub, laser_scan_l_pub, laser_scan_f_lr_pub, pose_array_pub, x_plane, y_plane, h_active, v_active,
    path_generation_status_pub);

    // algo_utils.stride = stride;
    


    // if (argc == 1) {
    //     std::cout << "Usage: \n 1. ./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine\n 2. ./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine images:1x3x512x512\n 3. ./yolo_onnx_zed yolov8s.engine <SVO path>" << std::endl;
    //     return 0;
    // }
    
    
    // Check Optim engine first
    if (engine_type.c_str() == "-s") {
        OptimDim dyn_dim_profile;
        Yolo::build_engine(onnx_path, engine_path, dyn_dim_profile);
        std::cout<<"serialize the engine"<<std::endl;
        return 0;
    }

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::Camera zed;
    sl::Pose pose;
    sl::InitParameters init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    // init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    init_parameters.coordinate_units = sl::UNIT::METER;
    init_parameters.depth_maximum_distance = 5;

    if(svo_open)
    {
        std::string zed_opt = svo_path;
        if (zed_opt.find(".svo") != std::string::npos)
        {
            init_parameters.input.setFromSVOFile(zed_opt.c_str());
        }
    }
    
        
    
    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }
    // zed.enablePositionalTracking();
    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.image_sync = true;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_segmentation = true;
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    if (detection_parameters.enable_tracking)
        zed.enablePositionalTracking(); 
    returned_state = zed.enableObjectDetection(detection_parameters);

    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }
    auto camera_config = zed.getCameraInformation().camera_configuration;
    sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
    auto camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;
    // Create OpenGL Viewer
    // GLViewer viewer;
    // viewer.init(argc, argv, camera_info.calibration_parameters.left_cam, true);


    // Creating the inference engine class
    // std::string engine_name = "";
    
    // if (argc > 0)
    //     engine_name = argv[1];
    // else {
    //     std::cout << "Error: missing engine name as argument" << std::endl;
    //     return EXIT_FAILURE;
    // }
    Yolo detector;
    if (detector.init(engine_name)) {
        std::cerr << "Detector init failed!" << std::endl;
        return EXIT_FAILURE;
    }

    auto display_resolution = zed.getCameraInformation().camera_configuration.resolution;
    sl::Mat left_sl, point_cloud;
    cv::Mat left_cv, object_mask_cv;
    sl::Plane plane;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    sl::Pose cam_w_pose;
    cam_w_pose.pose_data.setIdentity();
    std::vector<sl::float3> bbx_3d_laser;
    int cnt = 0;
    cv::Mat outImg;
    int ct = 0;
    while (rclcpp::ok() && zed.grab() == sl::ERROR_CODE::SUCCESS) {

        if(start_bed_detection_)
        {
            if(ct<1)
            {
                std::cerr << "<-- Bed_detection started --->" << std::endl;
            }
            ct+=1;
            // Get image for inference
            zed.retrieveImage(left_sl, sl::VIEW::LEFT);

            // Running inference
            auto detections = detector.run(left_sl, display_resolution.height, display_resolution.width, CONF_THRESH);

            // Get image for display
            left_cv = slMat2cvMat(left_sl);

            // Preparing for ZED SDK ingesting
            std::vector<sl::CustomBoxObjectData> objects_in;
            for (auto &it : detections) {
                sl::CustomBoxObjectData tmp;
                // Fill the detections into the correct format
                tmp.unique_object_id = sl::generate_unique_id();
                tmp.probability = it.prob;
                tmp.label = (int) it.label;
                tmp.bounding_box_2d = cvt(it.box);
                tmp.is_grounded = ((int) it.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
                // others are tracked in full 3D space                
                objects_in.push_back(tmp);

                auto bed_msg = std_msgs::msg::UInt8();
                bed_msg.data = 1;
                bed_detection_status_pub->publish(bed_msg);
            }

            if(objects.object_list.size()<1)
            {
                auto bed_msg = std_msgs::msg::UInt8();
                bed_msg.data = 0;
                bed_detection_status_pub->publish(bed_msg);
            }



            // Send the custom detected boxes to the ZED
            zed.ingestCustomBoxObjects(objects_in);


            // Displaying 'raw' objects
            for (size_t j = 0; j < detections.size(); j++) {
                cv::Rect r = get_rect(detections[j].box);
                cv::rectangle(left_cv, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(left_cv, std::to_string((int) detections[j].label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            


            // Retrieve the tracked objects, with 2D and 3D attributes
            zed.retrieveObjects(objects, objectTracker_parameters_rt);
            // GL Viewer
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::CPU);
            zed.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);
            
            // algo_utils.zed2_3d_bbx_pub(node->get_clock()->now(),objects);
            
            // for (auto data : objects.object_list)
            // {
            //     cv::rectangle(left_cv, cv::Point(data.bounding_box_2d[0][0],data.bounding_box_2d[0][1]), cv::Point(data.bounding_box_2d[2][0],data.bounding_box_2d[2][1]), cv::Scalar(255, 255, 255), 2);
            // }
            
            if(objects.object_list.size() > 0)
            {
                
                if (start_generate_flashpoint_ && bed_algo_loop_)
                {
                    // bbx_3d_laser = objects.object_list[0].bounding_box;
                    algo_utils.pose_bed_detection_algo(node->get_clock()->now(), objects, plane, pose, zed, left_cv, stride, point_cloud, dis_thresh);
                    // std::cout << "<------------------------- okay okay -----------------> "<<cnt<< std::endl;
                }
                    
                
                
            }

            if(svo_open)
            {
                cv::resize(left_cv, outImg, cv::Size(), 0.50, 0.50);
                // cv::imshow("Objects", outImg);
            }
            else
            {
                // cv::imshow("Objescts", left_cv);s
            }
            
            int key = cv::waitKey(1);
            if (key == 'q')
            {
                std::cout << "q key is pressed by the user. Stopping the video" << std::endl;
                break;
            }

            // algo_utils.camera_laser_pub(node->get_clock()->now(), bbx_3d_laser);
            cnt +=1;

        }
        rclcpp::spin_some(node);

        
        // viewer.updateData(point_cloud, objects.object_list, cam_w_pose.pose_data);
        
    }

    // viewer.exit();
    cv::destroyAllWindows();
    left_sl.free();
    objects.object_list.clear();

    // Disable modules
    zed.disableObjectDetection();
    zed.disablePositionalTracking();
    zed.close();
    return EXIT_SUCCESS;
}
