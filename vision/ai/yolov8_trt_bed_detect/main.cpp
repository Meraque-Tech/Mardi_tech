#include "main_fun.cpp"
#include "function_ex.cpp"
#include <sl/Camera.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>


#include <signal.h>
#include <stdio.h>
// #include <geometry_msgs/msg/transform_stamped.hpp>
#include "std_srvs/srv/trigger.hpp"


void sig_handler(int signal){
    std::cout << "\nCtrl+C pressed. Exiting..." << std::endl;
    exit(0); 

}

void bed_detection_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
          std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    // int bed_detection_data = request->start_bed_detection;
    start_bed_detection_ = 1;
    response->success = 1;
    response->message = "bed detection started";
    bool bed_detection_fb_ = 1;
    std::cout << "\nbed_detection_fb_ --> "<<bed_detection_fb_<< std::endl;
    

}

void bed_detection_plane_algo_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
          std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    start_bed_detection_plane_algo_ = 1;
    response->success = 1;
    response->message = "bed detection plane algo started";
}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    signal (SIGINT,sig_handler);
    node = rclcpp::Node::make_shared("yolov8_trt");
    auto conf_pub = node->create_publisher<std_msgs::msg::Float32>("conf", 10);
    float32_multi_array_f front_pose_pub = node->create_publisher<float32_multi_array>("front_pose", 10);
    float32_multi_array_f left_pose_pub = node->create_publisher<float32_multi_array>("left_pose", 10);
    float32_multi_array_f right_pose_pub = node->create_publisher<float32_multi_array>("right_pose", 10);
    float32_multi_array_f img_pose_pub = node->create_publisher<float32_multi_array>("img_pose", 10);
    static_tf2 static_target_tf = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
    viz_msg_f wp_navigation_markers_pub_ = node->create_publisher<viz_msg>("waypoints", rclcpp::QoS(1).transient_local());

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr bed_detection_service =
    node->create_service<std_srvs::srv::Trigger>("bed_detection", &bed_detection_cb);

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr bed_detection_plane_algo_service =
    node->create_service<std_srvs::srv::Trigger>("bed_detection_plane_algo", &bed_detection_plane_algo_cb);
    // LH_pose_stamped_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>("LH_pose_stamped", 10);
    // RH_pose_stamped_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>("RH_pose_stamped", 10);
    // FH_pose_stamped_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>("FH_pose_stamped", 10);

    left_pose_array_pub = node->create_publisher<nav_msgs::msg::Path>("left_pose_array", 10);
    right_pose_array_pub = node->create_publisher<nav_msgs::msg::Path>("right_pose_array", 10);
    font_pose_array_pub = node->create_publisher<nav_msgs::msg::Path>("font_pose_array", 10);
    bed_detection_nav2_call_pub = node->create_publisher<std_msgs::msg::UInt8>("bed_detection_status", 10);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // node->declare_parameter("conf", 0.80);
    // conf_ = node->get_parameter("conf").as_double();

    navigation_action_client_ =
        rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(
        node,
        "navigate_to_pose");
        
    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.coordinate_units = sl::UNIT::METER;

    // Open the camera
    sl::ERROR_CODE err = zed.open(init_parameters);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Failed to open the ZED camera. Error code: " << err << std::endl;
        return 1;
    }

    err = zed.enablePositionalTracking();
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Failed to enable positional tracking. Error code: " << err << std::endl;
        zed.close();
        return 1;
    }

    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    // detection_parameters.enable_segmentation = true; // designed to give person pixel mask
    // detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    err = zed.enableObjectDetection(detection_parameters);
    if (err != sl::ERROR_CODE::SUCCESS) {
        // print("enableObjectDetection", err, "\nExit program.");
        std::cout << "Failed to enableObjectDetection. Error code: " << err << std::endl;
        zed.close();
        return EXIT_FAILURE;
    }

    // sl::Mat image, point_cloud;
    int i = 0;
    sl::Mat left_sl, depth_image, point_cloud;
    cv::Mat left_cv_bgra, left_cv_bgr;
    sl::Pose pose;
    sl::Plane plane;
    sl::Mesh mesh;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    sl::CameraParameters zedCameraParams;
    // cv::Mat left_cv_bgra, left_cv_bgr;
    float tolerance = 0.01f;

    cudaSetDevice(kGpuId);
    std::string wts_name = "";
    std::string engine_name = "";
    std::string img_dir;
    std::string sub_type = "";
    std::string cuda_post_process="";
    int model_bboxes;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, sub_type, cuda_post_process)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolov8_trt -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov8_trt -d [.engine] ../samples  [c/g]// deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(wts_name, engine_name, sub_type);
        return 0;
    }

    // Deserialize the engine from file
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getBindingDimensions(1);
    model_bboxes = out_dims.d[0];
    // Prepare cpu and gpu buffers
    float *device_buffers[2];
    float *output_buffer_host = nullptr;
    float *decode_ptr_host=nullptr;
    float *decode_ptr_device=nullptr;

    // Read images from directory
    // std::vector<std::string> file_names;
    // if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    //     std::cerr << "read_files_in_dir failed." << std::endl;
    //     return -1;
    // }

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host, &decode_ptr_device, cuda_post_process);
    
    try {
        if(std::string(argv[5]) == "-conf")
        {
            conf_score_value = std::stof(argv[6]); // Convert the argument to a floating-point value
            std::cout<<"conf_score_value -> "<<conf_score_value<<std::endl;
        }
        if(std::string(argv[7]) == "-kh")
        {
            k_h = std::stof(argv[8]); // Convert the argument to a floating-point value
            std::cout<<"k_h_value -> "<<k_h<<std::endl;
        }
        if(std::string(argv[9]) == "-kv")
        {
            k_v = std::stof(argv[10]); // Convert the argument to a floating-point value
            std::cout<<"k_v_value -> "<<k_v<<std::endl;
        }
        if(std::string(argv[11]) == "-d_thresh")
        {
            distance_thresh = std::stof(argv[12]); // Convert the argument to a floating-point value
            std::cout<<"distance_thresh -> "<<distance_thresh<<std::endl;
        }
        if(std::string(argv[13]) == "-font_dis")
        {
            font_dis = std::stof(argv[14]); // Convert the argument to a floating-point value
            std::cout<<"font_dis -> "<<font_dis<<std::endl;
        }
        
    } catch (const std::exception &e) {
        std::cerr << "Error parsing conf_score: " << e.what() << std::endl;
        return -1;
    }

    // batch predict
    while(rclcpp::ok() && zed.grab() == sl::ERROR_CODE::SUCCESS)
    {
        if (start_bed_detection_ == 1)
        {
            zed.retrieveImage(left_sl, sl::VIEW::LEFT);
            zed.retrieveImage(depth_image, sl::VIEW::DEPTH);
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA);

            // Preparing inference
            left_cv_bgra = slMat2cvMat(left_sl);
            cv::cvtColor(left_cv_bgra, left_cv_bgr, cv::COLOR_BGRA2BGR);

            if (left_cv_bgr.empty()) continue;

            // Get a batch of images
            std::vector<cv::Mat> img_batch {left_cv_bgr};

            
            // std::vector<std::string> img_name_batch;
            // for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
            //     cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
            //     img_batch.push_back(img);
            //     img_name_batch.push_back(file_names[j]);
            // }
            // Preprocess
            cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
            // Run inference
            infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host, decode_ptr_device, model_bboxes, cuda_post_process);
            std::vector<std::vector<Detection>> res_batch;
            if (cuda_post_process == "c") {
                // NMS
                batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
            } else if (cuda_post_process == "g") {
                //Process gpu decode and nms results
                batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
            }
            
            // Draw bounding boxes
            
            // Preparing for ZED SDK ingesting
            for (size_t i = 0; i < img_batch.size(); i++) {
                auto &res = res_batch[i];
                cv::Mat img = img_batch[i];

                // cv::imshow("detected images",img);
                if(res.size() > 0)
                {
                    for (auto &it : res) 
                    {
                        
                        std::cout << "it.conf --->"<< it.conf<< std::endl;
                        std::cout << "------------------------------------------------------" << std::endl;
                        
                        if(it.conf > conf_score_value)
                        {
                            auto conf_msg = std::make_shared<std_msgs::msg::Float32>();
                            conf_msg->data = it.conf;
                            conf_pub->publish(*conf_msg);
                            auto bed_msg = std_msgs::msg::UInt8();
                            bed_msg.data = 1;
                            bed_detection_nav2_call_pub->publish(bed_msg);
                            bed_detection_fb_ = 1;
                            std::cout << "Detected bed: "<< std::endl;
                        }
                        else
                        {
                            auto bed_msg = std_msgs::msg::UInt8();

                            bed_msg.data = 0;
                            bed_detection_nav2_call_pub->publish(bed_msg);
                            bed_detection_fb_ = 0;
                            // break;

                        }

                        
                    }
                    
                }
                else
                {
                    auto bed_msg = std_msgs::msg::UInt8();

                    bed_msg.data = 0;
                    bed_detection_nav2_call_pub->publish(bed_msg);
                    bed_detection_fb_ = 0;
                    // break;

                }
                
            }
            





            


            // zed.retrieveObjects(objects, objectTracker_parameters_rt);

            // draw_bbox(img_batch, res_batch);
            // cv::imshow("detected images", left_cv_bgr);

            if(start_bed_detection_plane_algo_ == 1)
            {
                sl::POSITIONAL_TRACKING_STATE tracking_state = zed.getPosition(pose);
                if (tracking_state == sl::POSITIONAL_TRACKING_STATE::OK) 
                {
                    sl::Transform resetTrackingFloorFrame;
                    sl::ERROR_CODE plane_err = zed.findFloorPlane(plane, resetTrackingFloorFrame);

                    if (plane_err != sl::ERROR_CODE::SUCCESS)
                    {
                        std::cout << "No plane found" << std::endl;
                    }

                    if(plane_err == sl::ERROR_CODE::SUCCESS)
                    {
                        sl::float4 plane_eq = plane.getPlaneEquation();
                        rclcpp::Time now = node->get_clock()->now();
                        plane_detectio_algo(img_batch, 
                        res_batch,point_cloud, 
                        plane_eq, 
                        zedCameraParams,
                        front_pose_pub,
                        left_pose_pub,
                        right_pose_pub,
                        img_pose_pub,
                        static_target_tf,
                        wp_navigation_markers_pub_,
                        now
                        );
                        
                        // sofa_plane_detectio_algo(img_batch, res_batch,point_cloud, plane_eq, zedCameraParams);
                        // draw_bbox(img_batch, res_batch);
                        // cv::imshow("detected images", left_cv_bgr);

                    }
                }
            }
            


            
            


            
            // Send the custom detected boxes to the ZED
            // for (size_t i = 0; i < img_batch.size(); i++) 
            // {
            //     auto &res = res_batch[i];
            //     cv::Mat img = img_batch[i];
            //     cv::imshow("detected images", img);

            // }
            
            int key = cv::waitKey(1);
            if (key == 'q')
            {
                std::cout << "q key is pressed by the user. Stopping the video" << std::endl;
                break;
            }
        }

        
        rclcpp::spin_some(node);

        
        // Save images
        // for (size_t j = 0; j < img_batch.size(); j++) {
        //     cv::imwrite("_" + img_name_batch[j], img_batch[j]);
        // }
    }
    

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    CUDA_CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    delete[] output_buffer_host;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
    cv::destroyAllWindows();
    zed.disablePositionalTracking();
    zed.close();
    

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < kOutputSize; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;
    // rclcpp::spin(node);
    return 0;
}

