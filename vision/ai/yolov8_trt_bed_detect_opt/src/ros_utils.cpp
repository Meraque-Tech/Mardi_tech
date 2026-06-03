#include "ros_utils.h"
// #include "viz_utils.cpp"
// #include "tf2_utils.cpp"
// #include "scan_gen.cpp"
// #include  "math_equations.cpp"
// #include "bed_3d_gen.cpp"
// #include "gamma_logic.cpp"
// #include "plane_finder.cpp"
// #include "way_points.cpp"
// #include "cluster.cpp"
// #include "convex_hull.cpp"

AlgoUtils::AlgoUtils(std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_target_tf_,
        std::shared_ptr<tf2_ros::TransformBroadcaster> dynamic_target_tf_,
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr wp_navigation_markers_pub_,
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_,
        float k_dis, double stride_,
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_r_pub_,
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_f_pub_,
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_l_pub_,
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_f_lr_pub_,
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_array_pub_, double x_plane_, double y_plane_, bool h_active_, bool v_active_,
        rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr path_generation_status_pub_) {

    // Constructor implementation here
    static_target_tf = static_target_tf_;
    dynamic_target_tf = dynamic_target_tf_;
    k = k_dis;
    stride = stride_;
    wp_navigation_markers_pub = wp_navigation_markers_pub_;
    tf_buffer = std::move(tf_buffer_);
    laser_scan_r_pub = laser_scan_r_pub_;
    laser_scan_f_pub = laser_scan_f_pub_;
    laser_scan_l_pub = laser_scan_l_pub_;
    laser_scan_f_lr_pub = laser_scan_f_lr_pub_;
    pose_array_pub = pose_array_pub_;
    x_plane = x_plane_;
    y_plane = y_plane_;
    h_active = h_active_;
    v_active = v_active_;
    path_generation_status_pub = path_generation_status_pub_;


}


void AlgoUtils::pose_bed_detection_algo(rclcpp::Time now,
    sl::Objects &objects, sl::Plane &plane, sl::Pose &pose, sl::Camera &zed,
    cv::Mat &image, int stride, sl::Mat &point_cloud, double dis_thresh)
{
    for (auto object : objects.object_list) 
    {
        std::vector<sl::uint2> object_2Dbbox = object.bounding_box_2d; // Get the 2D bounding box of the object
        std::vector<sl::float3> bbx_3d = object.bounding_box;

        auto [is_plane_live, plane_eq] = find_plane_eq(plane, pose, zed);

        cv::Point left_up = cv::Point(object_2Dbbox[0][0],object_2Dbbox[0][1]);
        cv::Point left_down = cv::Point(object_2Dbbox[3][0],object_2Dbbox[3][1]);
        cv::Point right_up = cv::Point(object_2Dbbox[1][0],object_2Dbbox[1][1]);
        cv::Point right_down = cv::Point(object_2Dbbox[2][0],object_2Dbbox[2][1]);

        if (is_plane_live == 1)
        {
            // std::cout << "<------------------------- okay okay -----------------> "<< std::endl;
            // find_object_mask(now, image, object_2Dbbox, plane_eq, point_cloud);
            find_object_mask_convexhull(now, image, object_2Dbbox, plane_eq, point_cloud);

        }
        
        
    }
}