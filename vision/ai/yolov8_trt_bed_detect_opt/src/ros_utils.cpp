#include "ros_utils.h"

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
