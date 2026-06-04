#ifndef ROS_UTILS_H
#define ROS_UTILS_H

#include <iostream>
#include <fstream>
#include <signal.h>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/u_int8.hpp"

#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include "tf2/LinearMath/Quaternion.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "sensor_msgs/msg/laser_scan.hpp"
#include <bits/stdc++.h>

class AlgoUtils {
    public:
        AlgoUtils(std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_target_tf_,
        std::shared_ptr<tf2_ros::TransformBroadcaster> dynamic_target_tf_,
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr wp_navigation_markers_pub_,
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_,
        float k_dis, double stride_,
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_r_pub_,
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_f_pub_,
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_l_pub_,
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_f_lr_pub_,
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_array_pub_, double x_plane_, double y_plane_, bool h_active_, bool v_active_,
        rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr path_generation_status_pub_);

    private:
        std::unique_ptr<tf2_ros::Buffer> tf_buffer;
        std::shared_ptr<tf2_ros::TransformListener> transform_listener{nullptr};
        std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_target_tf;
        std::shared_ptr<tf2_ros::TransformBroadcaster> dynamic_target_tf;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr wp_navigation_markers_pub;

        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_r_pub;
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_f_pub;
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_l_pub;
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_f_lr_pub;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_array_pub;
        rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr path_generation_status_pub;

        bool mObjDetTracking = true;
        float dis_thresh = 1.0;
        float k;
        double stride;
        double unique_id;
        std::string camera_frame = "camera_link";
        std::string map_frame = "map";
        std::string laser_frame = "laser_link";
        std::string base_frame = "base_footprint";
        bool h_active;
        bool v_active;

        std::vector<geometry_msgs::msg::PoseStamped> tf2_pose_vec_;
        double x_plane, y_plane;

    public:
        void send_static_target_tf(rclcpp::Time now, std::string parent_frame, std::string child_frame, double x, double y, double z, double th_x, double th_y, double th_z);
        void send_dynamic_target_tf(rclcpp::Time now, std::string parent_frame, std::string child_frame, double x, double y, double z, double th_x, double th_y, double th_z);
        void updateWpNavigationMarkers(std::vector<geometry_msgs::msg::PoseStamped> &poses_);
        void resetUniqueId();
        int getUniqueId();
        geometry_msgs::msg::TransformStamped tf_sub(std::string parent_frame, std::string child_frame);
        geometry_msgs::msg::PoseStamped tf2_pose(rclcpp::Time now, geometry_msgs::msg::TransformStamped &tf_msg);
        void wp_vec_pub(rclcpp::Time now, std::vector<geometry_msgs::msg::PoseStamped> &tf2_pose_vec_);
};

#endif // ROS_UTILS_H
