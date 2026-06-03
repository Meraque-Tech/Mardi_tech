#ifndef ROS_UTILS_H
#define ROS_UTILS_H

#include <iostream>
#include <fstream>
#include <signal.h>

#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
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
        // sl::float3 pix_pcl_img_cm_;
        // sl::float4 f_PlaneCoefficients, l_PlaneCoefficients, r_PlaneCoefficients;
        std::string camera_frame = "camera_link";
        std::string map_frame = "map";
        std::string laser_frame = "laser_link";
        std::string base_frame = "base_footprint";
        bool h_active;
        bool v_active;

        std::vector<geometry_msgs::msg::PoseStamped> tf2_pose_vec_;
        sl::Mat zed2_img;

        // bed dia = 2.3m (1,3), w= 1.1m (4,7), l = 2.2m (0,1), h=1m (2,6)
        //  1 ------ 2
        // / l       /| h  
        // 0 ------ 3 |
        // | Object | 6
        // |        |/
        // 4 ------ 7
        //    w

        float bed_H = 1.0;
        float bed_W = 1.1;
        float bed_L = 2.25;
        float bed_dia = 2.3;
        sl::float3 nall_strc = {0.0, 0.0, 0.0};
        int cnt = 0;
        cv::Point l_up, ll_up, l_down, ll_down, r_up, rr_up, r_down, rr_down;
        cv::Mat img_;
        sl::float4 plane_eq_; 
        sl::Mat point_cloud_;
        double x_plane, y_plane;

        // float nall = 0.0;

    public:
        void send_static_target_tf(rclcpp::Time now, std::string parent_frame, std::string child_frame, double x, double y, double z, double th_x, double th_y, double th_z);
        void send_dynamic_target_tf(rclcpp::Time now, std::string parent_frame, std::string child_frame, double x, double y, double z, double th_x, double th_y, double th_z);
        sl::float3 midpoint_of_rectangle(const std::vector<sl::float3>& vertices);
        std::vector<sl::float3> zed2_3d_bbx_vec(sl::Objects &objects);
        void bbx_3d_tf_pub(rclcpp::Time now, const std::vector<sl::float3>& bbx_3d);
        std::tuple<sl::float3, sl::float3, sl::float3, sl::float3> mid_points_lfr(rclcpp::Time now, std::vector<sl::float3> &bbx_3d);
        std::tuple<float, float> find_HV_slop(float x1, float y1, float x2, float y2);
        std::tuple<float, float> WP_creation(float gth_m, float gpx, float gpy, int type_wp, float k);
        float dis_from_pane(sl::float4 &plane_eq,sl::float3 &pcl);
        float euclidean_distance_onearg(const sl::float3& point1);
        float euclidean_distance_twoarg(const sl::float3& point1, const sl::float3& point2);
        sl::float4 findPlaneCoefficients(sl::float3 pix_pcl_P, sl::float3 pix_pcl_Q, sl::float3 pix_pcl_R);
        sl::float3 find_foot_point_on_plane(const sl::float4 &plane_eq, const sl::float3 &pcl);
        std::tuple<bool, sl::float4> find_plane_eq(sl::Plane &plane, sl::Pose &pose, sl::Camera &zed);
        void zed2_3d_bbx_pub(rclcpp::Time now,sl::Objects &objects);
        std::tuple<float, float> clsuter_find(cv::Mat &img,cv::Point &topLeft, cv::Point &topRight, sl::float4 &plane_eq, sl::Mat &point_cloud);
        std::tuple<float, size_t, float, size_t> minmax_idx(const std::vector<auto>& v);
        sl::float3 pix_pcl(int pix_x, int pix_y, sl::Mat &point_cloud, cv::Mat &image);
        void updateWpNavigationMarkers(std::vector<geometry_msgs::msg::PoseStamped> &poses_);
        void resetUniqueId();
        int getUniqueId();
        geometry_msgs::msg::TransformStamped tf_sub(std::string parent_frame, std::string child_frame);
        geometry_msgs::msg::PoseStamped tf2_pose(rclcpp::Time now, geometry_msgs::msg::TransformStamped &tf_msg);
        std::vector<sl::float3> inner_2d_points_gen(sl::float3 &init_coordinates, sl::float3 &final_coordinates, int NOP);
        std::vector<sl::float2> inner_2d_polar_points_gen(std::vector<sl::float3> &inner_2d_points);
        sensor_msgs::msg::LaserScan polar_to_laser_scan(rclcpp::Time now, const std::vector<sl::float2> &polar_points);
        void camera_laser_pub(rclcpp::Time now, std::vector<sl::float3> &bbx_3d);
        void pose_bed_detection_algo(rclcpp::Time now,
            sl::Objects &objects, sl::Plane &plane, sl::Pose &pose, sl::Camera &zed,
            cv::Mat &image, int stride, sl::Mat &point_cloud, double dis_thresh);
        void find_object_mask(rclcpp::Time now, cv::Mat &img, std::vector<sl::uint2> object_2Dbbox, sl::float4 &plane_eq, sl::Mat &point_cloud);
        std::tuple<cv::Point, size_t, cv::Point, size_t> minmax_idx_pix(const std::vector<cv::Point>& v);
        int euclidean_distance_twoarg_pix(const cv::Point &point1, const cv::Point &point2);
        std::tuple<sl::float3, int> min_cluster_pix_mask_find(cv::Mat &img, std::vector<int>& cluster_points_pix_top ,std::vector<cv::Point>& cluster_points_pix_top_uv, 
        cv::Point& bbx_up, std::vector<sl::float3> &pix_pcl_top, sl::float3 &pix_pcl_img_cm);
        double Cosin(double a, double b, double c);
        double gamma_sign_correction(float gamma, float x);
        std::tuple<float, float> find_global_PQR(const sl::float3& img_frame, const sl::float3& bed_PQR);
        void find_object_mask_convexhull(rclcpp::Time now, cv::Mat &img, std::vector<sl::uint2> object_2Dbbox, sl::float4 &plane_eq, sl::Mat &point_cloud);
        std::tuple<sl::float3, sl::float3, int> d_lmr_logic(std::vector<sl::float3> &pix_pcl_top_min_lmr, std::vector<double> &gamma_lmr);
        sl::float3 mid_point_finder(sl::float3 &lef_bed_pcl, sl::float3 &right_bed_pcl);
        std::tuple<double, double, double> find_gamma_lmr(const sl::float3 &pix_pcl_img_cm, const sl::float3 &pix_pcl_top_min_left ,const sl::float3 &pix_pcl_top_min_mid, const sl::float3 &pix_pcl_top_min_right);
        void gen_wp_logic(rclcpp::Time now, sl::float3 main_l, sl::float3 main_r, int type);
        std::tuple<sl::float3, sl::float3, sl::float3, double, double> gen_wp_all(rclcpp::Time now, sl::float3 main_l, sl::float3 main_r, float bed_dimension, int type_wp);
        std::tuple<float, float, bool> left_space_3d_finder();
        std::tuple<float, float, bool> right_space_3d_finder();
        std::tuple<sl::float3, int> plane_cluster_mask_find(cv::Mat &img, std::vector<int> &cluster_points_pix_top ,
                                                            std::vector<cv::Point> &cluster_points_pix_top_uv, 
                                                            std::vector<sl::float3> &pix_pcl_top);
        void wp_vec(rclcpp::Time now, std::string parent_frame, std::string child_frame);
        void wp_vec_pub(rclcpp::Time now, std::vector<geometry_msgs::msg::PoseStamped> &tf2_pose_vec_);
        sl::float3 tf_sub_to_eular_angles(std::string parent_frame, std::string child_frame);
};

#endif // ROS_UTILS_H

