#pragma once

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <prime_msgs/msg/rtk_status.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>


struct Pose2D
{
    double x;
    double y;
    double yaw;
};



class DualRTKHeadingNode : public rclcpp::Node
{
public:
    DualRTKHeadingNode();

private:
    /* ---- Callbacks ---- */
    void computeHeading();
    void processAndPublishOdom(
        const nav_msgs::msg::Odometry::SharedPtr msg);
    
    void processAndPublishEkf(
        const nav_msgs::msg::Odometry::SharedPtr msg);

    /* ---- Helpers ---- */
    bool validFix(const sensor_msgs::msg::NavSatFix &fix);
    void initPointMarker();

    geometry_msgs::msg::TransformStamped publishTF(
                                const rclcpp::Time &stamp,
                                double x, double y, double z,
                                double roll, double pitch, double yaw,
                                const std::string &parent_frame,
                                const std::string &child_frame);
    
    
    void publishPose(
        const rclcpp::Time &stamp,
        double x, double y, double yaw,
        const std::string &frame_id,
        const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr &pub);
    
    
    void updateAndPublishPath(
        const rclcpp::Time &stamp,
        double x, double y, double yaw,
        const std::string &frame_id,
        nav_msgs::msg::Path &path,
        const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pub);
    
    void publishOdom(
        const rclcpp::Time &stamp,
        double x, double y, double z,
        double roll, double pitch, double yaw,
        double vx, double vy, double wz,
        const std::string &frame_id,
        const std::string &child_frame_id,
        const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pub);
    
    double odomDistance2D(const nav_msgs::msg::Odometry& a,
                      const nav_msgs::msg::Odometry& b);

    Pose2D rotatePose2D(double x, double y, double yaw, double yaw_offset);

    std::tuple<double, double, double> getXYYawFromOdom(const nav_msgs::msg::Odometry & odom);
    nav_msgs::msg::Odometry makeOdomFromXYYaw(double x, double y, double yaw);
    geometry_msgs::msg::PoseStamped odomToPoseStamped(const nav_msgs::msg::Odometry & odom);


    /* ---- Subscribers ---- */
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_left_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_right_;
    rclcpp::Subscription<prime_msgs::msg::RtkStatus>::SharedPtr sub_left_status_;
    rclcpp::Subscription<prime_msgs::msg::RtkStatus>::SharedPtr sub_right_status_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_ekf_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr odom_path_pub_ekf_;

    /* ---- Publishers ---- */
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_lio_offest_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_lio_offest_;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr points_pub_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    /* ---- IMU ---- */
    // void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // void lidarCallback(sensor_msgs::msg::PointCloud2::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;


    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_lidar_;
    

    /* ---- Messages ---- */
    nav_msgs::msg::Path path_msg_;
    nav_msgs::msg::Path path_rtk_;
    nav_msgs::msg::Path path_rtk_lio_offset;
    nav_msgs::msg::Path odom_path_pub_ekf_hb;

    visualization_msgs::msg::Marker points_marker_;

    /* ---- State ---- */
    sensor_msgs::msg::NavSatFix left_fix_;
    sensor_msgs::msg::NavSatFix right_fix_;

    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr mid_fix_pub_;
    
    bool left_received_{false};
    bool right_received_{false};
    std::string left_carrier_soln_{"NO FIX"};
    std::string right_carrier_soln_{"NO FIX"};

    double yaw_{0.0};
    double map_origin_x_{0.0};
    double map_origin_y_{0.0};
    bool map_origin_set_{false};

    gps_utils::GpsConverter converter_;

    double motion_threshold_xy_;
    std::string map_frame_{"map"};
    std::string odom_frame_{"odom"};
    std::string base_frame_{"rtk_pub"};

    gps_utils::ECEF init_rover_ecef_;
    bool init_rover_ecef_set_{false};
    bool init_rover_enu_set_{false};
    bool enu_mid_first_{false};
    gps_utils::ENU enu_mid_first_vale_;

    bool is_learned_yaw_offset_{false};

    double learned_yaw_offset{0.0};


    bool has_initial_odom_ {false};
    nav_msgs::msg::Odometry initial_odom_;
    nav_msgs::msg::Odometry current_odom;
    nav_msgs::msg::Odometry lio_rtk_odom;
    double dist_lio_from_start;
    double yaw_offset_sf {0.0};
    
    double position_offset_x_ {0.0};
    double position_offset_y_ {0.0};
};
