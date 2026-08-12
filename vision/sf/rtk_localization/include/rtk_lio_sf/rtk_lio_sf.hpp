#ifndef RTK_IMU_SF_HPP
#define RTK_IMU_SF_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <cmath>

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
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_left_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_right_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_ekf_;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr mid_fix_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_lio_offest_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_lio_offest_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr odom_path_pub_ekf_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr points_pub_;

    // TF Broadcaster
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // RTK Fix messages
    sensor_msgs::msg::NavSatFix left_fix_;
    sensor_msgs::msg::NavSatFix right_fix_;
    bool left_received_ = false;
    bool right_received_ = false;

    // GPS Converter
    gps_utils::GpsConverter converter_;
    bool init_rover_enu_set_ = false;

    // Odometry
    nav_msgs::msg::Odometry current_odom;
    nav_msgs::msg::Odometry lio_rtk_odom;

    // Calibration state
    bool offset_calibrated_ = false;
    double learned_yaw_offset_ = 0.0;
    double position_offset_x_ = 0.0;
    double position_offset_y_ = 0.0;

    // Initial poses for calibration
    nav_msgs::msg::Odometry initial_lio_odom_;
    gps_utils::ENU initial_rtk_enu_;
    bool has_initial_lio_ = false;
    bool has_initial_rtk_ = false;

    // Motion tracking
    double dist_lio_from_start = 0.0;
    double motion_threshold_xy_ = 0.3;

    // Path messages
    nav_msgs::msg::Path path_rtk_;
    nav_msgs::msg::Path path_rtk_lio_offset;
    nav_msgs::msg::Path odom_path_pub_ekf_hb;

    // Marker
    visualization_msgs::msg::Marker points_marker_;

    // Callbacks
    void computeHeading();
    void processAndPublishOdom(const nav_msgs::msg::Odometry::SharedPtr msg);
    void processAndPublishEkf(const nav_msgs::msg::Odometry::SharedPtr msg);

    // Helper functions
    bool validFix(const sensor_msgs::msg::NavSatFix &fix);
    void initPointMarker();
    
    geometry_msgs::msg::PoseStamped odomToPoseStamped(
        const nav_msgs::msg::Odometry & odom);
    
    nav_msgs::msg::Odometry makeOdomFromXYYaw(
        double x, double y, double yaw);
    
    std::tuple<double, double, double> getXYYawFromOdom(
        const nav_msgs::msg::Odometry & odom);
    
    Pose2D rotatePose2D(
        double x, double y, double yaw, double yaw_offset);
    
    double odomDistance2D(
        const nav_msgs::msg::Odometry & odom1,
        const nav_msgs::msg::Odometry & odom2);
    
    void publishOdom(
        rclcpp::Time stamp,
        double x, double y, double z,
        double roll, double pitch, double yaw,
        double vx, double vy, double omega,
        const std::string & frame_id,
        const std::string & child_frame_id,
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub);
    
    void updateAndPublishPath(
        rclcpp::Time stamp,
        double x, double y, double yaw,
        const std::string & frame_id,
        nav_msgs::msg::Path & path_msg,
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub);
    
    geometry_msgs::msg::TransformStamped publishTF(
        rclcpp::Time stamp,
        double x, double y, double z,
        double roll, double pitch, double yaw,
        const std::string & frame_id,
        const std::string & child_frame_id);
};

#endif // RTK_IMU_SF_HPP