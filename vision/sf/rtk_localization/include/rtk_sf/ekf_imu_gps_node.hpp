#ifndef EKF_FUSION_NODE_HPP
#define EKF_FUSION_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Dense>
#include <deque>

/**
 * @brief Extended Kalman Filter for fusing IMU and GPS data
 * 
 * State vector (15D):
 * [0:3]   position (x, y, z) in local ENU frame
 * [3:6]   velocity (vx, vy, vz)
 * [6:9]   orientation (roll, pitch, yaw) - Euler angles
 * [9:12]  accelerometer bias (bax, bay, baz)
 * [12:15] gyroscope bias (bgx, bgy, bgz)
 */
class EKFFusionNode : public rclcpp::Node
{
public:
    EKFFusionNode();

private:
    // ========== Callbacks ==========
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void gpsCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
    
    // ========== EKF Core Functions ==========
    void initializeEKF();
    void predictIMU(const sensor_msgs::msg::Imu::SharedPtr msg, double dt);
    void updateGPS(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
    void publishState();
    
    // ========== Helper Functions ==========
    Eigen::Matrix3d eulerToRotationMatrix(double roll, double pitch, double yaw);
    Eigen::Vector3d rotationMatrixToEuler(const Eigen::Matrix3d& R);
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v);
    Eigen::Vector3d llhToENU(double lat, double lon, double alt);
    bool isGPSValid(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
    
    // ========== ROS2 Subscribers ==========
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
    
    // ========== ROS2 Publishers ==========
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // ========== EKF State ==========
    Eigen::VectorXd x_;      // State vector (15x1)
    Eigen::MatrixXd P_;      // Covariance matrix (15x15)
    Eigen::MatrixXd Q_;      // Process noise covariance (15x15)
    Eigen::MatrixXd R_gps_;  // GPS measurement noise covariance (3x3)
    
    // ========== Reference Frame ==========
    bool origin_set_;
    double origin_lat_;
    double origin_lon_;
    double origin_alt_;
    
    // ========== Timing ==========
    rclcpp::Time last_imu_time_;
    bool initialized_;
    
    // ========== Parameters ==========
    double acc_noise_;
    double gyro_noise_;
    double acc_bias_noise_;
    double gyro_bias_noise_;
    double gps_noise_xy_;
    double gps_noise_z_;
    
    std::string world_frame_;
    std::string base_frame_;
    bool publish_tf_;
    
    // ========== Constants ==========
    static constexpr double GRAVITY = 9.81;
    static constexpr double DEG_TO_RAD = M_PI / 180.0;
    static constexpr int STATE_SIZE = 15;
};

#endif // EKF_FUSION_NODE_HPP