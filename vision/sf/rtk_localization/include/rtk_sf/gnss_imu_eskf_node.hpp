#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/nav_sat_status.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <gps_utils/gps_utils.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <deque>
#include <string>
#include <vector>

/*
 * Error-State Kalman Filter (ESKF) fusing:
 *   - /imu/data      (sensor_msgs/Imu)        complementary-filter initial
 *                                              attitude + high-rate prediction
 *   - /gnss/pvt      (std_msgs/String, JSON)  position and motion-heading update
 *                                              (same payload as
 *                                              gnss_pvt_enu_odom.cpp: lat/lon/alt/fix/carr/sats/hacc/vacc)
 *   - /gnss/rtk_status (std_msgs/Bool)        informational only (logged/published)
 *
 * Nominal state (integrated every IMU tick, quaternion-based -> no gimbal lock):
 *   p (3) position ENU of the IMU/body origin, v (3) velocity ENU (body frame origin),
 *   q (4) orientation (body->ENU), ab (3) accel bias, gb (3) gyro bias
 *
 * Error state (15-dim, corrected on each GNSS position/heading update):
 *   [dp(3), dv(3), dtheta(3), dab(3), dgb(3)]
 *
 * GNSS antenna is offset from the IMU/body origin by `gnss_lever_arm_` (body frame).
 * The predicted antenna position used in the measurement model is
 *   p_antenna = p + q * gnss_lever_arm_
 * so the lever arm is compensated for both static offset and rotation-induced
 * displacement (the antenna sweeps an arc during turns).
 *
 * GNSS is assumed to arrive with negligible/consistent latency relative to the
 * IMU rate (no timestamp is available on /gnss/pvt) -- the correction is applied
 * against whatever the nominal state is at the moment the message is processed.
 * If the upstream GNSS source starts publishing a timestamp, that should be used
 * with an IMU state buffer + rollback/replay instead of this simplification.
 */
class GnssImuEskfNode : public rclcpp::Node
{
public:
  GnssImuEskfNode();

private:
  static constexpr int STATE_SIZE = 15;
  static constexpr double GRAVITY = 9.80665;
  static constexpr size_t STATIONARY_INIT_SAMPLES = 100;

  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void pvtCallback(const std_msgs::msg::String::SharedPtr msg);
  void motionPosDeadbandCallback(const std_msgs::msg::Float32::SharedPtr msg);

  bool accumulateStationaryInit(const sensor_msgs::msg::Imu::SharedPtr msg);

  void predict(const Eigen::Vector3d & acc_meas, const Eigen::Vector3d & gyro_meas, double dt);
  void updatePosition(const Eigen::Vector3d & pos_meas, const Eigen::Matrix3d & R_pos);
  void updateHeading(double heading, double heading_variance);
  void injectErrorState(const Eigen::VectorXd & dx);

  void publishState(const rclcpp::Time & stamp);
  void publishGnssMarker(const Eigen::Vector3d & pos_meas, const rclcpp::Time & stamp);
  void publishGnssOnlyPath(const Eigen::Vector3d & pos_meas, const rclcpp::Time & stamp);
  void predictImuOnly(const Eigen::Vector3d & acc_meas, const Eigen::Vector3d & gyro_meas, double dt);
  void publishImuOnlyPath(const rclcpp::Time & stamp);
  void publishGnssOnlyOdom(const Eigen::Vector3d & pos_meas, const rclcpp::Time & stamp);
  void publishImuOnlyOdom(const Eigen::Vector3d & pos_meas, const rclcpp::Time & stamp);
  void publishMotionState(const Eigen::Vector3d & pos, const rclcpp::Time & stamp);
  void publishMotionStateRawGnss(const Eigen::Vector3d & pos_meas, const rclcpp::Time & stamp);

  static bool rtkStateFromCarr(int carr, bool & is_fixed, bool & is_float);
  static double wrapAngle(double angle);
  static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d & v);
  static tf2::Quaternion smallAngleQuat(const Eigen::Vector3d & dtheta);

  /* ===== Subscribers / Publishers ===== */
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pvt_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr rtk_status_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr motion_pos_deadband_sub_;
  rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr fix_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gnss_marker_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr gnss_only_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr imu_only_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr gnss_only_odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr imu_only_odom_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr motion_state_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr motion_state_raw_gnss_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  nav_msgs::msg::Path path_msg_;
  nav_msgs::msg::Path gnss_only_path_msg_;
  nav_msgs::msg::Path imu_only_path_msg_;
  gps_utils::GpsConverter converter_;

  /* ===== Parameters ===== */
  std::string map_frame_;
  std::string base_frame_;
  int min_fix_type_;
  double acc_noise_density_;
  double gyro_noise_density_;
  double acc_bias_rw_;
  double gyro_bias_rw_;
  bool publish_tf_;
  Eigen::Vector3d gnss_lever_arm_{Eigen::Vector3d::Zero()}; // antenna pos in body frame
  double default_hacc_;
  double default_vacc_;
  double min_heading_dist_;
  double gnss_heading_std_rad_;
  std::string motion_pos_deadband_topic_;
  double motion_pos_deadband_;
  double motion_idle_hold_sec_;

  /* ===== Nominal state ===== */
  Eigen::Vector3d p_{Eigen::Vector3d::Zero()};   // position, ENU (body/IMU origin)
  Eigen::Vector3d v_{Eigen::Vector3d::Zero()};   // velocity, ENU (body/IMU origin)
  Eigen::Quaterniond q_{Eigen::Quaterniond::Identity()}; // body -> ENU
  Eigen::Vector3d ab_{Eigen::Vector3d::Zero()};  // accel bias
  Eigen::Vector3d gb_{Eigen::Vector3d::Zero()};  // gyro bias

  /* ===== Error-state covariance (15x15) ===== */
  Eigen::MatrixXd P_;

  bool imu_initialized_{false};
  bool origin_set_{false};
  bool rtk_corrections_active_{false};
  rclcpp::Time last_imu_time_;
  Eigen::Vector3d last_acc_meas_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d last_gyro_meas_{Eigen::Vector3d::Zero()};
  /* ===== Motion-state hysteresis: entering forward/backward requires
   * motion_pos_deadband_ metres of displacement from the stationary anchor;
   * once moving, that state is held (re-evaluating direction each tick from
   * the still-advancing anchor) until position has stayed within
   * motion_pos_deadband_ of a fixed reference for motion_idle_hold_sec_,
   * confirming the vehicle actually stopped rather than pulsing back to
   * "idle" between successive 0.3 m crossings. ===== */
  Eigen::Vector3d motion_anchor_pos_{Eigen::Vector3d::Zero()};
  bool motion_anchor_set_{false};
  bool motion_is_moving_{false};
  Eigen::Vector3d motion_idle_ref_pos_{Eigen::Vector3d::Zero()};
  rclcpp::Time motion_idle_ref_time_;
  std::string last_motion_state_;

  /* ===== Raw-GNSS-only motion-state hysteresis: same deadband/hold logic as
   * above but driven entirely by raw GNSS fixes (pos_meas) -- position and
   * course-heading direction both come from consecutive GNSS fixes, with no
   * IMU/fused-state dependency at all. Independent state so it can diverge
   * from the fused /gnss_imu_eskf/motion_state output. ===== */
  Eigen::Vector3d motion_raw_anchor_pos_{Eigen::Vector3d::Zero()};
  bool motion_raw_anchor_set_{false};
  bool motion_raw_is_moving_{false};
  Eigen::Vector3d motion_raw_idle_ref_pos_{Eigen::Vector3d::Zero()};
  rclcpp::Time motion_raw_idle_ref_time_;
  std::string last_motion_raw_state_;

  /* ===== Single-antenna GNSS course heading ===== */
  Eigen::Vector3d heading_anchor_enu_{Eigen::Vector3d::Zero()};
  bool heading_anchor_set_{false};

  /* ===== GNSS-only odom debug/comparison (position = raw fix, orientation =
   * consecutive-fix course heading, independent of the min_heading_dist_
   * hysteresis used for the ESKF's own heading update) ===== */
  Eigen::Vector3d prev_gnss_pos_{Eigen::Vector3d::Zero()};
  bool prev_gnss_pos_set_{false};
  Eigen::Quaterniond gnss_only_orientation_{Eigen::Quaterniond::Identity()};

  /* ===== Raw GNSS fix visualization ===== */
  visualization_msgs::msg::MarkerArray gnss_marker_array_;
  int gnss_marker_id_{0};

  /* ===== IMU-only dead-reckoning (debug/comparison, never GNSS-corrected) =====
   * Seeded from the ESKF nominal state at the moment the ENU origin is latched,
   * then propagated with the same strapdown integration as predict() but with
   * no GNSS position/heading updates -- shows raw IMU drift for comparison
   * against the fused path and the raw GNSS-only path. */
  Eigen::Vector3d p_imu_only_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d v_imu_only_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond q_imu_only_{Eigen::Quaterniond::Identity()};

  /* ===== Stationary init accumulation ===== */
  std::vector<Eigen::Vector3d> init_acc_samples_;
  std::vector<Eigen::Vector3d> init_gyro_samples_;
};
