#include <gps_imu_fusion_ihw/kalman_filter.hpp>


#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

namespace gps_imu_fusion
{

class SensorFusionNode : public rclcpp::Node
{
public:
  SensorFusionNode()
  : Node("gps_imu_sensor_fusion"), initialised_(false)
  {
    /* ---------- 파라미터 ---------- */
    double sigma_a     = declare_parameter("sigma_a",    0.1);
    double sigma_omega = declare_parameter("sigma_omega",0.0245);
    double dt_default  = declare_parameter("dt_default", 0.01);
    gps_cov_           = declare_parameter("gps_cov",    0.000196);
    map_frame_         = declare_parameter("map_frame",  std::string("reference"));
    base_frame_        = declare_parameter("base_frame", std::string("gps_antenna"));
    publish_tf_        = declare_parameter("publish_tf", true);

    /* ---------- EKF ---------- */
    kf_ = std::make_unique<KalmanFilter>(dt_default, sigma_a, sigma_omega);

    /* ---------- 퍼블리셔 ---------- */
    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("/odometry/fusion", 10);
    pub_yaw_  = create_publisher<std_msgs::msg::Float32>("/global_yaw", 10);

    /* ---------- TF 브로드캐스터 ---------- */
    if (publish_tf_)
      tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

    /* ---------- 서브스크라이버 ---------- */
    using std::placeholders::_1;

    /* IMU: SensorDataQoS (BestEffort, depth 10) */
    auto imu_qos = rclcpp::SensorDataQoS();
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
                 "/ouster/imu", imu_qos,
                 std::bind(&SensorFusionNode::imuCallback, this, _1));

    /* GPS: 기본 Reliable QoS */
    sub_gps_ = create_subscription<geometry_msgs::msg::PointStamped>(
                 "/local_xy", 10,
                 std::bind(&SensorFusionNode::gpsCallback, this, _1));

    RCLCPP_INFO(get_logger(), "SensorFusionNode started – frame %s → %s",
                map_frame_.c_str(), base_frame_.c_str());
  }

private:
  /* ---------- IMU 콜백 (predict) ---------- */
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (!initialised_)
      return;

    rclcpp::Time curr_time(msg->header.stamp);

    /* Δt 계산 */
    if (prev_imu_time_.nanoseconds() == 0)   // 첫 프레임
    {
      prev_imu_time_ = curr_time;
      return;
    }
    double dt = (curr_time - prev_imu_time_).seconds();
    if (dt <= 0.0 || dt > 0.1)               // 100 Hz 기준 방어
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *this, 2000,
          "Abnormal dt=%.6f → set 0.01", dt);
      dt = 0.01;
    }
    prev_imu_time_ = curr_time;

    /* 입력 벡터 */
    Eigen::Vector2d acc_body(msg->linear_acceleration.x,
                             msg->linear_acceleration.y);
    double omega_z = msg->angular_velocity.z;

    /* 예측 */
    kf_->predict(acc_body, omega_z, dt);

    // 디버그 (0.5 s 간격)
    RCLCPP_INFO_THROTTLE(get_logger(), *this, 500,
        "ω=%.4f  dt=%.4f  ψ=%.4f rad", omega_z, dt, kf_->state()(4));

    publishOutputs(msg->header.stamp);
  }

  /* ---------- GPS 콜백 (update) ---------- */
  void gpsCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    Eigen::Vector2d gps_xy(msg->point.x, msg->point.y);
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * gps_cov_;

    if (!initialised_)           /* --- 첫 GPS 로 EKF 초기화 --- */
    {
      Vector5d x0; x0 << gps_xy(0), gps_xy(1), 0.0, 0.0, 0.0;
      Matrix5d P0 = Matrix5d::Identity();
      P0.diagonal() << 1.0, 1.0, 0.4, 0.4, 0.1;
      kf_->setInitialState(x0, P0);

      initialised_   = true;
      prev_imu_time_ = rclcpp::Time(msg->header.stamp);
      RCLCPP_INFO(get_logger(), "EKF initialised (GPS %.2f, %.2f)", x0(0), x0(1));
      publishOutputs(msg->header.stamp);
      return;
    }

    /* 업데이트 */
    kf_->update(gps_xy, R);
    publishOutputs(msg->header.stamp);
  }

  /* ---------- Odometry / yaw / TF 발행 ---------- */
  void publishOutputs(const builtin_interfaces::msg::Time &stamp)
  {
    const Vector5d &x = kf_->state();

    /* Odometry */
    nav_msgs::msg::Odometry odom;
    odom.header.stamp    = stamp;
    odom.header.frame_id = map_frame_;
    odom.child_frame_id  = base_frame_;
    odom.pose.pose.position.x = x(0);
    odom.pose.pose.position.y = x(1);
    odom.pose.pose.position.z = 0.0;

    tf2::Quaternion q; q.setRPY(0.0, 0.0, x(4));
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom.twist.twist.linear.x = x(2);
    odom.twist.twist.linear.y = x(3);

    pub_odom_->publish(odom);

    /* global_yaw (rad) */
    std_msgs::msg::Float32 yaw_msg;
    yaw_msg.data = static_cast<float>(x(4));
    pub_yaw_->publish(yaw_msg);

    /* TF 브로드캐스트 (선택) */
    if (publish_tf_ && tf_broadcaster_)
    {
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp    = stamp;
      tf_msg.header.frame_id = map_frame_;
      tf_msg.child_frame_id  = base_frame_;
      tf_msg.transform.translation.x = x(0);
      tf_msg.transform.translation.y = x(1);
      tf_msg.transform.translation.z = 0.0;
      tf_msg.transform.rotation.x = q.x();
      tf_msg.transform.rotation.y = q.y();
      tf_msg.transform.rotation.z = q.z();
      tf_msg.transform.rotation.w = q.w();
      tf_broadcaster_->sendTransform(tf_msg);
    }
  }

  /* ---------- 멤버 ---------- */
  std::unique_ptr<KalmanFilter> kf_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr      sub_imu_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_gps_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr       pub_odom_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr        pub_yaw_;
  std::shared_ptr<tf2_ros::TransformBroadcaster>              tf_broadcaster_;

  rclcpp::Time prev_imu_time_;
  bool         initialised_;
  double       gps_cov_;
  std::string  map_frame_, base_frame_;
  bool         publish_tf_;
};

} // namespace gps_imu_fusion

/* ---------------- main ---------------- */
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<gps_imu_fusion::SensorFusionNode>());
  rclcpp::shutdown();
  return 0;
}
