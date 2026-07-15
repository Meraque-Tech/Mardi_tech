// src/Localization/gps_imu_fusion_ihw/src/global_yaw_complementary_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float32.hpp>
#include <nav_msgs/msg/odometry.hpp>          // ★ 추가
#include <tf2/LinearMath/Quaternion.h>        // ★ 추가
#include <cmath>
#include <string>

namespace { inline double wrap(double a){ return std::atan2(std::sin(a), std::cos(a)); } }

class GlobalYawComplementary : public rclcpp::Node
{
public:
  GlobalYawComplementary() : Node("global_yaw_complementary")
  {
    /* ─ 파라미터 ─ */
    v_min_       = declare_parameter<double>("v_min",       0.5);
    dist_min_    = declare_parameter<double>("dist_min",    1.0);
    k_corr_      = declare_parameter<double>("k_corr",      0.02);
    gps_timeout_ = declare_parameter<double>("gps_timeout", 1.0);
    debug_       = declare_parameter<bool  >("debug",       false);

    // ★ 추가: odom 퍼블리시 관련 파라미터
    odom_topic_     = declare_parameter<std::string>("odom_topic",     "/odometry/filtered");
    odom_frame_id_  = declare_parameter<std::string>("odom_frame_id",  "odom");
    base_frame_id_  = declare_parameter<std::string>("base_frame_id",  "base_link");

    /* ─ 토픽 I/O ─ */
    sub_xy_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "/local_xy", 10, std::bind(&GlobalYawComplementary::xyCallback, this, std::placeholders::_1));

    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu/processed", rclcpp::SensorDataQoS(),
      std::bind(&GlobalYawComplementary::imuCallback, this, std::placeholders::_1));

    pub_yaw_  = create_publisher<std_msgs::msg::Float32>("/global_yaw/complementary", 10);
    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10); // ★ 추가

    RCLCPP_INFO(get_logger(), "global_yaw_complementary_node launched. Publishing %s and %s",
                "/global_yaw/complementary", odom_topic_.c_str());
  }

private:
  /* ===== GPS 콜백 ===== */
  void xyCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    const double x = msg->point.x;
    const double y = msg->point.y;
    const double z = msg->point.z;
    const double t = msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec;

    // 최신 위치 저장 (Odometry에 사용)
    latest_x_ = x; latest_y_ = y; latest_z_ = z;
    latest_xy_stamp_sec_ = t;
    have_xy_ = true;

    if (has_prev_xy_) {
      const double dx   = x - prev_x_;
      const double dy   = y - prev_y_;
      const double dist = std::hypot(dx, dy);
      if (dist < 1e-6) { prev_x_ = x; prev_y_ = y; last_xy_time_ = t; return; }

      // 속도 계산
      if (last_xy_time_ > 0.0) {
        const double dt = t - last_xy_time_;
        if (dt > 0.0) {
          vx_ = dx / dt;
          vy_ = dy / dt;
          speed_xy_ = dist / dt;
        }
      }

      const double yaw_gps = wrap(std::atan2(dy, dx));

      // 초기 yaw 결정
      if (!init_done_) {
        cum_dist_ += dist;
        if (cum_dist_ >= dist_min_ && speed_xy_ >= v_min_) {
          yaw_        = yaw_gps;
          init_done_  = true;
          RCLCPP_INFO(get_logger(), "initial yaw = %.2f°", yaw_ * 180.0 / M_PI);
        }
      }
      // 초기화 후 GPS yaw 기록 (속도 조건 만족 시)
      else if (speed_xy_ >= v_min_) {
        last_gps_yaw_  = yaw_gps;
        last_gps_time_ = t;
        gps_valid_     = true;
      }
    }

    // 상태 업데이트
    prev_x_ = x; prev_y_ = y; has_prev_xy_ = true; last_xy_time_ = t;
  }

  /* ===== IMU 콜백 ===== */
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (!init_done_) return;  // yaw 초기화 전에는 무시

    const double t = msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec;

    if (prev_imu_time_ < 0.0) { prev_imu_time_ = t; return; }

    const double dt = t - prev_imu_time_;
    prev_imu_time_  = t;
    if (dt <= 0.0 || dt > 0.2) return;  // 5 Hz 미만 or 시간 역전

    /* ➊ IMU 적분 */
    yaw_ = wrap(yaw_ + msg->angular_velocity.z * dt);

    /* ➋ GPS 보정 (최근값이 유효할 때만) */
    if (gps_valid_ && (t - last_gps_time_) <= gps_timeout_) {
      const double delta = wrap(last_gps_yaw_ - yaw_);
      yaw_ = wrap(yaw_ + k_corr_ * delta);
    } else {
      gps_valid_ = false;  // 타임아웃
    }

    /* 기존 Float32 퍼블리시 (디버깅용 유지) */
    std_msgs::msg::Float32 out;
    out.data = static_cast<float>(yaw_);
    pub_yaw_->publish(out);

    /* ★ Odometry 퍼블리시 */
    if (have_xy_) {
      nav_msgs::msg::Odometry odom;
      // 타임스탬프: IMU 시간(자세) 기준. 필요 시 latest_xy_stamp를 쓸 수도 있음.
      odom.header.stamp = msg->header.stamp;
      odom.header.frame_id = odom_frame_id_;     // 예: "odom"
      odom.child_frame_id  = base_frame_id_;     // 예: "base_link"

      // 위치: 최신 /local_xy
      odom.pose.pose.position.x = latest_x_;
      odom.pose.pose.position.y = latest_y_;
      odom.pose.pose.position.z = latest_z_;

      // 자세: 보정된 yaw → 쿼터니언
      tf2::Quaternion q; q.setRPY(0.0, 0.0, yaw_);
      odom.pose.pose.orientation.x = q.x();
      odom.pose.pose.orientation.y = q.y();
      odom.pose.pose.orientation.z = q.z();
      odom.pose.pose.orientation.w = q.w();

      // (선택) 공분산: 간단한 대각선 값으로 초기화
      // 위치(x,y,z)와 yaw에만 적당한 값, 나머지는 0
      for (double &c : odom.pose.covariance) c = 0.0;
      odom.pose.covariance[0]  = 1e-3;   // x
      odom.pose.covariance[7]  = 1e-3;   // y
      odom.pose.covariance[14] = 1e-1;   // z
      odom.pose.covariance[35] = 1e-2;   // yaw (row 5,col 5)

      // 속도: /local_xy로 계산한 vx, vy (기본 z=0), yaw rate은 IMU 사용
      odom.twist.twist.linear.x  = vx_;
      odom.twist.twist.linear.y  = vy_;
      odom.twist.twist.linear.z  = 0.0;
      odom.twist.twist.angular.x = 0.0;
      odom.twist.twist.angular.y = 0.0;
      odom.twist.twist.angular.z = msg->angular_velocity.z;

      // (선택) twist 공분산
      for (double &c : odom.twist.covariance) c = 0.0;
      odom.twist.covariance[0]  = 6e-4;  // vx
      odom.twist.covariance[7]  = 6e-4;  // vy
      odom.twist.covariance[35] = 6e-4;  // yaw rate

      pub_odom_->publish(odom);
    }

    if (debug_) {
      RCLCPP_DEBUG(get_logger(), "yaw = %.2f°  vx=%.3f  vy=%.3f",
                   yaw_ * 180.0 / M_PI, vx_, vy_);
    }
  }

  /* ─ 멤버 변수 ─ */
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_xy_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr            sub_imu_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr              pub_yaw_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr             pub_odom_; // ★ 추가

  // 파라미터
  double v_min_, dist_min_, k_corr_, gps_timeout_;
  bool   debug_;
  std::string odom_topic_, odom_frame_id_, base_frame_id_; // ★ 추가

  // /local_xy 상태(위치/속도)
  double prev_x_{0.0}, prev_y_{0.0};
  bool   has_prev_xy_{false};
  double last_xy_time_{-1.0};
  double cum_dist_{0.0};
  double vx_{0.0}, vy_{0.0}, speed_xy_{0.0};               // ★ 추가
  double latest_x_{0.0}, latest_y_{0.0}, latest_z_{0.0};   // ★ 추가
  double latest_xy_stamp_sec_{-1.0};
  bool   have_xy_{false};

  // IMU/보정 상태
  double prev_imu_time_{-1.0};
  double yaw_{0.0};
  bool   init_done_{false};

  // GPS yaw 보정
  double last_gps_yaw_{0.0};
  double last_gps_time_{-1.0};
  bool   gps_valid_{false};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalYawComplementary>());
  rclcpp::shutdown();
  return 0;
}
