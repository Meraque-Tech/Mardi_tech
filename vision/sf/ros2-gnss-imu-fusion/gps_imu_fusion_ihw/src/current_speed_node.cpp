#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float32.hpp>
#include <cmath>

class CurrentSpeedNode : public rclcpp::Node
{
public:
  CurrentSpeedNode() : Node("current_speed_node")
  {
    /* ─ 파라미터 ─ */
    v_min_       = declare_parameter<double>("v_min",       0.4);   // GPS 신뢰 최소속도 [m/s]
    k_corr_      = declare_parameter<double>("k_corr",      0.02);  // GPS→IMU 보정 비율 0~1
    gps_timeout_ = declare_parameter<double>("gps_timeout", 0.5);   // GPS 최신성 [s]

    /* ─ 토픽 I/O ─ */
    sub_xy_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "/local_xy", 10,
      std::bind(&CurrentSpeedNode::xyCallback, this, std::placeholders::_1));

    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu/processed", rclcpp::SensorDataQoS(),
      std::bind(&CurrentSpeedNode::imuCallback, this, std::placeholders::_1));

    pub_speed_ = create_publisher<std_msgs::msg::Float32>("/current_speed", 10);

    RCLCPP_INFO(get_logger(), "current_speed_node started.");
  }

private:
  /* =============== GPS 콜백 =============== */
  void xyCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    const double x = msg->point.x;
    const double y = msg->point.y;
    const double t = msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec;

    if (has_prev_xy_) {
      const double dx = x - prev_x_;
      const double dy = y - prev_y_;
      const double dt = t - last_xy_time_;
      if (dt > 0.0) {
        const double vx_gps = dx / dt;
        const double vy_gps = dy / dt;
        const double speed_gps = std::hypot(vx_gps, vy_gps);

        // 초기화 또는 보정
        if (!init_done_ && speed_gps >= v_min_) {
          vx_ = vx_gps; vy_ = vy_gps; init_done_ = true;
        } else if (speed_gps >= v_min_) {
          last_gps_vx_ = vx_gps;
          last_gps_vy_ = vy_gps;
          last_gps_time_ = t;
          gps_valid_ = true;
        }
      }
    }

    prev_x_ = x; prev_y_ = y; has_prev_xy_ = true; last_xy_time_ = t;
  }

  /* =============== IMU 콜백 =============== */
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (!init_done_) return;                     // 속도 초기화 전에는 무시

    const double t = msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec;
    const double dt = (prev_imu_time_ < 0.0) ? 0.0 : t - prev_imu_time_;
    prev_imu_time_ = t;
    if (dt <= 0.0 || dt > 0.2) return;           // 시간 역전/샘플 누락 시 무시

    /* 1) IMU 적분 */
    vx_ += msg->linear_acceleration.x * dt;
    vy_ += msg->linear_acceleration.y * dt;

    /* 2) GPS 보정 */
    if (gps_valid_ && (t - last_gps_time_) <= gps_timeout_) {
      vx_ += k_corr_ * (last_gps_vx_ - vx_);
      vy_ += k_corr_ * (last_gps_vy_ - vy_);
    } else {
      gps_valid_ = false;
    }

    /* 퍼블리시: 스칼라 속도 */
    std_msgs::msg::Float32 sp;
    sp.data = static_cast<float>(std::hypot(vx_, vy_));
    pub_speed_->publish(sp);
  }

  /* ───── 멤버 ───── */
  // 토픽
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_xy_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr            sub_imu_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr              pub_speed_;

  // 파라미터
  double v_min_, k_corr_, gps_timeout_;

  // GPS 상태
  double prev_x_{0.0}, prev_y_{0.0}, last_xy_time_{-1.0};
  bool   has_prev_xy_{false};

  // IMU 상태
  double prev_imu_time_{-1.0};
  double vx_{0.0}, vy_{0.0};
  bool   init_done_{false};

  // GPS 보정용
  double last_gps_vx_{0.0}, last_gps_vy_{0.0}, last_gps_time_{-1.0};
  bool   gps_valid_{false};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CurrentSpeedNode>());
  rclcpp::shutdown();
  return 0;
}
