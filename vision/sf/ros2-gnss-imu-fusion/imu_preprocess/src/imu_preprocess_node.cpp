#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <array>
#include <cmath>

class ImuPreprocessNode : public rclcpp::Node
{
public:
  ImuPreprocessNode() : Node("imu_preprocess_node")
  {
    /* ---------- 파라미터 ---------- */
    use_precalibrated_bias_ = declare_parameter<bool>("use_precalibrated_bias", false);
    lpf_cutoff_ = declare_parameter<double>("lpf_cutoff", 15.0);  // 1차 IIR LPF 컷오프 [Hz]
    // 중력 보정 옵션 (동작 중 보정)
    remove_gravity_   = declare_parameter<bool>("remove_gravity", false); // 기본은 비활성화 (역호환)
    gravity_magnitude_ = declare_parameter<double>("gravity_magnitude", 9.81);
    
    // Pre-calibrated bias parameters
    if (use_precalibrated_bias_) {
      bias_acc_[0] = declare_parameter<double>("bias_acc_x", 0.0);
      bias_acc_[1] = declare_parameter<double>("bias_acc_y", 0.0);
      bias_acc_[2] = declare_parameter<double>("bias_acc_z", 0.0);
      bias_gyro_[0] = declare_parameter<double>("bias_gyro_x", 0.0);
      bias_gyro_[1] = declare_parameter<double>("bias_gyro_y", 0.0);
      bias_gyro_[2] = declare_parameter<double>("bias_gyro_z", 0.0);
      
      calibrated_ = true;  // Skip calibration phase
      
      RCLCPP_INFO(get_logger(),
        "Using pre-calibrated IMU bias:\n"
        "  acc  = [%.4f, %.4f, %.4f] m/s²\n"
        "  gyro = [%.4f, %.4f, %.4f] rad/s",
        bias_acc_[0], bias_acc_[1], bias_acc_[2],
        bias_gyro_[0], bias_gyro_[1], bias_gyro_[2]);
    } else {
      // Only declare calib_duration if we're doing runtime calibration
      calib_duration_ = declare_parameter<double>("calib_duration", 20.0);
    }

    /* ---------- 토픽 I/O ---------- */
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu/data", rclcpp::SensorDataQoS(),
      // "/ouster/imu", rclcpp::SensorDataQoS(),
      std::bind(&ImuPreprocessNode::imuCallback, this, std::placeholders::_1));

    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(
      "/imu/processed", rclcpp::SensorDataQoS());

    start_time_ = now();
  }

private:
  /* ===== 유틸: 쿼터니언 정규화 ===== */
  static void normalizeQuaternion(double &w, double &x, double &y, double &z)
  {
    const double n = std::sqrt(w*w + x*x + y*y + z*z);
    if (n > 1e-12) {
      w /= n; x /= n; y /= n; z /= n;
    } else {
      // 기본 단위쿼터니언
      w = 1.0; x = y = z = 0.0;
    }
  }

  /* ===== 유틸: R^T * [0,0,g] (world→body) ===== */
  static void gravityInBodyFrame(double qw, double qx, double qy, double qz,
                                 double g, double &gx_b, double &gy_b, double &gz_b)
  {
    // q: body→world 회전. R은 body→world, 그러면 g_body = R^T * g_world.
    normalizeQuaternion(qw, qx, qy, qz);
    const double xx = qx*qx;
    const double yy = qy*qy;
    const double zz = qz*qz;
    const double wx = qw*qx;
    const double wy = qw*qy;
    const double wz = qw*qz;
    const double xy = qx*qy;
    const double xz = qx*qz;
    const double yz = qy*qz;

    // g_body = g * (R^T의 3번째 컬럼) = g * (R의 3번째 행)
    gx_b = g * (2.0*(xz - wy));
    gy_b = g * (2.0*(yz + wx));
    gz_b = g * (1.0 - 2.0*(xx + yy));
  }

  /* ===== 유틸: orientation 유효성 체크 ===== */
  static bool orientationValid(const sensor_msgs::msg::Imu &m)
  {
    // REP-145: orientation_covariance[0] == -1 이면 orientation 미제공 (권장)
    bool cov_valid = true;
    if (!m.orientation_covariance.empty()) {
      cov_valid = (m.orientation_covariance[0] >= 0.0);
    }
    const auto &q = m.orientation;
    const double n2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
    return cov_valid && (n2 > 1e-8) && std::isfinite(n2);
  }

  /* ===== IMU 콜백 ===== */
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const double dt =
      (last_time_.nanoseconds() == 0) ? 0.01
      : (now() - last_time_).seconds();
    last_time_ = now();

    /* 1) 정적 캘리브레이션 단계 ------------------------- */
    if (!calibrated_) {
      // This should only happen if use_precalibrated_bias_ is false
      accumulateBias(*msg);
      if ((now() - start_time_).seconds() >= calib_duration_) {
        finalizeBias();     // 편향 확정 + 로그 1회 출력
        calibrated_ = true;
      }
      return;               // 아직 퍼블리시하지 않음
    }

    /* 2) 편향 제거 -------------------------------------- */
    sensor_msgs::msg::Imu out = *msg;
    out.linear_acceleration.x -= bias_acc_[0];
    out.linear_acceleration.y -= bias_acc_[1];
    out.linear_acceleration.z -= bias_acc_[2];
    out.angular_velocity.x    -= bias_gyro_[0];
    out.angular_velocity.y    -= bias_gyro_[1];
    out.angular_velocity.z    -= bias_gyro_[2];

    /* 2.5) 중력 보정(선택) ------------------------------- */
    if (remove_gravity_) {
      const bool has_ori = orientationValid(*msg);
      if (has_ori) {
        const auto &q = msg->orientation;
        double gx_b, gy_b, gz_b;
        gravityInBodyFrame(q.w, q.x, q.y, q.z, gravity_magnitude_, gx_b, gy_b, gz_b);
        // 측정(바이어스 제거 후)에서 중력 성분 제거 → 정지 시 0 근처
        out.linear_acceleration.x -= gx_b;
        out.linear_acceleration.y -= gy_b;
        out.linear_acceleration.z -= gz_b;
      } else if (!warned_orientation_missing_) {
        warned_orientation_missing_ = true;
        RCLCPP_WARN(get_logger(), "remove_gravity 활성화 됐지만 orientation이 유효하지 않아 중력 보정을 건너뜁니다.");
      }
    }

    /* 3) 1차 IIR 저역통과필터 --------------------------- */
    if (lpf_cutoff_ > 0.0) {
      const double tau   = 1.0 / (2.0 * M_PI * lpf_cutoff_);
      const double alpha = dt / (tau + dt);

      auto lpf = [alpha](double x, double &prev) {
        double y = alpha * x + (1.0 - alpha) * prev;
        prev = y;
        return y;
      };

      out.linear_acceleration.x = lpf(out.linear_acceleration.x, acc_prev_[0]);
      out.linear_acceleration.y = lpf(out.linear_acceleration.y, acc_prev_[1]);
      out.linear_acceleration.z = lpf(out.linear_acceleration.z, acc_prev_[2]);
      out.angular_velocity.x    = lpf(out.angular_velocity.x,    gyro_prev_[0]);
      out.angular_velocity.y    = lpf(out.angular_velocity.y,    gyro_prev_[1]);
      out.angular_velocity.z    = lpf(out.angular_velocity.z,    gyro_prev_[2]);
    }

    imu_pub_->publish(out);
  }

  /* ===== 편향 누적 ===== */
  void accumulateBias(const sensor_msgs::msg::Imu &m)
  {
    // orientation이 유효하면 모든 축에 대해 중력 성분을 제거하여 바이어스 평균을 구한다.
    if (orientationValid(m)) {
      const auto &q = m.orientation;
      double gx_b, gy_b, gz_b;
      gravityInBodyFrame(q.w, q.x, q.y, q.z, gravity_magnitude_, gx_b, gy_b, gz_b);
      sum_acc_[0]  += (m.linear_acceleration.x - gx_b);
      sum_acc_[1]  += (m.linear_acceleration.y - gy_b);
      sum_acc_[2]  += (m.linear_acceleration.z - gz_b);
    } else {
      // orientation 미제공 시 기존 로직 유지 (Z축만 9.81 제거)
      sum_acc_[0]  += m.linear_acceleration.x;
      sum_acc_[1]  += m.linear_acceleration.y;
      sum_acc_[2]  += m.linear_acceleration.z - gravity_magnitude_; // 중력 제거(Z)
    }
    sum_gyro_[0] += m.angular_velocity.x;
    sum_gyro_[1] += m.angular_velocity.y;
    sum_gyro_[2] += m.angular_velocity.z;
    ++sample_cnt_;
  }

  /* ===== 편향 확정 + 단 1회 로그 ===== */
  void finalizeBias()
  {
    for (int i = 0; i < 3; ++i) {
      bias_acc_[i]  = sum_acc_[i]  / sample_cnt_;
      bias_gyro_[i] = sum_gyro_[i] / sample_cnt_;
    }
    RCLCPP_INFO(get_logger(),
      "IMU bias calibrated:\n"
      "  acc  = [%.4f, %.4f, %.4f] m/s²\n"
      "  gyro = [%.4f, %.4f, %.4f] rad/s",
      bias_acc_[0],  bias_acc_[1],  bias_acc_[2],
      bias_gyro_[0], bias_gyro_[1], bias_gyro_[2]);
  }

  /* ===== 멤버 ===== */
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr     imu_pub_;
  rclcpp::Time start_time_, last_time_;

  bool use_precalibrated_bias_;
  double calib_duration_;
  double lpf_cutoff_;
  bool remove_gravity_;
  double gravity_magnitude_;
  bool calibrated_{false};
  bool warned_orientation_missing_{false};

  std::array<double,3> sum_acc_{}, sum_gyro_{};
  std::array<double,3> bias_acc_{}, bias_gyro_{};
  std::array<double,3> acc_prev_{}, gyro_prev_{};
  size_t sample_cnt_{0};
};

/* ===== 메인 ===== */
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuPreprocessNode>());
  rclcpp::shutdown();
  return 0;
}
