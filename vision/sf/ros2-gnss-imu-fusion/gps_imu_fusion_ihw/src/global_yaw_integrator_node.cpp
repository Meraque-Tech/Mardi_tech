/* global_yaw_integrator_node.cpp
 * ───────────────────────────────
 *  /local_xy  → 초기 global yaw(ψ0) 산출
 *  /imu/processed → ωz 적분으로 yaw 지속 업데이트
 *  퍼블리시 : /global_yaw (std_msgs/Float32, [rad])
 */

 #include <rclcpp/rclcpp.hpp>
 #include <geometry_msgs/msg/point_stamped.hpp>
 #include <sensor_msgs/msg/imu.hpp>
 #include <std_msgs/msg/float32.hpp>
 #include <cmath>
 
 namespace {
 inline double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }
 }
 
 class GlobalYawIntegrator : public rclcpp::Node
 {
 public:
   GlobalYawIntegrator() : Node("global_yaw_integrator")
   {
     /* ──── 파라미터 ──── */
     v_min_    = declare_parameter<double>("v_min",    0.5);  // [m/s]
     dist_min_ = declare_parameter<double>("dist_min", 1.0);  // [m]
     debug_    = declare_parameter<bool  >("debug",    false);
 
     /* ──── 토픽 I/O ──── */
     sub_xy_ = create_subscription<geometry_msgs::msg::PointStamped>(
       "/local_xy", 10,
       std::bind(&GlobalYawIntegrator::xyCallback, this, std::placeholders::_1));
 
     sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
       "/imu/processed", rclcpp::SensorDataQoS(),
       std::bind(&GlobalYawIntegrator::imuCallback, this, std::placeholders::_1));
 
     pub_yaw_ = create_publisher<std_msgs::msg::Float32>("/global_yaw_1", 10);
 
     RCLCPP_INFO(get_logger(), "global_yaw_integrator_node launched.");
   }
 
 private:
   /* ===== GPS callback (local_xy) ===== */
   void xyCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
   {
     const double x = msg->point.x;
     const double y = msg->point.y;
     const double t = msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec;
 
     if (has_prev_xy_) {
       const double dx = x - prev_x_;
       const double dy = y - prev_y_;
       const double dist = std::hypot(dx, dy);
       if (dist < 1e-6) return;          // 잡음
 
       double speed = 0.0;
       if (last_xy_time_ > 0.0) {
         const double dt = t - last_xy_time_;
         if (dt > 0.0) speed = dist / dt;
       }
 
       /* 초기 yaw 결정 */
       if (!init_done_) {
         cum_dist_ += dist;
         if (cum_dist_ >= dist_min_ && speed >= v_min_) {
           yaw_ = wrap(std::atan2(dy, dx));
           init_done_ = true;
           RCLCPP_INFO(get_logger(), "initial yaw = %.2f°", yaw_ * 180.0 / M_PI);
         }
       }
     }
 
     prev_x_ = x;
     prev_y_ = y;
     has_prev_xy_ = true;
     last_xy_time_ = t;
   }
 
   /* ===== IMU callback (ωz 적분) ===== */
   void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
   {
     if (!init_done_) return;
 
     const double t = msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec;
     if (prev_imu_time_ < 0.0) {
       prev_imu_time_ = t;
       return;
     }
 
     const double dt = t - prev_imu_time_;
     prev_imu_time_ = t;
     if (dt <= 0.0 || dt > 0.2) return;  // 5 Hz 미만 또는 시간 역전 시 스킵
 
     yaw_ = wrap(yaw_ + msg->angular_velocity.z * dt);
 
     std_msgs::msg::Float32 out;
     out.data = static_cast<float>(yaw_);
     pub_yaw_->publish(out);
 
     if (debug_) {
       RCLCPP_DEBUG(get_logger(), "yaw = %.2f°", yaw_ * 180.0 / M_PI);
     }
   }
 
   /* ──── 멤버 ──── */
   rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_xy_;
   rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr            sub_imu_;
   rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr              pub_yaw_;
 
   /* params */
   double v_min_, dist_min_;
   bool   debug_;
 
   /* state */
   double prev_x_{0.0}, prev_y_{0.0};
   bool   has_prev_xy_{false};
   double last_xy_time_{-1.0};
   double cum_dist_{0.0};
 
   double prev_imu_time_{-1.0};
   double yaw_{0.0};
   bool   init_done_{false};
 };
 
 /* ===== main ===== */
 int main(int argc, char **argv)
 {
   rclcpp::init(argc, argv);
   rclcpp::spin(std::make_shared<GlobalYawIntegrator>());
   rclcpp::shutdown();
   return 0;
 }
 