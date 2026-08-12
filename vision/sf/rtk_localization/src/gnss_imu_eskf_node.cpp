#include <rtk_sf/gnss_imu_eskf_node.hpp>

#include <algorithm>
#include <cmath>

/* ================= Constructor ================= */
GnssImuEskfNode::GnssImuEskfNode()
: Node("gnss_imu_eskf_node")
{
  this->declare_parameter<std::string>("map_frame", "map");
  this->declare_parameter<std::string>("base_frame", "gnss_base_link");
  this->declare_parameter<int>("min_fix_type", 3);
  this->declare_parameter<double>("acc_noise_density", 0.05);   // m/s^2 / sqrt(Hz)
  this->declare_parameter<double>("gyro_noise_density", 0.005); // rad/s / sqrt(Hz)
  this->declare_parameter<double>("acc_bias_rw", 0.001);
  this->declare_parameter<double>("gyro_bias_rw", 0.0001);
  this->declare_parameter<bool>("publish_tf", true);
  this->declare_parameter<std::vector<double>>("gnss_lever_arm", {0.0, 0.0, 0.0});
  this->declare_parameter<double>("default_hacc", 1.0);
  this->declare_parameter<double>("default_vacc", 2.0);
  this->declare_parameter<double>("min_heading_dist", 0.1);
  this->declare_parameter<double>("gnss_heading_std_deg", 15.0);
  this->declare_parameter<double>("motion_pos_deadband", 0.3);       // m
  this->declare_parameter<double>("motion_idle_hold_sec", 1.0);      // s

  map_frame_ = this->get_parameter("map_frame").as_string();
  base_frame_ = this->get_parameter("base_frame").as_string();
  min_fix_type_ = this->get_parameter("min_fix_type").as_int();
  acc_noise_density_ = this->get_parameter("acc_noise_density").as_double();
  gyro_noise_density_ = this->get_parameter("gyro_noise_density").as_double();
  acc_bias_rw_ = this->get_parameter("acc_bias_rw").as_double();
  gyro_bias_rw_ = this->get_parameter("gyro_bias_rw").as_double();
  publish_tf_ = this->get_parameter("publish_tf").as_bool();
  default_hacc_ = this->get_parameter("default_hacc").as_double();
  default_vacc_ = this->get_parameter("default_vacc").as_double();
  min_heading_dist_ = std::max(
    1e-3, this->get_parameter("min_heading_dist").as_double());
  gnss_heading_std_rad_ = std::max(
    1e-3, this->get_parameter("gnss_heading_std_deg").as_double() * M_PI / 180.0);
  motion_pos_deadband_ = this->get_parameter("motion_pos_deadband").as_double();
  motion_idle_hold_sec_ = this->get_parameter("motion_idle_hold_sec").as_double();

  const auto lever_arm = this->get_parameter("gnss_lever_arm").as_double_array();
  if (lever_arm.size() == 3) {
    gnss_lever_arm_ = Eigen::Vector3d(lever_arm[0], lever_arm[1], lever_arm[2]);
  } else {
    RCLCPP_WARN(get_logger(),
      "gnss_lever_arm must have exactly 3 elements [x,y,z], got %zu; defaulting to zero",
      lever_arm.size());
  }

  // Initial error-state covariance
  P_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
  P_.block<3, 3>(0, 0) *= 10.0;    // position
  P_.block<3, 3>(3, 3) *= 5.0;     // velocity
  P_.block<3, 3>(6, 6) *= 0.1;     // attitude (small-angle)
  P_.block<3, 3>(9, 9) *= 0.01;    // accel bias
  P_.block<3, 3>(12, 12) *= 0.0001; // gyro bias

  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    "/imu/data", rclcpp::SensorDataQoS(),
    std::bind(&GnssImuEskfNode::imuCallback, this, std::placeholders::_1));

  pvt_sub_ = create_subscription<std_msgs::msg::String>(
    "/gnss/pvt", rclcpp::SystemDefaultsQoS(),
    std::bind(&GnssImuEskfNode::pvtCallback, this, std::placeholders::_1));

  rtk_status_sub_ = create_subscription<std_msgs::msg::Bool>(
    "/gnss/rtk_status", rclcpp::SystemDefaultsQoS(),
    [this](const std_msgs::msg::Bool::SharedPtr msg) {
      rtk_corrections_active_ = msg->data;
    });

  odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
    "/gnss_imu_eskf/odom", rclcpp::SystemDefaultsQoS());
  path_pub_ = create_publisher<nav_msgs::msg::Path>("/gnss_imu_eskf/path", 10);
  gnss_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    "/gnss_imu_eskf/raw_gnss_markers", rclcpp::SystemDefaultsQoS());
  gnss_only_path_pub_ = create_publisher<nav_msgs::msg::Path>(
    "/gnss_imu_eskf/gnss_only_path", 10);
  imu_only_path_pub_ = create_publisher<nav_msgs::msg::Path>(
    "/gnss_imu_eskf/imu_only_path", 10);
  gnss_only_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
    "/gnss_imu_eskf/gnss_only_odom", rclcpp::SystemDefaultsQoS());
  imu_only_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
    "/gnss_imu_eskf/imu_only_odom", rclcpp::SystemDefaultsQoS());
  motion_state_pub_ = create_publisher<std_msgs::msg::String>(
    "/gnss_imu_eskf/motion_state", rclcpp::SystemDefaultsQoS());
  motion_state_raw_gnss_pub_ = create_publisher<std_msgs::msg::String>(
    "/gnss_imu_eskf/motion_state_raw_gnss", rclcpp::SystemDefaultsQoS());
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  path_msg_.header.frame_id = map_frame_;
  gnss_only_path_msg_.header.frame_id = map_frame_;
  imu_only_path_msg_.header.frame_id = map_frame_;

  RCLCPP_INFO(get_logger(),
    "GNSS+IMU ESKF node started (map_frame='%s', base_frame='%s', lever_arm=[%.3f,%.3f,%.3f])",
    map_frame_.c_str(), base_frame_.c_str(),
    gnss_lever_arm_.x(), gnss_lever_arm_.y(), gnss_lever_arm_.z());
}

/* ================= Stationary init accumulation ================= */
bool GnssImuEskfNode::accumulateStationaryInit(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  const Eigen::Vector3d acc(
    msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  const Eigen::Vector3d gyro(
    msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  if (!acc.allFinite() || !gyro.allFinite()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "Non-finite IMU sample during stationary init, discarding");
    return false;
  }

  init_acc_samples_.push_back(acc);
  init_gyro_samples_.push_back(gyro);

  if (init_acc_samples_.size() < STATIONARY_INIT_SAMPLES) {
    return false;
  }

  // Average gyro gives the static gyro bias.  The complementary filter has
  // already fused accelerometer/gyro orientation, so its quaternion is the
  // preferred complete roll/pitch/yaw initial guess.
  Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < init_acc_samples_.size(); ++i) {
    acc_mean += init_acc_samples_[i];
    gyro_mean += init_gyro_samples_[i];
  }
  acc_mean /= static_cast<double>(init_acc_samples_.size());
  gyro_mean /= static_cast<double>(init_gyro_samples_.size());

  gb_ = gyro_mean;

  const Eigen::Quaterniond imu_orientation(
    msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
  const bool orientation_available = msg->orientation_covariance[0] >= 0.0 &&
    imu_orientation.coeffs().allFinite() && imu_orientation.norm() > 1e-6;

  if (orientation_available) {
    q_ = imu_orientation.normalized();
  } else {
    // sensor_msgs/Imu uses covariance[0] == -1 to mark orientation unavailable.
    // Preserve gravity-only roll/pitch alignment as a fallback; yaw is then 0
    // until GNSS course-over-ground becomes observable after motion.
    const Eigen::Vector3d up_body = acc_mean.normalized();
    const Eigen::Vector3d up_world(0.0, 0.0, 1.0);
    q_ = Eigen::Quaterniond::FromTwoVectors(up_body, up_world).normalized();
  }

  // Accelerometer stationary residual gives the accel bias: with the vehicle
  // still, acc_mean should equal +g rotated into the body frame (specific
  // force cancels gravity). Any leftover after removing that expected
  // gravity component is bias, not motion. Skipping this (leaving ab_ at its
  // zero default) was the source of the fast, small-amplitude wobble seen on
  // /gnss_imu_eskf/path: uncompensated accel bias integrates twice into
  // position every IMU tick and was only ever getting corrected indirectly
  // and slowly through GNSS position innovations via the ESKF error state.
  const Eigen::Vector3d gravity_world(0.0, 0.0, GRAVITY);
  const Eigen::Vector3d expected_acc_body = q_.conjugate() * gravity_world;
  ab_ = acc_mean - expected_acc_body;

  init_acc_samples_.clear();
  init_gyro_samples_.clear();

  RCLCPP_INFO(get_logger(),
    "Stationary init complete: gyro_bias=[%.5f,%.5f,%.5f], accel_bias=[%.5f,%.5f,%.5f], attitude from %s",
    gb_.x(), gb_.y(), gb_.z(), ab_.x(), ab_.y(), ab_.z(),
    orientation_available ? "IMU complementary filter" : "gravity fallback");
  return true;
}

/* ================= IMU Callback (prediction) ================= */
void GnssImuEskfNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  const Eigen::Vector3d acc_meas(
    msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  const Eigen::Vector3d gyro_meas(
    msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  if (!acc_meas.allFinite() || !gyro_meas.allFinite()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "Non-finite IMU measurement received, skipping sample");
    return;
  }

  const rclcpp::Time current_time(msg->header.stamp);

  if (!imu_initialized_) {
    if (!accumulateStationaryInit(msg)) {
      return;
    }
    imu_initialized_ = true;
    last_imu_time_ = current_time;
    last_acc_meas_ = acc_meas;
    last_gyro_meas_ = gyro_meas;
    RCLCPP_INFO(get_logger(), "ESKF initialized after stationary IMU averaging");
    return;
  }

  const double dt = (current_time - last_imu_time_).seconds();
  last_imu_time_ = current_time;

  if (dt <= 0.0 || dt > 1.0) {
    RCLCPP_WARN(get_logger(), "Invalid IMU dt: %.4f s, skipping prediction", dt);
    last_acc_meas_ = acc_meas;
    last_gyro_meas_ = gyro_meas;
    return;
  }

  // Midpoint IMU preintegration over [last_imu_time, current_time].  Keeping
  // the previous sample avoids the first-order (zero-order-hold) integration
  // that was used here before.
  const Eigen::Vector3d acc_mid = 0.5 * (last_acc_meas_ + acc_meas);
  const Eigen::Vector3d gyro_mid = 0.5 * (last_gyro_meas_ + gyro_meas);
  last_acc_meas_ = acc_meas;
  last_gyro_meas_ = gyro_meas;

  // Do not integrate position/velocity until the map origin is latched by the
  // first accepted GNSS fix -- otherwise p_/v_ accumulate drift in an
  // arbitrary, unanchored frame before the ENU origin even exists.
  if (!origin_set_) {
    return;
  }

  predict(acc_mid, gyro_mid, dt);
  publishState(current_time);

  predictImuOnly(acc_mid, gyro_mid, dt);
  publishImuOnlyPath(current_time);
}

/* ================= GNSS PVT Callback (update) ================= */
void GnssImuEskfNode::pvtCallback(const std_msgs::msg::String::SharedPtr msg)
{
  if (!imu_initialized_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "Waiting for IMU stationary init before consuming GNSS");
    return;
  }

  nlohmann::json j;
  try {
    j = nlohmann::json::parse(msg->data);
  } catch (const nlohmann::json::parse_error & e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "Failed to parse /gnss/pvt payload: %s", e.what());
    return;
  }

  if (!j.contains("lat") || !j.contains("lon") ||
      !j["lat"].is_number() || !j["lon"].is_number())
  {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "/gnss/pvt missing or non-numeric lat/lon, skipping");
    return;
  }

  const double lat = j.value("lat", 0.0);
  const double lon = j.value("lon", 0.0);
  const double alt = j.value("alt", 0.0);
  const int fix = j.value("fix", 0);
  const int carr = j.value("carr", 0);
  const int sats = j.value("sats", 0);
  const double hacc = j.value("hacc", default_hacc_);
  const double vacc = j.value("vacc", default_vacc_);

  if (!std::isfinite(lat) || !std::isfinite(lon) || !std::isfinite(alt) ||
      lat < -90.0 || lat > 90.0 || lon < -180.0 || lon > 180.0)
  {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "/gnss/pvt has invalid lat/lon/alt (%.7f, %.7f, %.3f), skipping", lat, lon, alt);
    return;
  }

  if (fix < min_fix_type_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "GNSS fix type %d below minimum %d, skipping", fix, min_fix_type_);
    return;
  }

  if (!converter_.is_origin_set()) {
    converter_.set_enu_origin(lat, lon, alt);
    origin_set_ = true;
    // Anchor the nominal state to this origin so integration starts from
    // (0,0,0) rather than whatever drifted while waiting for the first fix.
    p_.setZero();
    v_.setZero();
    // p_ represents the body/IMU origin, while the first ENU fix represents
    // the antenna.  Apply the lever arm at initialization as well as updates.
    p_ = -q_.toRotationMatrix() * gnss_lever_arm_;
    heading_anchor_enu_.setZero();
    heading_anchor_set_ = true;

    // Seed the IMU-only dead-reckoning debug track from the same origin/attitude
    // so it is directly comparable to the fused path and the raw GNSS-only path.
    p_imu_only_ = p_;
    v_imu_only_.setZero();
    q_imu_only_ = q_;
    RCLCPP_INFO(get_logger(), "ENU origin set at lat=%.7f lon=%.7f alt=%.3f", lat, lon, alt);
    return;  // origin fix consumed to seed the frame, not as a measurement
  }

  const auto enu = converter_.latlon_to_enu(lat, lon, alt);
  const Eigen::Vector3d pos_meas(enu.e, enu.n, enu.u);

  // Publish the raw (uncorrected) GNSS fix for every accepted PVT message so
  // RViz can be used to visually compare it against the IMU-smoothed
  // /gnss_imu_eskf/path -- large marker/path divergence indicates the IMU
  // prediction is doing meaningful work compensating GNSS noise/gaps.
  const rclcpp::Time gnss_stamp = this->now();
  publishGnssMarker(pos_meas, gnss_stamp);
  publishGnssOnlyPath(pos_meas, gnss_stamp);

  // Orientation for the GNSS-only debug odom: course heading between the
  // immediately previous and current fix (independent of the ESKF's own
  // min_heading_dist_ hysteresis below, so this always reflects the latest
  // fix-to-fix motion).
  if (!prev_gnss_pos_set_) {
    prev_gnss_pos_ = pos_meas;
    prev_gnss_pos_set_ = true;
  } else {
    const double course_dx = pos_meas.x() - prev_gnss_pos_.x();
    const double course_dy = pos_meas.y() - prev_gnss_pos_.y();
    if (std::hypot(course_dx, course_dy) > 1e-3) {
      const double course_heading = std::atan2(course_dy, course_dx);
      gnss_only_orientation_ =
        Eigen::Quaterniond(Eigen::AngleAxisd(course_heading, Eigen::Vector3d::UnitZ()));
    }
    prev_gnss_pos_ = pos_meas;
  }
  publishGnssOnlyOdom(pos_meas, gnss_stamp);
  publishMotionStateRawGnss(pos_meas, gnss_stamp);

  // "IMU-only" debug odom: position from the raw GNSS fix (same as above),
  // orientation from the IMU-only dead-reckoned attitude (q_imu_only_) --
  // shows how the IMU's own attitude estimate looks overlaid on the GNSS
  // track, without pulling in the IMU-only position drift.
  publishImuOnlyOdom(pos_meas, gnss_stamp);

  bool is_fixed = false, is_float = false;
  rtkStateFromCarr(carr, is_fixed, is_float);

  // hacc/vacc are 1-sigma accuracy estimates (meters) reported by the
  // receiver; fall back to fix/carr-quality defaults only if absent/invalid.
  double h_var = (hacc > 0.0 && std::isfinite(hacc)) ? hacc * hacc : default_hacc_ * default_hacc_;
  double v_var = (vacc > 0.0 && std::isfinite(vacc)) ? vacc * vacc : default_vacc_ * default_vacc_;
  if (!(hacc > 0.0 && std::isfinite(hacc))) {
    // No usable hacc: fall back to a coarse quality bucket from carr/fix.
    h_var = is_fixed ? 0.0004 : (is_float ? 0.01 : 1.0);
    v_var = h_var * 4.0;
  }

  Eigen::Matrix3d R_pos = Eigen::Matrix3d::Identity();
  R_pos(0, 0) = h_var;
  R_pos(1, 1) = h_var;
  R_pos(2, 2) = v_var;

  updatePosition(pos_meas, R_pos);

  // Single-antenna heading, following gnss_pvt_enu_odom.cpp: retain the
  // previous anchor while stationary/noisy and form course only after at
  // least min_heading_dist metres of horizontal displacement.
  if (!heading_anchor_set_) {
    heading_anchor_enu_ = pos_meas;
    heading_anchor_set_ = true;
  } else {
    const double dx = pos_meas.x() - heading_anchor_enu_.x();
    const double dy = pos_meas.y() - heading_anchor_enu_.y();
    const double heading_distance = std::hypot(dx, dy);
    if (heading_distance >= min_heading_dist_) {
      const double gnss_heading = std::atan2(dy, dx);
      updateHeading(
        gnss_heading, gnss_heading_std_rad_ * gnss_heading_std_rad_);
      heading_anchor_enu_ = pos_meas;

      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "GNSS heading correction: %.1f deg from %.3f m displacement",
        gnss_heading * 180.0 / M_PI, heading_distance);
    }
  }

  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
    "GNSS update: E=%.3f N=%.3f U=%.3f fix=%d carr=%d(%s) sats=%d hacc=%.3f vacc=%.3f rtk_corr=%s",
    pos_meas.x(), pos_meas.y(), pos_meas.z(), fix, carr,
    is_fixed ? "fixed" : (is_float ? "float" : "none"), sats, hacc, vacc,
    rtk_corrections_active_ ? "yes" : "no");
}

/* ================= ESKF Prediction (IMU strapdown) ================= */
void GnssImuEskfNode::predict(
  const Eigen::Vector3d & acc_meas, const Eigen::Vector3d & gyro_meas, double dt)
{
  // Bias-corrected measurements
  const Eigen::Vector3d acc = acc_meas - ab_;
  const Eigen::Vector3d gyro = gyro_meas - gb_;

  const Eigen::Vector3d dtheta = gyro * dt;
  const double angle = dtheta.norm();
  Eigen::Quaterniond dq;
  if (angle > 1e-12) {
    dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, dtheta.normalized()));
  } else {
    dq = Eigen::Quaterniond(1.0, 0.5 * dtheta.x(), 0.5 * dtheta.y(), 0.5 * dtheta.z());
    dq.normalize();
  }

  // Mid-interval attitude rotates the midpoint accelerometer sample into ENU.
  Eigen::Quaterniond dq_half = Eigen::Quaterniond::Identity().slerp(0.5, dq);
  const Eigen::Matrix3d R_bw = (q_ * dq_half).normalized().toRotationMatrix();
  const Eigen::Vector3d gravity(0.0, 0.0, -GRAVITY);
  const Eigen::Vector3d acc_world = R_bw * acc + gravity;

  // ===== Nominal state propagation =====
  p_ += v_ * dt + 0.5 * acc_world * dt * dt;
  v_ += acc_world * dt;

  q_ = (q_ * dq).normalized();
  // ab_, gb_ modeled as random walk -> unchanged in nominal propagation

  // ===== Error-state transition Jacobian (15x15) =====
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
  F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
  F.block<3, 3>(0, 6) = -0.5 * R_bw * skewSymmetric(acc) * dt * dt;
  F.block<3, 3>(0, 9) = -0.5 * R_bw * dt * dt;
  F.block<3, 3>(3, 6) = -R_bw * skewSymmetric(acc) * dt;
  F.block<3, 3>(3, 9) = -R_bw * dt;
  F.block<3, 3>(6, 6) = dq.conjugate().toRotationMatrix();
  F.block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity() * dt;

  // ===== Process noise (continuous-time densities, discretized by dt) =====
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(STATE_SIZE, STATE_SIZE);
  const double acc_var = acc_noise_density_ * acc_noise_density_;
  const double gyro_var = gyro_noise_density_ * gyro_noise_density_;
  Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * acc_var * dt * dt * dt / 3.0;
  Q.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * acc_var * dt * dt / 2.0;
  Q.block<3, 3>(3, 0) = Q.block<3, 3>(0, 3);
  Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * acc_var * dt;
  Q.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * gyro_var * dt;
  Q.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * acc_bias_rw_ * acc_bias_rw_ * dt;
  Q.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * gyro_bias_rw_ * gyro_bias_rw_ * dt;

  P_ = F * P_ * F.transpose() + Q;
  P_ = 0.5 * (P_ + P_.transpose()); // enforce symmetry against floating-point drift
}

/* ================= ESKF Update (GNSS position) ================= */
void GnssImuEskfNode::updatePosition(
  const Eigen::Vector3d & pos_meas, const Eigen::Matrix3d & R_pos)
{
  // Measurement model: predicted antenna position = body position + R * lever_arm.
  // H w.r.t. the attitude error state picks up -R*skew(lever_arm) because a
  // small rotation error dtheta perturbs the lever-arm's world-frame offset
  // by -R*skew(lever_arm)*dtheta (standard ESKF antenna/lever-arm Jacobian).
  const Eigen::Matrix3d R_bw = q_.toRotationMatrix();
  const Eigen::Vector3d antenna_offset_world = R_bw * gnss_lever_arm_;
  const Eigen::Vector3d p_antenna_pred = p_ + antenna_offset_world;

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, STATE_SIZE);
  H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  H.block<3, 3>(0, 6) = -R_bw * skewSymmetric(gnss_lever_arm_);

  const Eigen::Vector3d y = pos_meas - p_antenna_pred;  // innovation
  const Eigen::Matrix3d S = H * P_ * H.transpose() + R_pos;

  // Solve K*S = P*H^T via LDLT instead of forming S.inverse() explicitly.
  const Eigen::MatrixXd PHt = P_ * H.transpose();
  const Eigen::MatrixXd K = S.ldlt().solve(PHt.transpose()).transpose();

  const Eigen::VectorXd dx = K * y;

  // Joseph-form covariance update: numerically stable (stays symmetric PSD
  // even with imperfect K) versus the simplified (I - K*H)*P form.
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
  const Eigen::MatrixXd IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R_pos * K.transpose();
  P_ = 0.5 * (P_ + P_.transpose());
  // Inject only after the measurement covariance is formed. injectErrorState
  // then applies the attitude reset Jacobian to the corrected covariance.
  injectErrorState(dx);
}

/* ================= ESKF Update (GNSS course heading) ================= */
void GnssImuEskfNode::updateHeading(double heading, double heading_variance)
{
  const Eigen::Matrix3d R_bw = q_.toRotationMatrix();
  const double r00 = R_bw(0, 0);
  const double r10 = R_bw(1, 0);
  const double horizontal_norm_sq = r00 * r00 + r10 * r10;
  if (horizontal_norm_sq < 1e-8) {
    return;  // yaw is singular when the body x axis is vertical
  }

  const double predicted_heading = std::atan2(r10, r00);
  const double innovation = wrapAngle(heading - predicted_heading);

  // Right-multiplicative attitude error Jacobian for
  // yaw = atan2(R(1,0), R(0,0)). This remains correct with nonzero roll/pitch.
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, STATE_SIZE);
  H(0, 7) = (r10 * R_bw(0, 2) - r00 * R_bw(1, 2)) / horizontal_norm_sq;
  H(0, 8) = (r00 * R_bw(1, 1) - r10 * R_bw(0, 1)) / horizontal_norm_sq;

  const double innovation_variance = (H * P_ * H.transpose())(0, 0) + heading_variance;
  if (!std::isfinite(innovation_variance) || innovation_variance <= 0.0) {
    return;
  }

  const Eigen::VectorXd K = P_ * H.transpose() / innovation_variance;
  const Eigen::VectorXd dx = K * innovation;

  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
  const Eigen::MatrixXd IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + heading_variance * K * K.transpose();
  P_ = 0.5 * (P_ + P_.transpose());
  injectErrorState(dx);
}

/* ================= Inject error-state into nominal state ================= */
void GnssImuEskfNode::injectErrorState(const Eigen::VectorXd & dx)
{
  p_ += dx.segment<3>(0);
  v_ += dx.segment<3>(3);

  const Eigen::Vector3d dtheta = dx.segment<3>(6);
  const tf2::Quaternion dq_tf = smallAngleQuat(dtheta);
  Eigen::Quaterniond dq(dq_tf.w(), dq_tf.x(), dq_tf.y(), dq_tf.z());
  q_ = (q_ * dq).normalized();

  ab_ += dx.segment<3>(9);
  gb_ += dx.segment<3>(12);

  // ESKF attitude reset Jacobian: after injecting dtheta into the nominal
  // quaternion, the error state's attitude reference rotates too, so the
  // covariance's attitude rows/cols must be transformed by
  // G_theta = I - skew(0.5 * dtheta) (first-order reset Jacobian).
  Eigen::MatrixXd G = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
  G.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() - skewSymmetric(0.5 * dtheta);
  P_ = G * P_ * G.transpose();
  P_ = 0.5 * (P_ + P_.transpose());
}

/* ================= Publish fused odometry ================= */
void GnssImuEskfNode::publishState(const rclcpp::Time & stamp)
{
  tf2::Quaternion q(q_.x(), q_.y(), q_.z(), q_.w());

  // nav_msgs/Odometry.twist must be expressed in child_frame_id (body frame),
  // not the world/ENU frame the ESKF integrates in -- rotate world velocity
  // into body frame with q^{-1} (and its covariance with R^T P R).
  const Eigen::Matrix3d R_bw = q_.toRotationMatrix();
  const Eigen::Vector3d v_body = R_bw.transpose() * v_;
  const Eigen::Matrix3d P_vel_body = R_bw.transpose() * P_.block<3, 3>(3, 3) * R_bw;

  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = map_frame_;
  odom.child_frame_id = base_frame_;
  odom.pose.pose.position.x = p_.x();
  odom.pose.pose.position.y = p_.y();
  odom.pose.pose.position.z = p_.z();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  odom.twist.twist.linear.x = v_body.x();
  odom.twist.twist.linear.y = v_body.y();
  odom.twist.twist.linear.z = v_body.z();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      odom.pose.covariance[i * 6 + j] = P_(i, j);           // position block
      odom.pose.covariance[(i + 3) * 6 + (j + 3)] = P_(6 + i, 6 + j); // attitude block
      odom.twist.covariance[i * 6 + j] = P_vel_body(i, j);  // velocity block, body frame
    }
  }

  odom_pub_->publish(odom);

  publishMotionState(p_, stamp);

  if (publish_tf_) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = map_frame_;
    tf_msg.child_frame_id = base_frame_;
    tf_msg.transform.translation.x = p_.x();
    tf_msg.transform.translation.y = p_.y();
    tf_msg.transform.translation.z = p_.z();
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(tf_msg);
  }

  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = map_frame_;
  pose.pose = odom.pose.pose;
  path_msg_.header.stamp = stamp;
  path_msg_.poses.push_back(pose);
  path_pub_->publish(path_msg_);
}

/* ================= Publish motion-state classification ================= */
void GnssImuEskfNode::publishMotionState(const Eigen::Vector3d & pos, const rclcpp::Time & stamp)
{
  // This runs every IMU tick (high rate), but "moving" is a state with
  // hysteresis, not a per-tick distance test -- otherwise entering
  // forward/backward for the single tick that crosses motion_pos_deadband_
  // and then immediately falling back to "idle" while displacement
  // re-accumulates from zero produces a one-tick-forward/many-ticks-idle
  // pulse train instead of a held state.
  if (!motion_anchor_set_) {
    motion_anchor_pos_ = pos;
    motion_anchor_set_ = true;
    motion_idle_ref_pos_ = pos;
    motion_idle_ref_time_ = stamp;
  }

  std::string state;

  if (!motion_is_moving_) {
    // Stationary: wait for displacement from the anchor to cross the
    // deadband to enter "moving". The anchor stays fixed the whole time so
    // displacement genuinely accumulates against a stationary reference.
    const double dist_since_anchor = (pos - motion_anchor_pos_).norm();
    if (dist_since_anchor < motion_pos_deadband_) {
      state = "idle";
    } else {
      motion_is_moving_ = true;
      motion_idle_ref_pos_ = motion_anchor_pos_;  // baseline for direction below
      motion_idle_ref_time_ = stamp;
    }
  }

  if (motion_is_moving_) {
    // Moving: direction comes from the ENU displacement since
    // motion_idle_ref_pos_ (a stable, non-per-tick-noisy baseline -- it only
    // advances when the vehicle re-crosses the deadband from that
    // reference), projected onto the vehicle's forward (body +X) axis in
    // ENU. Positive => it actually moved the way it's facing (forward);
    // negative => it moved opposite to where it's facing (reverse), e.g.
    // backing up while still oriented forward.
    const Eigen::Vector3d displacement = pos - motion_idle_ref_pos_;
    const double dist_from_idle_ref = displacement.norm();

    if (dist_from_idle_ref >= motion_pos_deadband_) {
      const Eigen::Vector3d forward_axis_world = q_.toRotationMatrix().col(0);
      const double along_heading = displacement.dot(forward_axis_world);
      state = (along_heading >= 0.0) ? "forward" : "backward";
      motion_idle_ref_pos_ = pos;
      motion_idle_ref_time_ = stamp;
    } else if ((stamp - motion_idle_ref_time_).seconds() >= motion_idle_hold_sec_) {
      // Stayed within the deadband of a fixed reference for the full hold
      // window -- confirmed stopped, not just between two crossings.
      motion_is_moving_ = false;
      motion_anchor_pos_ = pos;
      state = "idle";
    } else {
      // Still within the deadband and hold window hasn't elapsed yet: hold
      // the last known direction rather than pulsing to idle.
      state = last_motion_state_.empty() ? "idle" : last_motion_state_;
    }
  }

  last_motion_state_ = state;

  std_msgs::msg::String msg;
  msg.data = state;
  motion_state_pub_->publish(msg);
}

/* ================= Publish motion-state classification (raw GNSS only) ================= */
void GnssImuEskfNode::publishMotionStateRawGnss(
  const Eigen::Vector3d & pos_meas, const rclcpp::Time & stamp)
{
  // Same deadband/hysteresis structure as publishMotionState(), but driven
  // entirely by raw GNSS: position is the raw fix (pos_meas) and "facing
  // direction" is gnss_only_orientation_ -- the fix-to-fix course heading
  // computed just above in pvtCallback -- instead of the ESKF's fused q_.
  // No IMU data is involved anywhere in this function.
  if (!motion_raw_anchor_set_) {
    motion_raw_anchor_pos_ = pos_meas;
    motion_raw_anchor_set_ = true;
    motion_raw_idle_ref_pos_ = pos_meas;
    motion_raw_idle_ref_time_ = stamp;
  }

  std::string state;

  if (!motion_raw_is_moving_) {
    const double dist_since_anchor = (pos_meas - motion_raw_anchor_pos_).norm();
    if (dist_since_anchor < motion_pos_deadband_) {
      state = "idle";
    } else {
      motion_raw_is_moving_ = true;
      motion_raw_idle_ref_pos_ = motion_raw_anchor_pos_;
      motion_raw_idle_ref_time_ = stamp;
    }
  }

  if (motion_raw_is_moving_) {
    const Eigen::Vector3d displacement = pos_meas - motion_raw_idle_ref_pos_;
    const double dist_from_idle_ref = displacement.norm();

    if (dist_from_idle_ref >= motion_pos_deadband_) {
      const Eigen::Vector3d course_axis_world =
        gnss_only_orientation_.toRotationMatrix().col(0);
      const double along_course = displacement.dot(course_axis_world);
      state = (along_course >= 0.0) ? "forward" : "backward";
      motion_raw_idle_ref_pos_ = pos_meas;
      motion_raw_idle_ref_time_ = stamp;
    } else if ((stamp - motion_raw_idle_ref_time_).seconds() >= motion_idle_hold_sec_) {
      motion_raw_is_moving_ = false;
      motion_raw_anchor_pos_ = pos_meas;
      state = "idle";
    } else {
      state = last_motion_raw_state_.empty() ? "idle" : last_motion_raw_state_;
    }
  }

  last_motion_raw_state_ = state;

  std_msgs::msg::String msg;
  msg.data = state;
  motion_state_raw_gnss_pub_->publish(msg);
}

/* ================= Publish raw GNSS fix marker ================= */
void GnssImuEskfNode::publishGnssMarker(
  const Eigen::Vector3d & pos_meas, const rclcpp::Time & stamp)
{
  // Each accepted raw GNSS fix becomes its own SPHERE marker (unique id) so
  // it is independently visible/selectable in RViz, rather than one growing
  // POINTS list. Comparing the marker cloud against /gnss_imu_eskf/path shows
  // how much the IMU prediction is smoothing out noisy/jumpy raw GNSS between
  // fixes.
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = map_frame_;
  marker.header.stamp = stamp;
  marker.ns = "raw_gnss_fixes";
  marker.id = gnss_marker_id_++;
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = pos_meas.x();
  marker.pose.position.y = pos_meas.y();
  marker.pose.position.z = pos_meas.z();
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.15;
  marker.scale.y = 0.15;
  marker.scale.z = 0.15;
  marker.color.r = 1.0f;
  marker.color.g = 0.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0f;
  marker.lifetime = rclcpp::Duration(0, 0);  // persist until node exit

  gnss_marker_array_.markers.push_back(marker);
  gnss_marker_pub_->publish(gnss_marker_array_);
}

/* ================= Publish GNSS-only path (raw fixes, no IMU fusion) ================= */
void GnssImuEskfNode::publishGnssOnlyPath(
  const Eigen::Vector3d & pos_meas, const rclcpp::Time & stamp)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = map_frame_;
  pose.pose.position.x = pos_meas.x();
  pose.pose.position.y = pos_meas.y();
  pose.pose.position.z = pos_meas.z();
  pose.pose.orientation.w = 1.0;

  gnss_only_path_msg_.header.stamp = stamp;
  gnss_only_path_msg_.poses.push_back(pose);
  gnss_only_path_pub_->publish(gnss_only_path_msg_);
}

/* ================= IMU-only dead-reckoning (debug, never GNSS-corrected) ================= */
void GnssImuEskfNode::predictImuOnly(
  const Eigen::Vector3d & acc_meas, const Eigen::Vector3d & gyro_meas, double dt)
{
  // Mirrors the nominal-state strapdown propagation in predict(), reusing the
  // same (converged) biases so this track isolates "what if GNSS never
  // corrected the filter" rather than also diverging on raw uncorrected bias.
  const Eigen::Vector3d acc = acc_meas - ab_;
  const Eigen::Vector3d gyro = gyro_meas - gb_;

  const Eigen::Vector3d dtheta = gyro * dt;
  const double angle = dtheta.norm();
  Eigen::Quaterniond dq;
  if (angle > 1e-12) {
    dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, dtheta.normalized()));
  } else {
    dq = Eigen::Quaterniond(1.0, 0.5 * dtheta.x(), 0.5 * dtheta.y(), 0.5 * dtheta.z());
    dq.normalize();
  }

  Eigen::Quaterniond dq_half = Eigen::Quaterniond::Identity().slerp(0.5, dq);
  const Eigen::Matrix3d R_bw = (q_imu_only_ * dq_half).normalized().toRotationMatrix();
  const Eigen::Vector3d gravity(0.0, 0.0, -GRAVITY);
  const Eigen::Vector3d acc_world = R_bw * acc + gravity;

  p_imu_only_ += v_imu_only_ * dt + 0.5 * acc_world * dt * dt;
  v_imu_only_ += acc_world * dt;
  q_imu_only_ = (q_imu_only_ * dq).normalized();
}

void GnssImuEskfNode::publishImuOnlyPath(const rclcpp::Time & stamp)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = map_frame_;
  pose.pose.position.x = p_imu_only_.x();
  pose.pose.position.y = p_imu_only_.y();
  pose.pose.position.z = p_imu_only_.z();
  pose.pose.orientation.x = q_imu_only_.x();
  pose.pose.orientation.y = q_imu_only_.y();
  pose.pose.orientation.z = q_imu_only_.z();
  pose.pose.orientation.w = q_imu_only_.w();

  imu_only_path_msg_.header.stamp = stamp;
  imu_only_path_msg_.poses.push_back(pose);
  imu_only_path_pub_->publish(imu_only_path_msg_);
}

/* ================= Publish GNSS-only odom (position = raw fix, orientation =
 * consecutive-fix course heading) ================= */
void GnssImuEskfNode::publishGnssOnlyOdom(
  const Eigen::Vector3d & pos_meas, const rclcpp::Time & stamp)
{
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = map_frame_;
  odom.child_frame_id = base_frame_;
  odom.pose.pose.position.x = pos_meas.x();
  odom.pose.pose.position.y = pos_meas.y();
  odom.pose.pose.position.z = pos_meas.z();
  odom.pose.pose.orientation.x = gnss_only_orientation_.x();
  odom.pose.pose.orientation.y = gnss_only_orientation_.y();
  odom.pose.pose.orientation.z = gnss_only_orientation_.z();
  odom.pose.pose.orientation.w = gnss_only_orientation_.w();

  gnss_only_odom_pub_->publish(odom);
}

/* ================= Publish IMU-only odom (position + orientation both from
 * the IMU-only dead-reckoning track) ================= */
void GnssImuEskfNode::publishImuOnlyOdom(
  const Eigen::Vector3d & pos_meas, const rclcpp::Time & stamp)
{
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = map_frame_;
  odom.child_frame_id = base_frame_;
  odom.pose.pose.position.x = pos_meas.x();
  odom.pose.pose.position.y = pos_meas.y();
  odom.pose.pose.position.z = pos_meas.z();
  odom.pose.pose.orientation.x = q_imu_only_.x();
  odom.pose.pose.orientation.y = q_imu_only_.y();
  odom.pose.pose.orientation.z = q_imu_only_.z();
  odom.pose.pose.orientation.w = q_imu_only_.w();

  const Eigen::Matrix3d R_bw = q_imu_only_.toRotationMatrix();
  const Eigen::Vector3d v_body = R_bw.transpose() * v_imu_only_;
  odom.twist.twist.linear.x = v_body.x();
  odom.twist.twist.linear.y = v_body.y();
  odom.twist.twist.linear.z = v_body.z();

  imu_only_odom_pub_->publish(odom);
}

/* ================= Helpers ================= */

// carr convention (matches gnss_pvt_enu_odom / u-blox style payload): 0 = none,
// 1 = RTK float, 2 = RTK fixed. Returns false (both flags false) if carr is
// out of range, meaning "no RTK carrier solution / use fix-only quality".
bool GnssImuEskfNode::rtkStateFromCarr(int carr, bool & is_fixed, bool & is_float)
{
  is_fixed = (carr == 2);
  is_float = (carr == 1);
  return is_fixed || is_float;
}

double GnssImuEskfNode::wrapAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

Eigen::Matrix3d GnssImuEskfNode::skewSymmetric(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d m;
  m <<     0, -v(2),  v(1),
        v(2),     0, -v(0),
       -v(1),  v(0),     0;
  return m;
}

tf2::Quaternion GnssImuEskfNode::smallAngleQuat(const Eigen::Vector3d & dtheta)
{
  tf2::Quaternion q;
  const double angle = dtheta.norm();
  if (angle > 1e-12) {
    const tf2::Vector3 axis(dtheta.x() / angle, dtheta.y() / angle, dtheta.z() / angle);
    q.setRotation(axis, angle);
  } else {
    q.setValue(0.5 * dtheta.x(), 0.5 * dtheta.y(), 0.5 * dtheta.z(), 1.0);
    q.normalize();
  }
  return q;
}

/* ================= Main ================= */
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GnssImuEskfNode>());
  rclcpp::shutdown();
  return 0;
}
