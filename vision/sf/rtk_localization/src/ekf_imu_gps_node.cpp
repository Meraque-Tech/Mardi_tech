#include <rtk_sf/ekf_imu_gps_node.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

/* ================= Constructor ================= */
EKFFusionNode::EKFFusionNode() 
    : Node("ekf_fusion_node"),
      initialized_(false),
      origin_set_(false)
{
    auto qos = rclcpp::SystemDefaultsQoS();
    
    // ========== Declare Parameters ==========
    this->declare_parameter("acc_noise", 0.1);
    this->declare_parameter("gyro_noise", 0.01);
    this->declare_parameter("acc_bias_noise", 0.001);
    this->declare_parameter("gyro_bias_noise", 0.0001);
    this->declare_parameter("gps_noise_xy", 0.5);
    this->declare_parameter("gps_noise_z", 1.0);
    this->declare_parameter("world_frame", "map");
    this->declare_parameter("base_frame", "base_link");
    this->declare_parameter("publish_tf", true);
    
    // Get parameters
    acc_noise_ = this->get_parameter("acc_noise").as_double();
    gyro_noise_ = this->get_parameter("gyro_noise").as_double();
    acc_bias_noise_ = this->get_parameter("acc_bias_noise").as_double();
    gyro_bias_noise_ = this->get_parameter("gyro_bias_noise").as_double();
    gps_noise_xy_ = this->get_parameter("gps_noise_xy").as_double();
    gps_noise_z_ = this->get_parameter("gps_noise_z").as_double();
    world_frame_ = this->get_parameter("world_frame").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    publish_tf_ = this->get_parameter("publish_tf").as_bool();
    
    // ========== Initialize EKF ==========
    initializeEKF();
    
    // ========== Subscribers ==========
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", qos,
        std::bind(&EKFFusionNode::imuCallback, this, std::placeholders::_1));
    
    gps_sub_ = create_subscription<sensor_msgs::msg::NavSatFix>(
        "/rtk/mid/fix", qos,
        std::bind(&EKFFusionNode::gpsCallback, this, std::placeholders::_1));
    
    // ========== Publishers ==========
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
        "/ekf/odometry", qos);
    
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/ekf/pose", qos);
    
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    
    RCLCPP_INFO(get_logger(), "EKF Fusion Node initialized");
    RCLCPP_INFO(get_logger(), "  acc_noise: %.4f, gyro_noise: %.4f", acc_noise_, gyro_noise_);
    RCLCPP_INFO(get_logger(), "  gps_noise: [%.2f, %.2f, %.2f]", gps_noise_xy_, gps_noise_xy_, gps_noise_z_);
}

/* ================= Initialize EKF ================= */
void EKFFusionNode::initializeEKF()
{
    // State vector: [pos, vel, euler, acc_bias, gyro_bias]
    x_ = Eigen::VectorXd::Zero(STATE_SIZE);
    
    // Covariance matrix - initial uncertainty
    P_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    P_.block<3, 3>(0, 0) *= 10.0;    // position uncertainty
    P_.block<3, 3>(3, 3) *= 5.0;     // velocity uncertainty
    P_.block<3, 3>(6, 6) *= 0.1;     // orientation uncertainty
    P_.block<3, 3>(9, 9) *= 0.01;    // acc bias uncertainty
    P_.block<3, 3>(12, 12) *= 0.001; // gyro bias uncertainty
    
    // Process noise covariance
    Q_ = Eigen::MatrixXd::Zero(STATE_SIZE, STATE_SIZE);
    Q_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.001; // position noise
    Q_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * acc_noise_ * acc_noise_;
    Q_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * gyro_noise_ * gyro_noise_;
    Q_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * acc_bias_noise_ * acc_bias_noise_;
    Q_.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * gyro_bias_noise_ * gyro_bias_noise_;
    
    // GPS measurement noise covariance
    R_gps_ = Eigen::Matrix3d::Identity();
    R_gps_(0, 0) = gps_noise_xy_ * gps_noise_xy_;
    R_gps_(1, 1) = gps_noise_xy_ * gps_noise_xy_;
    R_gps_(2, 2) = gps_noise_z_ * gps_noise_z_;
}

/* ================= IMU Callback ================= */
void EKFFusionNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    // Convert message timestamp to rclcpp::Time
    rclcpp::Time current_time(msg->header.stamp);
    
    if (!initialized_) {
        last_imu_time_ = current_time;
        initialized_ = true;
        
        // Initialize orientation from IMU if available
        if (msg->orientation_covariance[0] > 0) {
            tf2::Quaternion q(
                msg->orientation.x,
                msg->orientation.y,
                msg->orientation.z,
                msg->orientation.w
            );
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            x_(6) = roll;
            x_(7) = pitch;
            x_(8) = yaw;
        }
        
        RCLCPP_INFO(get_logger(), "EKF initialized with first IMU measurement");
        return;
    }
    
    // Calculate time delta
    double dt = (current_time - last_imu_time_).seconds();
    last_imu_time_ = current_time;
    
    if (dt <= 0.0 || dt > 1.0) {
        RCLCPP_WARN(get_logger(), "Invalid IMU dt: %.4f seconds, skipping prediction", dt);
        return;
    }
    
    // Prediction step
    predictIMU(msg, dt);
    
    // Publish state after prediction
    publishState();
}

/* ================= GPS Callback ================= */
void EKFFusionNode::gpsCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
    if (!initialized_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, 
            "Waiting for IMU to initialize EKF");
        return;
    }
    
    if (!isGPSValid(msg)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, 
            "GPS fix not valid, skipping update");
        return;
    }
    
    // Set origin on first valid GPS
    if (!origin_set_) {
        origin_lat_ = msg->latitude;
        origin_lon_ = msg->longitude;
        origin_alt_ = msg->altitude;
        origin_set_ = true;
        
        // Initialize position state with first GPS measurement
        x_.segment<3>(0) = Eigen::Vector3d::Zero();
        
        RCLCPP_INFO(get_logger(), "GPS origin set: [%.8f, %.8f, %.2f]", 
            origin_lat_, origin_lon_, origin_alt_);
        return;
    }
    
    // Update step
    updateGPS(msg);
}

/* ================= EKF Prediction (IMU) ================= */
void EKFFusionNode::predictIMU(const sensor_msgs::msg::Imu::SharedPtr msg, double dt)
{
    // Extract current state
    Eigen::Vector3d pos = x_.segment<3>(0);
    Eigen::Vector3d vel = x_.segment<3>(3);
    Eigen::Vector3d euler = x_.segment<3>(6);
    Eigen::Vector3d acc_bias = x_.segment<3>(9);
    Eigen::Vector3d gyro_bias = x_.segment<3>(12);
    
    // Get IMU measurements
    Eigen::Vector3d acc_meas(
        msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z
    );
    
    Eigen::Vector3d gyro_meas(
        msg->angular_velocity.x,
        msg->angular_velocity.y,
        msg->angular_velocity.z
    );
    
    // Correct measurements with bias
    Eigen::Vector3d acc = acc_meas - acc_bias;
    Eigen::Vector3d gyro = gyro_meas - gyro_bias;
    
    // Rotation matrix from body to world
    Eigen::Matrix3d R_wb = eulerToRotationMatrix(euler(0), euler(1), euler(2));
    
    // Gravity vector in world frame
    Eigen::Vector3d gravity(0.0, 0.0, -GRAVITY);
    
    // Acceleration in world frame
    Eigen::Vector3d acc_world = R_wb * acc + gravity;
    
    // ========== State Prediction ==========
    // Position: p_k+1 = p_k + v_k * dt + 0.5 * a_k * dt^2
    Eigen::Vector3d pos_new = pos + vel * dt + 0.5 * acc_world * dt * dt;
    
    // Velocity: v_k+1 = v_k + a_k * dt
    Eigen::Vector3d vel_new = vel + acc_world * dt;
    
    // Orientation: euler_k+1 = euler_k + gyro_k * dt (simplified)
    Eigen::Vector3d euler_new = euler + gyro * dt;
    
    // Normalize yaw to [-pi, pi]
    while (euler_new(2) > M_PI) euler_new(2) -= 2.0 * M_PI;
    while (euler_new(2) < -M_PI) euler_new(2) += 2.0 * M_PI;
    
    // Biases remain constant (random walk model)
    Eigen::Vector3d acc_bias_new = acc_bias;
    Eigen::Vector3d gyro_bias_new = gyro_bias;
    
    // Update state
    x_.segment<3>(0) = pos_new;
    x_.segment<3>(3) = vel_new;
    x_.segment<3>(6) = euler_new;
    x_.segment<3>(9) = acc_bias_new;
    x_.segment<3>(12) = gyro_bias_new;
    
    // ========== Jacobian Matrix (State Transition) ==========
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    
    // Position depends on velocity
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    
    // Velocity depends on acceleration (through rotation)
    Eigen::Matrix3d dR_droll = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dR_dpitch = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dR_dyaw = Eigen::Matrix3d::Zero();
    
    // Simplified: approximate derivatives
    // In full implementation, compute exact derivatives of rotation matrix
    F.block<3, 3>(3, 6) = skewSymmetric(R_wb * acc) * dt;
    F.block<3, 3>(3, 9) = -R_wb * dt;
    
    // Orientation depends on gyroscope
    F.block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity() * dt;
    
    // ========== Covariance Prediction ==========
    P_ = F * P_ * F.transpose() + Q_ * dt;
}

/* ================= EKF Update (GPS) ================= */
void EKFFusionNode::updateGPS(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
    // Convert GPS to local ENU coordinates
    Eigen::Vector3d gps_pos = llhToENU(msg->latitude, msg->longitude, msg->altitude);
    
    // Measurement model: z = H * x
    // We measure position directly, so H picks out position from state
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, STATE_SIZE);
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    
    // Innovation (measurement residual)
    Eigen::Vector3d z_pred = x_.segment<3>(0); // Predicted measurement
    Eigen::Vector3d y = gps_pos - z_pred;      // Innovation
    
    // Innovation covariance
    Eigen::Matrix3d S = H * P_ * H.transpose() + R_gps_;
    
    // Kalman gain
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
    
    // State update
    x_ = x_ + K * y;
    
    // Covariance update
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    P_ = (I - K * H) * P_;
    
    RCLCPP_DEBUG(get_logger(), "GPS update: innovation=[%.3f, %.3f, %.3f]", 
        y(0), y(1), y(2));
}

/* ================= Publish State ================= */
void EKFFusionNode::publishState()
{
    auto now = this->now();
    
    // Extract state
    Eigen::Vector3d pos = x_.segment<3>(0);
    Eigen::Vector3d vel = x_.segment<3>(3);
    Eigen::Vector3d euler = x_.segment<3>(6);
    
    // Convert Euler to Quaternion
    tf2::Quaternion q;
    q.setRPY(euler(0), euler(1), euler(2));
    
    // ========== Publish Odometry ==========
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = now;
    odom_msg.header.frame_id = world_frame_;
    odom_msg.child_frame_id = base_frame_;
    
    odom_msg.pose.pose.position.x = pos(0);
    odom_msg.pose.pose.position.y = pos(1);
    odom_msg.pose.pose.position.z = pos(2);
    
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();
    
    odom_msg.twist.twist.linear.x = vel(0);
    odom_msg.twist.twist.linear.y = vel(1);
    odom_msg.twist.twist.linear.z = vel(2);
    
    // Copy covariance (6x6 for pose, 6x6 for twist)
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (i < 3 && j < 3) {
                odom_msg.pose.covariance[i * 6 + j] = P_(i, j);
                odom_msg.twist.covariance[i * 6 + j] = P_(3 + i, 3 + j);
            }
        }
    }
    
    odom_pub_->publish(odom_msg);
    
    // ========== Publish Pose ==========
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header = odom_msg.header;
    pose_msg.pose = odom_msg.pose.pose;
    pose_pub_->publish(pose_msg);
    
    // ========== Publish TF ==========
    if (publish_tf_) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header = odom_msg.header;
        transform.child_frame_id = base_frame_;
        transform.transform.translation.x = pos(0);
        transform.transform.translation.y = pos(1);
        transform.transform.translation.z = pos(2);
        transform.transform.rotation.x = q.x();
        transform.transform.rotation.y = q.y();
        transform.transform.rotation.z = q.z();
        transform.transform.rotation.w = q.w();
        
        tf_broadcaster_->sendTransform(transform);
    }
}

/* ================= Helper Functions ================= */

Eigen::Matrix3d EKFFusionNode::eulerToRotationMatrix(double roll, double pitch, double yaw)
{
    double cr = cos(roll), sr = sin(roll);
    double cp = cos(pitch), sp = sin(pitch);
    double cy = cos(yaw), sy = sin(yaw);
    
    Eigen::Matrix3d R;
    R << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
         sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
         -sp,     cp * sr,                cp * cr;
    
    return R;
}

Eigen::Vector3d EKFFusionNode::rotationMatrixToEuler(const Eigen::Matrix3d& R)
{
    double roll = atan2(R(2, 1), R(2, 2));
    double pitch = atan2(-R(2, 0), sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
    double yaw = atan2(R(1, 0), R(0, 0));
    
    return Eigen::Vector3d(roll, pitch, yaw);
}

Eigen::Matrix3d EKFFusionNode::skewSymmetric(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d m;
    m <<     0, -v(2),  v(1),
          v(2),     0, -v(0),
         -v(1),  v(0),     0;
    return m;
}

Eigen::Vector3d EKFFusionNode::llhToENU(double lat, double lon, double alt)
{
    // WGS84 parameters
    const double a = 6378137.0;           // semi-major axis (m)
    const double e2 = 0.00669437999014;   // first eccentricity squared
    
    // Convert to radians
    double lat_rad = lat * DEG_TO_RAD;
    double lon_rad = lon * DEG_TO_RAD;
    double lat0_rad = origin_lat_ * DEG_TO_RAD;
    double lon0_rad = origin_lon_ * DEG_TO_RAD;
    
    // Differences
    double d_lat = lat_rad - lat0_rad;
    double d_lon = lon_rad - lon0_rad;
    double d_alt = alt - origin_alt_;
    
    // Radius of curvature
    double N = a / sqrt(1.0 - e2 * sin(lat0_rad) * sin(lat0_rad));
    
    // ENU coordinates (approximate for small distances)
    double east = d_lon * (N + origin_alt_) * cos(lat0_rad);
    double north = d_lat * (N * (1.0 - e2) + origin_alt_);
    double up = d_alt;
    
    return Eigen::Vector3d(east, north, up);
}

bool EKFFusionNode::isGPSValid(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
    // Check for valid fix
    if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX) {
        return false;
    }
    
    // Check for reasonable values
    if (std::isnan(msg->latitude) || std::isnan(msg->longitude) || std::isnan(msg->altitude)) {
        return false;
    }
    
    return true;
}

/* ================= Main ================= */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFFusionNode>());
    rclcpp::shutdown();
    return 0;
}