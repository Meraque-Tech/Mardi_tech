#include <gps_utils/gps_utils.hpp>
#include <rtk_sf/rtk_imu_sf.hpp>


/* ================= Constructor ================= */
DualRTKHeadingNode::DualRTKHeadingNode()
: Node("rtk_mid_raw_node")
{
    auto qos = rclcpp::SystemDefaultsQoS();
    this->set_parameter(rclcpp::Parameter("use_sim_time", false));

    this->declare_parameter<double>("motion_threshold_xy", 0.3);
    this->declare_parameter<std::string>("map_frame", "map");
    this->declare_parameter<std::string>("odom_frame", "odom");
    this->declare_parameter<std::string>("base_frame", "rtk_pub");

    map_frame_ = this->get_parameter("map_frame").as_string();
    odom_frame_ = this->get_parameter("odom_frame").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();

    sub_left_ = create_subscription<sensor_msgs::msg::NavSatFix>(
        "/rtk/left/fix",
        qos,
        [this](const sensor_msgs::msg::NavSatFix::SharedPtr msg)
        {
            left_fix_ = *msg;
            left_received_ = true;
            computeHeading();
        });

    sub_right_ = create_subscription<sensor_msgs::msg::NavSatFix>(
        "/rtk/right/fix",
        qos,
        [this](const sensor_msgs::msg::NavSatFix::SharedPtr msg)
        {
            right_fix_ = *msg;
            right_received_ = true;
            computeHeading();
        });

    sub_left_status_ = create_subscription<prime_msgs::msg::RtkStatus>(
        "/rtk/left/status", qos,
        [this](const prime_msgs::msg::RtkStatus::SharedPtr msg)
        {
            left_carrier_soln_ = msg->carrier_soln;
        });

    sub_right_status_ = create_subscription<prime_msgs::msg::RtkStatus>(
        "/rtk/right/status", qos,
        [this](const prime_msgs::msg::RtkStatus::SharedPtr msg)
        {
            right_carrier_soln_ = msg->carrier_soln;
        });

    mid_fix_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>("/rtk/mid/fix", qos);
    odom_pub_    = create_publisher<nav_msgs::msg::Odometry>("/rtk/pub/odom", qos);
    path_pub_    = create_publisher<nav_msgs::msg::Path>("/rtk/pub/path", 10);
    points_pub_  = create_publisher<visualization_msgs::msg::Marker>("/rtk/pub/points", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    initPointMarker();

    RCLCPP_INFO(get_logger(), "Rtk Mid Raw Node started");
    RCLCPP_INFO(
        get_logger(),
        "Frames: map_frame='%s', odom_frame='%s', base_frame='%s'",
        map_frame_.c_str(),
        odom_frame_.c_str(),
        base_frame_.c_str());
}


/* ================= Heading ================= */
void DualRTKHeadingNode::computeHeading()
{
    if (!left_received_ || !right_received_)
        return;

    // Firmware publishes STATUS_NO_FIX for RTK float — cannot use NavSatFix.status.
    // Use carrier_soln from /rtk/*/status: accept FIXED and FLOAT, reject NO FIX.
    // if (left_carrier_soln_ == "NO FIX" || right_carrier_soln_ == "NO FIX")
    //     return;

    auto ecef_left  = converter_.latlon_to_ecef(left_fix_.latitude,  left_fix_.longitude,  left_fix_.altitude);
    auto ecef_right = converter_.latlon_to_ecef(right_fix_.latitude, right_fix_.longitude, right_fix_.altitude);

    // ===== Midpoint in ECEF → LLA =====
    gps_utils::ECEF mid_ecef;
    mid_ecef.x = 0.5 * (ecef_left.x + ecef_right.x);
    mid_ecef.y = 0.5 * (ecef_left.y + ecef_right.y);
    mid_ecef.z = 0.5 * (ecef_left.z + ecef_right.z);

    auto mid_lla = converter_.ecef_to_latlon(mid_ecef.x, mid_ecef.y, mid_ecef.z);

    // ===== Publish NavSatFix =====
    sensor_msgs::msg::NavSatFix mid_fix;
    mid_fix.header.stamp    = this->now();
    mid_fix.header.frame_id = "rtk_mid";
    mid_fix.latitude   = mid_lla.lat;
    mid_fix.longitude  = mid_lla.lon;
    mid_fix.altitude   = mid_lla.alt;
    // Firmware reports STATUS_NO_FIX even for RTK float, so set status from carrier_soln
    mid_fix.status.service = left_fix_.status.service;
    if (left_carrier_soln_ == "FIXED" && right_carrier_soln_ == "FIXED")
        mid_fix.status.status = sensor_msgs::msg::NavSatStatus::STATUS_GBAS_FIX;  // RTK fixed
    else if (left_carrier_soln_ != "NO FIX" && right_carrier_soln_ != "NO FIX")
        mid_fix.status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;       // RTK float
    else
        mid_fix.status.status = sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX;
    mid_fix.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_KNOWN;
    for (int i = 0; i < 9; ++i)
        mid_fix.position_covariance[i] =
            0.25 * (left_fix_.position_covariance[i] + right_fix_.position_covariance[i]);
    
    mid_fix_pub_->publish(mid_fix);

    // ===== UTM midpoint + heading =====
    auto utm_left  = converter_.latlon_to_utm(left_fix_.latitude,  left_fix_.longitude);
    auto utm_right = converter_.latlon_to_utm(right_fix_.latitude, right_fix_.longitude);

    double utm_mid_e = 0.5 * (utm_left.easting  + utm_right.easting);
    double utm_mid_n = 0.5 * (utm_left.northing + utm_right.northing);

    double dx = utm_right.easting  - utm_left.easting;
    double dy = utm_right.northing - utm_left.northing;

    double baseline_yaw = std::atan2(dy, dx);
    double yaw = std::atan2(std::sin(baseline_yaw + M_PI_2),
                            std::cos(baseline_yaw + M_PI_2));

    // ===== Map origin on first fix =====
    if (!map_origin_set_)
    {
        map_origin_x_  = utm_mid_e;
        map_origin_y_  = utm_mid_n;
        map_origin_set_ = true;
        RCLCPP_INFO(get_logger(), "Map origin set: E=%.3f N=%.3f", map_origin_x_, map_origin_y_);
        return;
    }

    double x_local = utm_mid_e - map_origin_x_;
    double y_local = utm_mid_n - map_origin_y_;

    auto stamp = this->now();

    // ===== TF: map → rtk_pub =====
    auto tf_msg = publishTF(
        stamp, x_local, y_local, 0.0, 0.0, 0.0, yaw, map_frame_, base_frame_);
    tf_broadcaster_->sendTransform(tf_msg);

    // ===== Odometry =====
    publishOdom(stamp, x_local, y_local, 0.0, 0.0, 0.0, yaw,
                0.0, 0.0, 0.0, map_frame_, base_frame_, odom_pub_);

    // ===== Path =====
    updateAndPublishPath(stamp, x_local, y_local, yaw, map_frame_, path_rtk_, path_pub_);

    // ===== Marker (trail points) =====
    geometry_msgs::msg::Point p;
    p.x = x_local;
    p.y = y_local;
    p.z = 0.0;
    points_marker_.points.push_back(p);
    points_marker_.header.stamp = stamp;
    points_pub_->publish(points_marker_);
}


/* ================= Utils ================= */
bool DualRTKHeadingNode::validFix(
    const sensor_msgs::msg::NavSatFix &fix)
{
    return fix.status.status > sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX;
}

void DualRTKHeadingNode::initPointMarker()
{
    points_marker_.header.frame_id = map_frame_;
    points_marker_.ns   = "rtk_mid_points";
    points_marker_.id   = 0;
    points_marker_.type = visualization_msgs::msg::Marker::POINTS;
    points_marker_.action = visualization_msgs::msg::Marker::ADD;
    points_marker_.scale.x = 0.1;
    points_marker_.scale.y = 0.1;
    points_marker_.color.r = 0.0f;
    points_marker_.color.g = 1.0f;
    points_marker_.color.b = 0.5f;
    points_marker_.color.a = 1.0f;
}


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DualRTKHeadingNode>());
    rclcpp::shutdown();
    return 0;
}
