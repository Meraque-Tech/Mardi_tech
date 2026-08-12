#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <gps_utils/gps_utils.hpp>

#include <nlohmann/json.hpp>

#include <cmath>

/*
 * Subscribes to:
 *   /gnss/pvt        (std_msgs/String, JSON: {"type":"pvt","lat":..,"lon":..,"alt":..,
 *                      "fix":.., "carr":.., "sats":.., "hacc":.., "vacc":.., "pdop":..})
 *   /gnss/rtk_status (std_msgs/Bool)  -- true when base station correction is being received
 *
 * Publishes:
 *   /gnss/odom  (nav_msgs/Odometry) in a local ENU frame (map -> gnss_base_link),
 *   /gnss/path  (nav_msgs/Path)
 *   TF: map -> gnss_base_link
 *
 * ENU origin is latched on the first fix received (fix >= 3, i.e. 3D fix or better).
 *
 * Heading is derived from consecutive ENU fixes (current - previous) and only
 * updated once the displacement exceeds `min_heading_dist` (default 0.1 m) to
 * avoid noisy heading jumps while stationary or moving slowly.
 */
class GnssPvtEnuOdomNode : public rclcpp::Node
{
public:
  GnssPvtEnuOdomNode()
  : Node("gnss_pvt_enu_odom_node")
  {
    this->declare_parameter<std::string>("map_frame", "map");
    this->declare_parameter<std::string>("base_frame", "gnss_base_link");
    this->declare_parameter<int>("min_fix_type", 3);
    this->declare_parameter<double>("min_heading_dist", 0.1);

    map_frame_ = this->get_parameter("map_frame").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    min_fix_type_ = this->get_parameter("min_fix_type").as_int();
    min_heading_dist_ = this->get_parameter("min_heading_dist").as_double();

    auto qos = rclcpp::SystemDefaultsQoS();

    pvt_sub_ = create_subscription<std_msgs::msg::String>(
      "/gnss/pvt", qos,
      std::bind(&GnssPvtEnuOdomNode::pvtCallback, this, std::placeholders::_1));

    rtk_status_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/gnss/rtk_status", qos,
      [this](const std_msgs::msg::Bool::SharedPtr msg) {
        rtk_corrections_active_ = msg->data;
      });

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/gnss/odom", qos);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/gnss/path", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    path_msg_.header.frame_id = map_frame_;

    RCLCPP_INFO(get_logger(), "GNSS PVT -> ENU odom node started (map_frame='%s', base_frame='%s')",
      map_frame_.c_str(), base_frame_.c_str());
  }

private:
  void pvtCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    nlohmann::json j;
    try {
      j = nlohmann::json::parse(msg->data);
    } catch (const nlohmann::json::parse_error & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Failed to parse /gnss/pvt payload: %s", e.what());
      return;
    }

    if (!j.contains("lat") || !j.contains("lon")) {
      return;
    }

    const double lat = j.value("lat", 0.0);
    const double lon = j.value("lon", 0.0);
    const double alt = j.value("alt", 0.0);
    const int fix = j.value("fix", 0);
    const int sats = j.value("sats", 0);

    if (fix < min_fix_type_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "GNSS fix type %d below minimum %d, skipping", fix, min_fix_type_);
      return;
    }

    if (!converter_.is_origin_set()) {
      converter_.set_enu_origin(lat, lon, alt);
      RCLCPP_INFO(get_logger(), "ENU origin set at lat=%.7f lon=%.7f alt=%.3f", lat, lon, alt);
    }

    const auto enu = converter_.latlon_to_enu(lat, lon, alt);

    // ===== Heading from consecutive fixes (current - prev), gated by min displacement =====
    if (prev_enu_set_) {
      const double dx = enu.e - prev_enu_.e;
      const double dy = enu.n - prev_enu_.n;
      const double dist = std::hypot(dx, dy);
      if (dist >= min_heading_dist_) {
        heading_ = std::atan2(dy, dx);
        heading_set_ = true;
        prev_enu_ = enu;
      }
    } else {
      prev_enu_ = enu;
      prev_enu_set_ = true;
    }

    const auto stamp = this->now();

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, heading_);

    // ===== TF: map -> base_frame =====
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = map_frame_;
    tf_msg.child_frame_id = base_frame_;
    tf_msg.transform.translation.x = enu.e;
    tf_msg.transform.translation.y = enu.n;
    tf_msg.transform.translation.z = enu.u;
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(tf_msg);

    // ===== Odometry =====
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = map_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = enu.e;
    odom.pose.pose.position.y = enu.n;
    odom.pose.pose.position.z = enu.u;
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    const double pos_var = fixVariance(fix);
    // Heading is only observable once two fixes >= min_heading_dist apart have
    // been seen; until then report it as effectively unobserved.
    const double yaw_var = heading_set_ ? 0.05 : 1e6;
    odom.pose.covariance[0] = pos_var;   // x
    odom.pose.covariance[7] = pos_var;   // y
    odom.pose.covariance[14] = pos_var * 4.0; // z (vertical typically worse)
    odom.pose.covariance[21] = 1e6;      // roll (unobserved)
    odom.pose.covariance[28] = 1e6;      // pitch (unobserved)
    odom.pose.covariance[35] = yaw_var;  // yaw

    odom_pub_->publish(odom);

    // ===== Path =====
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = map_frame_;
    pose.pose = odom.pose.pose;
    path_msg_.header.stamp = stamp;
    path_msg_.poses.push_back(pose);
    path_pub_->publish(path_msg_);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "GNSS ENU: E=%.3f N=%.3f U=%.3f | heading=%.1fdeg fix=%d sats=%d rtk_corr=%s",
      enu.e, enu.n, enu.u, heading_ * 180.0 / M_PI, fix, sats,
      rtk_corrections_active_ ? "yes" : "no");
  }

  static double fixVariance(int fix)
  {
    // Rough covariance by fix type: 3D fix vs RTK float vs RTK fixed.
    // fix: 3 = 3D fix, carr(ier) solution distinguishes float/fixed upstream;
    // kept simple here since only "fix" is guaranteed present in the payload.
    switch (fix) {
      case 4: return 0.0004;   // RTK fixed  (~2 cm)
      case 5: return 0.01;     // RTK float  (~10 cm)
      default: return 1.0;     // plain 3D fix (~1 m)
    }
  }

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pvt_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr rtk_status_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  nav_msgs::msg::Path path_msg_;
  gps_utils::GpsConverter converter_;

  std::string map_frame_;
  std::string base_frame_;
  int min_fix_type_;
  double min_heading_dist_;
  bool rtk_corrections_active_{false};

  gps_utils::ENU prev_enu_{};
  bool prev_enu_set_{false};
  double heading_{0.0};
  bool heading_set_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GnssPvtEnuOdomNode>());
  rclcpp::shutdown();
  return 0;
}
