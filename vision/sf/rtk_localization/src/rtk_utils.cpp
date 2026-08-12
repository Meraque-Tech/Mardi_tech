
#include <angle_utils/angle_utils.hpp>
#include <gps_utils/gps_utils.hpp>
#include <rtk_sf/rtk_imu_sf.hpp>

/* ================= TF ================= */
geometry_msgs::msg::TransformStamped DualRTKHeadingNode::publishTF(
                const rclcpp::Time &stamp,
                double x, double y, double z,
                double roll, double pitch, double yaw,
                const std::string &parent_frame,
                const std::string &child_frame)
{
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    q.normalize();

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = parent_frame;
    tf.child_frame_id = child_frame;

    tf.transform.translation.x = x;
    tf.transform.translation.y = y;
    tf.transform.translation.z = z;

    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();

    return tf;
}

/* ================= Pose ================= */
void DualRTKHeadingNode::publishPose(
    const rclcpp::Time &stamp,
    double x, double y, double yaw,
    const std::string &frame_id,
    const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr &pub)
{
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);

    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = frame_id;
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();
    pose.pose.orientation.w = q.w();

    pub->publish(pose);
}

/* ================= Path ================= */
void DualRTKHeadingNode::updateAndPublishPath(
    const rclcpp::Time &stamp,
    double x, double y, double yaw,
    const std::string &frame_id,
    nav_msgs::msg::Path &path,
    const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pub)
{
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);

    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = frame_id;
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();
    pose.pose.orientation.w = q.w();

    path.header = pose.header;
    path.poses.push_back(pose);

    pub->publish(path);
}


/* ================= Odometry ================= */
void DualRTKHeadingNode::publishOdom(
    const rclcpp::Time &stamp,
    double x, double y, double z,
    double roll, double pitch, double yaw,
    double vx, double vy, double wz,
    const std::string &frame_id,
    const std::string &child_frame_id,
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pub)
{
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = frame_id;
    odom.child_frame_id = child_frame_id;

    /* ---- Pose ---- */
    odom.pose.pose.position.x = x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.position.z = z;

    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    q.normalize();

    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    /* ---- Twist (BODY FRAME) ---- */
    odom.twist.twist.linear.x  = vx;   // forward
    odom.twist.twist.linear.y  = vy;   // lateral
    odom.twist.twist.linear.z  = 0.0;

    odom.twist.twist.angular.x = 0.0;
    odom.twist.twist.angular.y = 0.0;
    odom.twist.twist.angular.z = wz;   // yaw rate

    pub->publish(odom);
}


double DualRTKHeadingNode::odomDistance2D(const nav_msgs::msg::Odometry& a,
                      const nav_msgs::msg::Odometry& b)
{
    const double dx = a.pose.pose.position.x - b.pose.pose.position.x;
    const double dy = a.pose.pose.position.y - b.pose.pose.position.y;

    return std::sqrt(dx * dx + dy * dy);
}