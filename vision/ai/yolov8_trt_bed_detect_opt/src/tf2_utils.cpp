#include "ros_utils.h"

void AlgoUtils::send_static_target_tf(rclcpp::Time now, std::string parent_frame, std::string child_frame, double x, double y, double z, double th_x, double th_y, double th_z)
{
    geometry_msgs::msg::TransformStamped tf_msg;
    // rclcpp::Time now = this->get_clock()->now();
    tf_msg.header.stamp = now;
    tf_msg.header.frame_id = parent_frame;
    tf_msg.child_frame_id = child_frame;

    tf_msg.transform.translation.x = x;
    tf_msg.transform.translation.y = y;
    tf_msg.transform.translation.z = z;

    tf2::Quaternion q;
    q.setRPY(th_x, th_y, th_z);
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();
    
    static_target_tf->sendTransform(tf_msg);
}

void AlgoUtils::send_dynamic_target_tf(rclcpp::Time now, std::string parent_frame, std::string child_frame, double x, double y, double z, double th_x, double th_y, double th_z)
{
    geometry_msgs::msg::TransformStamped tf_msg;
    // rclcpp::Time now = this->get_clock()->now();
    tf_msg.header.stamp = now;
    tf_msg.header.frame_id = parent_frame;
    tf_msg.child_frame_id = child_frame;

    tf_msg.transform.translation.x = x;
    tf_msg.transform.translation.y = y;
    tf_msg.transform.translation.z = z;

    tf2::Quaternion q;
    q.setRPY(th_x, th_y, th_z);
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();
    
    dynamic_target_tf->sendTransform(tf_msg);
}

geometry_msgs::msg::TransformStamped AlgoUtils::tf_sub(std::string parent_frame, std::string child_frame)
{
  geometry_msgs::msg::TransformStamped transformStamped;
  transformStamped = tf_buffer->lookupTransform(
    parent_frame.c_str(), child_frame.c_str(),
    tf2::TimePointZero);

    return transformStamped;
}

geometry_msgs::msg::PoseStamped AlgoUtils::tf2_pose(rclcpp::Time now, geometry_msgs::msg::TransformStamped &tf_msg)
{   
    geometry_msgs::msg::PoseStamped goal_pose;
    goal_pose.header.stamp = now;
    goal_pose.header.frame_id = "map";
    goal_pose.pose.position.x = tf_msg.transform.translation.x;
    goal_pose.pose.position.y = tf_msg.transform.translation.y;
    goal_pose.pose.position.z = 0.0;
    goal_pose.pose.orientation.x = 0.0;
    goal_pose.pose.orientation.y = 0.0;
    goal_pose.pose.orientation.z = tf_msg.transform.rotation.z;
    goal_pose.pose.orientation.w = tf_msg.transform.rotation.w;
    return goal_pose;
}


