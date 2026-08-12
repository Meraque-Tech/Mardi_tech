#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>     // ✅ REQUIRED
#include <tf2_ros/transform_broadcaster.h>

#include <std_msgs/msg/float32.hpp>

#include <mutex>
#include <cmath>

class LioRtkGeorefNode : public rclcpp::Node
{
public:
    LioRtkGeorefNode()
    : Node("lio_rtk_georef_node")
    {
        auto qos = rclcpp::SystemDefaultsQoS();

        rtk_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/rtk/odom", qos,
            std::bind(&LioRtkGeorefNode::rtkCallback, this, std::placeholders::_1));

        lio_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", qos,
            std::bind(&LioRtkGeorefNode::lioCallback, this, std::placeholders::_1));
        
        rtk_lio_yaw_pub_ = create_publisher<std_msgs::msg::Float32>(
            "/rtk_lio/yaw",
            rclcpp::SystemDefaultsQoS());

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        RCLCPP_INFO(get_logger(),
            "LIO–RTK georeference node started (atan2-based offset)");
    }

private:
    /* ---------------- Utilities ---------------- */

    double yawFromQuat(const geometry_msgs::msg::Quaternion &q)
    {
        tf2::Quaternion tfq(q.x, q.y, q.z, q.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(tfq).getRPY(roll, pitch, yaw);
        return yaw;
    }

    double normalizeYaw(double yaw)
    {
        return std::atan2(std::sin(yaw), std::cos(yaw));
    }

    /* ---------------- Callbacks ---------------- */

    void rtkCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!has_init_rtk_)
        {
            init_rtk_odom_ = *msg;
            has_init_rtk_ = true;
            RCLCPP_INFO(get_logger(), "RTK initial pose captured");
        }

        rtk_odom_ = *msg;
        rtk_ready_ = true;
        tryComputeOffset();
    }

    void lioCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!has_init_lio_)
        {
            init_lio_odom_ = *msg;
            has_init_lio_ = true;
            RCLCPP_INFO(get_logger(), "LIO initial pose captured");
        }

        lio_odom_ = *msg;
        lio_ready_ = true;
        tryComputeOffset();
    }


    /* ---------------- Core Logic ---------------- */

    void tryComputeOffset()
    {
         if (init_offset_computed_) return;

        if (!rtk_ready_ || !lio_ready_ ||
            !has_init_rtk_ || !has_init_lio_)
            return;

        // --- RTK displacement from initial ---
        double dx_rtk = rtk_odom_.pose.pose.position.x -
                        init_rtk_odom_.pose.pose.position.x;
        double dy_rtk = rtk_odom_.pose.pose.position.y -
                        init_rtk_odom_.pose.pose.position.y;

        // --- LIO displacement from initial ---
        double dx_lio = lio_odom_.pose.pose.position.x -
                        init_lio_odom_.pose.pose.position.x;
        double dy_lio = lio_odom_.pose.pose.position.y -
                        init_lio_odom_.pose.pose.position.y;

        double dist_rtk = std::hypot(dx_rtk, dy_rtk);
        double dist_lio = std::hypot(dx_lio, dy_lio);

        // Require meaningful movement
        if (dist_rtk < 1.0 || dist_lio < 1.0)
            return;

        double yaw_rtk = std::atan2(dy_rtk, dx_rtk);
        double yaw_lio = std::atan2(dy_lio, dx_lio);

        double yaw_offset = normalizeYaw(yaw_lio - yaw_rtk);

        
        RCLCPP_INFO(get_logger(),
            "INIT-based motion | RTK dist=%.2f yaw=%.2f deg | "
            "LIO dist=%.2f yaw=%.2f deg | offset=%.2f deg",
            dist_rtk, yaw_rtk * 180.0 / M_PI,
            dist_lio, yaw_lio * 180.0 / M_PI,
            yaw_offset * 180.0 / M_PI);

        std_msgs::msg::Float32 yaw_msg;
        yaw_msg.data = yaw_offset;  // or processed value

        rtk_lio_yaw_pub_->publish(yaw_msg);

        init_offset_computed_ = true;


        // map → odom TF (yaw only)
        // tf2::Quaternion q;
        // q.setRPY(0.0, 0.0, yaw_offset);

        // map_to_odom_.setOrigin(tf2::Vector3(0.0, 0.0, 0.0));
        // map_to_odom_.setRotation(q);

        // publishTf();
    }


    void publishTf()
    {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = now();
        tf_msg.header.frame_id = "map";
        tf_msg.child_frame_id = "odom";

        tf_msg.transform.translation.x = map_to_odom_.getOrigin().x();
        tf_msg.transform.translation.y = map_to_odom_.getOrigin().y();
        tf_msg.transform.translation.z = 0.0;

        tf_msg.transform.rotation.x = map_to_odom_.getRotation().x();
        tf_msg.transform.rotation.y = map_to_odom_.getRotation().y();
        tf_msg.transform.rotation.z = map_to_odom_.getRotation().z();
        tf_msg.transform.rotation.w = map_to_odom_.getRotation().w();

        tf_broadcaster_->sendTransform(tf_msg);
    }

    /* ---------------- Members ---------------- */

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr rtk_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr lio_sub_;

    nav_msgs::msg::Odometry rtk_odom_;
    nav_msgs::msg::Odometry lio_odom_;

    bool rtk_ready_{false};
    bool lio_ready_{false};
    bool offset_computed_{false};

    tf2::Transform map_to_odom_;   // ✅ now valid

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::mutex mutex_;

    nav_msgs::msg::Odometry prev_rtk_odom_;
    nav_msgs::msg::Odometry prev_lio_odom_;
    bool has_prev_rtk_{false};
    bool has_prev_lio_{false};

    nav_msgs::msg::Odometry init_rtk_odom_;
    nav_msgs::msg::Odometry init_lio_odom_;
    bool has_init_rtk_{false};
    bool has_init_lio_{false};
    bool init_offset_computed_{false};

    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr rtk_lio_yaw_pub_;


};

/* ---------------- main ---------------- */

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LioRtkGeorefNode>());
    rclcpp::shutdown();
    return 0;
}
