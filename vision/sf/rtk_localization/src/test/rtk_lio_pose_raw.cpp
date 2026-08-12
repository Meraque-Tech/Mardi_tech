#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <gps_utils/gps_utils.hpp>
#include <angle_utils/angle_utils.hpp>


#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <std_srvs/srv/set_bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <Eigen/Dense>


#include <cmath>
#include <mutex>

class DualRTKHeadingNode : public rclcpp::Node
{
public:
    DualRTKHeadingNode()
    : Node("dual_rtk_heading_node")
    {
        this->declare_parameter<double>("actual_baseline", 1.20);   // meters
        this->declare_parameter<double>("baseline_tolerance", 0.25); // meters

        this->get_parameter("actual_baseline", actual_baseline_);
        this->get_parameter("baseline_tolerance", baseline_tolerance_);

        auto qos = rclcpp::SystemDefaultsQoS();

        sub_left_ = create_subscription<sensor_msgs::msg::NavSatFix>(
            "/rtk/left/fix",
            qos,
            std::bind(&DualRTKHeadingNode::leftCallback, this, std::placeholders::_1));

        sub_right_ = create_subscription<sensor_msgs::msg::NavSatFix>(
            "/rtk/right/fix",
            qos,
            std::bind(&DualRTKHeadingNode::rightCallback, this, std::placeholders::_1));
        
        

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/rtk/odom", qos);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        path_pub_ = create_publisher<nav_msgs::msg::Path>("/rtk/path", 10);
        odom_path_pub_ = create_publisher<nav_msgs::msg::Path>("/odom_path", 10);

        points_pub_ = create_publisher<visualization_msgs::msg::Marker>("/rtk/points", 10);


        odom_raw_pub_ = create_publisher<nav_msgs::msg::Odometry>(
            "/odom", rclcpp::SystemDefaultsQoS());

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/Odometry",
            rclcpp::SystemDefaultsQoS(),
            std::bind(&DualRTKHeadingNode::odomCallback, this, std::placeholders::_1));
        
        stop_rtk_srv_ = create_service<std_srvs::srv::SetBool>(
            "/rtk/enable",
            std::bind(
                &DualRTKHeadingNode::handleRtkEnable,
                this,
                std::placeholders::_1,
                std::placeholders::_2));
        
        rtk_lio_yaw_sub_ = create_subscription<std_msgs::msg::Float32>(
            "/rtk_lio/yaw",
            rclcpp::SystemDefaultsQoS(),
            std::bind(&DualRTKHeadingNode::rtkLioYawCallback, this, std::placeholders::_1));

        initPointMarker();


        RCLCPP_INFO(get_logger(), "Dual RTK Heading Node started");
    }

private:
    void rtkLioYawCallback(const std_msgs::msg::Float32::SharedPtr msg)
    {
        rtk_lio_yaw_ = msg->data;

        RCLCPP_DEBUG(get_logger(),
            "RTK-LIO yaw received: %.2f deg", rtk_lio_yaw_);
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        nav_msgs::msg::Odometry odom_raw = *msg;

        // Extract current odometry pose directly into Eigen vector
        curr_odom_[0] = msg->pose.pose.position.x;
        curr_odom_[1] = msg->pose.pose.position.y;
        angle_utils::Euler euler = angle_utils::eulerFromQuaternion(msg->pose.pose.orientation);
        curr_odom_[2] = euler.yaw;

          // Get yaw only (for 2D)
        // double yaw = tf2::getYaw(q);
        // curr_odom_[2] = yaw;

        // Optional: enforce naming consistency
        odom_raw.header.frame_id = "map";
        odom_raw.child_frame_id = "base_link";
        odom_raw_pub_->publish(odom_raw);

        // 90 degree rotation offset is there
        
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = msg->header.stamp;   // keep original time
        pose.header.frame_id = "map";
        pose.pose = msg->pose.pose;

        odom_path_.header.frame_id = "map";
        odom_path_.header.stamp = pose.header.stamp;
        odom_path_.poses.push_back(pose);
        odom_path_pub_->publish(odom_path_);
    }
    /* ---------------- Callbacks ---------------- */

    void leftCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        left_fix_ = *msg;
        left_received_ = true;
        computeHeading();
    }

    void rightCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        right_fix_ = *msg;
        right_received_ = true;
        computeHeading();
    }

    /* ---------------- Core Logic ---------------- */

    Eigen::Vector3d transformToGlobal(const Eigen::Vector3d& local_pose) {
        // Extract components
        double x_lio = local_pose[0];
        double y_lio = local_pose[1];
        double yaw_lio = local_pose[2];
        
        // Calculate delta from initial odometry pose
        double dx = x_lio - init_odom_[0];
        double dy = y_lio - init_odom_[1];
        double dyaw = yaw_lio - init_odom_[2];
        
        // Apply rotation alignment
        double cos_align = std::cos(yaw_alignment_);
        double sin_align = std::sin(yaw_alignment_);
        
        double E_local = dx * cos_align - dy * sin_align;
        double N_local = dx * sin_align + dy * cos_align;
        
        // Add GNSS origin offset
        double E_global = E_local + init_gnss_[0];
        double N_global = N_local + init_gnss_[1];
        double yaw_global = dyaw + yaw_alignment_;
        
        return Eigen::Vector3d(E_global, N_global, yaw_global);
    }

    void computeHeading()
    {
        if (!left_received_ || !right_received_)
            return;

        if (!validFix(left_fix_) || !validFix(right_fix_))
            return;

        


        auto utm_left = converter_.latlon_to_utm(left_fix_.latitude, left_fix_.longitude);
        auto utm_right = converter_.latlon_to_utm(right_fix_.latitude, right_fix_.longitude);

        // RCLCPP_INFO_THROTTLE(
        //     get_logger(),
        //     *get_clock(),
        //     1000,
        //     "Left RTK UTM -> E: %.3f, N: %.3f, Zone: %d%c | "
        //     "Right RTK UTM -> E: %.3f, N: %.3f, Zone: %d%c",
        //     utm_left.easting,
        //     utm_left.northing,
        //     utm_left.zone,
        //     utm_left.band,
        //     utm_right.easting,
        //     utm_right.northing,
        //     utm_right.zone,
        //     utm_right.band);
        
        // [x′ y′​]=[[0 -1]
        //         [1 0]] [x y​]

        double dx = utm_right.easting - utm_left.easting;   // East difference
        double dy = utm_right.northing - utm_left.northing; // North difference

        // CORRECTED: atan2 expects (north, east) for ENU heading
        // Heading = 0° is North, 90° is East
        double yaw = std::atan2(dx, dy);  // atan2(E, N) gives heading from North

        // yaw = yaw + M_PI / 2.0;         // Adjust to vehicle heading

        x = (utm_left.easting + utm_right.easting) / 2.0;
        y = (utm_left.northing + utm_right.northing) / 2.0;

        if (!map_origin_set_)
        {
            map_origin_x_ = x;
            map_origin_y_ = y;
            map_origin_set_ = true;

            RCLCPP_INFO(
                get_logger(),
                "Map origin set at UTM: E=%.3f N=%.3f",
                map_origin_x_, map_origin_y_);
        }

        double x_local = x - map_origin_x_;
        double y_local = y - map_origin_y_;

        if(rtk_lio_yaw_){
            auto rotated = angle_utils::rotatePointByYaw(x_local, y_local, rtk_lio_yaw_);
            // x_local = rotated.first;
            // y_local = rotated.second;



            // init_gnss_[0] = rotated.first;   // East
            // init_gnss_[1] = rotated.second;  // North

            // init_odom_ = curr_odom_;
            // yaw_alignment_ = rtk_lio_yaw_;

            // found_gnss_

        }
        


        geometry_msgs::msg::Point p;
        p.x = x_local;
        p.y = y_local;
        p.z = 0.0;

        points_marker_.points.push_back(p);
        points_marker_.header.stamp = now();

        points_pub_->publish(points_marker_);


        yaw = std::atan2(std::sin(yaw), std::cos(yaw));

        double yaw_deg = yaw * 180.0 / M_PI;

        // RCLCPP_INFO_THROTTLE(
        //     get_logger(),
        //     *get_clock(),
        //     1000,
        //     "RTK Heading: %.2f deg (%.3f rad) | Midpoint UTM -> E: %.3f, N: %.3f",
        //     yaw_deg,
        //     yaw,
        //     x,
        //     y);
        
        double baseline = std::hypot(dx, dy);
        // RCLCPP_INFO_THROTTLE(
        //     get_logger(),
        //     *get_clock(),
        //     1000,
        //     "RTK Baseline Distance: %.3f m",
        //     baseline);

        baseline_error_ = std::abs(baseline - actual_baseline_);
        if (baseline_error_ > baseline_tolerance_)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                1000,
                "RTK Baseline Error: %.3f m exceeds tolerance of %.3f m",
                baseline_error_,
                baseline_tolerance_);
        }

        if(publish_rtk_odom_){

            nav_msgs::msg::Odometry odom;
            odom.header.stamp = now();
            odom.header.frame_id = "map";
            odom.child_frame_id = "base_link";

            odom.pose.pose.position.x = x_local;
            odom.pose.pose.position.y = y_local;
            odom.pose.pose.position.z = 0.0;

            tf2::Quaternion q;
            q.setRPY(0.0, 0.0, yaw);
            odom.pose.pose.orientation.x = q.x();
            odom.pose.pose.orientation.y = q.y();
            odom.pose.pose.orientation.z = q.z();
            odom.pose.pose.orientation.w = q.w();


            nav_msgs::msg::Odometry global_odom = createOdom(
                x_local,
                y_local,
                yaw
            );
            odom_pub_->publish(global_odom);
            
            // geometry_msgs::msg::TransformStamped tf_msg;
            // tf_msg.header.stamp = now();
            // tf_msg.header.frame_id = "map";
            // tf_msg.child_frame_id = "base_link";

            // tf_msg.transform.translation.x = x_local;
            // tf_msg.transform.translation.y = y_local;
            // tf_msg.transform.translation.z = 0.0;

            // tf_msg.transform.rotation.x = q.x();
            // tf_msg.transform.rotation.y = q.y();
            // tf_msg.transform.rotation.z = q.z();
            // tf_msg.transform.rotation.w = q.w();
            // tf_broadcaster_->sendTransform(tf_msg);


            geometry_msgs::msg::PoseStamped pose;
            pose.header.stamp = now();
            pose.header.frame_id = "map";

            pose.pose.position.x = x_local;
            pose.pose.position.y = y_local;
            pose.pose.position.z = 0.0;

            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            pose.pose.orientation.w = q.w();

            path_msg_.header.frame_id = "map";
            // Add pose to path
            path_msg_.poses.push_back(pose);
            path_msg_.header.stamp = pose.header.stamp;

            // Publish path
            path_pub_->publish(path_msg_);

        }

        


        



    }

    private:
    /* ---------------- Simple Odometry Creation ---------------- */

     nav_msgs::msg::Odometry createOdom(double x, double y, double yaw)
    {
        nav_msgs::msg::Odometry odom;
        
        // Use provided timestamp or current time
        odom.header.stamp = now();
        odom.header.frame_id = "map";
        odom.child_frame_id = "base_link";
        
        // Position
        odom.pose.pose.position.x = x;
        odom.pose.pose.position.y = y;
        odom.pose.pose.position.z = 0.0;
        
        // Orientation
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();
        
        return odom;
    }

    /* ---------------- Utilities ---------------- */

    bool validFix(const sensor_msgs::msg::NavSatFix &fix)
    {
        return fix.status.status >= sensor_msgs::msg::NavSatStatus::STATUS_FIX;
    }

    void initPointMarker()
    {
        points_marker_.header.frame_id = "map";
        points_marker_.ns = "rtk_points";
        points_marker_.id = 0;
        points_marker_.type = visualization_msgs::msg::Marker::POINTS;
        points_marker_.action = visualization_msgs::msg::Marker::ADD;

        // Point size
        points_marker_.scale.x = 0.1;
        points_marker_.scale.y = 0.1;

        // Green color
        points_marker_.color.r = 0.0f;
        points_marker_.color.g = 1.0f;
        points_marker_.color.b = 0.0f;
        points_marker_.color.a = 1.0f;
    }

    void handleRtkEnable(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                            std::shared_ptr<std_srvs::srv::SetBool::Response> response)
    {
        publish_rtk_odom_ = request->data;

        response->success = true;
        response->message = publish_rtk_odom_
            ? "RTK odom publishing ENABLED"
            : "RTK odom publishing DISABLED";

        RCLCPP_WARN(
            get_logger(),
            "%s",
            response->message.c_str());
    }

    /* ---------------- Members ---------------- */

    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_left_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_right_;

    sensor_msgs::msg::NavSatFix left_fix_;
    sensor_msgs::msg::NavSatFix right_fix_;

    bool left_received_{false};
    bool right_received_{false};
    bool origin_set_{false};

    double lat0_{0.0};
    double lon0_{0.0};

    std::mutex mutex_;
    gps_utils::GpsConverter converter_;
    

    double x_left, y_left,x_right, y_right, x, y;

    double actual_baseline_;        // meters (physical antenna separation)
    double baseline_tolerance_;     // meters (allowed error)
    double baseline_error_;         // meters (computed error)

    bool map_origin_set_{false};
    double map_origin_x_{0.0};
    double map_origin_y_{0.0};

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr odom_path_pub_;

    nav_msgs::msg::Path path_msg_;
    nav_msgs::msg::Path odom_path_;


    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr points_pub_;
    visualization_msgs::msg::Marker points_marker_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_raw_pub_;

    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr stop_rtk_srv_;

    bool publish_rtk_odom_{true};

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr rtk_lio_yaw_sub_;

    float rtk_lio_yaw_;

    Eigen::Vector3d init_gnss_;
    Eigen::Vector3d init_odom_;
    Eigen::Vector3d curr_odom_;
    double yaw_alignment_; 


};

/* ---------------- main ---------------- */

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DualRTKHeadingNode>());
    rclcpp::shutdown();
    return 0;
}
