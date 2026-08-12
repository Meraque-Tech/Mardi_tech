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
#include <Eigen/Dense>



#include <cmath>
#include <mutex>
#include <deque>

class DualRTKHeadingNode : public rclcpp::Node
{
public:
    DualRTKHeadingNode()
    : Node("geo_ref_lio_node")
    {
        this->declare_parameter<double>("actual_baseline", 1.20);   // meters
        this->declare_parameter<double>("baseline_tolerance", 0.25); // meters
        this->declare_parameter<int>("min_gnss_samples", 10);       // min samples for calibration
        this->declare_parameter<int>("min_odom_samples", 100);       // min samples for calibration
        this->declare_parameter<double>("max_calibration_time", 5.0); // seconds
        this->declare_parameter<bool>("auto_calibrate", true);


        this->get_parameter("actual_baseline", actual_baseline_);
        this->get_parameter("baseline_tolerance", baseline_tolerance_);

        this->get_parameter("min_gnss_samples", min_gnss_samples_);
        this->get_parameter("min_odom_samples", min_odom_samples_);
        this->get_parameter("max_calibration_time", max_calibration_time_);
        this->get_parameter("auto_calibrate", auto_calibrate_);

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

        initPointMarker();


        RCLCPP_INFO(get_logger(), "Dual RTK Heading Node started");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {

        // if(collecting_calibration_data_ && !calibration_complete_){
        //     lidar_odom_queue_.push_back(*msg);

        //      // Check if we have enough samples
        //     if (lidar_odom_queue_.size() >= min_odom_samples_) {
        //         performCalibration();
        //     }
        // }

        // if (calibration_complete_) {
        //     // Transform LIDAR odometry to geo-referenced frame
        //     // publishGeoReferencedOdometry(msg);
        // }

        // nav_msgs::msg::Odometry odom_raw = *msg;

        // odom_path_.header.frame_id = "map";

        // Optional: enforce naming consistency
        // odom_raw.header.frame_id = "map";
        // odom_raw.child_frame_id = "base_link";
        // odom_raw_pub_->publish(odom_raw);
        
        // geometry_msgs::msg::PoseStamped pose;
        // pose.header.stamp = msg->header.stamp;   // keep original time
        // pose.header.frame_id = "map";
        // pose.pose = msg->pose.pose;
        // odom_path_.header.stamp = pose.header.stamp;
        // odom_path_.poses.push_back(pose);

        // odom_path_pub_->publish(odom_path_);
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

    void computeHeading()
    {
        if (!left_received_ || !right_received_)
            return;

        if (!validFix(left_fix_) || !validFix(right_fix_))
            return;
        
        auto utm_left = converter_.latlon_to_utm(left_fix_.latitude, left_fix_.longitude);
        auto utm_right = converter_.latlon_to_utm(right_fix_.latitude, right_fix_.longitude);

        double dx = utm_right.easting - utm_left.easting;
        double dy = utm_right.northing - utm_left.northing;
        double yaw = std::atan2(dy, dx);   // radians
        // yaw = yaw + M_PI / 2.0;          // Adjust to vehicle heading

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



        auto rotated = angle_utils::rotatePointByYaw(x, y, -M_PI / 2.0);

        x_local = rotated.first;
        y_local = rotated.second;



        yaw = std::atan2(y_local, x_local);



        yaw = std::atan2(std::sin(yaw), std::cos(yaw));

        RCLCPP_INFO(
            get_logger(),
            "RTK Position: E=%.3f N=%.3f Yaw=%.3f",
            x_local, y_local, yaw);

        double yaw_deg = yaw * 180.0 / M_PI;
        
        double baseline = std::hypot(dx, dy);

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

        // GNSSSample sample;
        // sample.timestamp = now();
        // sample.x = x;
        // sample.y = y;
        // sample.yaw = yaw;
        // sample.utm_zone = utm_left.zone;
        // sample.utm_band = utm_left.band;

        // if(collecting_calibration_data_ && !calibration_complete_){
        //     gnss_samples_.push_back(sample);

        //      // Check if we have enough samples
        //     if (gnss_samples_.size() >= min_gnss_samples_) {
        //         performCalibration();
        //     }
        // }
        

        


        



    }

    void performCalibration()
    {
        if (gnss_samples_.empty() || lidar_odom_queue_.empty()) {
            RCLCPP_ERROR(get_logger(), "Not enough data for calibration");
            return;
        }


        // Calculate average GNSS position and orientation
        double avg_x = 0, avg_y = 0, avg_yaw = 0;
        for (const auto& sample : gnss_samples_) {
            avg_x += sample.x;
            avg_y += sample.y;
            avg_yaw += sample.yaw;
        }
        avg_x /= gnss_samples_.size();
        avg_y /= gnss_samples_.size();
        avg_yaw /= gnss_samples_.size();


        RCLCPP_INFO(
            get_logger(),
            "Calibration complete. Avg GNSS Position: (%.3f, %.3f), Yaw: %.3f deg",
            avg_x, avg_y, avg_yaw * 180.0 / M_PI);
        
        // Get corresponding LIDAR odometry (average of last few)
        Eigen::Vector3d lidar_pos(0, 0, 0);
        double lidar_yaw = 0;
        int count = lidar_odom_queue_.size();
        
        auto it = lidar_odom_queue_.rbegin();
        for (int i = 0; i < count && it != lidar_odom_queue_.rend(); ++i, ++it) {
            lidar_pos.x() += it->pose.pose.position.x;
            lidar_pos.y() += it->pose.pose.position.y;
            
            // Extract yaw from quaternion
            tf2::Quaternion q(
                it->pose.pose.orientation.x,
                it->pose.pose.orientation.y,
                it->pose.pose.orientation.z,
                it->pose.pose.orientation.w);
            double roll, pitch, yaw;
            tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
            lidar_yaw += yaw;
        }
        lidar_pos /= count;
        lidar_yaw /= count;

        RCLCPP_INFO(
            get_logger(),
            "Avg LIDAR Position: (%.3f, %.3f), Yaw: %.3f deg",
            lidar_pos.x(), lidar_pos.y(), lidar_yaw * 180.0 / M_PI);
        
        

        transform_.translation.x = avg_x;
        transform_.translation.y = avg_y;
        transform_.translation.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, avg_yaw);
        // Set the rotation (identity for now)
        transform_.rotation.x = q.x();
        transform_.rotation.y = q.y();
        transform_.rotation.z = q.z();
        transform_.rotation.w = q.w();


        lidar_ref_pose_ = lidar_pos;
        lidar_ref_yaw_ = lidar_yaw;

        calibration_complete_ = true;
        collecting_calibration_data_ = false;


        RCLCPP_INFO(get_logger(), "Calibration complete!");
        // RCLCPP_INFO(get_logger(), "Map origin: (%.3f, %.3f)", avg_x, avg_y);
        // RCLCPP_INFO(get_logger(), "Map yaw: %.3f rad (%.1f deg)", avg_yaw, avg_yaw * 180.0 / M_PI);

    }

     void publishGeoReferencedOdometry(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
    {
        if(!calibration_complete_) return;

        Eigen::Vector3d lidar_pos(
            odom_msg->pose.pose.position.x,
            odom_msg->pose.pose.position.y,
            odom_msg->pose.pose.position.z);

        tf2::Quaternion q(
            odom_msg->pose.pose.orientation.x,
            odom_msg->pose.pose.orientation.y,
            odom_msg->pose.pose.orientation.z,
            odom_msg->pose.pose.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        
        Eigen::Vector3d real_pos = lidar_pos - lidar_ref_pose_;
        double real_yaw = yaw - lidar_ref_yaw_;

        // RCLCPP_INFO(
        //     get_logger(),
        //     "Real Position: (%.3f, %.3f), Yaw: %.3f deg",
        //     real_pos.x(), real_pos.y(), real_yaw * 180.0 / M_PI);


        // double cos_yaw = std::cos(real_yaw);
        // double sin_yaw = std::sin(real_yaw);

        // Eigen::Vector3d map_pose;
        // map_pose.x() = transform_.translation.x + (cos_yaw * real_pos.x() - sin_yaw * real_pos.y());
        // map_pose.y() = transform_.translation.y + (sin_yaw * real_pos.x() + cos_yaw * real_pos.y());
        // map_pose.z() = transform_.translation.z + real_pos.z();

        // double map_yaw = real_yaw;


        // nav_msgs::msg::Odometry geo_referenced_odom = *odom_msg;
        // geo_referenced_odom.pose.pose.position.x = real_pos.x();
        // geo_referenced_odom.pose.pose.position.y = real_pos.y();
        // geo_referenced_odom.pose.pose.position.z = real_pos.z();

        // tf2::Quaternion q_geo;
        // q_geo.setRPY(0.0, 0.0, real_yaw);
        // geo_referenced_odom.pose.pose.orientation.x = q_geo.x();
        // geo_referenced_odom.pose.pose.orientation.y = q_geo.y();
        // geo_referenced_odom.pose.pose.orientation.z = q_geo.z();
        // geo_referenced_odom.pose.pose.orientation.w = q_geo.w();

        // odom_pub_->publish(geo_referenced_odom);
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

    bool publish_rtk_odom_{false};

    std::deque<GNSSSample> gnss_samples_;
    std::deque<nav_msgs::msg::Odometry> lidar_odom_queue_;

    // Calibration
    bool collecting_calibration_data_{true};
    bool calibration_complete_{false};

    int min_gnss_samples_, min_odom_samples_;
    double max_calibration_time_;
    bool auto_calibrate_;
    
    rclcpp::Time calibration_start_time_;
    geometry_msgs::msg::Transform transform_;

    Eigen::Vector3d lidar_ref_pose_;
    double lidar_ref_yaw_{0.0};




};

/* ---------------- main ---------------- */

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DualRTKHeadingNode>());
    rclcpp::shutdown();
    return 0;
}
