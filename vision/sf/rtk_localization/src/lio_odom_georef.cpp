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
    : Node("dual_rtk_odom_node")
    {
        this->declare_parameter<double>("actual_baseline", 1.20);   // meters
        this->declare_parameter<double>("baseline_tolerance", 0.25); // meters

        // Physical offset from LIDAR to vehicle center (RTK mid)
        // LIDAR is at (0.196, 0.618) in vehicle frame
        // Vehicle center (RTK mid) is at (0, 0)
        this->declare_parameter<double>("lidar_to_vehicle_x", 0.196);   // LIDAR X in vehicle frame
        this->declare_parameter<double>("lidar_to_vehicle_y", 0.618);   // LIDAR Y in vehicle frame


        this->get_parameter("actual_baseline", actual_baseline_);
        this->get_parameter("baseline_tolerance", baseline_tolerance_);

        this->get_parameter("lidar_to_vehicle_x", lidar_to_vehicle_x_);
        this->get_parameter("lidar_to_vehicle_y", lidar_to_vehicle_y_);

        auto qos = rclcpp::SystemDefaultsQoS();

        sub_left_ = create_subscription<sensor_msgs::msg::NavSatFix>(
            "/rtk/left/fix", qos,
            [this](const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
                left_fix_ = *msg;
                left_received_ = true;
                computeHeading();
            });

         sub_right_ = create_subscription<sensor_msgs::msg::NavSatFix>(
            "/rtk/right/fix", qos,
            [this](const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
                right_fix_ = *msg;
                right_received_ = true;
                computeHeading();
            });
        
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/Odometry",
            qos,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                 processAndPublishOdom(msg);
            });
        
        
        path_pub_ = create_publisher<nav_msgs::msg::Path>("/rtk/path", 10);
        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/rtk/odom", qos);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        
        points_pub_ = create_publisher<visualization_msgs::msg::Marker>("/rtk/points", 10);

        odom_raw_pub_ = create_publisher<nav_msgs::msg::Odometry>("/lio/odom", qos);
        odom_path_pub_ = create_publisher<nav_msgs::msg::Path>("/lio/odom/path", 10);

        rtk_lio_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/rtk/lio/odom", qos);
        rtk_lio_odom_path_pub_ = create_publisher<nav_msgs::msg::Path>("/rtk/lio/odom/path", 10);

        
        stop_rtk_srv_ = create_service<std_srvs::srv::SetBool>(
            "/rtk/enable",
            std::bind(
                &DualRTKHeadingNode::handleRtkEnable,
                this,
                std::placeholders::_1,
                std::placeholders::_2));
        
        gnss_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>("/gnss/global_pose/latlon", 10);
        lio_gnss_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>("/lio/global_pose/latlon", 10);
        
        

        initPointMarker();

        RCLCPP_INFO(get_logger(), "Dual RTK Heading Node started");
    }

private:
    /* ---------------- Callbacks ---------------- */

    void tryComputeOffset()
    {
        if (init_offset_computed_) return;

        if (!has_init_rtk_ || !has_init_lio_)
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

        if (dist_rtk < 0.5 || dist_lio < 0.5)
            return;

        // RCLCPP_INFO(get_logger(),
        //         "INIT-based motion | RTK dist=%.2f "
        //         "LIO dist=%.2f",
        //         dist_rtk,
        //         dist_lio);


        double yaw_rtk = std::atan2(dy_rtk, dx_rtk);
        double yaw_lio = std::atan2(dy_lio, dx_lio);

        //  If you want to go from LIO to RTK frame for yaw offset
        // double yaw_offset = angle_utils::normalizeAngle(yaw_lio - yaw_rtk);
        //  If you want to go from RTK to LIO frame for yaw offset
        double yaw_offset = angle_utils::normalizeAngle(yaw_rtk - yaw_lio);



        
        RCLCPP_INFO(get_logger(),
            "INIT-based motion | RTK dist=%.2f yaw=%.2f deg | "
            "LIO dist=%.2f yaw=%.2f deg | offset=%.2f deg",
            dist_rtk, yaw_rtk * 180.0 / M_PI,
            dist_lio, yaw_lio * 180.0 / M_PI,
            yaw_offset * 180.0 / M_PI);

        
        // actual_yaw_offset_ = yaw_offset;
        yaw_alignment_ = yaw_offset;


        // std_msgs::msg::Float32 yaw_msg;
        // yaw_msg.data = yaw_offset;  // or processed value

        // rtk_lio_yaw_pub_->publish(yaw_msg);

        
        init_gnss_[0] = curr_gnss_[0];
        init_gnss_[1] = curr_gnss_[1];
        init_gnss_[2] = 0.0;
        

        RCLCPP_INFO(
            get_logger(),
            "RTK GNSS Origin initialized at: E=%.3f, N=%.3f",
            init_gnss_[0],
            init_gnss_[1]);
        

            
        
        init_offset_computed_ = true;

        


        // map → odom TF (yaw only)
        // tf2::Quaternion q;
        // q.setRPY(0.0, 0.0, yaw_offset);

        // map_to_odom_.setOrigin(tf2::Vector3(0.0, 0.0, 0.0));
        // map_to_odom_.setRotation(q);

        // publishTf();
    }

    void processAndPublishOdom(
        const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        /* ===============================
        * 1. Extract odometry → Eigen
        * =============================== */
        curr_odom_[0] = msg->pose.pose.position.x;
        curr_odom_[1] = msg->pose.pose.position.y;

        angle_utils::Euler euler = angle_utils::eulerFromQuaternion(msg->pose.pose.orientation);
        curr_odom_[2] = euler.yaw;   // yaw only (2D)

        /* ===============================
        * 2. Publish raw odometry
        * =============================== */
        nav_msgs::msg::Odometry odom_raw = *msg;
        odom_raw.header.frame_id = "map";
        odom_raw.child_frame_id = "base_link";

        Eigen::Vector3d lio_pose  = curr_odom_;

        

        // RCLCPP_INFO(get_logger(),
        //     "LIO Pose: X=%.2f, Y=%.2f, Yaw=%.2f ",
        //     curr_odom_[0],
        //     curr_odom_[1],
        //     curr_odom_[2]);
        
        if (yaw_alignment_){
            Eigen::Vector3d global_pose = transformToGlobalSimple(lio_pose);
            // RCLCPP_INFO(get_logger(),
            //     "LIO Pose Transformed to Global: E=%.3f, N=%.3f, Yaw=%.3f deg",
            //     global_pose[0],
            //     global_pose[1],
            //     global_pose[2] * 180.0 / M_PI);

                if(utm_initialized_){
                    auto latlon = converter_.utm_to_latlon(global_pose[0], global_pose[1],  utm_zone_, utm_band_);
                    // RCLCPP_INFO(get_logger(), "Global LatLon: Lat=%.8f, Lon=%.8f", latlon.lat, latlon.lon);

                    sensor_msgs::msg::NavSatFix msg;
                    msg.header.stamp = now();
                    msg.header.frame_id = "gps";

                    msg.latitude  = latlon.lat;
                    msg.longitude = latlon.lon;
                    msg.altitude  = 0.0;

                    msg.status.status  = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
                    msg.status.service = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;

                    msg.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;

                    lio_gnss_pub_->publish(msg);

                }

                global_pose[0] = global_pose[0] - init_gnss_[0];
                global_pose[1] = global_pose[1] - init_gnss_[1];

                nav_msgs::msg::Odometry odom_rtk_lio =  createOdom(
                    global_pose[0],
                    global_pose[1],
                    global_pose[2]
                );

                rtk_lio_odom_pub_->publish(odom_rtk_lio);

                

                
                                        

                auto odom_rtk_lio_pose = makePoseStamped(global_pose[0], global_pose[1], global_pose[2] , now(), "map");

                odom_rtk_lio_path_.header.frame_id = "map";
                odom_rtk_lio_path_.header.stamp = odom_rtk_lio_pose.header.stamp;
                odom_rtk_lio_path_.poses.push_back(odom_rtk_lio_pose);


                rtk_lio_odom_path_pub_->publish(odom_rtk_lio_path_);

                tryComputeOffset();




        }
        

        // RCLCPP_INFO(
        //         get_logger(),
        //         "Map origin seted");

        lio_odom_ = odom_raw;

        if (!has_init_lio_)
        {
            init_lio_odom_ = odom_raw;
            has_init_lio_ = true;
            RCLCPP_INFO(get_logger(), "LIO initial pose captured");
        }

        odom_raw_pub_->publish(odom_raw);

        /* ===============================
        * 3. PoseStamped for path
        * =============================== */
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = msg->header.stamp;   // keep original timestamp
        pose.header.frame_id = "map";
        pose.pose = msg->pose.pose;

        /* ===============================
        * 4. Path publishing
        * =============================== */
        odom_path_.header.frame_id = "map";
        odom_path_.header.stamp = pose.header.stamp;
        odom_path_.poses.push_back(pose);
        

        tryComputeOffset();

        odom_path_pub_->publish(odom_path_);
    }

    Eigen::Vector3d transformToGlobalSimple(const Eigen::Vector3d& lio_pose) {
        // Since LIO starts at (0,0,0) and init_odom_ = (0,0,0)
        // dx = lio_pose[0] - 0, dy = lio_pose[1] - 0, dyaw = lio_pose[2] - 0

        // need to find the pose from lidar to imu and rtk and use them here 
        
        double dx = lio_pose[0];
        double dy = lio_pose[1];
        double dyaw = lio_pose[2];

        // dx = dx - lidar_to_vehicle_x_;
        // dy = dy - lidar_to_vehicle_y_;
        
        double cos_align = std::cos(yaw_alignment_);
        double sin_align = std::sin(yaw_alignment_);
        
        double E = dx * cos_align - dy * sin_align + init_gnss_[0];
        double N = dx * sin_align + dy * cos_align + init_gnss_[1];
        double yaw = yaw_alignment_ + dyaw;
        
        // Normalize yaw
        yaw = std::atan2(std::sin(yaw), std::cos(yaw));
        
        return Eigen::Vector3d(E, N, yaw);
    }


    void computeHeading()
    {
        if (!left_received_ || !right_received_)
            return;

        if (!validFix(left_fix_) || !validFix(right_fix_))
            return;

        

        auto utm_left = converter_.latlon_to_utm(left_fix_.latitude, left_fix_.longitude);
        auto utm_right = converter_.latlon_to_utm(right_fix_.latitude, right_fix_.longitude);

        double dx = utm_right.easting - utm_left.easting;   // East difference
        double dy = utm_right.northing - utm_left.northing; // North difference

        // RCLCPP_INFO_THROTTLE(
        //                     get_logger(),
        //                     *get_clock(),
        //                     1000,
        //                     "UTM: E=%.3f N=%.3f Zone=%d%c",
        //                     utm_left.easting,
        //                     utm_left.northing,
        //                     utm_left.zone,
        //                     utm_left.band);
        
        utm_zone_ = utm_left.zone;
        utm_band_ = utm_left.band;  // 'N' or 'S'
        utm_initialized_ = true;



        // double yaw = std::atan2(dy, dx);  // atan2(N, E) gives heading from North
        double baseline_yaw = std::atan2(dy, dx);
        double yaw = baseline_yaw + M_PI/2.0;

        // yaw = yaw + M_PI / 2.0;         // Adjust to vehicle heading

        x = (utm_left.easting + utm_right.easting) / 2.0;
        y = (utm_left.northing + utm_right.northing) / 2.0;

        auto latlon = converter_.utm_to_latlon(x, y,  utm_zone_, utm_band_);
        // RCLCPP_INFO(get_logger(), "Global LatLon: Lat=%.8f, Lon=%.8f", latlon.lat, latlon.lon);

        sensor_msgs::msg::NavSatFix msg;
        msg.header.stamp = now();
        msg.header.frame_id = "gps";

        msg.latitude  = latlon.lat;
        msg.longitude = latlon.lon;
        msg.altitude  = 0.0;

        msg.status.status  = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
        msg.status.service = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;

        msg.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;

        gnss_pub_->publish(msg);

        if (!prev_initialized_)
        {
            prev_x_ = x;
            prev_y_ = y;
            prev_initialized_ = true;
            return;  // wait for movement
        }

        double dx_tr = x - prev_x_;
        double dy_tr = y - prev_y_;

        double distance_tr = std::hypot(dx_tr, dy_tr);

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

        // auto rotated = angle_utils::rotatePointByYaw(x_local, y_local, -M_PI / 2.0);
        // x_local = rotated.first;
        // y_local = rotated.second;


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

            curr_gnss_[0] = x;
            curr_gnss_[1] = y;
            curr_gnss_[2] = 0.0;

            

            nav_msgs::msg::Odometry global_odom = createOdom(
                x_local,
                y_local,
                yaw
            );
            

            if (!has_init_rtk_)
            {
                init_rtk_odom_ = global_odom;
                has_init_rtk_ = true;
                RCLCPP_INFO(get_logger(), "RTK initial pose captured");
            }

            rtk_odom_ = global_odom;
            odom_pub_->publish(global_odom);



            tf2::Quaternion q;
            q.setRPY(0.0, 0.0, yaw);

            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header.stamp = now();
            tf_msg.header.frame_id = "map";
            tf_msg.child_frame_id = "base_link";

            tf_msg.transform.translation.x = x_local;
            tf_msg.transform.translation.y = y_local;
            tf_msg.transform.translation.z = 0.0;

            tf_msg.transform.rotation.x = q.x();
            tf_msg.transform.rotation.y = q.y();
            tf_msg.transform.rotation.z = q.z();
            tf_msg.transform.rotation.w = q.w();
            tf_broadcaster_->sendTransform(tf_msg);


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


        // prev_x_ = x;
        // prev_y_ = y;
        // tryComputeOffset();

        

        


        



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
    
    geometry_msgs::msg::PoseStamped makePoseStamped(double x_x, double y_y, double yaw_yaw, 
                                                    const rclcpp::Time & stamp,
                                                    const std::string & frame_id = "map")
    {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = stamp;
        pose.header.frame_id = frame_id;

        pose.pose.position.x = x_x;
        pose.pose.position.y = y_y;
        pose.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw_yaw);

        // pose.pose.orientation = tf2::toMsg(q);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();

        return pose;
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
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr rtk_lio_odom_path_pub_;

    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr lio_gnss_pub_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr gnss_pub_;

    nav_msgs::msg::Path path_msg_;
    nav_msgs::msg::Path odom_rtk_lio_path_;
    nav_msgs::msg::Path odom_path_;


    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr points_pub_;
    visualization_msgs::msg::Marker points_marker_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_raw_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr rtk_lio_odom_pub_;

    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr stop_rtk_srv_;

    bool publish_rtk_odom_{true};
    bool init_gnss_set_{false};

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr rtk_lio_yaw_sub_;

    float rtk_lio_yaw_;

    Eigen::Vector3d init_gnss_;
    Eigen::Vector3d curr_gnss_;
    Eigen::Vector3d init_odom_;
    Eigen::Vector3d curr_odom_;
    double yaw_alignment_;


    double prev_x_{0.0};
    double prev_y_{0.0};

    bool prev_initialized_{false};

    nav_msgs::msg::Odometry init_rtk_odom_;
    nav_msgs::msg::Odometry init_lio_odom_;
    nav_msgs::msg::Odometry rtk_odom_;
    nav_msgs::msg::Odometry lio_odom_;

    bool has_init_rtk_{false};
    bool has_init_lio_{false};
    bool init_offset_computed_{false};

    int utm_zone_ = 47;        // UTM zone number (e.g. 47)
    char utm_band_ = 'N';     // Hemisphere band ('N' or 'S')
    bool utm_initialized_ = false;

    // Physical offsets
    double lidar_to_vehicle_x_;  // LIDAR X in vehicle frame
    double lidar_to_vehicle_y_;  // LIDAR Y in vehicle frame

    // double

};

/* ---------------- main ---------------- */

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DualRTKHeadingNode>());
    rclcpp::shutdown();
    return 0;
}
