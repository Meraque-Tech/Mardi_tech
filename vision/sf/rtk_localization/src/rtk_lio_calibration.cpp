#include <gps_utils/gps_utils.hpp>
#include <rtk_sf/rtk_imu_sf.hpp>
// #include <angle_utils/angle_utils.hpp>


/* ================= Constructor ================= */
DualRTKHeadingNode::DualRTKHeadingNode()
: Node("rtk_mid_raw_node")
{
    auto qos = rclcpp::SystemDefaultsQoS();
    // this->declare_parameter<bool>("use_sim_time", false);
    this->set_parameter(rclcpp::Parameter("use_sim_time", false));

    // meters (XY plane)
    this->declare_parameter<double>("motion_threshold_xy", 0.3);

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

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/dlio/odom_node/odom",
        qos,
        std::bind(
            &DualRTKHeadingNode::processAndPublishOdom,
            this,
            std::placeholders::_1));
    
    odom_sub_ekf_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered",
        qos,
        std::bind(
            &DualRTKHeadingNode::processAndPublishEkf,
            this,
            std::placeholders::_1));
    
    odom_path_pub_ekf_ = create_publisher<nav_msgs::msg::Path>("/odometry/filtered/path", qos);

    mid_fix_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>("/rtk/mid/fix", qos);


    odom_pub_  = create_publisher<nav_msgs::msg::Odometry>("/rtk/mid/odom", qos);
    odom_pub_lio_offest_  = create_publisher<nav_msgs::msg::Odometry>("/odom/lio/offset", qos);
    
    path_pub_lio_offest_  = create_publisher<nav_msgs::msg::Path>("/odom/lio/path", qos);
    path_pub_  = create_publisher<nav_msgs::msg::Path>("/rtk/mid/path", 10);
    points_pub_= create_publisher<visualization_msgs::msg::Marker>("/rtk/mid/points", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    initPointMarker();

    RCLCPP_INFO(get_logger(), "Rtk Mid Raw Node started");
}


geometry_msgs::msg::PoseStamped 
    DualRTKHeadingNode::odomToPoseStamped(const nav_msgs::msg::Odometry & odom)
{
    geometry_msgs::msg::PoseStamped pose;

    pose.header = odom.header;
    pose.pose   = odom.pose.pose;

    return pose;
}

nav_msgs::msg::Odometry DualRTKHeadingNode::makeOdomFromXYYaw(
    double x, double y, double yaw)
{
    nav_msgs::msg::Odometry odom;

    // ---------------- Header ----------------
    odom.header.stamp = this->now();          // current ROS time
    odom.header.frame_id = "odom";        // parent frame
    odom.child_frame_id  = "base_link";       // robot frame

    // ---------------- Position ----------------
    odom.pose.pose.position.x = x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.position.z = 0.0;

    // ---------------- Orientation (yaw → quaternion) ----------------
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    q.normalize();

    // Assign quaternion manually
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    return odom;
}

std::tuple<double, double, double> DualRTKHeadingNode::getXYYawFromOdom(const nav_msgs::msg::Odometry & odom)
{
    // Position
    double x = odom.pose.pose.position.x;
    double y = odom.pose.pose.position.y;
    
    tf2::Quaternion q(
        odom.pose.pose.orientation.x,
        odom.pose.pose.orientation.y,
        odom.pose.pose.orientation.z,
        odom.pose.pose.orientation.w);

    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    return {x, y, yaw};
}


Pose2D DualRTKHeadingNode::rotatePose2D(double x, double y, double yaw, double yaw_offset)
{
    Pose2D out;

    // Rotate position
    out.x = x * std::cos(yaw_offset) -
            y * std::sin(yaw_offset);

    out.y = x * std::sin(yaw_offset) +
            y * std::cos(yaw_offset);

    // Rotate yaw too
    out.yaw = yaw + yaw_offset;

    // Normalize yaw to [-pi, pi]
    out.yaw = std::atan2(std::sin(out.yaw),
                         std::cos(out.yaw));

    return out;
}

/* ================= Heading ================= */
void DualRTKHeadingNode::computeHeading()
{

    if (!left_received_ || !right_received_)
        return;

    if (!validFix(left_fix_) || !validFix(right_fix_))
        return;

    auto ecef_left = converter_.latlon_to_ecef(left_fix_.latitude,
                                    left_fix_.longitude,
                                    left_fix_.altitude);
    auto ecef_right = converter_.latlon_to_ecef(right_fix_.latitude,
                                     right_fix_.longitude,
                                     right_fix_.altitude);
    
    // auto enu_left = converter_.ecef_to_enu(ecef_left);
    // auto enu_right = converter_.ecef_to_enu(ecef_right);


    

    // ===== Midpoint in ECEF =====
    gps_utils::ECEF mid_ecef;
    mid_ecef.x = 0.5 * (ecef_left.x + ecef_right.x);
    mid_ecef.y = 0.5 * (ecef_left.y + ecef_right.y);
    mid_ecef.z = 0.5 * (ecef_left.z + ecef_right.z);

    // ===== Convert back to LLA =====
    auto mid_lla = converter_.ecef_to_latlon(mid_ecef.x,
                                            mid_ecef.y,
                                            mid_ecef.z);

    

    

    if (!init_rover_enu_set_)   // meters
    {

        converter_.set_enu_origin(mid_lla.lat, 
                                mid_lla.lon, 
                                mid_lla.alt);
        
        init_rover_enu_set_ = true;
        RCLCPP_INFO(get_logger(), "Origin set. Computing ENU coordinates.");

        return;
    }

    try
    {
        auto enu_mid = converter_.ecef_to_enu(mid_ecef);
        double yaw = converter_.computeYawFromECEF(ecef_left, ecef_right, 0.0);


        if(!enu_mid_first_){

            enu_mid_first_vale_ = enu_mid;
            enu_mid_first_ = true;
            return;
        }

        // compute displacement from first accepted ENU
        auto enu_displacement_2d = converter_.enuDistance2D(enu_mid, enu_mid_first_vale_);
        this->get_parameter("motion_threshold_xy", motion_threshold_xy_);

        if(enu_displacement_2d > motion_threshold_xy_ && dist_lio_from_start > motion_threshold_xy_ && !is_learned_yaw_offset_){

            double dE = enu_mid.e - enu_mid_first_vale_.e;
            double dN = enu_mid.n - enu_mid_first_vale_.n;

            double motion_yaw = std::atan2(dN, dE);
            motion_yaw = std::atan2(std::sin(motion_yaw), std::cos(motion_yaw));

            learned_yaw_offset =std::atan2(std::sin(motion_yaw - yaw), std::cos(motion_yaw - yaw));

            RCLCPP_INFO(
                            this->get_logger(),
                            "Yaw | RTK: %.3f rad (%.2f deg) | Motion: %.3f rad (%.2f deg) | Offset: %.3f rad (%.2f deg)",
                            yaw,
                            yaw * 180.0 / M_PI,
                            motion_yaw,
                            motion_yaw * 180.0 / M_PI,
                            learned_yaw_offset,
                            learned_yaw_offset * 180.0 / M_PI
                        );
            
            

            // ---------- Print RTK Odom Coordinates ----------
            RCLCPP_INFO(this->get_logger(),
                        "RTK Odom Initial: x=%.3f  y=%.3f",
                        enu_mid_first_vale_.e,
                        enu_mid_first_vale_.n);
            
            RCLCPP_INFO(this->get_logger(),
                        "RTK Odom Current: x=%.3f  y=%.3f",
                        enu_mid.e,
                        enu_mid.n);


            // ---------- Print LIO Odom Coordinates ----------
            RCLCPP_INFO(this->get_logger(),
                        "LIO Odom Initial: x=%.3f  y=%.3f",
                        initial_odom_.pose.pose.position.x,
                        initial_odom_.pose.pose.position.y);

            RCLCPP_INFO(this->get_logger(),
                        "LIO Odom Current: x=%.3f  y=%.3f",
                        current_odom.pose.pose.position.x,
                        current_odom.pose.pose.position.y);
            
            
            // ---------- Compute RTK Motion Vector ----------
            double dx_rtk = enu_mid.e - enu_mid_first_vale_.e;
            double dy_rtk = enu_mid.n - enu_mid_first_vale_.n;

            // ---------- Compute LIO Motion Vector ----------
            double dx_lio = current_odom.pose.pose.position.x -
                            initial_odom_.pose.pose.position.x;

            double dy_lio = current_odom.pose.pose.position.y -
                            initial_odom_.pose.pose.position.y;
            
            
            // ---------- Compute Heading Angles ----------
            double yaw_rtk = std::atan2(dy_rtk, dx_rtk);
            double yaw_lio = std::atan2(dy_lio, dx_lio);

            // ---------- Compute Yaw Offset ----------
            // double yaw_offset = yaw_lio - yaw_rtk;
            yaw_offset_sf = yaw_rtk - yaw_lio;

            // Normalize to [-pi, pi]
            yaw_offset_sf = std::atan2(std::sin(yaw_offset_sf),
                                    std::cos(yaw_offset_sf));
            
            // Rotate initial LIO position into RTK frame
            auto [x_lio_init, y_lio_init, yaw_lio_init] = getXYYawFromOdom(initial_odom_);
            auto rotated_init = rotatePose2D(x_lio_init, y_lio_init, yaw_lio_init, yaw_offset_sf);

            // Position offset = RTK_initial - LIO_rotated_initial
            position_offset_x_ = enu_mid_first_vale_.e - rotated_init.x;
            position_offset_y_ = enu_mid_first_vale_.n - rotated_init.y;

            RCLCPP_INFO(this->get_logger(), 
                        "Position Offset: dx=%.3f dy=%.3f", 
                        position_offset_x_, position_offset_y_);
            
            
            
            // LIO frame→RTK frame
            // Δθlio→rtk​
            // RCLCPP_INFO(this->get_logger(),
            //             "Yaw Offset (LIO - RTK) = %.3f rad (%.2f deg)",
            //             yaw_offset_sf,
            //             yaw_offset_sf * 180.0 / M_PI);

            RCLCPP_INFO(this->get_logger(),
                        "Yaw Offset (RTK - LIO) = %.3f rad (%.2f deg)",
                        yaw_offset_sf,
                        yaw_offset_sf * 180.0 / M_PI);
            
            
            is_learned_yaw_offset_ = true;
            


        }

        
    
        // ***********************************************************************************
        auto [x_lio, y_lio, yaw_lio] = getXYYawFromOdom(current_odom);
            
        // RCLCPP_INFO(this->get_logger(),
        //         "Current Odom Pose: x_lio=%.3f y_lio=%.3f yaw_lio=%.2f deg",
        //         x_lio, y_lio,
        //         yaw_lio * 180.0 / M_PI);
        
        // 1. Rotate
        auto [x_rot, y_rot, yaw_rot] = rotatePose2D(x_lio, y_lio, yaw_lio, yaw_offset_sf);

        // 2. Translate
        double x_corr = x_rot + position_offset_x_;
        double y_corr = y_rot + position_offset_y_;
        double yaw_corr = yaw_rot;

        // Call function
        nav_msgs::msg::Odometry corrected_odom = makeOdomFromXYYaw(x_corr, y_corr, yaw_corr);

        // Publish
        odom_pub_lio_offest_->publish(corrected_odom);

        // ---------------- Path Publishing ----------------
        updateAndPublishPath(
                now(),
                x_corr,
                y_corr,
                yaw_corr,
                "odom",
                path_rtk_lio_offset,
                path_pub_lio_offest_);

        // // Update path timestamp
        // path_msg_.header.stamp = corrected_odom.header.stamp;
        // geometry_msgs::msg::PoseStamped pose_stamped = odomToPoseStamped(corrected_odom);
        // path_msg_.poses.push_back(pose_stamped);
        // // Publish path
        // path_pub_lio_offest_->publish(path_msg_);

        

        // RCLCPP_INFO(this->get_logger(),
        //         "Corrected Pose: x=%.3f y=%.3f yaw=%.2f deg",
        //         x_corr, y_corr,
        //         yaw_corr * 180.0 / M_PI);
        
        // ***********************************************************************************

        yaw = converter_.computeYawFromECEF(ecef_left, ecef_right, learned_yaw_offset);

        publishOdom(
                    now(),
                    enu_mid.e,
                    enu_mid.n,
                    0.0,          // z ignored for ground robot
                    0.0, 0.0,
                    yaw,
                    0.0,      // m/s
                    0.0,      // m/s
                    0.0,    // rad/s
                    "odom",
                    "base_link",
                    odom_pub_);
        
        updateAndPublishPath(
                now(),
                enu_mid.e,
                enu_mid.n,
                yaw,
                "odom",
                path_rtk_,
                path_pub_);
        
        // tf publish --->
        // auto tf_msg = publishTF(
        //                     now(),
        //                     enu_mid.e,
        //                     enu_mid.n,
        //                     0.0,
        //                     0.0,
        //                     0.0,
        //                     yaw,
        //                     "odom",
        //                     "base_link"
        //                 );
        
        // tf_broadcaster_->sendTransform(tf_msg);
        
        
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    


    
    
    
    // ===== Publish NavSatFix =====
    sensor_msgs::msg::NavSatFix mid_fix;
    mid_fix.header.stamp = this->now();
    mid_fix.header.frame_id = "rtk_mid";

    mid_fix.latitude  = mid_lla.lat;
    mid_fix.longitude = mid_lla.lon;
    mid_fix.altitude  = mid_lla.alt;

    // Copy status (RTK FIX stays FIX)
    mid_fix.status = left_fix_.status;

    mid_fix.position_covariance_type =
        sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;

    mid_fix_pub_->publish(mid_fix);




   
    // double dx = right_fix_.longitude - left_fix_.longitude;
    // double dy = right_fix_.latitude  - left_fix_.latitude;

    // yaw_ = std::atan2(dy, dx) + M_PI_2;
}

void DualRTKHeadingNode::processAndPublishEkf(
    const nav_msgs::msg::Odometry::SharedPtr msg)
{
    lio_rtk_odom = *msg;

    auto [x_fused, y_fused, yaw_fused] =
        getXYYawFromOdom(lio_rtk_odom);

    // Publish only if valid numbers
    if (!std::isfinite(x_fused) ||
        !std::isfinite(y_fused) ||
        !std::isfinite(yaw_fused))
        return;

    updateAndPublishPath(
        now(),
        x_fused,
        y_fused,
        yaw_fused,
        "odom",
        odom_path_pub_ekf_hb,
        odom_path_pub_ekf_);
}


/* ================= Odometry ================= */
void DualRTKHeadingNode::processAndPublishOdom(
    const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // Current odom
    current_odom = *msg;

    double x = current_odom.pose.pose.position.x;
    double y = current_odom.pose.pose.position.y;

    // Store the first odom as initial reference
    if (!has_initial_odom_)
    {
        initial_odom_ = current_odom;
        has_initial_odom_ = true;

        RCLCPP_INFO(this->get_logger(),
                    "Initial odom set at (%.3f, %.3f)", x, y);

        return;
    }

    // Compute distance from initial odom → current odom
    dist_lio_from_start = odomDistance2D(initial_odom_, current_odom);

    // RCLCPP_INFO(this->get_logger(),
    //             "Distance Lio from start: %.3f meters", dist_lio_from_start);

    // Continue processing...
}


/* ================= Utils ================= */
bool DualRTKHeadingNode::validFix(
    const sensor_msgs::msg::NavSatFix &fix)
{
    return fix.status.status >= sensor_msgs::msg::NavSatStatus::STATUS_FIX;
}

void DualRTKHeadingNode::initPointMarker()
{
    points_marker_.header.frame_id = "map";
    points_marker_.type = visualization_msgs::msg::Marker::POINTS;
    points_marker_.scale.x = 0.1;
    points_marker_.scale.y = 0.1;
    points_marker_.color.g = 1.0;
    points_marker_.color.a = 1.0;
}


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DualRTKHeadingNode>());
    rclcpp::shutdown();
    return 0;
}
