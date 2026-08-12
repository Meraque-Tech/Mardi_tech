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
        "/odom",
        qos,
        std::bind(
            &DualRTKHeadingNode::processAndPublishOdom,
            this,
            std::placeholders::_1));
    
    
    
    mid_fix_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>("/rtk/mid/fix", qos);


    odom_pub_  = create_publisher<nav_msgs::msg::Odometry>("/rtk/mid/odom", qos);
    path_pub_  = create_publisher<nav_msgs::msg::Path>("/rtk/mid/path", 10);
    points_pub_= create_publisher<visualization_msgs::msg::Marker>("/rtk/mid/points", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    initPointMarker();

    RCLCPP_INFO(get_logger(), "Rtk Mid Raw Node started");
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

        if(enu_displacement_2d > motion_threshold_xy_ && !is_learned_yaw_offset_){

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
            
            is_learned_yaw_offset_ = true;

        }

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

/* ================= Odometry ================= */
void DualRTKHeadingNode::processAndPublishOdom(
    const nav_msgs::msg::Odometry::SharedPtr msg)
{
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;

    if (!map_origin_set_)
    {
        map_origin_x_ = x;
        map_origin_y_ = y;
        map_origin_set_ = true;
    }

    double x_local = x - map_origin_x_;
    double y_local = y - map_origin_y_;

    (void)x_local;
    (void)y_local;
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
