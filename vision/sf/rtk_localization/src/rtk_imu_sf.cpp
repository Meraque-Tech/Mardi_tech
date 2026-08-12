#include <gps_utils/gps_utils.hpp>
#include <rtk_sf/rtk_imu_sf.hpp>
// #include <angle_utils/angle_utils.hpp>


/* ================= Constructor ================= */
DualRTKHeadingNode::DualRTKHeadingNode()
: Node("dual_rtk_heading_node")
{
    auto qos = rclcpp::SystemDefaultsQoS();

    // meters (XY plane)
    this->declare_parameter<double>("motion_threshold_xy", 0.5);

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

    odom_pub_  = create_publisher<nav_msgs::msg::Odometry>("/rtk/odom", qos);
    path_pub_  = create_publisher<nav_msgs::msg::Path>("/rtk/path", 10);
    points_pub_= create_publisher<visualization_msgs::msg::Marker>("/rtk/points", 10);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    initPointMarker();

    RCLCPP_INFO(get_logger(), "RTK IMU SF Node started");
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

    // Rover center = midpoint
    gps_utils::ECEF rover_ecef;
    rover_ecef.x = 0.5 * (ecef_left.x + ecef_right.x);
    rover_ecef.y = 0.5 * (ecef_left.y + ecef_right.y);
    rover_ecef.z = 0.5 * (ecef_left.z + ecef_right.z);
    

    if (!init_rover_ecef_set_)
    {
        init_rover_ecef_.x = rover_ecef.x;
        init_rover_ecef_.y = rover_ecef.y;
        init_rover_ecef_set_ = true;

        RCLCPP_INFO(
            get_logger(),
            "Initial rover XY (ECEF) set: X=%.3f Y=%.3f",
            init_rover_ecef_.x,
            init_rover_ecef_.y);

        return;
    }

    // Compare current vs initial (XY only)
    double dx = rover_ecef.x - init_rover_ecef_.x;
    double dy = rover_ecef.y - init_rover_ecef_.y;

    double distance_xy = std::sqrt(dx*dx + dy*dy);

    this->get_parameter("motion_threshold_xy", motion_threshold_xy_);

    if (distance_xy > motion_threshold_xy_ && !init_rover_enu_set_)   // meters
    {
        RCLCPP_INFO(
            get_logger(),
            "Rover moved %.3f meters from INITIAL (XY only)",
            distance_xy);

        converter_.set_enu_origin(left_fix_.latitude, 
                                left_fix_.longitude, 
                                left_fix_.altitude);
        
        init_rover_enu_set_ = true;
        RCLCPP_INFO(get_logger(), "Origin set. Computing ENU coordinates.");

        // return;
    }

    if(converter_.is_origin_set()){
        
        auto rover_enu = converter_.ecef_to_enu(rover_ecef);
        

        // Convert both antennas to ENU
        auto left_enu  = converter_.ecef_to_enu(ecef_left);
        auto right_enu = converter_.ecef_to_enu(ecef_right);

        // Baseline vector in ENU
        double dE = right_enu.e - left_enu.e;
        double dN = right_enu.n - left_enu.n;

        // // Yaw from East → North (ROS convention)
        // double yaw = std::atan2(dN, dE);

        // Method 1: Swap arguments (RECOMMENDED)
        double yaw = atan2(dN, dE) + M_PI/2.0;  // Note: (N, E) not (E, N)

        // Normalize yaw to [-pi, pi]
        yaw = std::atan2(std::sin(yaw), std::cos(yaw));
        // yaw = atan2(ΔN, ΔE)

        // RCLCPP_INFO_THROTTLE(
        //     get_logger(),
        //     *get_clock(),
        //     1000,
        //     "Rover ENU: E: %.3f m, N: %.3f m, U: %.3f m, Yaw: %.3f deg",
        //     rover_enu.e,
        //     rover_enu.n,
        //     rover_enu.u,
        //     yaw * 180.0 / M_PI);
        
        
        publishOdom(
                now(),
                rover_enu.e,
                rover_enu.n,
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
                rover_enu.e,
                rover_enu.n,
                yaw,
                "odom",
                path_rtk_,
                path_pub_);

        
        
    }

    
    // RCLCPP_INFO_THROTTLE(
    //     get_logger(),
    //     *get_clock(),
    //     1000,
    //     "Left ECEF:  X: %.3f m, Y: %.3f m, Z: %.3f m",
    //     ecef_left.x, ecef_left.y, ecef_left.z);
    
    // RCLCPP_INFO_THROTTLE(
    //     get_logger(),
    //     *get_clock(),
    //     1000,
    //     "Right ECEF: X: %.3f m, Y: %.3f m, Z: %.3f m",
    //     ecef_right.x, ecef_right.y, ecef_right.z);

    


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
