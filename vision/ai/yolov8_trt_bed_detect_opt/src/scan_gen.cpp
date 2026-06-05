#include "ros_utils.h"

std::vector<sl::float2> AlgoUtils::inner_2d_polar_points_gen(std::vector<sl::float3> &inner_2d_points)
{

    std::vector<sl::float2> polar_points;

    for (size_t i = 0; i < inner_2d_points.size(); ++i)
    {
        double d = euclidean_distance_onearg(inner_2d_points[i]);
        double theta = std::atan2(inner_2d_points[i].y , inner_2d_points[i].x);

        sl::float2 polar_point = {d, theta};
        polar_points.push_back(polar_point);
    }

    return polar_points;
}


sensor_msgs::msg::LaserScan AlgoUtils::polar_to_laser_scan(rclcpp::Time now, const std::vector<sl::float2> &polar_points)
{
    sensor_msgs::msg::LaserScan laser_scan;
    laser_scan.header.stamp = now;
    laser_scan.header.frame_id = camera_frame; // Replace with your desired frame ID
    laser_scan.angle_min = polar_points.front().y; // Minimum angle (adjust as needed)
    laser_scan.angle_max = polar_points.back().y; // Maximum angle (adjust as needed)
    laser_scan.angle_increment = (laser_scan.angle_max - laser_scan.angle_min)/polar_points.size();
    laser_scan.scan_time = 0.1; // Scan time (adjust as needed)
    laser_scan.range_min = 0.0; // Minimum range (adjust as needed)
    laser_scan.range_max = 10.0; // Maximum range (adjust as needed)
    // laser_scan.ranges.resize(polar_points.size());
    // laser_scan.intensities.resize(polar_points.size());

    for (const sl::float2& polar_point : polar_points)
    {
        // Calculate Cartesian coordinates from polar coordinates
        // laser_scan.angle_increment = abs(polar_point.y);  // Angle increment
        laser_scan.intensities.push_back(47.0);
        laser_scan.ranges.push_back(polar_point.x);
    }

    return laser_scan;
}

void AlgoUtils::camera_laser_pub(rclcpp::Time now, std::vector<sl::float3> &bbx_3d)
{
    std::vector<sl::float3> inner_2d_points_gen_r = inner_2d_points_gen(bbx_3d.at(3), bbx_3d.at(2), 200);
    std::vector<sl::float3> inner_2d_points_gen_f = inner_2d_points_gen(bbx_3d.at(3), bbx_3d.at(0), 200);
    std::vector<sl::float3> inner_2d_points_gen_l = inner_2d_points_gen(bbx_3d.at(0), bbx_3d.at(1), 200);
    std::vector<sl::float3> inner_2d_points_gen_f_lr = inner_2d_points_gen(bbx_3d.at(1), bbx_3d.at(2), 200);

    std::vector<sl::float2> inner_2d_polar_points_gen_r = inner_2d_polar_points_gen(inner_2d_points_gen_r);
    std::vector<sl::float2> inner_2d_polar_points_gen_f = inner_2d_polar_points_gen(inner_2d_points_gen_f);
    std::vector<sl::float2> inner_2d_polar_points_gen_l = inner_2d_polar_points_gen(inner_2d_points_gen_l);
    std::vector<sl::float2> inner_2d_polar_points_gen_f_lr = inner_2d_polar_points_gen(inner_2d_points_gen_f_lr);

    sensor_msgs::msg::LaserScan laser_scan_r = polar_to_laser_scan(now, inner_2d_polar_points_gen_r);
    sensor_msgs::msg::LaserScan laser_scan_f = polar_to_laser_scan(now, inner_2d_polar_points_gen_f);
    sensor_msgs::msg::LaserScan laser_scan_l = polar_to_laser_scan(now, inner_2d_polar_points_gen_l);
    sensor_msgs::msg::LaserScan laser_scan_f_lr = polar_to_laser_scan(now, inner_2d_polar_points_gen_f_lr);

    laser_scan_r_pub->publish(laser_scan_r);
    laser_scan_f_pub->publish(laser_scan_f);
    laser_scan_l_pub->publish(laser_scan_l);
    laser_scan_f_lr_pub->publish(laser_scan_f_lr);

}
