#pragma once

#include <cmath>
#include <stdexcept>
#include <ostream>

#include <geometry_msgs/msg/quaternion.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>  // ✅ THIS IS THE KEY
#include <rclcpp/rclcpp.hpp>

struct GNSSSample {
        rclcpp::Time timestamp;
        double x, y;          // UTM coordinates
        double yaw;           // Heading in radians
        int utm_zone;
        char utm_band;
    };


namespace angle_utils
{

    // ============================================================================
    // Constants
    // ============================================================================
    inline constexpr double PI     = M_PI;
    inline constexpr double TWO_PI = M_PI * 2.0;

    struct Euler
    {
        double roll;
        double pitch;
        double yaw;
    };

    inline double deg2rad(double deg)
    {
        return deg * PI / 180.0;
    }

    inline double rad2deg(double rad)
    {
        return rad * 180.0 / PI;
    }



    // ============================================================================
    // Quaternion  →  Euler   (radians)
    // ============================================================================
    inline Euler eulerFromQuaternion(const geometry_msgs::msg::Quaternion &q_msg)
    {
        tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        return {roll, pitch, yaw};
    }

    double normalizeAngle(double angle_rad)
    {
        double yaw = std::atan2(std::sin(angle_rad), std::cos(angle_rad));
        return yaw;
    }

    // ============================================================================
    // Euler (radians)  →  Quaternion msg
    // ============================================================================
    inline geometry_msgs::msg::Quaternion quaternionFromEuler(
            double roll, double pitch, double yaw)
    {
        tf2::Quaternion q;
        q.setRPY(roll, pitch, yaw);

        geometry_msgs::msg::Quaternion q_msg;
        q_msg.x = q.x();
        q_msg.y = q.y();
        q_msg.z = q.z();
        q_msg.w = q.w();
        return q_msg;
    }

    std::pair<double, double> rotatePointByYaw(double x, double y, double yaw_rad) {
        const double cos_yaw = std::cos(yaw_rad);
        const double sin_yaw = std::sin(yaw_rad);
        
        return {
            x * cos_yaw - y * sin_yaw,
            x * sin_yaw + y * cos_yaw
        };
    }



}