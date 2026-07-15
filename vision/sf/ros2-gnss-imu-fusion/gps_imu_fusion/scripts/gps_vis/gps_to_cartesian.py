#!/usr/bin/env python3
"""
GPS to Local Cartesian Converter Node
Converts GPS lat/lon to local Cartesian coordinates for robot_localization
Publishes odometry messages with position in local frame
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from std_msgs.msg import Header
import utm
from tf_transformations import quaternion_from_euler
import tf2_ros
from geometry_msgs.msg import TransformStamped


class GpsToCartesianConverter(Node):
    def __init__(self):
        super().__init__('gps_to_cartesian_converter')
        
        # Declare parameters
        self.declare_parameter('reference_latitude', 37.540091)  # Konkuk University Ilgamho
        self.declare_parameter('reference_longitude', 127.076555)
        self.declare_parameter('reference_altitude', 39.5)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('child_frame', 'gps')
        
        # Get parameters
        self.ref_lat = self.get_parameter('reference_latitude').value
        self.ref_lon = self.get_parameter('reference_longitude').value
        self.ref_alt = self.get_parameter('reference_altitude').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.world_frame = self.get_parameter('world_frame').value
        self.child_frame = self.get_parameter('child_frame').value
        
        # Convert reference to UTM
        self.utm_x_ref, self.utm_y_ref, self.utm_zone, self.utm_letter = utm.from_latlon(self.ref_lat, self.ref_lon)
        
        # Subscribers
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/ublox_gps_node/fix',
            self.gps_callback,
            10
        )
        
        self.gps_vel_sub = self.create_subscription(
            TwistWithCovarianceStamped,
            '/ublox_gps_node/fix_velocity',
            self.gps_vel_callback,
            10
        )
        
        # Publishers
        self.odom_pub = self.create_publisher(
            Odometry,
            '/gps/odometry',
            10
        )
        
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/gps/pose',
            10
        )
        
        # TF broadcaster
        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # State
        self.last_velocity = None
        self.first_fix_received = False
        
        self.get_logger().info(
            f'GPS to Cartesian converter initialized\n'
            f'Reference: {self.ref_lat:.6f}°N, {self.ref_lon:.6f}°E\n'
            f'UTM Zone: {self.utm_zone}{self.utm_letter}\n'
            f'UTM Reference: {self.utm_x_ref:.2f}E, {self.utm_y_ref:.2f}N'
        )
    
    def gps_vel_callback(self, msg: TwistWithCovarianceStamped):
        """Store latest velocity for use in odometry message"""
        self.last_velocity = msg
    
    def gps_callback(self, msg: NavSatFix):
        """Convert GPS fix to local Cartesian coordinates"""
        # Check if we have a valid fix
        if msg.status.status < 0:  # No fix
            self.get_logger().warn('No GPS fix available')
            return
        
        # Convert lat/lon to UTM
        try:
            utm_x, utm_y, zone, letter = utm.from_latlon(msg.latitude, msg.longitude)
        except Exception as e:
            self.get_logger().error(f'Failed to convert GPS to UTM: {e}')
            return
        
        # Check if we're in the same UTM zone
        if zone != self.utm_zone or letter != self.utm_letter:
            self.get_logger().warn(
                f'GPS position in different UTM zone: {zone}{letter} vs {self.utm_zone}{self.utm_letter}'
            )
        
        # Calculate local Cartesian position
        x = utm_x - self.utm_x_ref
        y = utm_y - self.utm_y_ref
        z = msg.altitude - self.ref_alt
        
        if not self.first_fix_received:
            self.first_fix_received = True
            self.get_logger().info(f'First GPS fix received. Local position: ({x:.2f}, {y:.2f}, {z:.2f})')
        
        # Create odometry message
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.child_frame
        
        # Position
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        
        # Orientation (identity - GPS doesn't provide orientation)
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = 0.0
        odom.pose.pose.orientation.w = 1.0
        
        # Position covariance
        # Convert GPS covariance to local frame (same values)
        if msg.position_covariance_type == 2:  # Diagonal known
            odom.pose.covariance[0] = msg.position_covariance[0]  # x variance
            odom.pose.covariance[7] = msg.position_covariance[4]  # y variance
            odom.pose.covariance[14] = msg.position_covariance[8]  # z variance
        else:
            # Default covariance based on fix type
            if msg.status.status == 2:  # RTK Fix
                cov_xy = 0.02 * 0.02
                cov_z = 0.04 * 0.04
            elif msg.status.status == 1:  # RTK Float
                cov_xy = 0.3 * 0.3
                cov_z = 0.5 * 0.5
            else:  # Single
                cov_xy = 2.0 * 2.0
                cov_z = 5.0 * 5.0
            
            odom.pose.covariance[0] = cov_xy
            odom.pose.covariance[7] = cov_xy
            odom.pose.covariance[14] = cov_z
        
        # Orientation covariance (very high - no orientation from GPS)
        odom.pose.covariance[21] = 1e6
        odom.pose.covariance[28] = 1e6
        odom.pose.covariance[35] = 1e6
        
        # Velocity (if available)
        if self.last_velocity is not None:
            # GPS velocity is in ENU frame, which matches our local frame
            odom.twist.twist.linear.x = self.last_velocity.twist.twist.linear.x
            odom.twist.twist.linear.y = self.last_velocity.twist.twist.linear.y
            odom.twist.twist.linear.z = self.last_velocity.twist.twist.linear.z
            
            # Copy velocity covariance
            for i in range(36):
                odom.twist.covariance[i] = self.last_velocity.twist.covariance[i]
        
        self.odom_pub.publish(odom)
        
        # Also publish as PoseWithCovarianceStamped for robot_localization
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = odom.header.stamp
        pose_msg.header.frame_id = self.world_frame  # 'odom' frame
        pose_msg.pose = odom.pose
        self.pose_pub.publish(pose_msg)
        
        # Publish TF if enabled
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = msg.header.stamp
            t.header.frame_id = self.world_frame
            t.child_frame_id = self.child_frame
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = GpsToCartesianConverter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()