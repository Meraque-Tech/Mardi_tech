#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped
from std_msgs.msg import Float64


class VelocityMagnitudeNode(Node):
    def __init__(self):
        super().__init__('velocity_magnitude_node')
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            10
        )
        
        self.gps_vel_sub = self.create_subscription(
            TwistWithCovarianceStamped,
            '/ublox_gps_node/fix_velocity',
            self.gps_velocity_callback,
            10
        )
        
        # Publishers
        self.odom_mag_pub = self.create_publisher(
            Float64,
            '/velocity_magnitude/odometry',
            10
        )
        
        self.gps_mag_pub = self.create_publisher(
            Float64,
            '/velocity_magnitude/gps',
            10
        )
        
        self.get_logger().info('Velocity magnitude node started')
        self.get_logger().info('Subscribing to /odometry/filtered and /ublox_gps_node/fix_velocity')
        self.get_logger().info('Publishing to /velocity_magnitude/odometry and /velocity_magnitude/gps')
    
    def odom_callback(self, msg):
        # Extract linear velocity from odometry
        linear = msg.twist.twist.linear
        
        # Calculate magnitude (only x and y)
        magnitude = math.sqrt(linear.x**2 + linear.y**2)
        
        # Publish magnitude
        mag_msg = Float64()
        mag_msg.data = magnitude
        self.odom_mag_pub.publish(mag_msg)
        
        self.get_logger().debug(f'Odometry velocity magnitude: {magnitude:.3f} m/s')
    
    def gps_velocity_callback(self, msg):
        # Extract linear velocity from GPS
        linear = msg.twist.twist.linear
        
        # Calculate magnitude (only x and y)
        magnitude = math.sqrt(linear.x**2 + linear.y**2)
        
        # Publish magnitude
        mag_msg = Float64()
        mag_msg.data = magnitude
        self.gps_mag_pub.publish(mag_msg)
        
        self.get_logger().debug(f'GPS velocity magnitude: {magnitude:.3f} m/s')


def main(args=None):
    rclpy.init(args=args)
    
    node = VelocityMagnitudeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()