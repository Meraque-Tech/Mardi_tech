#!/usr/bin/env python3
"""
Fixed GPS Publisher for AerialMap
Publishes a fixed GPS location to keep the aerial map centered at a constant position
regardless of the actual GPS data from bag files
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Header


class FixedGPSPublisher(Node):
    def __init__(self):
        super().__init__('fixed_gps_publisher')
        
        # Fixed reference point (Konkuk University Ilgamho)
        self.declare_parameter('latitude', 37.540091)
        self.declare_parameter('longitude', 127.076555)
        self.declare_parameter('altitude', 39.5)
        self.declare_parameter('publish_rate', 10.0)
        
        # Get parameters
        self.fixed_lat = self.get_parameter('latitude').value
        self.fixed_lon = self.get_parameter('longitude').value
        self.fixed_alt = self.get_parameter('altitude').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Publisher for fixed GPS position
        self.gps_pub = self.create_publisher(
            NavSatFix,
            '/aerialmap/fix',  # Dedicated topic for AerialMap
            10
        )
        
        # Create timer for periodic publishing
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_fixed_gps
        )
        
        self.get_logger().info(
            f'Fixed GPS Publisher initialized\n'
            f'Publishing to: /aerialmap/fix\n'
            f'Fixed position: {self.fixed_lat:.6f}°N, {self.fixed_lon:.6f}°E\n'
            f'Rate: {self.publish_rate} Hz'
        )
    
    def publish_fixed_gps(self):
        """Publish fixed GPS position"""
        msg = NavSatFix()
        
        # Header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        # GPS coordinates
        msg.latitude = self.fixed_lat
        msg.longitude = self.fixed_lon
        msg.altitude = self.fixed_alt
        
        # Status (RTK Fix)
        msg.status.status = NavSatStatus.STATUS_FIX
        msg.status.service = NavSatStatus.SERVICE_GPS
        
        # Covariance (small values for fixed position)
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        msg.position_covariance[0] = 0.01  # x variance (1cm)
        msg.position_covariance[4] = 0.01  # y variance (1cm)
        msg.position_covariance[8] = 0.02  # z variance (2cm)
        
        self.gps_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FixedGPSPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()