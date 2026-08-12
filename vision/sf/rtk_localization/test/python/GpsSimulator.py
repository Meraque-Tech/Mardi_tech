#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import NavSatStatus
import math
import time

class GpsSimulator(Node):
    def __init__(self):
        super().__init__('gps_simulator')
        self.pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.timer = self.create_timer(0.5, self.publish_fix)  # 2 Hz

        # Your real test points (Darmstadt area)
        self.waypoints = [
            (49.899600976838286, 8.899341648789813, 110.0),
            (49.8998894578428,   8.898535985040668, 112.0),
            (49.90096619250414,  8.899532919301544, 115.0),
            (49.90066376911161,  8.900237165860988, 118.0),
        ]
        self.idx = 0
        self.get_logger().info("GPS Simulator started – publishing on /gps/fix")

    def publish_fix(self):
        lat, lon, alt = self.waypoints[self.idx]
        self.idx = (self.idx + 1) % len(self.waypoints)

        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gps"

        msg.status.status = NavSatStatus.STATUS_FIX          # good fix
        msg.status.service = NavSatStatus.SERVICE_GPS

        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt

        # Fake realistic covariance (diagonal: ~1m accuracy)
        msg.position_covariance = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 4.0
        ]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self.pub.publish(msg)
        self.get_logger().info(f"Published GPS: {lat:.9f}, {lon:.9f}")


def main():
    rclpy.init()
    node = GpsSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()