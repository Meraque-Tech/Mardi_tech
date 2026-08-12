#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

import csv
import os
from datetime import datetime


class GpsToCsvNode(Node):

    def __init__(self):
        super().__init__('gps_to_csv_node')

        # CSV file name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = f'gnss_global_pose_latlon.csv'

        self.file = open(self.csv_file, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['latitude', 'longitude'])

        self.subscription = self.create_subscription(
            NavSatFix,
            '/lio/global_pose/latlon',
            self.callback,
            10)

        self.get_logger().info(
            f'Logging /lio/global_pose/latlon to {self.csv_file}')

    def callback(self, msg: NavSatFix):

        # Ignore invalid GPS
        if msg.status.status == msg.status.STATUS_NO_FIX:
            return

        lat = msg.latitude
        lon = msg.longitude

        self.writer.writerow([f'{lat:.8f}', f'{lon:.8f}'])
        self.file.flush()

    def destroy_node(self):
        self.file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GpsToCsvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
