#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math


class ImuYawListener(Node):

    def __init__(self):
        super().__init__('imu_yaw_listener')

        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.get_logger().info('IMU yaw listener started')


    def imu_callback(self, msg: Imu):
        # Quaternion
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        # Quaternion → yaw (ENU / ROS)
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)
        )

        yaw_deg = math.degrees(yaw)

        self.get_logger().info(
            f"Yaw: {yaw:.4f} rad | {yaw_deg:.2f} deg"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ImuYawListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
