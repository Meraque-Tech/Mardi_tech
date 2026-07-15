#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import utm

class GpsPathPublisher(Node):
    def __init__(self):
        super().__init__('gps_path_publisher')

        # Parameters for consistency with EKF path
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('ref_lat', 37.237394)
        self.declare_parameter('ref_lon', 126.770827)
        self.declare_parameter('gps_topic', '/ublox_gps_node/fix')

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.ref_lat = self.get_parameter('ref_lat').get_parameter_value().double_value
        self.ref_lon = self.get_parameter('ref_lon').get_parameter_value().double_value
        self.gps_topic = self.get_parameter('gps_topic').get_parameter_value().string_value

        # Compute reference UTM once, so GPS points map to the same local origin as EKF
        self.ref_utm_x, self.ref_utm_y, self.ref_zone, self.ref_letter = utm.from_latlon(self.ref_lat, self.ref_lon)
        self.get_logger().info(
            f'GPS Online Vis: frame_id={self.frame_id}, ref=({self.ref_lat:.6f}, {self.ref_lon:.6f}) '
            f'-> UTM {self.ref_zone}{self.ref_letter} ({self.ref_utm_x:.2f}, {self.ref_utm_y:.2f})'
        )

        self.subscription = self.create_subscription(
            NavSatFix,
            self.gps_topic,  # Ublox GPS 토픽
            self.listener_callback,
            10)
        self.path_publisher = self.create_publisher(Path, '/gps_path', 10)
        self.odom_path_publisher = self.create_publisher(Path, '/odom_path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id
        self.odom_path_msg = Path()
        self.odom_path_msg.header.frame_id = self.frame_id

        # Subscribe to EKF odometry (conditional publish when received)
        self.odom_received = False
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            50
        )

    def listener_callback(self, msg: NavSatFix):
        # Convert WGS84 (lat, lon) to UTM
        utm_x, utm_y, zone, letter = utm.from_latlon(msg.latitude, msg.longitude)

        # Warn if zone differs from reference (should not happen in local operation)
        if zone != self.ref_zone or letter != self.ref_letter:
            self.get_logger().warn(
                f'UTM zone changed: {zone}{letter} (ref {self.ref_zone}{self.ref_letter})'
            )

        # PoseStamped 메시지 생성
        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = self.path_msg.header.frame_id  # Path의 frame_id와 동일하게 설정

        # Local cartesian relative to fixed reference (match EKF origin)
        pose.pose.position.x = utm_x - self.ref_utm_x
        pose.pose.position.y = utm_y - self.ref_utm_y
        pose.pose.position.z = 0.0  # Keep 2D path; adjust if needed
        pose.pose.orientation.w = 1.0 # 방향은 일단 1.0으로 설정

        # 경로에 새로운 좌표 추가
        self.path_msg.poses.append(pose)
        self.path_msg.header.stamp = self.get_clock().now().to_msg()

        # 경로 발행
        self.path_publisher.publish(self.path_msg)
        self.get_logger().info(f'Published path with {len(self.path_msg.poses)} poses.')

    def odom_callback(self, msg: Odometry):
        # Use odometry positions directly; EKF already outputs local cartesian relative to same origin
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.odom_path_msg.header.frame_id
        pose.pose = msg.pose.pose

        self.odom_path_msg.poses.append(pose)
        self.odom_path_msg.header.stamp = pose.header.stamp
        self.odom_path_publisher.publish(self.odom_path_msg)
        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info('Started publishing odometry path (/odom_path) to compare with /gps_path')

def main(args=None):
    rclpy.init(args=args)
    gps_path_publisher = GpsPathPublisher()
    rclpy.spin(gps_path_publisher)
    gps_path_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
