#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import TwistWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import StaticTransformBroadcaster
import time

class LeverArmTestNode(Node):
    def __init__(self):
        super().__init__('lever_arm_test_node')
        
        # Publishers for sensor data
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.gps_vel_pub = self.create_publisher(TwistWithCovarianceStamped, '/gps/vel', 10)
        
        # Subscribe to EKF output to monitor results
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        
        # Static TF broadcaster for lever arms
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        
        # Publish static transforms for lever arms
        self.publish_static_transforms()
        
        # Timer for sensor data publishing
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Simulation state
        self.t = 0.0
        self.dt = 0.1
        
        # Lever arms (meters)
        self.imu_lever_arm = [0.5, 0.0, 0.3]  # IMU 50cm forward, 30cm up from base_link
        self.gps_lever_arm = [-0.3, 0.0, 1.0]  # GPS 30cm back, 1m up from base_link
        
        # Motion profile: rotating platform with oscillating pitch
        self.angular_vel_z = 0.5  # rad/s (yaw rate)
        self.pitch_amplitude = 0.2  # rad
        self.pitch_frequency = 0.3  # Hz
        
        self.get_logger().info('Lever arm test node started')
        self.get_logger().info(f'IMU lever arm: {self.imu_lever_arm}')
        self.get_logger().info(f'GPS lever arm: {self.gps_lever_arm}')
        
    def publish_static_transforms(self):
        """Publish static transforms for IMU and GPS lever arms"""
        static_transforms = []
        
        # IMU transform
        t_imu = TransformStamped()
        t_imu.header.stamp = self.get_clock().now().to_msg()
        t_imu.header.frame_id = 'base_link'
        t_imu.child_frame_id = 'imu_link'
        t_imu.transform.translation.x = self.imu_lever_arm[0]
        t_imu.transform.translation.y = self.imu_lever_arm[1]
        t_imu.transform.translation.z = self.imu_lever_arm[2]
        t_imu.transform.rotation.w = 1.0
        static_transforms.append(t_imu)
        
        # GPS transform
        t_gps = TransformStamped()
        t_gps.header.stamp = self.get_clock().now().to_msg()
        t_gps.header.frame_id = 'base_link'
        t_gps.child_frame_id = 'gps'
        t_gps.transform.translation.x = self.gps_lever_arm[0]
        t_gps.transform.translation.y = self.gps_lever_arm[1]
        t_gps.transform.translation.z = self.gps_lever_arm[2]
        t_gps.transform.rotation.w = 1.0
        static_transforms.append(t_gps)
        
        self.tf_static_broadcaster.sendTransform(static_transforms)
        
    def timer_callback(self):
        """Generate and publish sensor data with lever arm effects"""
        self.t += self.dt
        
        # Current vehicle state (base_link)
        yaw = self.angular_vel_z * self.t
        pitch = self.pitch_amplitude * np.sin(2 * np.pi * self.pitch_frequency * self.t)
        pitch_rate = self.pitch_amplitude * 2 * np.pi * self.pitch_frequency * \
                     np.cos(2 * np.pi * self.pitch_frequency * self.t)
        
        # Angular velocity and acceleration in base frame
        omega_base = np.array([0, pitch_rate, self.angular_vel_z])
        alpha_base = np.array([0, 
                               -self.pitch_amplitude * (2*np.pi*self.pitch_frequency)**2 * 
                               np.sin(2*np.pi*self.pitch_frequency*self.t),
                               0])
        
        # Base acceleration (simple forward motion + gravity)
        a_base = np.array([0.5, 0, 9.81])  # 0.5 m/s² forward + gravity
        
        # === IMU measurements (with lever arm effects) ===
        # a_imu = a_base + alpha × r + omega × (omega × r)
        r_imu = np.array(self.imu_lever_arm)
        a_imu = a_base + np.cross(alpha_base, r_imu) + \
                np.cross(omega_base, np.cross(omega_base, r_imu))
        
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'
        imu_msg.angular_velocity.x = omega_base[0]
        imu_msg.angular_velocity.y = omega_base[1]
        imu_msg.angular_velocity.z = omega_base[2]
        imu_msg.linear_acceleration.x = a_imu[0]
        imu_msg.linear_acceleration.y = a_imu[1]
        imu_msg.linear_acceleration.z = a_imu[2]
        
        self.imu_pub.publish(imu_msg)
        
        # === GPS measurements ===
        # Use a reference position near Konkuk University
        ref_lat = 37.540091
        ref_lon = 127.076555
        
        # Simple circular motion for testing
        radius = 10.0  # meters
        x_base = radius * np.cos(yaw)
        y_base = radius * np.sin(yaw)
        z_base = 0.0
        
        # GPS position includes lever arm effect
        # In world frame: p_gps = p_base + R_wb * r_gps
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])
        R_pitch = np.array([[1, 0, 0],
                           [0, np.cos(pitch), -np.sin(pitch)],
                           [0, np.sin(pitch), np.cos(pitch)]])
        R_wb = R_yaw @ R_pitch
        
        r_gps_world = R_wb @ np.array(self.gps_lever_arm)
        x_gps = x_base + r_gps_world[0]
        y_gps = y_base + r_gps_world[1]
        z_gps = z_base + r_gps_world[2]
        
        # Convert to lat/lon (simple approximation)
        lat_gps = ref_lat + y_gps / 111111.0
        lon_gps = ref_lon + x_gps / (111111.0 * np.cos(np.radians(ref_lat)))
        
        gps_msg = NavSatFix()
        gps_msg.header.stamp = self.get_clock().now().to_msg()
        gps_msg.header.frame_id = 'gps'
        gps_msg.latitude = lat_gps
        gps_msg.longitude = lon_gps
        gps_msg.altitude = z_gps + 50.0  # Add reference altitude
        gps_msg.status.status = 0  # STATUS_FIX
        gps_msg.status.service = 1  # SERVICE_GPS
        
        self.gps_pub.publish(gps_msg)
        
        # GPS velocity
        v_base = np.array([-radius * self.angular_vel_z * np.sin(yaw),
                          radius * self.angular_vel_z * np.cos(yaw),
                          0])
        v_gps = v_base + np.cross(omega_base, R_wb @ np.array(self.gps_lever_arm))
        
        gps_vel_msg = TwistWithCovarianceStamped()
        gps_vel_msg.header.stamp = self.get_clock().now().to_msg()
        gps_vel_msg.header.frame_id = 'gps'
        gps_vel_msg.twist.twist.linear.x = v_gps[0]
        gps_vel_msg.twist.twist.linear.y = v_gps[1]
        gps_vel_msg.twist.twist.linear.z = v_gps[2]
        
        self.gps_vel_pub.publish(gps_vel_msg)
        
        if int(self.t * 10) % 50 == 0:  # Log every 5 seconds
            self.get_logger().info(f't={self.t:.1f}s: yaw={np.degrees(yaw):.1f}°, '
                                 f'pitch={np.degrees(pitch):.1f}°, '
                                 f'lever arm effect on IMU acc: '
                                 f'[{a_imu[0]-a_base[0]:.3f}, '
                                 f'{a_imu[1]-a_base[1]:.3f}, '
                                 f'{a_imu[2]-a_base[2]:.3f}] m/s²')
        
    def odom_callback(self, msg):
        """Monitor EKF output"""
        if int(self.t * 10) % 100 == 0:  # Log every 10 seconds
            self.get_logger().info(f'EKF output: x={msg.pose.pose.position.x:.2f}, '
                                 f'y={msg.pose.pose.position.y:.2f}, '
                                 f'z={msg.pose.pose.position.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = LeverArmTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()