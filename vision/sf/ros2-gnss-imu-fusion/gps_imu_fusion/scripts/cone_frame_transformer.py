#!/usr/bin/env python3
"""
Cone Frame Transformer Node - Optimized Version
Transforms cone detections from sensor frame to map frame
Supports both TrackedConeArray and ModifiedFloat32MultiArray (legacy)
Uses numpy vectorization for performance optimization
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros
import tf2_geometry_msgs
from custom_interface.msg import TrackedConeArray, TrackedCone
# Legacy message type - commented out for migration to TrackedConeArray
# from custom_interface.msg import ModifiedFloat32MultiArray
from geometry_msgs.msg import PointStamped, Point
import numpy as np
from tf2_ros import TransformException
import tf2_py as tf2


class ConeFrameTransformer(Node):
    def __init__(self):
        super().__init__('cone_frame_transformer')
        
        # Parameters
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('timeout_sec', 0.1)
        # use_sim_time is automatically declared by ROS2, don't redeclare
        
        # Get parameters
        self.target_frame = self.get_parameter('target_frame').value
        self.timeout_sec = self.get_parameter('timeout_sec').value
        
        # TF2 with longer buffer time
        self.tf_buffer = tf2_ros.Buffer(rclpy.time.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # QoS for best effort (LiDAR data)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # TrackedConeArray (UKF output)
        self.ukf_cone_sub = self.create_subscription(
            TrackedConeArray,
            '/cone/fused/ukf',
            self.ukf_cone_callback,
            qos
        )
        
        self.ukf_cone_pub = self.create_publisher(
            TrackedConeArray,
            '/cone/fused/ukf/map',
            qos
        )
        
        # TrackedConeArray v2 (non-UKF output from cone_detection)
        self.v2_cone_sub = self.create_subscription(
            TrackedConeArray,
            '/cone/lidar',
            self.v2_cone_callback,
            qos
        )
        
        self.v2_cone_pub = self.create_publisher(
            TrackedConeArray,
            '/cone/lidar/map',
            qos
        )
        
        # ModifiedFloat32MultiArray (Legacy - will be removed) - commented out for migration
        # self.legacy_cone_sub = self.create_subscription(
        #     ModifiedFloat32MultiArray,
        #     '/sorted_cones_time',
        #     self.legacy_cone_callback,
        #     qos
        # )
        # 
        # self.legacy_cone_pub = self.create_publisher(
        #     ModifiedFloat32MultiArray,
        #     '/sorted_cones_time_map',
        #     qos
        # )
        
        self.get_logger().info(f'Cone Frame Transformer started')
        self.get_logger().info(f'Target frame: {self.target_frame}')
        self.get_logger().info(f'Publishing TrackedConeArray to: /cone/fused/ukf/map')
        self.get_logger().info(f'Publishing TrackedConeArray to: /cone/lidar/map')
        # Legacy format - commented out
        # self.get_logger().info(f'Publishing ModifiedFloat32MultiArray to: /sorted_cones_time_map')
        
        self._msg_count = 0
        
    def v2_cone_callback(self, msg):
        """Transform TrackedConeArray v2 from source frame to target frame"""
        self._transform_tracked_cones(msg, self.v2_cone_pub, 'v2')
    
    def ukf_cone_callback(self, msg):
        """Transform TrackedConeArray UKF from source frame to target frame"""
        self._transform_tracked_cones(msg, self.ukf_cone_pub, 'ukf')
    
    def _transform_tracked_cones(self, msg, publisher, label=''):
        """Common transform logic for TrackedConeArray messages - Optimized with numpy"""
        
        # Skip if already in target frame
        if msg.header.frame_id == self.target_frame:
            publisher.publish(msg)
            return
        
        # Skip if no cones
        if not msg.cones:
            output_msg = TrackedConeArray()
            output_msg.header.stamp = msg.header.stamp
            output_msg.header.frame_id = self.target_frame
            publisher.publish(output_msg)
            return
            
        try:
            # Get transform once - use smaller timeout for better responsiveness
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    msg.header.frame_id,
                    msg.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=0.01)  # Reduced timeout
                )
            except (tf2_ros.ExtrapolationException, tf2_ros.LookupException):
                # Fallback to latest available transform with no wait
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    msg.header.frame_id,
                    rclpy.time.Time()  # Latest available
                )
            
            # Extract transform matrix components for numpy operations
            trans = transform.transform.translation
            rot = transform.transform.rotation
            
            # Convert quaternion to rotation matrix
            # Using tf2's quaternion to matrix conversion
            rot_matrix = self.quaternion_to_rotation_matrix(rot.x, rot.y, rot.z, rot.w)
            translation = np.array([trans.x, trans.y, trans.z])
            
            # Create output message
            output_msg = TrackedConeArray()
            output_msg.header.stamp = msg.header.stamp
            output_msg.header.frame_id = self.target_frame
            
            # Vectorized transform if many cones
            if len(msg.cones) > 5:  # Use vectorization for many cones
                # Extract all cone positions into numpy array
                cone_positions = np.array([[cone.position.x, cone.position.y, cone.position.z] 
                                          for cone in msg.cones])
                
                # Apply transformation: R * p + t (vectorized)
                transformed_positions = (rot_matrix @ cone_positions.T).T + translation
                
                # Create transformed cones
                for i, cone in enumerate(msg.cones):
                    transformed_cone = TrackedCone()
                    transformed_cone.position = Point(
                        x=transformed_positions[i, 0],
                        y=transformed_positions[i, 1],
                        z=transformed_positions[i, 2]
                    )
                    transformed_cone.color = cone.color
                    transformed_cone.track_id = cone.track_id
                    output_msg.cones.append(transformed_cone)
            else:
                # For few cones, use simpler approach
                for cone in msg.cones:
                    pos = np.array([cone.position.x, cone.position.y, cone.position.z])
                    transformed_pos = rot_matrix @ pos + translation
                    
                    transformed_cone = TrackedCone()
                    transformed_cone.position = Point(
                        x=transformed_pos[0],
                        y=transformed_pos[1],
                        z=transformed_pos[2]
                    )
                    transformed_cone.color = cone.color
                    transformed_cone.track_id = cone.track_id
                    output_msg.cones.append(transformed_cone)
            
            # Publish
            publisher.publish(output_msg)
            
            # Log periodically
            self._msg_count += 1
            if self._msg_count % 100 == 0:
                self.get_logger().info(
                    f'Transformed {len(msg.cones)} {label} TrackedCones from {msg.header.frame_id} to {self.target_frame}'
                )
                
        except TransformException as e:
            # Log only occasionally to avoid spam
            if not hasattr(self, '_last_warn_time') or (self.get_clock().now() - self._last_warn_time).nanoseconds > 5e9:
                self.get_logger().warning(f'TF transform failed for TrackedConeArray: {e}')
                self._last_warn_time = self.get_clock().now()
    
    def quaternion_to_rotation_matrix(self, x, y, z, w):
        """Convert quaternion to 3x3 rotation matrix"""
        # Normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Calculate rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        return np.array([
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
        ])
    
    # Legacy callback function - commented out for migration to TrackedConeArray
    # def legacy_cone_callback(self, msg):
    #     """Transform ModifiedFloat32MultiArray from source frame to target frame (LEGACY) - Optimized"""
    #     
    #     # Skip if already in target frame
    #     if msg.header.frame_id == self.target_frame:
    #         self.legacy_cone_pub.publish(msg)
    #         return
    #         
    #     try:
    #         # Get transform once with smaller timeout
    #         try:
    #             transform = self.tf_buffer.lookup_transform(
    #                 self.target_frame,
    #                 msg.header.frame_id,
    #                 msg.header.stamp,
    #                 timeout=rclpy.duration.Duration(seconds=0.01)
    #             )
    #         except (tf2_ros.ExtrapolationException, tf2_ros.LookupException):
    #             transform = self.tf_buffer.lookup_transform(
    #                 self.target_frame,
    #                 msg.header.frame_id,
    #                 rclpy.time.Time()
    #             )
    #         
    #         # Extract transform matrix components
    #         trans = transform.transform.translation
    #         rot = transform.transform.rotation
    #         rot_matrix = self.quaternion_to_rotation_matrix(rot.x, rot.y, rot.z, rot.w)
    #         translation = np.array([trans.x, trans.y, trans.z])
    #         
    #         # Create output message
    #         output_msg = ModifiedFloat32MultiArray()
    #         output_msg.header.stamp = msg.header.stamp
    #         output_msg.header.frame_id = self.target_frame
    #         output_msg.layout = msg.layout
    #         output_msg.class_names = msg.class_names
    #         
    #         # Parse and transform cone data efficiently
    #         if len(msg.data) > 0 and msg.layout.dim[1].size > 0:
    #             cone_size = msg.layout.dim[1].size
    #             if len(msg.data) % cone_size == 0:
    #                 num_cones = len(msg.data) // cone_size
    #                 
    #                 # Convert to numpy array for efficient processing
    #                 data_array = np.array(msg.data).reshape(num_cones, cone_size)
    #                 
    #                 # Extract and transform positions (first 3 elements of each cone)
    #                 if cone_size >= 3:
    #                     positions = data_array[:, :3]
    #                     transformed_positions = (rot_matrix @ positions.T).T + translation
    #                     data_array[:, :3] = transformed_positions
    #                 
    #                 # Convert back to list
    #                 output_msg.data = data_array.flatten().tolist()
    #             else:
    #                 # Fallback for malformed data
    #                 output_msg.data = msg.data
    #         else:
    #             output_msg.data = msg.data
    #         
    #         # Publish
    #         self.legacy_cone_pub.publish(output_msg)
    #         
    #         if self._msg_count % 100 == 0:
    #             self.get_logger().info(
    #                 f'Transformed ModifiedFloat32MultiArray from {msg.header.frame_id} to {self.target_frame}'
    #             )
    #             
    #     except TransformException as e:
    #         # Log only occasionally to avoid spam
    #         if not hasattr(self, '_last_warn_time_legacy') or (self.get_clock().now() - self._last_warn_time_legacy).nanoseconds > 5e9:
    #             self.get_logger().warning(f'TF transform failed for ModifiedFloat32MultiArray: {e}')
    #             self._last_warn_time_legacy = self.get_clock().now()


def main(args=None):
    rclpy.init(args=args)
    node = ConeFrameTransformer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()