#!/usr/bin/env python3
"""
GPS Visualization Node
Loads GPS data from CSV files and publishes as Path markers for RViz visualization
Reference point (Konkuk University) is set as the origin (0,0)
"""

import rclpy
from rclpy.node import Node
import os
import glob
import csv
import utm
from pathlib import Path
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header, ColorRGBA
import numpy as np
from ament_index_python.packages import get_package_share_directory


class GPSVisualizationNode(Node):
    def __init__(self):
        super().__init__('gps_visualization_node')
        
        # Reference point (Konkuk University Ilgamho)
        self.ref_lat = 37.540091
        self.ref_lon = 127.076555
        
        # Convert reference to UTM
        self.utm_x_ref, self.utm_y_ref, self.utm_zone, self.utm_letter = utm.from_latlon(
            self.ref_lat, self.ref_lon
        )
        
        self.get_logger().info(
            f'GPS Visualization Node initialized\n'
            f'Reference: {self.ref_lat:.6f}°N, {self.ref_lon:.6f}°E\n'
            f'UTM Zone: {self.utm_zone}{self.utm_letter}\n'
            f'UTM Reference: {self.utm_x_ref:.2f}E, {self.utm_y_ref:.2f}N'
        )
        
        # Color mapping for each course
        self.colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Green
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # Yellow
            ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),  # Magenta
            ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),  # Cyan
        ]
        
        # Publishers for each course
        self.path_publishers = {}
        self.marker_publishers = {}
        
        # Load all CSV files
        self.courses_data = {}
        self.load_csv_files()
        
        # Create timer for periodic publishing (1Hz)
        self.timer = self.create_timer(1.0, self.publish_paths)
        
        self.get_logger().info(f'Loaded {len(self.courses_data)} course files')
    
    def load_csv_files(self):
        """Load all course*.csv files from the data directory"""
        # Try package share directory first
        data_dir = None
        try:
            package_share_dir = get_package_share_directory('gps_imu_fusion')
            data_dir = os.path.join(package_share_dir, 'scripts', 'gps_vis', 'data')
            if not os.path.exists(data_dir):
                data_dir = None
        except Exception:
            pass
        
        # Fallback to script location
        if data_dir is None:
            script_dir = Path(__file__).parent
            local_data_dir = script_dir / 'data'
            
            if local_data_dir.exists():
                data_dir = str(local_data_dir)
                self.get_logger().info(f'Using local data directory: {data_dir}')
            else:
                # Try relative path from workspace
                workspace_data_dir = script_dir.parent.parent.parent / 'scripts' / 'gps_vis' / 'data'
                if workspace_data_dir.exists():
                    data_dir = str(workspace_data_dir)
                    self.get_logger().info(f'Using workspace data directory: {data_dir}')
        
        if data_dir is None or not os.path.exists(data_dir):
            self.get_logger().error('No data directory found!')
            return
        
        # Find all course*.csv files
        csv_files = sorted(glob.glob(os.path.join(data_dir, 'course*.csv')))
        
        for idx, csv_file in enumerate(csv_files):
            course_name = os.path.splitext(os.path.basename(csv_file))[0]
            course_num = int(course_name.replace('course', ''))
            
            # Create publishers for this course
            path_topic = f'/gps_vis/{course_name}'
            marker_topic = f'/gps_vis/{course_name}_markers'
            
            self.path_publishers[course_name] = self.create_publisher(
                Path, path_topic, 10
            )
            self.marker_publishers[course_name] = self.create_publisher(
                MarkerArray, marker_topic, 10
            )
            
            # Load CSV data
            path_data = []
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        lat = float(row['Lat'])
                        lon = float(row['Long'])
                        
                        # Convert to UTM
                        utm_x, utm_y, zone, letter = utm.from_latlon(lat, lon)
                        
                        # Check UTM zone consistency
                        if zone != self.utm_zone or letter != self.utm_letter:
                            self.get_logger().warn(
                                f'{course_name}: Different UTM zone detected '
                                f'{zone}{letter} vs {self.utm_zone}{self.utm_letter}'
                            )
                        
                        # Calculate local coordinates
                        x = utm_x - self.utm_x_ref
                        y = utm_y - self.utm_y_ref
                        
                        path_data.append((x, y))
                
                self.courses_data[course_name] = {
                    'points': path_data,
                    'color': self.colors[min(idx, len(self.colors)-1)]
                }
                
                self.get_logger().info(
                    f'Loaded {course_name}: {len(path_data)} points'
                )
                
            except Exception as e:
                self.get_logger().error(f'Failed to load {csv_file}: {e}')
    
    def create_path_message(self, course_name, points, color):
        """Create a Path message from GPS points"""
        path = NavPath()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            
            # Orientation (identity quaternion)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            
            path.poses.append(pose)
        
        return path
    
    def create_marker_array(self, course_name, points, color):
        """Create markers for visualization (points and lines)"""
        marker_array = MarkerArray()
        
        # Line list marker (LINE_LIST allows width control in RViz2)
        line_marker = Marker()
        line_marker.header.frame_id = 'map'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = f'{course_name}_line'
        line_marker.id = 0
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 5.0  # Line width (works with LINE_LIST)
        line_marker.color = color
        line_marker.lifetime.sec = 0  # Persistent
        
        # Create line segments between consecutive points
        for i in range(len(points) - 1):
            # Start point of line segment
            p1 = Point()
            p1.x = points[i][0]
            p1.y = points[i][1]
            p1.z = 0.0
            line_marker.points.append(p1)
            
            # End point of line segment
            p2 = Point()
            p2.x = points[i+1][0]
            p2.y = points[i+1][1]
            p2.z = 0.0
            line_marker.points.append(p2)
        
        marker_array.markers.append(line_marker)
        
        # Add start/end markers
        if len(points) > 0:
            # Start marker (sphere)
            start_marker = Marker()
            start_marker.header = line_marker.header
            start_marker.ns = f'{course_name}_start'
            start_marker.id = 1
            start_marker.type = Marker.SPHERE
            start_marker.action = Marker.ADD
            start_marker.pose.position.x = points[0][0]
            start_marker.pose.position.y = points[0][1]
            start_marker.pose.position.z = 0.0
            start_marker.pose.orientation.w = 1.0
            start_marker.scale.x = 0.5
            start_marker.scale.y = 0.5
            start_marker.scale.z = 0.5
            start_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
            start_marker.lifetime.sec = 0
            marker_array.markers.append(start_marker)
            
            # End marker (cube)
            end_marker = Marker()
            end_marker.header = line_marker.header
            end_marker.ns = f'{course_name}_end'
            end_marker.id = 2
            end_marker.type = Marker.CUBE
            end_marker.action = Marker.ADD
            end_marker.pose.position.x = points[-1][0]
            end_marker.pose.position.y = points[-1][1]
            end_marker.pose.position.z = 0.0
            end_marker.pose.orientation.w = 1.0
            end_marker.scale.x = 0.5
            end_marker.scale.y = 0.5
            end_marker.scale.z = 0.5
            end_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red
            end_marker.lifetime.sec = 0
            marker_array.markers.append(end_marker)
        
        return marker_array
    
    def publish_paths(self):
        """Publish all paths and markers"""
        for course_name, data in self.courses_data.items():
            points = data['points']
            color = data['color']
            
            # Publish Path
            path_msg = self.create_path_message(course_name, points, color)
            self.path_publishers[course_name].publish(path_msg)
            
            # Publish MarkerArray
            marker_array = self.create_marker_array(course_name, points, color)
            self.marker_publishers[course_name].publish(marker_array)
        
        # Log once every 10 seconds
        if int(self.get_clock().now().nanoseconds / 1e9) % 10 == 0:
            self.get_logger().info(
                f'Publishing {len(self.courses_data)} GPS paths...'
            )


def main(args=None):
    rclpy.init(args=args)
    node = GPSVisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()