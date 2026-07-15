from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('gps_imu_fusion_ihw')
    param_file = os.path.join(pkg_share, 'config', 'gps_imu_fusion.yaml')

    return LaunchDescription([
        Node(
            package='gps_imu_fusion_ihw',
            executable='sensor_fusion_node',
            name='gps_imu_sensor_fusion',
            parameters=[param_file],
            output='screen'
        )
    ])
