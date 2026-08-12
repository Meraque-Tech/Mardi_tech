import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('rtk_localization'),
        'config',
        'gnss_imu_eskf.yaml',
    )

    config_argument = DeclareLaunchArgument(
        'config',
        default_value=default_config,
        description='GNSS/IMU ESKF parameter file',
    )

    fusion_node = Node(
        package='rtk_localization',
        executable='gnss_imu_eskf_node',
        name='gnss_imu_eskf_node',
        output='screen',
        parameters=[LaunchConfiguration('config')],
    )

    return LaunchDescription([config_argument, fusion_node])

# colcon build --packages-select rtk_localization --merge-install
# ros2 launch rtk_localization gnss_imu_eskf.launch.py