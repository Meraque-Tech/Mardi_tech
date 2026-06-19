import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('yolov8_trt_bed_detect')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg, 'config', 'trt_params.yaml'),
        description='Path to the TensorRT parameter YAML file',
    )

    engine_arg = DeclareLaunchArgument(
        'engine_name',
        default_value='',
        description='Override engine path (leave empty to use value from params_file)',
    )

    yolov8_node = Node(
        package='yolov8_trt_bed_detect',
        executable='yolov8_trt_bed_detect',
        name='yolov8_trt',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            # command-line override wins over yaml when non-empty
            {'engine_name': LaunchConfiguration('engine_name')},
        ],
    )

    return LaunchDescription([
        params_file_arg,
        engine_arg,
        yolov8_node,
    ])
