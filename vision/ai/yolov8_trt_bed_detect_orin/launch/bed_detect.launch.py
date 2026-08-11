import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('yolov8_trt_bed_detect_orin')

    web_server_path = os.path.join(pkg, 'web_server', 'web_server.py')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg, 'config', 'trt_params.yaml'),
        description='Full path to the TensorRT parameter YAML file',
    )

    yolov8_node = Node(
        package='yolov8_trt_bed_detect_orin',
        executable='yolov8_trt_bed_detect_orin',
        name='yolov8_trt',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        yolov8_node
    ])
