import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('yolov8_trt_bed_detect')

    web_server_path = os.path.join(pkg, 'web_server', 'web_server.py')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg, 'config', 'trt_params.yaml'),
        description='Full path to the TensorRT parameter YAML file',
    )

    yolov8_node = Node(
        package='yolov8_trt_bed_detect',
        executable='yolov8_trt_bed_detect',
        name='yolov8_trt',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    web_server = ExecuteProcess(
        cmd=['python3', web_server_path],
        output='screen',
        env={
            **os.environ,
            'SAVE_DIR':    '/saved_frames',
            'MJPEG_PORT':  '8080',
            'API_PORT':    '8090',
            'HISTORY_DB':  '/saved_frames/count_history.db',
        }
    )

    return LaunchDescription([
        params_file_arg,
        yolov8_node,
        web_server,
    ])
