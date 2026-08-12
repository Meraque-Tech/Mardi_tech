import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('yolov8_trt_bed_detect_orin')
    web_server_path = os.path.join(pkg, 'web_server', 'web_server.py')

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='inference',
        choices=['serialize', 'inference'],
        description='Run TensorRT engine serialization or YOLOv8 inference',
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg, 'config', 'trt_params.yaml'),
        description='Full path to the TensorRT parameter YAML file',
    )

    serialize_node = Node(
        package='yolov8_trt_bed_detect_orin',
        executable='yolov8_trt_bed_detect_orin',
        name='yolov8_trt',
        output='screen',
        arguments=['-s'],
        parameters=[LaunchConfiguration('params_file')],
        condition=LaunchConfigurationEquals('mode', 'serialize'),
    )

    yolov8_node = Node(
        package='yolov8_trt_bed_detect_orin',
        executable='yolov8_trt_bed_detect_orin',
        name='yolov8_trt',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        condition=LaunchConfigurationEquals('mode', 'inference'),
    )

    web_server = ExecuteProcess(
        cmd=['python3', web_server_path],
        output='screen',
        env={
            **os.environ,
            'SAVE_DIR': '/saved_frames',
            'MJPEG_PORT': '8080',
            'API_PORT': '8090',
            'HISTORY_DB': '/saved_frames/count_history.db',
            'AUTO_SAVE_INTERVAL': '0.5',
            'MOTION_STATE_TOPIC': os.environ.get(
                'MOTION_STATE_TOPIC', '/gnss_imu_eskf/motion_state_raw_gnss'
            ),
            'MOTION_POS_DEADBAND_TOPIC': os.environ.get(
                'MOTION_POS_DEADBAND_TOPIC', '/gnss_imu_eskf/motion_pos_deadband'
            ),
            'MOTION_POS_DEADBAND_M': os.environ.get('MOTION_POS_DEADBAND_M', '0.3'),
            'GNSS_ONLY_ODOM_TOPIC': os.environ.get(
                'GNSS_ONLY_ODOM_TOPIC', '/gnss_imu_eskf/gnss_only_odom'
            ),
            'DIRECTION_STALE_TIMEOUT': os.environ.get(
                'DIRECTION_STALE_TIMEOUT', '3.0'
            ),
        },
        condition=LaunchConfigurationEquals('mode', 'inference'),
    )

    return LaunchDescription([
        mode_arg,
        params_file_arg,
        serialize_node,
        yolov8_node,
        web_server,
    ])
