import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('yolov8_trt_bed_detect_orin')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg, 'config', 'trt_params.yaml'),
        description='Full path to the TensorRT parameter YAML file '
                     '(wts_name, engine_name, model_type, input_h, input_w)',
    )

    serialize_node = Node(
        package='yolov8_trt_bed_detect_orin',
        executable='yolov8_trt_bed_detect_orin',
        name='yolov8_trt',
        output='screen',
        arguments=['-s'],
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        serialize_node,
    ])
