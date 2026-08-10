import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory('yolov8_trt_bed_detect_orin')
    pkg_prefix = os.path.dirname(os.path.dirname(pkg_share))  # .../install/<pkg>
    exe = os.path.join(pkg_prefix, 'lib', 'yolov8_trt_bed_detect_orin', 'yolov8_trt_bed_detect_orin')

    wts_arg = DeclareLaunchArgument(
        'wts',
        default_value='/weights/yolov8n.wts',
        description='Path to the input .wts weights file',
    )
    engine_arg = DeclareLaunchArgument(
        'engine',
        default_value='/weights/yolov8n_orin_nano_ros2_humble.engine',
        description='Path to write the serialised TensorRT engine',
    )
    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='n',
        description='YOLOv8 model size: n, s, m, l, x',
    )
    input_h_arg = DeclareLaunchArgument(
        'input_h',
        default_value='512',
        description='Input height (must match trt_params.yaml input_h)',
    )
    input_w_arg = DeclareLaunchArgument(
        'input_w',
        default_value='512',
        description='Input width (must match trt_params.yaml input_w)',
    )

    serialize_process = ExecuteProcess(
        cmd=[
            exe, '-s',
            LaunchConfiguration('wts'),
            LaunchConfiguration('engine'),
            LaunchConfiguration('model_type'),
            LaunchConfiguration('input_h'),
            LaunchConfiguration('input_w'),
        ],
        output='screen',
    )

    return LaunchDescription([
        wts_arg,
        engine_arg,
        model_type_arg,
        input_h_arg,
        input_w_arg,
        serialize_process,
    ])
