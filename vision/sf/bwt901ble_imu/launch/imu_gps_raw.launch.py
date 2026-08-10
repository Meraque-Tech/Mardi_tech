from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ---------------- IMU Arguments ---------------- #

    device_address_arg = DeclareLaunchArgument(
        'device_address',
        default_value='F7:75:A3:1D:9C:AB'
    )

    device_name_filter_arg = DeclareLaunchArgument(
        'device_name_filter',
        default_value='WT'
    )

    topic_arg = DeclareLaunchArgument(
        'topic',
        default_value='imu/data_raw'
    )

    mag_topic_arg = DeclareLaunchArgument(
        'mag_topic',
        default_value='imu/mag'
    )

    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='imu_link'
    )

    publish_tf_arg = DeclareLaunchArgument(
        'publish_tf',
        default_value='false'
    )

    tf_parent_frame_arg = DeclareLaunchArgument(
        'tf_parent_frame',
        default_value='imu_reference'
    )

    magnetic_declination_arg = DeclareLaunchArgument(
        'magnetic_declination_radians',
        default_value='0.0'
    )

    yaw_offset_arg = DeclareLaunchArgument(
        'yaw_offset_radians',
        default_value='0.0'
    )

    # ---------------- GPS Launch ---------------- #

    gps_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('base_receiver'),
                'launch',
                'receiver.launch.py'
            )
        )
    )

    # ---------------- RTK Localization ---------------- #

    localization_node = Node(
        package='rtk_localization',
        executable='gnss_fix_enu_odom',
        name='rtk_localization',
        output='screen',
        parameters=[
            os.path.join(
                get_package_share_directory('rtk_localization'),
                'config',
                'gnss_odom.yaml'
            )
        ],
    )

    shutdown_on_localization_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=localization_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason='rtk_localization exited')
                )
            ],
        )
    )

    # ---------------- Complementary Filter ---------------- #

    imu_comp_filter = Node(
        package='imu_complementary_filter',
        executable='complementary_filter_node',
        name='complementary_filter_node',
        output='screen'
    )

    # ---------------- IMU Publisher ---------------- #

    imu_publisher = Node(
        package='bwt901ble_imu',
        executable='imu_publisher',
        name='bwt901ble_imu_publisher',
        output='screen',
        parameters=[{
            'device_address': LaunchConfiguration('device_address'),
            'device_name_filter': LaunchConfiguration('device_name_filter'),
            'topic': LaunchConfiguration('topic'),
            'mag_topic': LaunchConfiguration('mag_topic'),
            'frame_id': LaunchConfiguration('frame_id'),
            'publish_tf': False,
            'tf_parent_frame': LaunchConfiguration('tf_parent_frame'),
        }],
    )

    # ---------------- Madgwick Filter (Optional) ---------------- #

    imu_madgwick_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter_madgwick_node',
        output='screen',
        parameters=[{
            'use_mag': False,
            'publish_tf': ParameterValue(
                LaunchConfiguration('publish_tf'),
                value_type=bool,
            ),
            'fixed_frame': LaunchConfiguration('tf_parent_frame'),
            'world_frame': 'enu',
            'declination': ParameterValue(
                LaunchConfiguration('magnetic_declination_radians'),
                value_type=float,
            ),
            'yaw_offset': ParameterValue(
                LaunchConfiguration('yaw_offset_radians'),
                value_type=float,
            ),
        }],
        remappings=[
            ('imu/data_raw', LaunchConfiguration('topic')),
            ('imu/mag', LaunchConfiguration('mag_topic')),
            ('imu/data', 'imu/data'),
        ],
    )

    return LaunchDescription([
        device_address_arg,
        device_name_filter_arg,
        topic_arg,
        mag_topic_arg,
        frame_id_arg,
        publish_tf_arg,
        tf_parent_frame_arg,
        magnetic_declination_arg,
        yaw_offset_arg,

        # GPS
        gps_launch,
        shutdown_on_localization_exit,
        localization_node,

        # IMU
        imu_publisher,
        imu_comp_filter,

        # Uncomment if you want the Madgwick filter
        # imu_madgwick_filter,
    ])
