"""Launch the BWT901 BLE IMU publisher."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    device_address_arg = DeclareLaunchArgument(
        'device_address',
        default_value='F7:75:A3:1D:9C:AB',
        description=(
            'BLE MAC address. Set empty to use the first matching name.'
        ),
    )
    device_name_filter_arg = DeclareLaunchArgument(
        'device_name_filter',
        default_value='WT',
        description='BLE device-name substring used when no MAC is set.',
    )
    topic_arg = DeclareLaunchArgument(
        'topic',
        default_value='imu/data_raw',
        description='Raw sensor_msgs/Imu topic consumed by Madgwick.',
    )
    mag_topic_arg = DeclareLaunchArgument(
        'mag_topic',
        default_value='imu/mag',
        description='Raw sensor_msgs/MagneticField topic for Madgwick.',
    )
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='imu_link',
        description='Frame placed in each Imu message header.',
    )
    publish_tf_arg = DeclareLaunchArgument(
        'publish_tf',
        default_value='false',
        description='Publish Madgwick absolute orientation as a transform.',
    )
    tf_parent_frame_arg = DeclareLaunchArgument(
        'tf_parent_frame',
        default_value='imu_reference',
        description='Parent frame of the optional rotation transform.',
    )
    magnetic_declination_arg = DeclareLaunchArgument(
        'magnetic_declination_radians',
        default_value='0.0',
        description='Local magnetic declination added to absolute yaw.',
    )
    yaw_offset_arg = DeclareLaunchArgument(
        'yaw_offset_radians',
        default_value='0.0',
        description='Additional mounting correction applied to yaw.',
    )

    imu_comp_filter = Node(
                package='imu_complementary_filter',
                executable='complementary_filter_node',
                name='complementary_filter_node')

    bwt901ble_imu_publisher = Node(
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
        bwt901ble_imu_publisher,
        imu_comp_filter,
        # imu_madgwick_filter,
    ])
