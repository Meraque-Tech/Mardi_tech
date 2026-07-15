from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Optional: temporary compatibility alias for reference frame
        DeclareLaunchArgument('publish_reference_tf', default_value='false', description='Publish map->reference identity for legacy Planning nodes'),

        # Sensor offsets relative to base_link
        DeclareLaunchArgument('os_sensor_x', default_value='0.483785'),
        DeclareLaunchArgument('os_sensor_y', default_value='0.0'),
        DeclareLaunchArgument('os_sensor_z', default_value='0.912377'),

        DeclareLaunchArgument('gps_x', default_value='1.3'),
        DeclareLaunchArgument('gps_y', default_value='0.0'),
        DeclareLaunchArgument('gps_z', default_value='0.357'),

        DeclareLaunchArgument('imu_x', default_value='1.307'),
        DeclareLaunchArgument('imu_y', default_value='0.0'),
        DeclareLaunchArgument('imu_z', default_value='-0.142'),

        # # ignore lever arm
        # DeclareLaunchArgument('os_sensor_x', default_value='0.0'),
        # DeclareLaunchArgument('os_sensor_y', default_value='0.0'),
        # DeclareLaunchArgument('os_sensor_z', default_value='0.0'),

        # DeclareLaunchArgument('gps_x', default_value='0.0'),
        # DeclareLaunchArgument('gps_y', default_value='0.0'),
        # DeclareLaunchArgument('gps_z', default_value='0.0'),

        # DeclareLaunchArgument('imu_x', default_value='0.0'),
        # DeclareLaunchArgument('imu_y', default_value='0.0'),
        # DeclareLaunchArgument('imu_z', default_value='0.0'),

        # map -> odom (identity until SLAM/localization owns it)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
        ),

        # base_link -> os_sensor
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_os_sensor',
            arguments=[
                LaunchConfiguration('os_sensor_x'),
                LaunchConfiguration('os_sensor_y'),
                LaunchConfiguration('os_sensor_z'),
                '0', '0', '0',
                'base_link', 'os_sensor'
            ]
        ),
        
        # ###############################################################
        # ##               최종 수정된 os_sensor to camera TF          ##
        # ###############################################################

        # os_sensor -> camera_1
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='os_sensor_to_cam1',
            arguments=[
                # Translation (x y z)
                '0.12429922', '0.08384356', '-0.10144900',
                # Quaternion (qx qy qz qw)
                '0.62267844', '-0.37316381', '0.37352383', '-0.57749482',
                # Parent -> Child
                'os_sensor', 'camera_1'
            ]
        ),

        # os_sensor -> camera_2
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='os_sensor_to_cam2',
            arguments=[
                # Translation (x y z)
                '0.14285914', '-0.09870662', '-0.07061942',
                # Quaternion (qx qy qz qw)
                '-0.38646373', '0.61188201', '-0.57029629', '0.38859790',
                # Parent -> Child
                'os_sensor', 'camera_2'
            ]
        ),
        
        # ###############################################################

        # base_link -> gps
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_gps',
            arguments=[
                LaunchConfiguration('gps_x'),
                LaunchConfiguration('gps_y'),
                LaunchConfiguration('gps_z'),
                '0', '0', '0',
                'base_link', 'gps'
            ]
        ),

        # base_link -> imu_link
        # Rotate IMU frame by 180 deg about Z to align sensor X with vehicle forward
        # Use quaternion form to avoid Euler order ambiguity: q = [0, 0, 1, 0] (yaw=pi)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu',
            arguments=[
                LaunchConfiguration('imu_x'),
                LaunchConfiguration('imu_y'),
                LaunchConfiguration('imu_z'),
                '0', '0', '1', '0',
                'base_link', 'imu_link'
            ]
        ),
    ])
