from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    """
    Launch file for Nocheon autonomous vehicle with complete TF tree.
    
    TF Tree Structure:
    map
     └── odom (static identity transform)
          └── base_link (dynamic, published by EKF)
               ├── os_sensor (static: -0.3m back, 0.7m up)
               │   └── os_imu (static: -0.3m back, 0.7m up)
               └── gps (static: 0.5m forward, 0.2m up) 
    """
    
    pkg_share = FindPackageShare('gps_imu_fusion')
    imu_preprocess_pkg = FindPackageShare('imu_preprocess')
    
    # Parameter file paths
    params_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'fusion_params.yaml'
    ])
    
    # Calibration file path (default)
    default_calib_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'improved_imu_calibration.json'
    ])
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'params_file',
            default_value=params_file,
            description='Path to the ROS2 parameters file'
        ),
        DeclareLaunchArgument(
            'imu_calibration_file',
            default_value=default_calib_file,
            description='Path to the IMU calibration JSON file'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',  # Changed to true for rosbag playback
            description='Use simulation (Gazebo) clock if true'
        ),
        
        # Sensor offset parameters (for easy adjustment)
        DeclareLaunchArgument('os_sensor_x', default_value='0.0', description='Ouster sensor X offset from base_link (m)'),
        DeclareLaunchArgument('os_sensor_y', default_value='0.0', description='Ouster sensor Y offset from base_link (m)'),
        DeclareLaunchArgument('os_sensor_z', default_value='0.7', description='Ouster sensor Z offset from base_link (m)'),
        
        DeclareLaunchArgument('gps_x', default_value='0.5', description='GPS antenna X offset from base_link (m)'),
        DeclareLaunchArgument('gps_y', default_value='0.0', description='GPS antenna Y offset from base_link (m)'),
        DeclareLaunchArgument('gps_z', default_value='0.0', description='GPS antenna Z offset from base_link (m)'),
        
        DeclareLaunchArgument('imu_x', default_value='0.0', description='IMU X offset from base_link (m)'),
        DeclareLaunchArgument('imu_y', default_value='0.0', description='IMU Y offset from base_link (m)'),
        DeclareLaunchArgument('imu_z', default_value='0.0', description='IMU Z offset from base_link (m)'),
        
        # # GPS to Cartesian converter node
        # Node(
        #     package='gps_imu_fusion',
        #     executable='gps_to_cartesian.py',
        #     name='gps_to_cartesian',
        #     output='screen',
        #     parameters=[{
        #         'reference_latitude': 37.540091,  # Konkuk University Ilgamho
        #         'reference_longitude': 127.076555,
        #         'reference_altitude': 39.5,
        #         'publish_tf': False,  # Let EKF handle odom->base_link
        #         'world_frame': 'map',
        #         'child_frame': 'gps',
        #         'use_sim_time': LaunchConfiguration('use_sim_time')
        #     }]
        # ),
        
        # EKF Fusion Node
        Node(
            package='gps_imu_fusion',
            executable='ekf_fusion_node',
            name='ekf_fusion_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'imu_calibration_file': LaunchConfiguration('imu_calibration_file'),
                    'reference_altitude': 39.5
                }
            ],
            remappings=[
                ('imu/processed', '/imu/processed'),
                ('/ublox_gps_node/fix', '/ublox_gps_node/fix'),
                ('/ublox_gps_node/fix_velocity', '/ublox_gps_node/fix_velocity')
            ]
        ),
        
        # Static transform: map -> odom (identity transform)
        # NOTE: Commented out when using cone_stellation SLAM, as SLAM manages map->odom transform
        # Uncomment this when running without SLAM
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),
        
        # Static transform: base_link -> os_sensor (Ouster LiDAR)
        # 30cm backward, 70cm upward from base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_os_sensor',
            output='screen',
            arguments=[
                LaunchConfiguration('os_sensor_x'),
                LaunchConfiguration('os_sensor_y'), 
                LaunchConfiguration('os_sensor_z'),
                '0', '0', '0',  # No rotation (quaternion xyzw = 0,0,0,1)
                'base_link', 'os_sensor'
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),
        
        # Static transform: base_link -> gps (GPS antenna)
        # 50cm forward, 20cm upward from base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_gps',
            output='screen',
            arguments=[
                LaunchConfiguration('gps_x'),
                LaunchConfiguration('gps_y'),
                LaunchConfiguration('gps_z'),
                '0', '0', '0',  # No rotation
                'base_link', 'gps'
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),
        
        # Cone frame transformer node
        Node(
            package='gps_imu_fusion',
            executable='cone_frame_transformer.py',
            name='cone_frame_transformer',
            output='screen',
            parameters=[{
                'target_frame': 'map',
                'timeout_sec': 0.1,
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }]
        ),
    ])