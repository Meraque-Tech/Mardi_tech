import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Package path
    pkg_rtk_localization = get_package_share_directory('rtk_localization')

    # UKF config file
    robot_localization_file_path = os.path.join(
        pkg_rtk_localization,
        'config/navsat_transform.yaml'
    )

    # rtk_mid
    rtk_localization_pose = Node(
        package='rtk_localization',
        executable='rtk_mid',
        name='rtk_mid',
        output='screen'
    )

    # navsat_transform node
    sensor_fusion_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        parameters=[robot_localization_file_path],
        remappings=[
            # GPS Fix input
            ("/gps/fix", "/rtk/mid/fix"),
            ("/imu/data", "/vectornav/imu_uncompensated"),
            ("/odometry/filtered", "/dlio/odom_node/odom"),
            # Output GPS odometry
            # ("/odometry/gps", "/rtk/gps/odom")
        ]
    )

    return LaunchDescription([
        rtk_localization_pose,
        sensor_fusion_node
    ])
