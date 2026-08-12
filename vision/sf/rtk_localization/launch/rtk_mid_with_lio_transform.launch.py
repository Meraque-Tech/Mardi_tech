import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── launch arguments ──────────────────────────────────────────────────────
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Launch RViz2 visualizer'
    )

    lio_odom_topic_arg = DeclareLaunchArgument(
        'lio_odom_topic',
        default_value='/Odometry/mapping',
        description='FAST-LIO2 odometry topic'
    )

    rtk_odom_topic_arg = DeclareLaunchArgument(
        'rtk_odom_topic',
        default_value='/rtk/pub/odom',
        description='RTK odometry in UTM (nav_msgs/Odometry)'
    )

    # ── rtk_mid.launch.py ─────────────────────────────────────────────────────
    rtk_mid_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rtk_localization'),
                'launch', 'rtk_mid.launch.py'
            )
        ),
        launch_arguments={
            'use_rviz': LaunchConfiguration('use_rviz'),
        }.items()
    )

    # ── lio_rtk_transform.launch.py ───────────────────────────────────────────
    lio_rtk_transform_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gnss_lio_eskf'),
                'launch', 'lio_rtk_transform.launch.py'
            )
        ),
        launch_arguments={
            'lio_odom_topic': LaunchConfiguration('lio_odom_topic'),
            'rtk_odom_topic': LaunchConfiguration('rtk_odom_topic'),
        }.items()
    )

    return LaunchDescription([
        use_rviz_arg,
        lio_odom_topic_arg,
        rtk_odom_topic_arg,
        rtk_mid_launch,
        lio_rtk_transform_launch,
    ])
