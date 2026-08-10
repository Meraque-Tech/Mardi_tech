from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("rtk_localization")
    default_config = PathJoinSubstitution(
        [package_share, "config", "gnss_odom.yaml"]
    )
    default_rviz_config = PathJoinSubstitution(
        [package_share, "rviz", "gnss_odom.rviz"]
    )

    config_file = LaunchConfiguration("config_file")
    use_rviz = LaunchConfiguration("use_rviz")

    odometry_node = Node(
        package="rtk_localization",
        executable="gnss_fix_enu_odom",
        name="rtk_localization",
        output="screen",
        parameters=[config_file],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="gnss_odometry_rviz",
        output="screen",
        arguments=["-d", default_rviz_config],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="GNSS odometry parameter file",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="false",
                description="Start RViz with GNSS odometry displays",
            ),
            odometry_node,
            rviz_node,
        ]
    )
