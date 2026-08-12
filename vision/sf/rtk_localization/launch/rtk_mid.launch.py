import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg = get_package_share_directory('rtk_localization')
    rviz_config = os.path.join(pkg, 'rviz', 'rtk_mid_with_lio_transform.rviz')

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 visualizer'
    )

    rtk_mid_node = Node(
        package='rtk_localization',
        executable='rtk_mid',
        name='rtk_mid',
        parameters=[{
            'map_frame': 'odom',
            'odom_frame': 'odom',
            'base_frame': 'rtk_pub',
        }],
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    return LaunchDescription([
        use_rviz_arg,
        rtk_mid_node,
        rviz_node,
    ])
