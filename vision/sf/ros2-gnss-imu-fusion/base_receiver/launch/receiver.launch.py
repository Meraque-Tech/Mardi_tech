from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="base_receiver",
            executable="base_receiver",
            name="base_receiver",
            output="screen",
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare("base_receiver"),
                    "config",
                    "receiver.yaml",
                ])
            ],
        ),
        Node(
            package="base_receiver",
            executable="gnss_enu",
            name="gnss_enu",
            output="screen",
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare("base_receiver"),
                    "config",
                    "receiver.yaml",
                ])
            ],
        ),
    ])
