from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare("base_receiver"),
        "config",
        "receiver.yaml",
    ])

    config_file = LaunchConfiguration("config_file")
    port = LaunchConfiguration("port")
    baud = LaunchConfiguration("baud")
    stale_timeout = LaunchConfiguration("stale_timeout")
    reconnect_interval = LaunchConfiguration("reconnect_interval")

    receiver_node = Node(
        package="base_receiver",
        executable="base_receiver",
        name="base_receiver",
        output="screen",
        parameters=[
            config_file,
            {
                "port": ParameterValue(port, value_type=str),
                "baud": ParameterValue(baud, value_type=int),
                "stale_timeout": ParameterValue(
                    stale_timeout,
                    value_type=float,
                ),
                "reconnect_interval": ParameterValue(
                    reconnect_interval,
                    value_type=float,
                ),
            },
        ],
    )

    enu_node = Node(
        package="base_receiver",
        executable="gnss_enu",
        name="gnss_enu",
        output="screen",
        parameters=[config_file],
    )

    motion_classifier_node = Node(
        package="base_receiver",
        executable="gnss_motion_classifier",
        name="gnss_motion_classifier",
        output="screen",
        parameters=[config_file],
    )

    def shutdown_when_node_exits(node, reason):
        return RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=[EmitEvent(event=Shutdown(reason=reason))],
            )
        )

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config,
            description="ROS parameter file used by all receiver nodes",
        ),
        DeclareLaunchArgument(
            "port",
            default_value="",
            description="Serial device path; empty enables USB auto-discovery",
        ),
        DeclareLaunchArgument(
            "baud",
            default_value="115200",
            description="Serial baud rate",
        ),
        DeclareLaunchArgument(
            "stale_timeout",
            default_value="3.0",
            description="Seconds before GNSS PVT data is considered stale",
        ),
        DeclareLaunchArgument(
            "reconnect_interval",
            default_value="2.0",
            description="Seconds between serial reconnect attempts",
        ),
        shutdown_when_node_exits(
            receiver_node,
            "base_receiver exited",
        ),
        shutdown_when_node_exits(
            enu_node,
            "gnss_enu exited",
        ),
        shutdown_when_node_exits(
            motion_classifier_node,
            "gnss_motion_classifier exited",
        ),
        receiver_node,
        enu_node,
        motion_classifier_node,
    ])
