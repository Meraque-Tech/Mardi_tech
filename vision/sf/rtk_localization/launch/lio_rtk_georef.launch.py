import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_rtk_localization= get_package_share_directory('rtk_localization')
    robot_localization_file_path = os.path.join(pkg_rtk_localization, 'config/ekf_rtk_lio.yaml')

    rtk_localization_pose = Node(
            package='rtk_localization',
            executable='rtk_lio_pose_raw',
            name='rtk_lio_pose_raw',
            output='screen'
        )
    
    lio_rtk_georef_node = Node(
            package='rtk_localization',
            executable='lio_rtk_georef_node',
            name='lio_rtk_georef_node',
            output='screen'
        )
    
    # witmotion_imu = IncludeLaunchDescription(
    #     os.path.join(get_package_share_directory("witmotion"), "launch", "imu.launch.py")
    # )

    # lom = IncludeLaunchDescription(
    #     os.path.join(get_package_share_directory("rf2o_laser_odometry"), "launch", "rf2o_laser_odometry.launch.py")
    # )

    # witmotion_imu = IncludeLaunchDescription(
    #     os.path.join(get_package_share_directory("witmotion"), "launch", "livox_front_cmf.launch.py")
    # )
    
    return LaunchDescription([
        rtk_localization_pose,
        lio_rtk_georef_node
        # witmotion_imu
    ])