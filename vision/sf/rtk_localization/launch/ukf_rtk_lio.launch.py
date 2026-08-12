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
        'config/ukf_rtk_lio.yaml'
    )

    # RTK + LIO calibration node
    rtk_localization_pose = Node(
        package='rtk_localization',
        executable='rtk_lio_calibration',
        name='rtk_lio_calibration',
        output='screen'
    )

    # UKF sensor fusion node
    sensor_fusion_node = Node(
        package='robot_localization',
        executable='ukf_node',          # ✅ UKF node instead of EKF
        name='robot_localization_ukf',
        output='screen',
        parameters=[robot_localization_file_path]
    )

    return LaunchDescription([
        rtk_localization_pose,
        sensor_fusion_node
    ])
