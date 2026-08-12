#!/bin/bash
colcon build --packages-select rtk_localization --merge-install

source install/setup.bash
ros2 launch rtk_localization gnss_imu_eskf.launch.py

# colcon build --packages-select imu_gnss_fusion
# source install/setup.bash
# ros2 launch imu_gnss_fusion imu_gnss_fusion.launch.py

# ros2 run rtk_localization gnss_imu_eskf_node

# cd ~/Downloads/bags && ros2 bag play imu_gps_20260804_100543