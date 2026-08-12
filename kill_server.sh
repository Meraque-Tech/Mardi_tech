#!/bin/bash

docker rm -f $(docker ps -aq)

# ARCH=$(uname -m)

# case "$ARCH" in
#   x86_64)
#     docker compose down imu_gnss_raw-x86 yolov8-trt-bed-detect-x86-jazzy

#     ;;
#   aarch64)
#     docker compose down imu_gnss_raw-aarch64 yolov8-trt-bed-detect-orin

#     ;;
#   *)
#     echo "Unsupported architecture: $ARCH"
#     exit 1
#     ;;
# esac


# ros2 daemon stop
# ros2 daemon start
# ros2 node list
# docker compose config --quiet