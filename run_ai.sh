#!/bin/bash

ARCH=$(uname -m)

case "$ARCH" in
  x86_64)
    ./kill_server.sh
    docker compose build imu_gnss_raw-x86 yolov8-trt-bed-detect-x86-jazzy
    # docker compose push imu_gnss_raw-x86 yolov8-trt-bed-detect-x86-jazzy
    docker compose up -d imu_gnss_raw-x86 yolov8-trt-bed-detect-x86-jazzy

    docker compose logs -f imu_gnss_raw-x86

    ;;
  aarch64)
    docker compose build imu_gnss_raw-aarch64 yolov8-trt-bed-detect-orin
    # docker compose push imu_gnss_raw-aarch64 yolov8-trt-bed-detect-orin
    docker compose up -d imu_gnss_raw-aarch64 yolov8-trt-bed-detect-orin

    ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac


# ros2 daemon stop
# ros2 daemon start
# ros2 node list
# docker compose config --quiet