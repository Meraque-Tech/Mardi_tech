#!/bin/bash

ARCH=$(uname -m)

case "$ARCH" in
  x86_64)
    ./kill_server.sh
    docker compose build imu_gnss_raw-x86 yolov8-trt-bed-detect-x86-jazzy sensor_data_raw-x86
    # docker compose push imu_gnss_raw-x86 yolov8-trt-bed-detect-x86-jazzy sensor_data_raw-x86
    docker compose up -d imu_gnss_raw-x86 yolov8-trt-bed-detect-x86-jazzy
    # docker compose up -d imu_gnss_raw-x86 yolov8-trt-bed-detect-x86-jazzy sensor_data_raw-x86

    docker compose logs -f imu_gnss_raw-x86

    ;;
  aarch64)
    docker compose build imu_gnss_raw-aarch64 \
          yolov8-trt-bed-detect-orin \
          sensor_data_raw-aarch64
    # docker compose push imu_gnss_raw-aarch64 yolov8-trt-bed-detect-orin sensor_data_raw-aarch64
    docker compose up -d imu_gnss_raw-aarch64 \
          yolov8-trt-bed-detect-orin \
          sensor_data_raw-aarch64

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