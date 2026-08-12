#!/bin/bash

ARCH=$(uname -m)

case "$ARCH" in
  x86_64)
    ./kill_server.sh
    docker compose build yolov8-trt-bed-detect-x86-jazzy
    docker compose up -d yolov8-trt-bed-detect-x86-jazzy
    docker compose up -d imu_gnss_raw
    docker compose logs -f imu_gnss_raw
    ;;
  aarch64)
    docker compose up -d imu_gnss_raw yolov8-trt-bed-detect-orin
    
    ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac
