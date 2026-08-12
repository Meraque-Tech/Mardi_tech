#!/bin/bash

ARCH=$(uname -m)

case "$ARCH" in
  x86_64)
    docker compose up -d yolov8-trt-bed-detect-x86-jazzy
    # docker compose up -d yolov8-trt-bed-detect-x86-jazzy --build
    docker compose up imu_gnss_raw
    
    ;;
  aarch64)
    docker compose up -d imu_gnss_raw yolov8-trt-bed-detect-orin
    
    ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac
