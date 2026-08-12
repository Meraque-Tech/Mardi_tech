#!/bin/bash

ARCH=$(uname -m)

case "$ARCH" in
  x86_64)
    docker compose up -d yolov8-trt-bed-detect-x86-jazzy
    ;;
  aarch64)
    docker compose up -d bwt901ble_imu yolov8-trt-bed-detect-orin
    ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac
