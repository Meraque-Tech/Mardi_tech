#!/bin/bash

# Usage: ./run_yolov8_det_video.sh [cam_id] [c/g] [show: true/false]
# Defaults: cam_id=0, postprocess=g, show=true

CAM_ID="${1:-0}"
POSTPROCESS="${2:-g}"
SHOW="${3:-true}"

xhost +local:root

docker run -it --rm --net=host \
    --runtime nvidia \
    --gpus all \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/root/.Xauthority \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    -v ./yolov8/images:/workspace/yolov8/build/images:ro \
    -v ./yolov8/weights:/workspace/yolov8/build/weights:ro \
    meraquetech/race_nav:yolov8-trt-x86 \
    bash -c "cd /workspace/yolov8/build && ./yolov8_det_video ./weights/yolov8n.engine ${CAM_ID} ${POSTPROCESS} ${SHOW}"
