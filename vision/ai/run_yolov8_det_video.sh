#!/bin/bash

# Usage: ./run_yolov8_det_video.sh [engine] [cam_id] [c/g] [show: true/false]
# Defaults: engine=yolov8n.engine, cam_id=0, postprocess=g, show=true

ENGINE="${1:-yolov8n.engine}"
CAM_ID="${2:-0}"
POSTPROCESS="${3:-g}"
SHOW="${4:-true}"

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
    bash -c "cd /workspace/yolov8/build && ./yolov8_det_video ./weights/${ENGINE} ${CAM_ID} ${POSTPROCESS} ${SHOW}"
