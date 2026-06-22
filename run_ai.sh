#!/bin/bash
cd vision/ai

docker run -it --rm --net=host \
  --runtime nvidia \
  --privileged \
  --name=yolov8-trt-bed-detect-nano \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --device /dev/video0:/dev/video0 \
  -v $PWD/yolov8/weights:/weights \
  -v $PWD/yolov8_trt_bed_detect/config:/ros2_ws/install/yolov8_trt_bed_detect/share/yolov8_trt_bed_detect/config \
  -v $PWD/yolov8_trt_bed_detect/launch:/ros2_ws/install/yolov8_trt_bed_detect/share/yolov8_trt_bed_detect/launch \
  meraquetech/race_nav:yolov8-trt-bed-detect-nano.v5 \
  bash -c "ros2 launch yolov8_trt_bed_detect yolov8_trt_bed_detect.launch.py"

