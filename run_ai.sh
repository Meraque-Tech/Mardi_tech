#!/bin/bash
cd vision/ai

docker run --rm --net=host \
  --runtime nvidia \
  --privileged \
  --name=yolov8-trt-bed-detect-nano \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --device /dev/video0:/dev/video0 \
  -v $PWD/yolov8/weights:/weights \
  -v $PWD/yolov8_trt_bed_detect/config:/ros2_ws/install/yolov8_trt_bed_detect/share/yolov8_trt_bed_detect/config \
  -v $PWD/yolov8_trt_bed_detect/launch:/ros2_ws/install/yolov8_trt_bed_detect/share/yolov8_trt_bed_detect/launch \
  meraquetech/race_nav:yolov8-trt-bed-detect-nano.v5 \
  bash -c "source /opt/ros/humble/install/setup.bash && source /ros2_ws/install/setup.bash && ros2 launch yolov8_trt_bed_detect bed_detect.launch.py"


# docker exec -it yolov8-trt-bed-detect-nano bash
# ros2 service call /bed_detection std_srvs/srv/Trigger {}
