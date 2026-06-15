# yolov8_trt_bed_detect

YOLOv8 TensorRT bed detection ROS 2 node for Jetson Nano. Uses a USB webcam as input — no ZED/depth camera required.

---

## How it works

- Activates via a ROS 2 service call (`bed_detection`)
- Runs YOLOv8 inference using TensorRT on every webcam frame
- Publishes detection confidence and a binary status (`0` / `1`) over ROS topics

### ROS Interface

| Type | Name | Msg Type | Description |
|------|------|----------|-------------|
| Service | `bed_detection` | `std_srvs/Trigger` | Start detection loop |
| Publisher | `conf` | `std_msgs/Float32` | Confidence score of detection |
| Publisher | `bed_detection_status` | `std_msgs/UInt8` | `1` = bed detected, `0` = not detected |

---

## Prerequisites

- Jetson Nano with JetPack 4.6.1 (L4T r32.7)
- Docker with NVIDIA runtime
- USB webcam on `/dev/video0`
- Trained YOLOv8 `.pt` model

---

## Step 1 — Generate `.wts` from `.pt`

Install ultralytics on your host (x86 or Jetson):

```bash
pip3 install ultralytics
```

Copy your `.pt` file into the package directory and run:

```bash
cd vision/ai/yolov8_trt_bed_detect/
python3 gen_wts.py -w yolov8s_bed.pt
# output: yolov8s_bed.wts
```

---

## Step 2 — Build the Docker image

From `vision/ai/`:

```bash
docker build \
  -f Dockerfile.yolov8_trt_bed_detect_jetson_nano \
  -t meraquetech/race_nav:yolov8-trt-bed-detect-nano.v1 \
  .
```

> The build context expects `yolov8_trt_bed_detect/` to be present in `vision/ai/`.

---

## Step 3 — Build the TensorRT engine (inside container)

Run the container with the weights folder mounted:

```bash
docker run -it --rm --net=host \
  --runtime nvidia \
  --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --device /dev/video0:/dev/video0 \
  -v $PWD/yolov8_trt_bed_detect/weights:/ros2_ws/src/yolov8_trt_bed_detect/weights \
  meraquetech/race_nav:yolov8-trt-bed-detect-nano.v1
```

Inside the container, serialize the engine:

```bash
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect \
  -s weights/yolov8s_bed.wts weights/yolov8s_bed.engine s
```

This produces `weights/yolov8s_bed.engine` (only needs to be done once per model).

---

## Step 4 — Run the detection node

```bash
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect \
  -d weights/yolov8s_bed.engine ./ g -conf 0.8
```

| Argument | Description |
|----------|-------------|
| `-d` | Deserialize and run inference mode |
| `weights/yolov8s_bed.engine` | Path to TensorRT engine file |
| `./` | Sample/image dir (not used in webcam mode, pass `./`) |
| `g` / `c` | Post-processing on GPU (`g`) or CPU (`c`) |
| `-conf <val>` | Confidence threshold (e.g. `0.8`) |

**Example:**

```bash
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect \
  -d weights/yolov8s_bed.engine ./ g -conf 0.85
```

---

## Step 5 — Trigger detection from another node / terminal

```bash
ros2 service call /bed_detection std_srvs/srv/Trigger {}
```

Then monitor output:

```bash
ros2 topic echo /bed_detection_status
ros2 topic echo /conf
```

---

## One-line Docker run (after engine is built)

```bash
docker run -it --rm --net=host \
  --runtime nvidia \
  --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --device /dev/video0:/dev/video0 \
  -v $PWD/yolov8_trt_bed_detect/weights:/ros2_ws/src/yolov8_trt_bed_detect/weights:ro \
  meraquetech/race_nav:yolov8-trt-bed-detect-nano.v1 \
  bash -c "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect \
  -d /ros2_ws/src/yolov8_trt_bed_detect/weights/yolov8s_bed.engine ./ g -conf 0.85"
```

---

## Webcam index

Default is `/dev/video0` (index `0`). To use a different camera, change the index in `main.cpp`:

```cpp
cv::VideoCapture cap(0);  // change 0 to 1, 2, etc.
```

---

## Notes

- Press `q` in the OpenCV window to stop the node
- The detection loop is **inactive until** the `bed_detection` service is called
- Engine serialization is device-specific — rebuild the `.engine` if switching hardware
