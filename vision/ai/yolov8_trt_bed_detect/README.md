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
  -t meraquetech/race_nav:yolov8-trt-bed-detect-nano.v4 \
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
  -v $PWD/yolov8/weights:/weights \
  -v $PWD/yolov8_trt_bed_detect/weights:/ros2_ws/src/yolov8_trt_bed_detect/weights \
  -v $PWD/yolov8_trt_bed_detect/config:/ros2_ws/install/yolov8_trt_bed_detect/share/yolov8_trt_bed_detect/config \
  meraquetech/race_nav:yolov8-trt-bed-detect-nano.v4

```

Inside the container, serialize the engine:

```bash
# for nano ->
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect \
  -s /weights/yolov8n.wts /weights/yolov8n.engine n

# for small ->
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect \
  -s /weights/yolov8s.wts /weights/yolov8s.engine s

```

This produces `weights/yolov8s.engine` (only needs to be done once per model).

---

## Step 4 — Configure parameters

All TensorRT and camera settings live in [`config/trt_params.yaml`](config/trt_params.yaml). Edit before running:

```yaml
yolov8_trt:
  ros__parameters:
    engine_name:        "weights/yolov8n.engine"
    input_h:            416        # must match engine build resolution
    input_w:            416
    precision:          "fp16"     # "fp16" or "int8"
    cuda_post_process:  "g"        # "g" = GPU NMS (faster), "c" = CPU NMS
    conf_thresh:        0.5
    nms_thresh:         0.45
    conf_score_value:   0.8        # gate for /bed_detection_status publish
    max_output_bbox:    100
    mjpeg_port:         8080
    camera_index:       0
    camera_width:       1280
    camera_height:      720
```

> **Resolution note:** `input_h` / `input_w` are baked into the TensorRT engine at serialize time.
> If you change them here, delete the old `.engine` and re-run Step 3.

### Inference speed tuning

| Change | Expected gain |
|--------|---------------|
| `input_h/w: 416` (from 640) | ~34% faster |
| `input_h/w: 320` (from 640) | ~61% faster |
| `precision: "int8"` | ~2× faster vs FP16 (needs calibration data) |
| `cuda_post_process: "g"` | GPU NMS, lower CPU load |
| `max_output_bbox: 100` | Less postprocess memory vs default 1000 |

---

## Step 5 — Build and source the workspace

Inside the container (or on the Jetson directly):

```bash
cd /ros2_ws
colcon build --packages-select yolov8_trt_bed_detect --symlink-install
source install/setup.bash
```

> Run `source install/setup.bash` in every new terminal before using `ros2 launch` or `ros2 run`.
> Add it to `~/.bashrc` to avoid repeating it:
> ```bash
> echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc
> ```

---

## Step 6 — Run the detection node


**Recommended — launch file (loads config automatically):**

```bash
ros2 launch yolov8_trt_bed_detect bed_detect.launch.py
```

Override a single parameter without editing the YAML:

```bash
# Use a completely different params file
ros2 launch yolov8_trt_bed_detect bed_detect.launch.py \
  params_file:=/path/to/int8_params.yaml

# Override just the engine path at runtime
ros2 launch yolov8_trt_bed_detect bed_detect.launch.py \
  params_file:=/ros2_ws/install/yolov8_trt_bed_detect/share/yolov8_trt_bed_detect/config/trt_params.yaml \
  --ros-args -p engine_name:=/ros2_ws/src/yolov8_trt_bed_detect/weights/yolov8n.engine
```

**Alternative — run directly with params file:**

```bash
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect \
  --ros-args --params-file config/trt_params.yaml
```

---

## Step 7 — Trigger detection from another node / terminal

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
  -v $PWD/yolov8_trt_bed_detect/config:/ros2_ws/install/yolov8_trt_bed_detect/share/yolov8_trt_bed_detect/config \
  meraquetech/race_nav:yolov8-trt-bed-detect-nano.v4 \
  bash -c "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  ros2 launch yolov8_trt_bed_detect bed_detect.launch.py \
  engine_name:=/ros2_ws/src/yolov8_trt_bed_detect/weights/yolov8s_bed.engine"
```

---

## Webcam index

Default is `/dev/video0` (index `0`). Change via the params file — no recompile needed:

```yaml
camera_index:  1   # /dev/video1
camera_width:  1280
camera_height: 720
```

---

## Notes

- Press `q` in the OpenCV window to stop the node
- The detection loop is **inactive until** the `bed_detection` service is called
- Engine serialization is device-specific — rebuild the `.engine` if switching hardware
