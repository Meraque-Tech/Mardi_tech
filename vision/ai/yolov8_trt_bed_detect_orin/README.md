# yolov8_trt_bed_detect_orin

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
| Service | `bed_detection_stop` | `std_srvs/Trigger` | Stop detection loop |
| Service | `reset_tracker` | `std_srvs/Trigger` | Clear cumulative unique counts |
| Service | `set_tracking` | `std_srvs/SetBool` | Enable/disable unique tracking |
| Publisher | `conf` | `std_msgs/Float32` | Confidence score of detection |
| Publisher | `bed_detection_status` | `std_msgs/UInt8` | `1` = bed detected, `0` = not detected |
| Publisher | `detection_active` | `std_msgs/UInt8` | `1` = inference running, `0` = stopped |
| Publisher | `tracking_enabled` | `std_msgs/UInt8` | `1` = unique tracking, `0` = per-frame counting |
| Publisher | `class_counts` | `std_msgs/String` | Per-class object counts e.g. `class0:2 class1:1` |

---

### Object Counting & Tracking

Controlled by `is_track` in `trt_params.yaml`:

| `is_track` | Behaviour | Output example |
|---|---|---|
| `false` | Detections visible in **current frame** | `class0:2 class1:1` |
| `true` | **Cumulative unique** objects seen since start (MOSSE tracker) | `class0:5 class1:3` |

Monitor counts:
```bash
ros2 topic echo /class_counts
```

#### Tracker comparison on Jetson Nano

| Tracker | Speed | CPU load | RAM | Accuracy | Suitable for Nano |
|---|---|---|---|---|---|
| **MOSSE** (current) | ~1-3ms/obj | Very low | ~5MB | Decent | ✅ Best choice |
| KCF | ~5-15ms/obj | Low | ~10MB | Good | ✅ Yes |
| CSRT | ~25-50ms/obj | High | ~30MB | Best | ⚠️ Risky |
| DeepSORT | ~100ms+ | Very high | ~200MB | Excellent | ❌ No |

MOSSE uses an FFT-based correlation filter — designed for high-speed tracking on resource-constrained hardware. Beds are slow-moving and large, making MOSSE a perfect fit.

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
cd vision/ai/yolov8_trt_bed_detect_orin/
python3 gen_wts.py -w yolov8s_bed.pt
# output: yolov8s_bed.wts
```

---

## Step 2 — Build the Docker image

From `vision/ai/`:

```bash
docker build \
  -f Dockerfile.yolov8_trt_bed_detect_orin_jetson_nano \
  -t meraquetech/race_nav:yolov8-trt-bed-detect-nano.v6 \
  .
```

> The build context expects `yolov8_trt_bed_detect_orin/` to be present in `vision/ai/`.

---

## Step 3 — Build the TensorRT engine (inside container)

Run the container with the weights folder mounted:

```bash
docker run -it --rm --net=host \
  --runtime nvidia \
  --privileged \
  --name=yolov8-trt-bed-detect-orin-nano \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --device /dev/video0:/dev/video0 \
  -v $PWD/yolov8/weights:/weights \
  -v $PWD/yolov8_trt_bed_detect_orin/config:/ros2_ws/install/yolov8_trt_bed_detect_orin/share/yolov8_trt_bed_detect_orin/config \
  -v $PWD/yolov8_trt_bed_detect_orin/launch:/ros2_ws/install/yolov8_trt_bed_detect_orin/share/yolov8_trt_bed_detect_orin/launch \
  meraquetech/race_nav:yolov8-trt-bed-detect-orin-nano

  




```

Inside the container, serialize the engine:

```bash
# for nano ->
ros2 run yolov8_trt_bed_detect_orin yolov8_trt_bed_detect_orin \
  -s /weights/yolov8n.wts /weights/yolov8n_416.engine n 416 416

ros2 run yolov8_trt_bed_detect_orin yolov8_trt_bed_detect_orin \
  -s /weights/yolov8n.wts /weights/yolov8n.engine n 512 512

ros2 run yolov8_trt_bed_detect_orin yolov8_trt_bed_detect_orin \
  -s /weights/yolov8n.wts /weights/yolov8n_orien_v1.engine n 512 512

# for small ->
ros2 run yolov8_trt_bed_detect_orin yolov8_trt_bed_detect_orin \
  -s /weights/yolov8s.wts /weights/yolov8s.engine s

```

<!-- RUN -->
```
  ros2 launch yolov8_trt_bed_detect_orin bed_detect.launch.py
  docker exec -it yolov8-trt-bed-detect-nano bash
  ros2 service call /bed_detection std_srvs/srv/Trigger {}

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
colcon build --packages-select yolov8_trt_bed_detect_orin --symlink-install
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
ros2 launch yolov8_trt_bed_detect_orin bed_detect.launch.py
```

Override a single parameter without editing the YAML:

```bash
# Use a completely different params file
ros2 launch yolov8_trt_bed_detect_orin bed_detect.launch.py \
  params_file:=/path/to/int8_params.yaml

# Override just the engine path at runtime
ros2 launch yolov8_trt_bed_detect_orin bed_detect.launch.py \
  params_file:=/ros2_ws/install/yolov8_trt_bed_detect_orin/share/yolov8_trt_bed_detect_orin/config/trt_params.yaml \
  --ros-args -p engine_name:=/ros2_ws/src/yolov8_trt_bed_detect_orin/weights/yolov8n.engine
```

**Alternative — run directly with params file:**

```bash
ros2 run yolov8_trt_bed_detect_orin yolov8_trt_bed_detect_orin \
  --ros-args --params-file config/trt_params.yaml
```

---

## Step 7 — Trigger detection from another node / terminal

```bash
ros2 service call /bed_detection std_srvs/srv/Trigger {}

# Stop it again
ros2 service call /bed_detection_stop std_srvs/srv/Trigger {}
```

Then monitor output:

```bash
ros2 topic echo /bed_detection_status
ros2 topic echo /conf
ros2 topic echo /class_counts
```

Toggle tracking live without restarting the node:

```bash
# enable unique object tracking (cumulative counts)
ros2 param set /yolov8_trt is_track true

# disable (per-frame counts only)
ros2 param set /yolov8_trt is_track false
```

---

## Web Dashboard & REST API

After launch, the dashboard is available at:

```
http://<host-ip>:8090/          ← dashboard UI
http://<host-ip>:8080/          ← MJPEG live stream
```

Saved frames are written to `./vision/ai/saved_frames/` on the host (mounted into the container at `/saved_frames`).
Live counts update continuously in the dashboard. They are persisted when `POST /api/save_count` is called, **Save current count** is pressed, or automatic saving is enabled. Automatic saving stores a paired frame and count every 0.5 seconds only while detection runs and fresh ROS direction data is exactly `/gnss/is_forward=true` and `/gnss/is_backward=false`. All other direction states, including stale or missing data, pause collection without switching off the Auto-save toggle; collection resumes automatically when the forward-only condition returns. Saved history lives in `/saved_frames/count_history.db`, so it remains available after a page refresh or container restart. The dashboard table is paginated so every stored sample remains viewable without making the page progressively slower.

> Automatic saving can create up to 172,800 JPEG files per day. Keep it disabled when it is not required and monitor free disk space.

The dashboard is self-contained and does not require internet access or CDN scripts.

### Serialize / deserialize from the dashboard

- **Convert to .engine** (in the *Model weights* card, next to a `.wts` file) requires a **# classes** value entered in the field beside the button — the number of classes that `.wts` was trained on. The button updates `trt_params.yaml` (`wts_name`, `engine_name`, and `num_class`) and then calls `POST /api/serialize/model`, which runs `serialize_engine.launch.py` in the background. The number of classes is not guessed or defaulted; the request is rejected client-side and server-side (`400`) if it's missing or not a positive integer, since serializing with the wrong class count silently produces a broken engine.
- **Deserialize model** (in the *Controls* card) calls `POST /api/deserialize/model`, which runs `bed_detect.launch.py` in the background to (re)load the current `.engine` and start the detection node.
- **Start detection** / **Stop detection** call `POST /api/start` / `POST /api/stop` directly (the ROS `Trigger` services) — unchanged, and independent of the serialize/deserialize launch jobs.
- Live `ros2 launch` output for the current or most recent serialize/deserialize job streams into the **Serialize / detect log** panel underneath the model file lists, polled from `GET /api/models/launch/log`.

### REST API reference

Base URL: `http://<host-ip>:8090`

| Method | Endpoint | Request input | Successful response | Purpose |
|---|---|---|---|---|
| `GET` | `/api/status` | None | Live state object plus `mjpeg_port` | Read detection, object-result, confidence, counts, and tracking state |
| `GET` | `/api/counts` | None | `{"0":2,"1":1}` | Read the latest live per-class count |
| `POST` | `/api/start` | None | `{"success":true,"message":"bed detection started"}` | Start inference through ROS |
| `POST` | `/api/stop` | None | `{"success":true,"message":"bed detection stopped"}` | Stop inference through ROS |
| `POST` | `/api/set_track` | JSON: `{"enabled":true}` | `{"success":true,"message":"...","is_track":true}` | Select unique-object or per-frame counting |
| `POST` | `/api/reset_tracker` | None | `{"success":true,"message":"tracker reset requested"}` | Clear cumulative unique-object counts |
| `POST` | `/api/save_count` | None | Saved record ID, time, counts, and total | Save exactly one current live-count snapshot to SQLite |
| `POST` | `/api/auto_save` | JSON: `{"enabled":true}` | Mode, message, and `interval_seconds` | Enable or disable forward-only paired frame-and-count saving |
| `GET` | `/api/history` | Query: `limit` (1–5000), `offset` (≥0) | Paginated history object | Read manually saved counts, newest first |
| `POST` | `/api/save` | None | `{"success":true,"filename":"frame_....jpg"}` | Save the current MJPEG frame as JPEG |
| `GET` | `/api/images` | None | Array of saved-image metadata | List saved JPEG frames |
| `DELETE` | `/api/images/{filename}` | Filename in URL | `{"success":true}` | Delete one saved JPEG frame |
| `DELETE` | `/api/data` | None | Numbers of deleted records and images | Disable auto-save and permanently clear all count history and JPEG frames |
| `GET` | `/saved/{filename}` | Filename in URL | JPEG bytes | Display or download a saved frame |
| `POST` | `/api/models/build_engine` | JSON: `{"filename":"model.wts","num_class":80}` | `{"success":true,"message":"...","engine_filename":"...","num_class":80}` | Write `wts_name`, `engine_name`, and `num_class` into `trt_params.yaml` for the next serialize. `num_class` is required — `400` if missing or not a positive integer |
| `DELETE` | `/api/models/{kind}/{filename}` | `kind` (`pt`\|`wts`\|`engine`) and filename in URL | `{"success":true}` | Permanently delete one `.pt`, `.wts`, or `.engine` file from the weights directory |
| `POST` | `/api/serialize/model` | None | `{"success":true,"message":"serialize started"}` | Launch `serialize_engine.launch.py` in the background to build the `.engine` from the configured `.wts` (and `num_class`). Stops a running deserialize first. Skipped (`"skipped":true`) if the configured `.engine` file already exists |
| `POST` | `/api/deserialize/model` | None | `{"success":true,"message":"deserialize started"}` | Launch `bed_detect.launch.py` in the background to load the `.engine` and run detection. Stops a running serialize first |
| `GET` | `/api/models/launch/status` | None | `{"running":bool,"mode":"serialize"\|"deserialize"\|null,"message":str\|null,"ok":bool\|null}` | Poll the status of the most recent serialize/deserialize launch. `ok` is `null` while running or if the launch was stopped by the other mode |
| `GET` | `/api/models/launch/log` | None | `{"lines":["...","..."]}` | Tail of the most recent serialize/deserialize `ros2 launch` output (last 500 lines, oldest first) |

#### Live state object

The `/api/status` endpoint and WebSocket messages share these fields:

| Field | Type | Meaning |
|---|---|---|
| `type` | string | WebSocket event type: `snapshot` or `status`; omitted by `/api/status` |
| `counts` | object | Latest class-to-count map, for example `{"0":2,"1":1}` |
| `bed_status` | integer | Internal ROS-compatible field: `1` = object found, `0` = clear |
| `conf` | number | Highest confidence from the latest detection frame |
| `detecting` | boolean | Whether TensorRT inference is running |
| `is_track` | boolean | `true` = cumulative unique tracking; `false` = per-frame counting |
| `auto_save` | boolean | Whether paired frame-and-count automatic saving is enabled |
| `is_forward` | boolean/null | Latest forward flag, or `null` when direction data is missing or stale |
| `is_backward` | boolean/null | Latest backward flag, or `null` when direction data is missing or stale |
| `direction_valid` | boolean | Whether both direction flags were received within the stale timeout |
| `auto_save_direction_eligible` | boolean | Whether fresh direction data is exactly forward=`true`, backward=`false` |
| `auto_save_active` | boolean | Whether Auto-save is enabled, detection is running, and direction is eligible |
| `last_updated` | string/null | ISO-8601 time of the latest live count |
| `mjpeg_port` | integer | MJPEG port; present only in `/api/status` |

#### History response

| Field | Type | Meaning |
|---|---|---|
| `total` | integer | Total manually saved records in SQLite |
| `limit` | integer | Maximum records returned on this page |
| `offset` | integer | Number of newer records skipped |
| `items` | array | History records containing `id`, `time`, `counts`, `total`, `bed`, `conf`, `tracking`, and optional `frame` filename |

Common error responses use `{"success":false,"message":"..."}`. Expected status codes include `400` for invalid pagination or filenames, `409` when saving before any live count exists or when the *same* serialize/deserialize launch is already running (calling the *other* mode instead stops the running one and switches), `500` for frame/database failures, and `503` when a ROS service is unavailable.

> `DELETE /api/data` is destructive and cannot be undone. The SQLite schema is retained so new records can be saved immediately afterward.

### WebSocket reference

Connection URL: `ws://<host-ip>:8090/ws` (use `wss://` when the dashboard is served through HTTPS).

| Direction | Event/input | When sent | Payload |
|---|---|---|---|
| Server → client | `snapshot` | Immediately after connection and after every `/class_counts` update | Complete live state object with `type: "snapshot"` |
| Server → client | `status` | Detection or tracking state changes | Complete live state object with `type: "status"` |
| Client → server | `ping` | Dashboard sends every 20 seconds | Plain text `ping`; keeps the socket open |

Example server message:

```json
{
  "type": "snapshot",
  "counts": {"0": 2, "1": 1},
  "bed_status": 1,
  "conf": 0.9123,
  "detecting": true,
  "is_track": false,
  "auto_save": true,
  "last_updated": "2026-06-22T08:00:00+00:00"
}
```

### Example curl calls

```bash
# start detection
curl -X POST http://<host-ip>:8090/api/start

# stop detection
curl -X POST http://<host-ip>:8090/api/stop

# save a frame
curl -X POST http://<host-ip>:8090/api/save

# get current counts
curl http://<host-ip>:8090/api/counts

# read stored count history
curl 'http://<host-ip>:8090/api/history?limit=50&offset=0'

# save the current live count to history
curl -X POST http://<host-ip>:8090/api/save_count

# enable paired frame + count saving every 0.5 seconds while detecting
curl -X POST http://<host-ip>:8090/api/auto_save \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

# permanently delete all saved counts and JPEG frames
curl -X DELETE http://<host-ip>:8090/api/data

# enable tracking
curl -X POST http://<host-ip>:8090/api/set_track \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

# list saved frames
curl http://<host-ip>:8090/api/images

# point trt_params.yaml at a .wts and its class count before serializing
curl -X POST http://<host-ip>:8090/api/models/build_engine \
  -H "Content-Type: application/json" \
  -d '{"filename": "yolov8n.wts", "num_class": 80}'

# serialize the .wts into a TensorRT .engine (runs serialize_engine.launch.py)
curl -X POST http://<host-ip>:8090/api/serialize/model

# deserialize the .engine and start detection (runs bed_detect.launch.py)
curl -X POST http://<host-ip>:8090/api/deserialize/model

# poll the serialize/deserialize launch status
curl http://<host-ip>:8090/api/models/launch/status

# tail the ros2 launch output for the current/last serialize or deserialize job
curl http://<host-ip>:8090/api/models/launch/log

# permanently delete a weights file (.pt, .wts, or .engine)
curl -X DELETE http://<host-ip>:8090/api/models/wts/yolov8n.wts
```

---

## One-line Docker run (after engine is built)

```bash
docker run -it --rm --net=host \
  --runtime nvidia \
  --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --device /dev/video0:/dev/video0 \
  -v $PWD/yolov8_trt_bed_detect_orin/weights:/ros2_ws/src/yolov8_trt_bed_detect_orin/weights:ro \
  -v $PWD/yolov8_trt_bed_detect_orin/config:/ros2_ws/install/yolov8_trt_bed_detect_orin/share/yolov8_trt_bed_detect_orin/config \
  -v $PWD/yolov8_trt_bed_detect_orin/launch:/ros2_ws/install/yolov8_trt_bed_detect_orin/share/yolov8_trt_bed_detect_orin/launch \
  meraquetech/race_nav:yolov8-trt-bed-detect-nano.v6 \
  bash -c "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  ros2 launch yolov8_trt_bed_detect_orin bed_detect.launch.py \
  engine_name:=/ros2_ws/src/yolov8_trt_bed_detect_orin/weights/yolov8s_bed.engine"
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
