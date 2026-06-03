# data_logger

Captures and saves frames or video from a camera device.
Two C++ OpenCV tools built from a single CMakeLists.txt, with a REST API and interactive controller.

---

## Project Structure

```
data_logger/
├── CMakeLists.txt
├── main.cpp                  # interactive terminal controller
├── api_server.py             # Flask REST API
├── frame_logger/
│   └── frame_logger.cpp      # saves timestamped JPEGs
├── video_logger/
│   └── video_logger.cpp      # saves .mp4 video file
├── build/                    # generated after cmake build
│   ├── data_logger_ctrl
│   ├── frame_logger
│   └── video_logger
└── README.md
```

---

## Requirements

```bash
sudo apt install g++ libopencv-dev pkg-config cmake
pip install flask
```

---

## Build

```bash
cd vision/data_logger

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

## REST API

Start the API server:

```bash
python3 api_server.py --port 5000
```

### Endpoints

#### `GET /status`
Returns current logger state.
```json
{ "running": true, "type": "frame", "pid": 12345 }
```

#### `POST /start/frame`
Start the frame logger. Body (all optional):
```json
{
  "device": 0,
  "output": "logs/frames",
  "fps": 5.0,
  "max_frames": 100,
  "show": false
}
```

#### `POST /start/video`
Start the video logger. Body (all optional):
```json
{
  "device": 2,
  "output": "logs/videos",
  "fps": 30.0,
  "duration": 60,
  "show": false
}
```

#### `POST /stop`
Stop the running logger.

### Example with curl

```bash
# Start frame logger
curl -X POST http://localhost:5000/start/frame \
     -H "Content-Type: application/json" \
     -d '{"device": 0, "fps": 5}'

# Check status
curl http://localhost:5000/status

# Stop
curl -X POST http://localhost:5000/stop
```

---

## Interactive Controller

```bash
./build/data_logger_ctrl
```

```
=== Data Logger Control ===
  1 - Start frame logger
  2 - Start video logger
  3 - Stop logger
  q - Quit
```

---

## frame_logger

Captures individual frames and saves them as timestamped JPEGs.

**Run:**
```bash
./build/frame_logger --device 0 --fps 5 --output logs/frames --show
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | Camera index (`/dev/videoN`) |
| `--output` | `logs/frames` | Output directory |
| `--fps` | `1.0` | Capture rate (frames/sec) |
| `--max-frames` | unlimited | Stop after N frames |
| `--show` | off | Display live window (press `q` to quit) |

**Output:**
```
logs/frames/frame_20260603_121500_123456.jpg
```

---

## video_logger

Records a continuous video and saves it as a single `.mp4` file.

**Run:**
```bash
./build/video_logger --device 2 --fps 30 --duration 60 --output logs/videos --show
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | Camera index (`/dev/videoN`) |
| `--output` | `logs/videos` | Output directory |
| `--fps` | `20.0` | Recording frame rate (Pi camera stable at 20) |
| `--duration` | unlimited | Stop after N seconds |
| `--width` | `640` | Capture width |
| `--height` | `640` | Capture height |
| `--yolo` | off | Letterbox to square YOLO format (`--yolo 640`) |
| `--show` | off | Display live window (press `q` to quit) |

**Output:**
```
logs/videos/video_20260603_121500.mp4
```

---

## YOLO Frame Size Guide — Agriculture / Pineapple Detection

### Recommended: 640×640

Best balance of accuracy and speed for pineapple crown detection. Default for YOLOv8/v5.

```bash
# Collect video dataset
./build/video_logger --device 2 --fps 20 --yolo 640 --duration 60

# Collect image dataset for labeling
./build/frame_logger --device 2 --fps 2 --yolo 640 --output dataset/raw
```

### Size comparison

| Size | Speed | Accuracy | Use case |
|------|-------|----------|----------|
| `320×320` | Very fast | Low | Edge device, far-away counting only |
| `416×416` | Fast | Medium | Jetson Nano, RPi with limited RAM |
| **`640×640`** | **Balanced** | **Good** | **Best for pineapple detection** |
| `1280×1280` | Slow | High | Drone top-down, dense small crowns |

### Why 640 for pineapples

- Pineapple crowns are **medium-size objects** — 640 captures enough detail
- Field lighting varies — resolution beyond 640 doesn't compensate for bad data
- Shooting **top-down from a drone**: use `1280` (crowns become small objects)
- Shooting **side view from ground**: `640` is sufficient

### Letterbox explained

The `--yolo` flag scales the frame to fit inside the target square and pads the remainder with grey (value 114), matching YOLO's internal preprocessing exactly:

```
Camera 640×480             YOLO 640×640
┌──────────────────┐       ┌──────────────────┐
│                  │  -->  │░░░░ padding ░░░░░│
│   camera frame   │       │   camera frame   │
│                  │       │░░░░ padding ░░░░░│
└──────────────────┘       └──────────────────┘
```

### Training with YOLOv8

```bash
# Label frames with Roboflow or LabelImg, then:
pip install ultralytics
yolo train data=pineapple.yaml model=yolov8n.pt imgsz=640
```
