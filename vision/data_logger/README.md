# data_logger

Captures and saves frames or video from a camera device.
Two C++ OpenCV tools built from a single CMakeLists.txt.

---

## Project Structure

```
data_logger/
├── CMakeLists.txt
├── frame_logger/
│   └── frame_logger.cpp      # saves timestamped JPEGs
├── video_logger/
│   └── video_logger.cpp      # saves .avi video file
├── build/                    # generated after cmake build
│   ├── frame_logger
│   └── video_logger
└── README.md
```

---

## Requirements

```bash
sudo apt install g++ libopencv-dev pkg-config cmake
```

---

## Build

```bash
cd vision/data_logger

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Binaries will be at `build/frame_logger` and `build/video_logger`.

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

Records a continuous video and saves it as a single `.avi` file.

**Run:**
```bash
./build/video_logger --device 0 --fps 30 --duration 10 --output logs/videos --show
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | Camera index (`/dev/videoN`) |
| `--output` | `logs/videos` | Output directory |
| `--fps` | `30.0` | Recording frame rate |
| `--duration` | unlimited | Stop after N seconds |
| `--show` | off | Display live window (press `q` to quit) |

**Output:**
```
logs/videos/video_20260603_121500.avi
```
