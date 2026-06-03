# data_logger

Captures and saves frames from a camera device.
C++ OpenCV backend called from a Python wrapper.

---

## Requirements

- g++ with C++17
- OpenCV 4.x (system install)
- Python 3

```bash
sudo apt install g++ libopencv-dev pkg-config
```

---

## Build

```bash
cd vision/data_logger

g++ -std=c++17 frame_logger.cpp -o frame_logger \
    $(pkg-config --cflags --libs opencv4)
```

---

## Run

**Via Python:**
```bash
python3 data_logger.py --fps 5 --show
```

**Direct C++ binary:**
```bash
./frame_logger --device 0 --fps 10 --show
```

**Save 50 frames then stop:**
```bash
python3 data_logger.py --fps 10 --max-frames 50
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `/dev/video0` | Camera device path |
| `--output` | `logs/` | Output directory for JPEGs |
| `--fps` | `1.0` | Capture rate (frames/sec) |
| `--max-frames` | unlimited | Stop after N frames |
| `--show` | off | Open live display window (press `q` to quit) |

---

## Output

Frames saved as timestamped JPEGs:
```
logs/frame_20260603_121500_123456.jpg
```
