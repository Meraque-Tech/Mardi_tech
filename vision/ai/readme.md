
# NVIDIA Container Setup (Jetson)

> **Note:** On Jetson, use `--runtime nvidia` — **not** `--gpus all` (which is x86-only).
> GPU access in `docker-compose` is handled via env vars; no `runtime:` or `deploy` key needed.

## 1. Install NVIDIA Container Toolkit

Skip if already on JetPack — it is pre-installed. Otherwise:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

## 2. Set NVIDIA as the Default Docker Runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker

docker info | grep -i runtime
```

Or edit `/etc/docker/daemon.json` manually:

```json
{
  “default-runtime”: “nvidia”,
  “runtimes”: {
    “nvidia”: {
      “path”: “nvidia-container-runtime”,
      “runtimeArgs”: []
    }
  }
}
```

Then restart Docker:

```bash
sudo systemctl restart docker
```

## 3. Verify

```bash
# Confirm nvidia is the default runtime
docker info | grep -i runtime
# Expected: Default Runtime: nvidia

# Confirm GPU device nodes are visible inside container
docker run --rm --runtime nvidia \
  meraquetech/race_nav:r36.4.0 \
  ls /dev/nvhost-ctrl /dev/nvmap

# Confirm CUDA toolchain is present
docker run --rm --runtime nvidia \
  meraquetech/race_nav:r36.4.0 \
  /usr/local/cuda/bin/nvcc --version

# Confirm CUDA libs are mounted in
docker run --rm --runtime nvidia \
  meraquetech/race_nav:r36.4.0 \
  ldconfig -p | grep libcuda
```

> **Why not `--gpus all` on Jetson?**
> `--gpus all` requires the NVIDIA Container Toolkit's CDI driver, which is not supported on L4T/Jetson.
> The `--runtime nvidia` flag (or setting it as default) is the correct Jetson approach.

## 4. docker-compose GPU Access

With `nvidia` set as the default runtime, `docker-compose` services get GPU access automatically via:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=all
```

No `runtime:` key or `deploy.resources` block is needed — both are unsupported in older `docker-compose` versions on Jetson.

# NVIDIA L4T PyTorch
```
    https://github.com/dusty-nv/jetson-containers/tree/master


    https://github.com/dusty-nv/jetson-containers/tree/master/packages/physicalAI/ros

    https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch?version=r35.2.1-pth2.0-py3

    docker pull dustynv/ros:humble-pytorch-l4t-r32.7.1

    docker tag dustynv/ros:humble-pytorch-l4t-r32.7.1 meraquetech/race_nav:humble-pytorch-l4t-r32.7.1

    docker push meraquetech/race_nav:humble-pytorch-l4t-r32.7.1

    docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -it --rm  meraquetech/race_nav:yolov8-trt-nano.v1

    docker run -it --rm --runtime nvidia --network host -v /home/user/project:/location/in/container meraquetech/race_nav:humble-pytorch-l4t-r32.7.1



    # noetic-pytorch-l4t-r32.7.1

    docker tag dustynv/ros:noetic-pytorch-l4t-r32.7.1 meraquetech/race_nav:noetic-pytorch-l4t-r32.7.1

    docker run -it --rm --runtime nvidia --network host -v /home/user/project:/location/in/container meraquetech/race_nav:noetic-pytorch-l4t-r32.7.1

    docker push meraquetech/race_nav:noetic-pytorch-l4t-r32.7.1


    docker-compose -f docker-compose.yolov8-trt-jetson-nano.yml build

```

---

# YOLOv8 TensorRT — Jetson Nano (TRT 8, JetPack 4.6)

Source: https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano/tree/tensorrt8

Dockerfile: `Dockerfile.yolov8_trt_jetson_nano`
Base image: `meraquetech/race_nav:humble-pytorch-l4t-r32.7.1`

### Build image

```bash
cd vision/ai

docker build \
  -f Dockerfile.yolov8_trt_jetson_nano \
  -t meraquetech/race_nav:yolov8-trt8-jetson-nano \
  .
```

### Convert ONNX → TRT engine (run once on the Nano)

The engine must be built **on the target Jetson** — it is not portable across different hardware or TensorRT versions.

**Step 1 — place your `.onnx` file in the `models/` folder on the host:**

```bash
ls models/
# yolov8n.onnx  ← must exist before running the container
```

**Step 2 — start the container with the models volume mounted:**

```bash
docker run -it --rm --runtime nvidia \
  -v $(pwd)/models:/yolov8_ws/models \
  meraquetech/race_nav:yolov8-trt8-jetson-nano \
  bash
```

**Step 3 — inside the container, run trtexec to export:**

```bash
# FP16 (recommended — best speed/accuracy trade-off)
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/yolov8n.onnx \
  --saveEngine=models/yolov8n.engine \
  --fp16

# INT8 (smaller, faster, lower accuracy)
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/yolov8n.onnx \
  --saveEngine=models/yolov8n_int8.engine \
  --int8
```

This takes several minutes. The resulting `models/yolov8n.engine` is written back to the **host** via the volume mount.

**One-liner (no interactive shell):**

```bash
docker run --rm --runtime nvidia \
  -v $(pwd)/models:/yolov8_ws/models \
  meraquetech/race_nav:yolov8-trt8-jetson-nano \
  /usr/src/tensorrt/bin/trtexec \
    --onnx=models/yolov8n.onnx \
    --saveEngine=models/yolov8n.engine \
    --fp16
```

### Run detection on an image

```bash
docker run --rm --runtime nvidia \
  -v $(pwd)/models:/yolov8_ws/models \
  -v $(pwd)/images:/yolov8_ws/images \
  meraquetech/race_nav:yolov8-trt8-jetson-nano \
  ./YoloV8rt models/yolov8n.engine images/test.jpg
```

---

# YOLOv8 TensorRT — Bed Detection (ROS2 Humble, Jetson, OpenCV camera)

Dockerfile: `Dockerfile.yolo8_trt.bed_detect`
Base image: `meraquetech/race_nav:humble-pytorch-l4t-r32.7.1`

### Build image

```bash
cd vision/ai

docker build \
  -f Dockerfile.yolo8_trt.bed_detect \
  -t meraquetech/race_nav:humble-pytorch-l4t-r32.7.1-bed-detect-yolov8-trt \
  .
```

### Run

```bash
# camera 0 (default)
docker run --rm --runtime nvidia --network host \
  --device /dev/video0 \
  -v $(pwd)/models:/agv_ws/models \
  meraquetech/race_nav:humble-pytorch-l4t-r32.7.1-bed-detect-yolov8-trt \
  ros2 run yolov8_trt_bed_detect_opt yolov8_trt_bed_detect_opt \
    --ros-args \
    -p engine_name:=models/bed_detect.engine \
    -p camera_id:=0

# camera 1 or 2
docker run --rm --runtime nvidia --network host \
  --device /dev/video1 \
  -v $(pwd)/models:/agv_ws/models \
  meraquetech/race_nav:humble-pytorch-l4t-r32.7.1-bed-detect-yolov8-trt \
  ros2 run yolov8_trt_bed_detect_opt yolov8_trt_bed_detect_opt \
    --ros-args \
    -p engine_name:=models/bed_detect.engine \
    -p camera_id:=1 \
    -p conf_thresh:=0.5
```

---

# YOLOv8 TensorRT CPP — Jetson (TRT 10, JetPack 6)

Source: https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP

Dockerfile: `YOLOv8-TensorRT-CPP/Dockerfile.jetson`
Base image: `nvcr.io/nvidia/l4t-jetpack:r36.4.0` (JetPack 6.1, TRT 10.x)

> **Note:** Requires Jetson Orin / AGX Orin / Orin NX. Not compatible with Jetson Nano (JetPack 4.6).

> **Troubleshooting — `CUDA driver version is insufficient for CUDA runtime version`:**
> This error means the host JetPack version does not match what the container expects.
> Verify your host JetPack version:
> ```bash
> cat /etc/nv_tegra_release
> # R32 → JetPack 4.x (Jetson Nano, CUDA 10.2) — cannot run this image
> # R35 → JetPack 5.x (Orin, CUDA 11.4)
> # R36 → JetPack 6.x (Orin, CUDA 12.x) ← required for this image
> ```
> This image uses TensorRT 10.x / CUDA 12.x (JetPack 6.x packages). Running it on a Jetson Nano
> (R32, CUDA 10.2) will always fail — use the `yolov8-trt8-jetson-nano` image instead.
> Also confirm the container is started with `--runtime nvidia`; without it the CUDA stub driver
> is used and produces the same error even on a compatible host.

# Tesing

````
docker run -it --rm --runtime nvidia \
  -v $(pwd)/models:/yolov8_ws/models \
  nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime bash


docker run -it --rm --runtime nvidia \
  -v $(pwd)/models:/yolov8_ws/models \
  meraquetech/race_nav:humble-pytorch-l4t-r32.7.1 bash

docker run --rm -it --runtime nvidia \
    -v $(pwd)/models:/yolov8_ws/models \
    nvcr.io/nvidia/l4t-base:r32.7.1 \
    bash

docker run --rm -it --runtime nvidia \
    -v $(pwd)/models:/yolov8_ws/models \
    dustynv/ros:humble-pytorch-l4t-r32.7.1 \
    bash


docker run --rm -it --runtime nvidia \
    -v $(pwd)/models:/yolov8_ws/models \
    meraquetech/race_nav:bionic-humble-pytorch-l4t-r32.7.1-zed \
    bash




/usr/src/tensorrt/bin/trtexec \
    --onnx=models/yolov8n.onnx \
    --saveEngine=models/yolov8n.engine \
    --fp16
````


### Build image

```bash
cd vision/ai/YOLOv8-TensorRT-CPP

docker build \
  -f Dockerfile.jetson \
  -t yolov8-trt-cpp-jetson \
  .
```

### Run detection on camera

```bash
docker run --rm --runtime nvidia -it \
  --device /dev/video0 \
  -v $(pwd)/models:/app/models \
  yolov8-trt-cpp-jetson \
  ./build/detect_object_video --model models/yolov8n.onnx --input 0
```