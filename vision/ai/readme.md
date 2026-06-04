
# Nvidia container sertup
```
    1. Install NVIDIA Container Toolkit

        If not already installed:

        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

        curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -

        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

        sudo apt update
        sudo apt install -y nvidia-container-toolkit
    

    2. Configure Docker to Use NVIDIA Runtime

    Edit Docker daemon config:

    sudo nano /etc/docker/daemon.json

    Add or modify like this:

    {
    # "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
        }
    }
    }

    👉 This is what people mean by “docker nvidia runtime true” — setting NVIDIA as default runtime.

    ✅ 3. Restart Docker
    sudo systemctl restart docker

    ✅ 4. Test GPU Access
    
    # Confirm CUDA toolchain is present
    docker run --rm --runtime nvidia \
      meraquetech/race_nav:r36.4.0 \
      /usr/local/cuda/bin/nvcc --version

    # Check CUDA libs are mounted in
    docker run --rm --runtime nvidia \
      meraquetech/race_nav:r36.4.0 \
      ldconfig -p | grep libcuda

    
    
    <!-- docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi -->



```

# NVIDIA L4T PyTorch
```
    https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch?version=r35.2.1-pth2.0-py3

    docker pull dustynv/ros:humble-pytorch-l4t-r32.7.1

    docker tag dustynv/ros:humble-pytorch-l4t-r32.7.1 meraquetech/race_nav:humble-pytorch-l4t-r32.7.1

    docker push meraquetech/race_nav:humble-pytorch-l4t-r32.7.1

    docker run -it --rm --runtime nvidia --network host -v /home/user/project:/location/in/container meraquetech/race_nav:humble-pytorch-l4t-r32.7.1



    # noetic-pytorch-l4t-r32.7.1

    docker tag dustynv/ros:noetic-pytorch-l4t-r32.7.1 meraquetech/race_nav:noetic-pytorch-l4t-r32.7.1

    docker run -it --rm --runtime nvidia --network host -v /home/user/project:/location/in/container meraquetech/race_nav:noetic-pytorch-l4t-r32.7.1

    docker push meraquetech/race_nav:noetic-pytorch-l4t-r32.7.1
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