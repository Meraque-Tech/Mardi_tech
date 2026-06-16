
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

docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -it --rm  meraquetech/race_nav:yolov8-trt-nano.v1

check ->

ldconfig -p | grep libcuda
/usr/local/cuda/bin/nvcc --version
ls /dev/nvhost-ctrl /dev/nvmap
dpkg -l | grep -i tensorrt


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

    docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -it --rm  meraquetech/race_nav:yolov8-trt-nano.v1
    docker push meraquetech/race_nav:yolov8-trt-nano.v1

   

```

# Model building
```
  docker run -it --rm --net=host \
        --runtime nvidia \
        --privileged \
        --gpus all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e XAUTHORITY=/root/.Xauthority \
        -v $HOME/.Xauthority:/root/.Xauthority:ro \
        -v $PWD/yolov8/images:/workspace/yolov8/build/images:ro \
        -v $PWD/yolov8/weights:/workspace/yolov8/build/weights:ro \
        -v $PWD/yolov8/weights:/output \
        meraquetech/race_nav:yolov8-trt-nano.v1

  # Test Nvidia ->

  ldconfig -p | grep libcuda
  /usr/local/cuda/bin/nvcc --version
  ls /dev/nvhost-ctrl /dev/nvmap
  dpkg -l | grep -i tensorrt

  # if output okay then all okay

  Serialize ->
  cd build
  ./yolov8_det -s ./weights/yolov8n.wts yolov8n.engine n
  cp yolov8n.engine /output/


  Serialize Engine Using Docker container ->
  # build in once -->
  docker run -it --rm --net=host \
        --runtime nvidia \
        --privileged \
        --gpus all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e XAUTHORITY=/root/.Xauthority \
        -v $HOME/.Xauthority:/root/.Xauthority:ro \
        -v $PWD/yolov8/images:/workspace/yolov8/build/images:ro \
        -v $PWD/yolov8/weights:/workspace/yolov8/build/weights:ro \
        -v $PWD/yolov8/weights:/output \
        meraquetech/race_nav:yolov8-trt-nano.v1 \
        bash -c "cd /workspace/yolov8/build && ./yolov8_det -s ./weights/yolov8n.wts yolov8n.engine n && cp yolov8n.engine /output/"
  

  DeSerialize ->
    ./yolov8_det -d /output/yolov8n.engine ./images g

```

# yolov8_trt_bed_detect (ROS 2 Node)

Runs YOLOv8 TensorRT bed detection with a USB webcam, publishing results over ROS 2 topics.

## Build image

From `vision/ai/` (with `yolov8_trt_bed_detect/` folder present):

```bash
docker build \
  -f Dockerfile.yolov8_trt_bed_detect_jetson_nano \
  -t meraquetech/race_nav:yolov8-trt-bed-detect-nano.v1 \
  .
```

## Run

```bash
docker run -it --rm --net=host \
  --runtime nvidia \
  --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --name=yolov8_trt_bed_detect \
  --device /dev/video0:/dev/video0 \
  -v $PWD/yolov8/weights:/ros2_ws/src/yolov8_trt_bed_detect/weights:ro \
  -v $PWD/yolov8/weights:/output \
  meraquetech/race_nav:yolov8-trt-bed-detect-nano.v1
```

Then inside the container:

```bash
# Serialize engine (once per model)
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect -s /ros2_ws/src/yolov8_trt_bed_detect/weights/yolov8n.wts /output/yolov8n.engine n

# Run detection node
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect \
  -d /output/yolov8n.engine ./ g -conf 0.2

# Trigger detection from another terminal
ros2 service call /bed_detection std_srvs/srv/Trigger {}
```

---

# Live Camera Stream with Detection (yolov8_stream)

Streams camera input with YOLOv8 OBB detections as **MJPEG over HTTP** — viewable in any browser, no plugins needed.

## Build

```bash
cd yolov8/build
cmake .. && make -j$(nproc) yolov8_stream
```

## Run

```bash
cd build
# Camera 0, GPU postprocess, default port 8080
./yolov8_stream -d weights/yolov8n.engine 0 g

# Camera 1, custom port 9090
./yolov8_stream -d weights/yolov8n.engine 1 g 9090

# Camera 2 or 3
./yolov8_stream -d weights/yolov8n.engine 2 g
./yolov8_stream -d weights/yolov8n.engine 3 g

# With horizontal flip (mirror correction)
./yolov8_stream -d weights/yolov8n.engine 0 g 8080 1
```

**Arguments:**
| Arg | Description |
|-----|-------------|
| `-d` | Run inference mode |
| `<engine>` | Path to serialized `.engine` file |
| `<cam 0-3>` | Camera device index (`/dev/video0` – `/dev/video3`) |
| `<c\|g>` | Postprocess on CPU (`c`) or GPU (`g`) |
| `[port]` | HTTP port for MJPEG stream (default: `8080`) |
| `[flip]` | `1`=horizontal, `0`=vertical, `2`=both (omit = no flip) |
| `[input_size]` | Model input resolution e.g. `640` or `1280` (default: `640`) |

## View Stream

Open in browser:
```
http://<host-ip>:8080/
```

Or with VLC / ffplay:
```bash
vlc http://localhost:8080/
ffplay http://localhost:8080/
```

## Run in Docker

### USB Camera (`/dev/video0`)

```bash
docker run -it --rm --net=host \
      --runtime nvidia \
      --privileged \
      -e NVIDIA_VISIBLE_DEVICES=all \
      --device /dev/video0:/dev/video0 \
      -v $PWD/yolov8/weights:/workspace/yolov8/build/weights:ro \
      meraquetech/race_nav:yolov8-trt-nano.v1 \
      bash -c "cd /workspace/yolov8/build && ./yolov8_stream -d ./weights/yolov8n.engine 0 g 8080 0"
```

> Add `--device /dev/video1:/dev/video1` etc. for cameras 1–3.

### CSI Camera (Jetson — requires nvargus-daemon)

CSI cameras on Jetson go through `nvargus-daemon` on the **host**, not through `/dev/video*`.
Start the daemon on the host first, then pass the socket into the container:

```bash
# On the host (once per boot)
sudo systemctl start nvargus-daemon

# Run container with argus socket
docker run -it --rm --net=host \
      --runtime nvidia \
      --privileged \
      -e NVIDIA_VISIBLE_DEVICES=all \
      --name=yolov8_trt \
      -v /tmp/argus_socket:/tmp/argus_socket \
      -v $PWD/yolov8/weights:/workspace/yolov8/build/weights:ro \
      meraquetech/race_nav:yolov8-trt-nano.v1 \
      bash -c "cd /workspace/yolov8/build && ./yolov8_stream -d ./weights/yolov8n.engine 0 g 8080 0 640"

docker rm -f yolov8_trt
```

> If `nvargus-daemon` is not installed: `sudo apt install nvidia-l4t-jetson-multimedia-api`