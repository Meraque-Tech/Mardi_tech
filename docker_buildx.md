# Docker Buildx — Multi-Architecture Builds

Guide for building Docker images targeting `x86 (amd64)`, `arm64`, and `aarch64` from an x86 host.

---

## Overview

The standard approach is **Docker Buildx** with `--platform` flags, publishing a multi-arch manifest to Docker Hub.

> `arm64` and `aarch64` are the same architecture. Docker uses `linux/arm64` to cover both.

---

## 1. One-Time Setup

### Enable QEMU (cross-compilation emulator)

```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```

**What this does:** Registers QEMU binary format handlers in the Linux kernel via `binfmt_misc`. When Docker encounters an `arm64` binary during build, the kernel transparently routes it through `qemu-aarch64-static` — no extra effort needed from Docker.

> **Note:** This registration does not survive a reboot. For persistence:
> ```bash
> sudo apt install qemu-user-static
> ```

### Verify QEMU is registered

```bash
ls /proc/sys/fs/binfmt_misc/ | grep qemu
# Expected: qemu-aarch64, qemu-arm, qemu-riscv64, etc.

cat /proc/sys/fs/binfmt_misc/qemu-aarch64
# Must show: enabled 1
# If not: echo 1 > /proc/sys/fs/binfmt_misc/qemu-aarch64
```

### Create a Buildx builder

```bash
docker buildx create --name multiarch --driver docker-container --use
docker buildx inspect --bootstrap

# Verify
docker buildx ls
```

---

## 2. Building Multi-Arch Images

### Build and push (amd64 + arm64)

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t meraquetech/race_nav:<tag> \
  --push \
  .
```

### Local load (one platform at a time)

```bash
docker buildx build \
  --platform linux/arm64 \
  -t meraquetech/race_nav:<tag> \
  --load \
  .
```

### Verify the manifest

```bash
docker buildx imagetools inspect meraquetech/race_nav:<tag>
# Expected output shows both linux/amd64 and linux/arm64 digests
```

---

## 3. Building This Repo's Image Chain

Images must be built **bottom-up** since each depends on the previous.

### Step 1 — Base ROS2 Humble image

`server/Dockerfile.ros2.humble.base` uses `ubuntu:22.04` as root — fully compatible with arm64.

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f server/Dockerfile.ros2.humble.base \
  -t meraquetech/race_nav:humble-ros-core.x \
  --push \
  .


docker buildx imagetools inspect meraquetech/race_nav:humble-ros-core.x

Output mustbe have this -> 
  Platform:    linux/amd64
  Platform:    linux/arm64

```

### Other builder
```
  docker run --privileged --rm tonistiigi/binfmt --install all

  docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f server/Dockerfile.amr_task_manager.dep \
  -t meraquetech/race_nav:amr_task_manager.dep.x \
  --push \
  .

  docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f agv_ws/src/Utilities/Dockerfile.utilities.dep \
  -t meraquetech/race_nav:humble-ros-core.utilities.dep.x \
  --push \
  .



  cd ~/robotics/Meraque/race_nav/agv_ws/src/Utilities

  docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.msg_srv_acs \
  -t meraquetech/race_nav:msg_srv_acs.x \
  --push \
  .


  docker buildx imagetools inspect meraquetech/race_nav:msg_srv_acs.x

  # docker build -f agv_ws/src/Control/Dockerfile.control.dep -t meraquetech/race_nav:humble.control.dep.x .

  docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f agv_ws/src/Control/Dockerfile.control.dep \
  -t meraquetech/race_nav:humble.control.dep.x \
  --push \
  .


  docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f agv_ws/src/Control/Dockerfile.vcu \
  -t meraquetech/race_nav:agv_control_vcu.x \
  --push \
  .


  docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f agv_ws/src/Control/Dockerfile.vcu \
  -t meraquetech/race_nav:agv_control_vcu.sprayer_360_node.x \
  --push \
  .


  docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f agv_ws/src/Perception/Lidar/Dockerfile.lidar.dep \
  -t meraquetech/race_nav:humble.lidar.dep.x \
  --push \
  .

  docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f agv_ws/src/Localization/IMU/Dockerfile.imu.dep \
  -t meraquetech/race_nav:humble.imu.dep.x \
  --push \
  .

  
  
  

  



```

#### Docker compose build buildx
```
  ## Always need to run it before docker buildx
  docker run --privileged --rm tonistiigi/binfmt --install all

  docker compose build --push --platform linux/amd64,linux/arm64 <service_name>
  For example:

  docker buildx bake \
  --set "*.platform=linux/amd64,linux/arm64" \
  --push \
  agv_control_vcu

  docker buildx bake \
  --set "*.platform=linux/amd64,linux/arm64" \
  --push \
  sprayer_360_node
  

  docker buildx bake \
  --set "*.platform=linux/amd64,linux/arm64" \
  --push \
  agv_server_js \
  direct_graphserver

  docker buildx bake \
  --set "*.platform=linux/amd64,linux/arm64" \
  --push \
  rs_lidar


  docker buildx bake \
  --set "*.platform=linux/amd64,linux/arm64" \
  --push \
  prime_description \
  sprayer_360_node \
  agv_control_vcu \
  rs_lidar

  

  


```

### Step 2 — Control dependencies image

`agv_ws/src/Control/Dockerfile.control.dep` uses only ROS2 packages and `python-can` — no GPU dependencies, works on all platforms.

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f agv_ws/src/Control/Dockerfile.control.dep \
  -t meraquetech/race_nav:humble.control.dep \
  --push \
  .
```

---

## 4. Nvidia Jetson Support

### The key distinction

Jetson is `linux/arm64` architecture, but has two cases:

| Use case | Base image needed |
|----------|-------------------|
| CPU-only workloads | `ubuntu:22.04 arm64` — works fine on Jetson |
| GPU workloads (CUDA/TensorRT) | Nvidia L4T base image — required for GPU access |

A plain `linux/arm64` image **runs** on Jetson but has **no GPU access**. GPU access on Jetson requires Nvidia's L4T kernel + drivers baked into the base image.

### Strategy for this repo

| Container | GPU? | Platform flag | Notes |
|-----------|------|---------------|-------|
| `agv_control_core` | No | `linux/amd64,linux/arm64` | Works on Jetson |
| `gnss_factor_graph` | No | `linux/amd64,linux/arm64` | Works on Jetson |
| `dlio_odometry`, `fast_lio` | No | `linux/amd64,linux/arm64` | Works on Jetson |
| `agv_server_js` | No | `linux/amd64,linux/arm64` | Works on Jetson |
| `navigation`, `mapping` | No | `linux/amd64,linux/arm64` | Works on Jetson |
| `yolov8_seg_ros2` | **Yes** | Jetson only | Needs L4T base |
| `zed_visual_slam` | **Yes** | Jetson only | Needs L4T + ZED SDK |

### GPU containers — Jetson Dockerfile

Check your JetPack version first:

```bash
# Run on the Jetson
cat /etc/nv_tegra_release
# or
dpkg -l | grep jetpack
# JetPack 6.x → L4T r36.x
# JetPack 5.x → L4T r35.x
```

Use the `dustynv` community image as base (pre-built ROS2 Humble + L4T):

```dockerfile
# Dockerfile.*.jetson
FROM dustynv/ros:humble-ros-base-l4t-r36.4.0
# ... same RUN steps as the CPU Dockerfile
```

```bash
docker build \
  -f agv_ws/src/Perception/Vision/ai/yolov8_trt_ros2/Dockerfile.jetson \
  -t meraquetech/race_nav:yolov8.jetson \
  --push \
  .
```

> Build GPU containers **natively on the Jetson** for best results. QEMU emulation works but is very slow for heavy C++ builds.

### Nvidia Docker (nvidia-container-toolkit)

`nvidia-container-toolkit` is a **runtime** tool — it exposes the Jetson GPU to containers at runtime. It does not affect image builds.

```
Build time:  linux/arm64 image (no GPU libs needed for CPU containers)
Run time:    nvidia-container-toolkit exposes GPU → container sees GPU
```

> TensorRT is **statically linked** at build time, so YOLOv8/TRT containers still need the L4T base image regardless of the runtime toolkit.

---

## 5. Known QEMU Cross-Build Issues

### `--symlink-install` breaks arm64 builds

**Symptom:**
```
Package 'builtin_interfaces' exports the library
'builtin_interfaces__rosidl_generator_c' which couldn't be found
```

**Cause:** `--symlink-install` creates symlinks instead of copying files into the install space. Under QEMU emulation, symlink resolution across the virtual filesystem is unreliable — the linker fails to find `.so` files even though they exist on disk.

**Fix:** Remove `--symlink-install` from colcon builds in Dockerfiles used for multi-arch builds:

```dockerfile
# BAD — breaks under QEMU arm64
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --merge-install --symlink-install

# GOOD
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --merge-install
```

> `--symlink-install` is fine for native x86 builds and local development. Remove it only in Dockerfiles built with `--platform linux/arm64`.

### `ldconfig` segfaults under QEMU

**Symptom:**
```
qemu: uncaught target signal 11 (Segmentation fault) - core dumped
Segmentation fault  ldconfig
exit code: 139
```

**Cause:** `ldconfig` crashes under QEMU arm64 emulation — known QEMU bug.

**Fix:** Only call `ldconfig` when installing actual shared libraries (`.so` files). If the package installs only headers or CMake config files (e.g. `rs_driver`), remove `ldconfig` entirely — it's a no-op anyway.

```dockerfile
# BAD — crashes under QEMU if no .so files were installed
RUN make install && ldconfig

# GOOD — safe when install only has headers/cmake files
RUN make install
```

---

## 6. Faster Builds — Native ARM Builder Node

QEMU is slow for heavy C++ packages (DLIO, FAST-LIO, GTSAM). If you have a Jetson or ARM machine available, add it as a native builder node:

```bash
# On the ARM machine — install Docker, then register it
docker buildx create --name arm-node --platform linux/arm64 ssh://user@arm-host

# On x86 machine — append it to your multiarch builder
docker buildx create --name multiarch --driver docker-container --use
docker buildx create --append --name multiarch --platform linux/arm64 ssh://user@arm-host
docker buildx inspect multiarch --bootstrap
```

This builds `amd64` natively on x86 and `arm64` natively on the ARM node **simultaneously**.

---

## 6. Summary

```
CPU containers  →  --platform linux/amd64,linux/arm64   (covers x86, standard arm64, Jetson CPU)
GPU containers  →  Separate Dockerfile with L4T base     (Jetson GPU only)
```

```bash
# Full rebuild order for this repo (CPU containers)
docker buildx build --platform linux/amd64,linux/arm64 -f server/Dockerfile.ros2.humble.base        -t meraquetech/race_nav:humble-ros-core       --push .
docker buildx build --platform linux/amd64,linux/arm64 -f agv_ws/src/Control/Dockerfile.control.dep -t meraquetech/race_nav:humble.control.dep     --push .
```
