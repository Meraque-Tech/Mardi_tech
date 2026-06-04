FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Build tools + TensorRT 10.x + OpenCV + deps
RUN apt-get update && apt-get install -y \
        build-essential git pkg-config cmake ninja-build ccache wget \
        libopencv-dev libspdlog-dev libfmt-dev \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y \
        libnvinfer-dev \
        libnvinfer-headers-dev \
        libnvinfer-plugin-dev \
        libnvinfer-headers-plugin-dev \
        libnvonnxparsers-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# Build (no TensorRT_DIR needed — libnvinfer-dev installs to standard paths)
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_BENCHMARKS=OFF \
    && cmake --build build --parallel $(nproc)

# Usage:
#   docker run --rm --runtime nvidia -it \
#     -v /path/to/models:/app/models \
#     --device /dev/video0 \
#     <image> \
#     ./build/detect_object_video --model models/yolov8n.onnx --input 0
ENTRYPOINT ["/bin/bash"]
