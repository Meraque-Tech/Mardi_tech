#!/usr/bin/env bash
set -Eeuo pipefail

log() {
    printf '\n\033[1;34m==> %s\033[0m\n' "$1"
}

error() {
    printf '\n\033[1;31mERROR: %s\033[0m\n' "$1" >&2
    exit 1
}

if [[ "${EUID}" -eq 0 ]]; then
    error "Run this script as your normal user, not as root. It will use sudo when required."
fi

log "Checking Jetson platform"

ARCH="$(uname -m)"
[[ "${ARCH}" == "aarch64" ]] || error "Expected aarch64, but detected ${ARCH}."

if [[ ! -f /etc/nv_tegra_release ]]; then
    error "/etc/nv_tegra_release was not found. This does not appear to be NVIDIA Jetson Linux."
fi

cat /etc/nv_tegra_release

log "Updating APT package indexes"
sudo apt-get update

log "Installing required packages"
sudo apt-get install -y \
    ca-certificates \
    curl \
    nvidia-container \
    nano

log "Installing Docker Engine"
TMP_SCRIPT="$(mktemp)"
trap 'rm -f "${TMP_SCRIPT}"' EXIT

curl -fsSL https://get.docker.com -o "${TMP_SCRIPT}"
sudo sh "${TMP_SCRIPT}"

log "Configuring NVIDIA Container Runtime for Docker"

if ! command -v nvidia-ctk >/dev/null 2>&1; then
    error "nvidia-ctk was not installed. Check the NVIDIA JetPack repositories and rerun the script."
fi

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl daemon-reload
sudo systemctl enable --now docker
sudo systemctl restart docker

log "Adding ${USER} to the docker group"
sudo usermod -aG docker "${USER}"

log "Checking Docker and NVIDIA runtime"
sudo docker --version
nvidia-ctk --version
sudo docker info | grep -i runtime || true

log "Testing standard Docker"
sudo docker run --rm hello-world




dpkg-query -W nvidia-jetpack
apt-cache policy nvidia-jetpack
dpkg-query -W -f='${Version}\n' nvidia-jetpack


cat /etc/nv_tegra_release
ldconfig -p | grep libcuda
/usr/local/cuda/bin/nvcc --version
ls /dev/nvhost-ctrl /dev/nvmap
dpkg -l | grep -i tensorrt


sudo docker info | grep -i runtime


docker run --rm -it \
    --runtime=nvidia \
    nvcr.io/nvidia/l4t-jetpack:r36.4.0 bash


log "Testing Jetson NVIDIA device access"

docker run --rm \
    --runtime=nvidia \
    nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
    bash -c '
        echo "Jetson release:"
        echo
        echo "NVIDIA device nodes:"
        ls -l /dev/nvhost-gpu /dev/nvhost* /dev/nvidia* 2>/dev/null || true
        /usr/src/tensorrt/bin/trtexec --help | head
        cat /etc/nv_tegra_release
        ldconfig -p | grep libcuda
        /usr/local/cuda/bin/nvcc --version
        ls /dev/nvhost-ctrl /dev/nvmap
        dpkg -l | grep -i tensorrt
        lsb_release -a
    '

cat <<'EOF'


docker run --rm -it \
    --runtime=nvidia \
    dustynv/ros:humble-desktop-l4t-r36.4.0 bash


docker run --rm \
    --runtime=nvidia \
    dustynv/ros:humble-desktop-l4t-r36.4.0 \
    bash -c '
        echo "Jetson release:"
        echo
        echo "NVIDIA device nodes:"
        ls -l /dev/nvhost-gpu /dev/nvhost* /dev/nvidia* 2>/dev/null || true
        /usr/src/tensorrt/bin/trtexec --help | head
        cat /etc/nv_tegra_release
        ldconfig -p | grep libcuda
        /usr/local/cuda/bin/nvcc --version
        ls /dev/nvhost-ctrl /dev/nvmap
        dpkg -l | grep -i tensorrt
    '






Installation completed.

To use Docker without sudo, either:

  1. Log out and log back in, or
  2. Run: newgrp docker

Then verify with:

  docker ps
  docker run --rm hello-world

EOF
