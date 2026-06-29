#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as a normal user, not root. It will use sudo when needed."
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required."
  exit 1
fi

. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
ARCH="$(dpkg --print-architecture)"
DISTRIBUTION="${ID}${VERSION_ID}"

if [[ -z "${CODENAME}" ]]; then
  echo "Could not detect Ubuntu codename from /etc/os-release."
  exit 1
fi

echo "==> Installing Docker Engine and Docker Compose V2"
sudo apt update
sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${CODENAME}
Components: stable
Architectures: ${ARCH}
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> Enabling Docker service"
sudo systemctl enable docker
sudo systemctl start docker

echo "==> Adding ${USER} to docker group"
if ! getent group docker >/dev/null; then
  sudo groupadd docker
fi
sudo usermod -aG docker "${USER}"

echo "==> Installing NVIDIA Container Toolkit"
sudo curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  -o /etc/apt/keyrings/nvidia-container-toolkit.asc
sudo chmod a+r /etc/apt/keyrings/nvidia-container-toolkit.asc

curl -fsSL "https://nvidia.github.io/libnvidia-container/${DISTRIBUTION}/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.asc] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt update
sudo apt install -y nvidia-container-toolkit

echo "==> Setting NVIDIA as Docker default runtime"
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker

echo "==> Docker versions"
docker --version || sudo docker --version
docker compose version || sudo docker compose version

echo "==> Docker runtime check"
sudo docker info | grep -i 'runtime'

echo "==> Optional Jetson performance commands"
echo "Run these when you want max performance:"
echo "  sudo nvpmodel -q"
echo "  sudo nvpmodel -m 0"
echo "  sudo jetson_clocks"
echo "  jtop"

echo
echo "Setup complete."
echo "Log out and log back in, or reboot, so docker group membership takes effect."
echo "Then verify without sudo:"
echo "  docker ps"
echo "  docker compose version"
