#!/bin/bash
set -e

REPO="meraquetech/race_nav"
TAG="bionic-humble-pytorch-l4t-r32.7.1-zed"
DOCKERFILE="Dockerfile.zed"

# ZED SDK version
ZED_SDK_MAJOR=4
ZED_SDK_MINOR=0
ZED_SDK_PATCH=7

# L4T version
L4T_MAJOR=32
L4T_MINOR=7
L4T_PATCH=1

cd "$(dirname "$0")"

echo "==> Building ${REPO}:${TAG}"

docker build \
  -f "${DOCKERFILE}" \
  --build-arg ZED_SDK_MAJOR=${ZED_SDK_MAJOR} \
  --build-arg ZED_SDK_MINOR=${ZED_SDK_MINOR} \
  --build-arg ZED_SDK_PATCH=${ZED_SDK_PATCH} \
  --build-arg L4T_MAJOR_VERSION=${L4T_MAJOR} \
  --build-arg L4T_MINOR_VERSION=${L4T_MINOR} \
  --build-arg L4T_PATCH_VERSION=${L4T_PATCH} \
  -t "${REPO}:${TAG}" \
  .

echo "==> Build complete: ${REPO}:${TAG}"

read -rp "Push to Docker Hub? [y/N] " PUSH
if [[ "${PUSH}" =~ ^[Yy]$ ]]; then
  docker push "${REPO}:${TAG}"
  echo "==> Pushed ${REPO}:${TAG}"
fi
