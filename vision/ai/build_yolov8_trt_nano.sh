#!/bin/bash
set -e

REPO="meraquetech/race_nav"
TAG="yolov8-trt8-jetson-nano.v1"
DOCKERFILE="Dockerfile.yolov8_trt_jetson_nano"

cd "$(dirname "$0")"

echo "==> Building ${REPO}:${TAG}"

docker build \
  -f "${DOCKERFILE}" \
  -t "${REPO}:${TAG}" \
  .

echo "==> Build complete: ${REPO}:${TAG}"

read -rp "Push to Docker Hub? [y/N] " PUSH
if [[ "${PUSH}" =~ ^[Yy]$ ]]; then
  docker push "${REPO}:${TAG}"
  echo "==> Pushed ${REPO}:${TAG}"
fi

docker rmi -f $(docker images -f "dangling=true" -q) 2>/dev/null || true
