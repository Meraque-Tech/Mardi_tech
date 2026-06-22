#!/bin/bash

IMAGE_TAG="meraquetech/race_nav:yolov8-trt-bed-detect-nano.v6"
DOCKERFILE="Dockerfile.yolov8_trt_bed_detect_jetson_nano"
CONTEXT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building Docker image: $IMAGE_TAG"
echo "Dockerfile : $DOCKERFILE"
echo "Context    : $CONTEXT_DIR"
echo ""

docker build \
    -f "$CONTEXT_DIR/$DOCKERFILE" \
    -t "$IMAGE_TAG" \
    "$CONTEXT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful: $IMAGE_TAG"
else
    echo ""
    echo "Build failed." >&2
    exit 1
fi


docker rmi -f $(docker images -f "dangling=true" -q) 2>/dev/null || true
docker rm -f $(docker ps -aq) 2>/dev/null || true

docker push "$IMAGE_TAG"