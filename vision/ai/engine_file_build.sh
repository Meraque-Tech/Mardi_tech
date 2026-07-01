#!/bin/bash

# Usage: ./engine_file_build.sh [model_name] [model_type] [num_classes]
# Defaults: model_name=yolov8n, model_type=n, num_classes=80
# Example: ./engine_file_build.sh mardi_pineapple_yolo_v8 n 3

MODEL_NAME="${1:-yolov8n}"
MODEL_TYPE="${2:-n}"
NC="${3:-80}"

WTS_FILE="${MODEL_NAME}.wts"
ENGINE_FILE="${MODEL_NAME}.engine"

echo "Model:   ${MODEL_NAME}.pt"
echo "Wts:     ${WTS_FILE}"
echo "Engine:  ${ENGINE_FILE}"
echo "Type:    ${MODEL_TYPE}"
echo "Classes: ${NC}"
echo ""

# # Step 1: Download .pt weights if not present
# if [ ! -f "./yolov8/weights/${MODEL_NAME}.pt" ]; then
#     echo "==> Step 1: Downloading ${MODEL_NAME}.pt ..."
#     mkdir -p ./yolov8/weights
#     wget -P ./yolov8/weights \
#         "https://github.com/ultralytics/assets/releases/download/v8.3.0/${MODEL_NAME}.pt"
#     if [ $? -ne 0 ]; then
#         echo "ERROR: Download failed."
#         exit 1
#     fi
# else
#     echo "==> Step 1: ${MODEL_NAME}.pt already exists, skipping download."
# fi
# echo ""

# Step 2: Generate .wts from .pt
if [ -f "./yolov8/weights/${WTS_FILE}" ]; then
    echo "==> Step 2: ${WTS_FILE} already exists, skipping generation."
else
    echo "==> Step 2: Generating ${WTS_FILE} from ${MODEL_NAME}.pt ..."
    docker run --rm --net=host \
        --runtime nvidia \
        --gpus all \
        --privileged \
        -v ./yolov8/weights:/workspace/yolov8/build/weights \
        -v ./yolov8:/yolov8 \
        meraquetech/tensorrt-yolov8:ultralytics \
        bash -c "cd /yolov8 && python3 gen_wts.py -w /workspace/yolov8/build/weights/${MODEL_NAME}.pt -o /workspace/yolov8/build/weights/${WTS_FILE} -t detect"

    if [ $? -ne 0 ]; then
        echo "ERROR: .wts generation failed."
        exit 1
    fi
    echo "==> ${WTS_FILE} generated successfully."
fi
echo ""

# Step 3: Patch kNumClass, recompile, serialize .wts → .engine
echo "==> Step 3: Building engine with kNumClass=${NC} ..."
docker run --rm --net=host \
    --runtime nvidia \
    --gpus all \
    --privileged \
    -v "$(pwd)/yolov8/images:/workspace/yolov8/build/images:ro" \
    -v "$(pwd)/yolov8/weights:/workspace/yolov8/build/weights" \
    -v "$(pwd)/yolov8/weights:/output" \
    meraquetech/race_nav:yolov8-trt-x86 \
    bash -c "
        set -e
        echo '--- Patching kNumClass to ${NC} ---'
        sed -i 's/const static int kNumClass = [0-9]\\+/const static int kNumClass = ${NC}/' \
            /workspace/yolov8/include/config.h
        grep 'kNumClass' /workspace/yolov8/include/config.h

        echo '--- Recompiling yolov8_det ---'
        cd /workspace/yolov8/build
        make -j\$(nproc) yolov8_det

        echo '--- Serializing engine ---'
        ./yolov8_det -s ./weights/${WTS_FILE} ${ENGINE_FILE} ${MODEL_TYPE}
        cp ${ENGINE_FILE} /output/
        echo '--- Done ---'
    "

if [ $? -ne 0 ]; then
    echo "ERROR: Engine serialization failed."
    exit 1
fi
echo "==> ${ENGINE_FILE} built and copied to ./yolov8/weights/"
