#!/bin/bash
# Build a YOLOv8 TensorRT engine from a .pt model.
#
# Usage:
#   ./engine_file_build.sh <model_name> <model_type> <num_classes> [width] [height] [--target x86|nano]
#
# Arguments:
#   model_name   Base name of the model (no extension).  Default: yolov8n
#   model_type   YOLOv8 variant letter: n / s / m / l / x.  Default: n
#   num_classes  Number of output classes in the custom model.  Default: 80
#   width        Input width  (optional, used only with --target nano).  Default: 640
#   height       Input height (optional, used only with --target nano).  Default: 640
#   --target     x86  → Docker x86 TRT image, make-based  (default)
#                nano → Jetson Nano Docker image, colcon + ros2 run
#
# Examples:
#   ./engine_file_build.sh yolov8n n 80
#   ./engine_file_build.sh mardi_pineapple_yolo_v8 n 3
#   ./engine_file_build.sh mardi_pineapple_yolo_v8 n 3 640 640 --target nano
# ./engine_file_build.sh mardi_pineapple_yolo_v8 n 3 512 512 --target nano

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME="yolov8n"
MODEL_TYPE="n"
NC=80
WIDTH=640
HEIGHT=640
TARGET="x86"

# ── Parse positional + optional args ─────────────────────────────────────────
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)   TARGET="$2"; shift 2 ;;
        --nc)       NC="$2";     shift 2 ;;
        --width)    WIDTH="$2";  shift 2 ;;
        --height)   HEIGHT="$2"; shift 2 ;;
        -*)         echo "Unknown option: $1"; exit 1 ;;
        *)          POSITIONAL+=("$1"); shift ;;
    esac
done
[[ ${#POSITIONAL[@]} -ge 1 ]] && MODEL_NAME="${POSITIONAL[0]}"
[[ ${#POSITIONAL[@]} -ge 2 ]] && MODEL_TYPE="${POSITIONAL[1]}"
[[ ${#POSITIONAL[@]} -ge 3 ]] && NC="${POSITIONAL[2]}"
[[ ${#POSITIONAL[@]} -ge 4 ]] && WIDTH="${POSITIONAL[3]}"
[[ ${#POSITIONAL[@]} -ge 5 ]] && HEIGHT="${POSITIONAL[4]}"

WTS_FILE="${MODEL_NAME}.wts"
ENGINE_FILE="${MODEL_NAME}.engine"

echo "================================================="
echo " YOLOv8 TRT Engine Builder"
echo "================================================="
echo " Model    : ${MODEL_NAME}.pt"
echo " Wts      : ${WTS_FILE}"
echo " Engine   : ${ENGINE_FILE}"
echo " Type     : ${MODEL_TYPE}"
echo " Classes  : ${NC}"
echo " Input    : ${WIDTH} x ${HEIGHT}"
echo " Target   : ${TARGET}"
echo "================================================="
echo ""

# ── Step 1: Generate .wts from .pt ───────────────────────────────────────────
if [ -f "./yolov8/weights/${WTS_FILE}" ]; then
    echo "==> Step 1: ${WTS_FILE} already exists — skipping."
else
    echo "==> Step 1: Generating ${WTS_FILE} from ${MODEL_NAME}.pt ..."
    docker run --rm --net=host \
        --runtime nvidia --gpus all --privileged \
        -v "$(pwd)/yolov8/weights:/workspace/yolov8/build/weights" \
        -v "$(pwd)/yolov8:/yolov8" \
        meraquetech/tensorrt-yolov8:ultralytics \
        bash -c "cd /yolov8 && python3 gen_wts.py \
            -w /workspace/yolov8/build/weights/${MODEL_NAME}.pt \
            -o /workspace/yolov8/build/weights/${WTS_FILE} \
            -t detect"
    echo "==> ${WTS_FILE} generated."
fi
echo ""

# # ── Step 2: Build engine ──────────────────────────────────────────────────────
# if [ "${TARGET}" = "nano" ]; then
#     # ── Nano: colcon build inside bed_detect image → ros2 run -s ─────────────
#     echo "==> Step 2: Building engine via ROS2 node (Nano target) ..."
#     echo "    kNumClass=${NC}  input=${WIDTH}x${HEIGHT}"

#     docker run --rm --net=host \
#         --runtime nvidia --gpus all --privileged \
#         -v "$(pwd)/yolov8/weights:/weights" \
#         -v "$(pwd)/yolov8_trt_bed_detect:/ros2_ws/src/yolov8_trt_bed_detect" \
#         meraquetech/race_nav:yolov8-trt-bed-detect-nano.v7 \
#         bash -c "
#             set -e
#             echo '--- Patching kNumClass to ${NC} ---'
#             sed -i 's/const static int kNumClass = [0-9]\\+/const static int kNumClass = ${NC}/' \
#                 /ros2_ws/src/yolov8_trt_bed_detect/include/config.h
#             grep 'kNumClass' /ros2_ws/src/yolov8_trt_bed_detect/include/config.h

#             echo '--- Rebuilding package ---'
#             ROS_SETUP=\$(find /opt/ros /root -name 'setup.bash' 2>/dev/null | head -1)
#             source \"\${ROS_SETUP}\"
#             cd /ros2_ws
#             colcon build --packages-select yolov8_trt_bed_detect \
#                 --cmake-args -DCMAKE_BUILD_TYPE=Release
#             source /ros2_ws/install/setup.bash

#             echo '--- Serializing engine ---'
#             ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect \
#                 -s /weights/${WTS_FILE} /weights/${ENGINE_FILE} ${MODEL_TYPE} ${WIDTH} ${HEIGHT}
#             echo '--- Done: /weights/${ENGINE_FILE} ---'
#         "

# else
#     # ── x86: make inside x86 TRT image → ./yolov8_det -s ────────────────────
#     echo "==> Step 2: Building engine via x86 TRT image ..."
#     echo "    kNumClass=${NC}"

#     docker run --rm --net=host \
#         --runtime nvidia --gpus all --privileged \
#         -v "$(pwd)/yolov8/images:/workspace/yolov8/build/images:ro" \
#         -v "$(pwd)/yolov8/weights:/workspace/yolov8/build/weights" \
#         -v "$(pwd)/yolov8/weights:/output" \
#         meraquetech/race_nav:yolov8-trt-x86 \
#         bash -c "
#             set -e
#             echo '--- Patching kNumClass to ${NC} ---'
#             sed -i 's/const static int kNumClass = [0-9]\\+/const static int kNumClass = ${NC}/' \
#                 /workspace/yolov8/include/config.h
#             grep 'kNumClass' /workspace/yolov8/include/config.h

#             echo '--- Recompiling yolov8_det ---'
#             cd /workspace/yolov8/build
#             make -j\$(nproc) yolov8_det

#             echo '--- Serializing engine ---'
#             ./yolov8_det -s ./weights/${WTS_FILE} ${ENGINE_FILE} ${MODEL_TYPE}
#             cp ${ENGINE_FILE} /output/
#             echo '--- Done: /output/${ENGINE_FILE} ---'
#         "
# fi

# echo ""
# echo "==> Engine ready: ./yolov8/weights/${ENGINE_FILE}"
