#!/bin/bash

pt_file=""
wts_file=""
engine_file=""
model_type=""

while [ "$1" != "" ]; do
    case $1 in
        -pt_file | --pt_file-name )         shift
                                            pt_file=$1
                                            ;;
        -wts_file | --wts_file-name )       shift
                                            wts_file=$1
                                            ;;
        -engine_file | --engine_file-name ) shift
                                            engine_file=$1
                                            ;;
        -model_type | --model-type-name )   shift
                                            model_type=$1
                                            ;;
    esac
    shift
done

# Steps for building wts file
echo "Pt file name --> $pt_file"
echo "Wts file name --> $wts_file"
echo "Engine file name --> $engine_file"
echo "Model type name --> $model_type"

if [ "$(uname -m)" == "x86_64" ]; then
    python3 /workspaces/sar/docker/arm64/packages/ros2_src/yolov8_trt_bed_detect/gen_wts.py -w "$pt_file"

    # Steps for building engine file
    if [ $? -eq 0 ]; then
        echo "wts file generation successfully."
        echo "engine file generation Started."
        source /workspaces/sar/docker/arm64/packages/ros2_src/install/setup.bash
        # ./yolov8_dev -s "$wts_file" "$engine_file" "$model_type"
        ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect -s "$wts_file" "$engine_file" "$model_type"
    else
        echo "wts file generation encountered an error."
    fi
elif [ "$(uname -m)" == "aarch64" ]; then
    python3 ~/sar/docker/arm64/packages/ros2_src/yolov8_trt_bed_detect/gen_wts.py -w "$pt_file"

    # Steps for building engine file
    if [ $? -eq 0 ]; then
        echo "wts file generation successfully."
        echo "engine file generation Started."
        source ~/sar/docker/arm64/packages/ros2_src/install/setup.bash
        # ./yolov8_nano -s "$wts_file" "$engine_file" "$model_type"
        ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect -s "$wts_file" "$engine_file" "$model_type"
    else
        echo "wts file generation encountered an error."
    fi
else
    echo "Unsupported architecture: $(uname -m)"
fi
