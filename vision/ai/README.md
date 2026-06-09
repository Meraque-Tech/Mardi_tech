# YOLOv8

# Default: yolov8n, type n Buidling model
```
./engine_file_build.sh
```


# Custom model
```
./engine_file_build.sh yolov8n n
./engine_file_build.sh yolov8s s

Two-step process:
Step 1 — runs meraquetech/tensorrt-yolov8:ultralytics to convert .pt → .wts via gen_wts.py
Step 2 — runs meraquetech/race_nav:yolov8-trt-x86 to serialize .wts → .engine and copies it back to ./yolov8/weights/

```

# Default: cam 0, GPU postprocess, show window
```
./run_yolov8_det_video.sh [engine] [cam_id] [c/g] [show]

./run_yolov8_det_video.sh                          # defaults: yolov8n.engine 0 g true

./run_yolov8_det_video.sh yolov8s.engine 0 g true

<!-- ./run_yolov8_det_video.sh yolov8s.engine 0 g true -->

```

# Custom args: cam_id, postprocess (c/g), show (true/false)
```
./run_yolov8_det_video.sh 1 c false
```


# Yolov8 model building Step by step with example yolo nano model ->
```
    

    docker run -it --rm --gpus all --privileged \
    --name yolo_export \
    -v ~/yolo_models:/workspace/models \
    -v ./yolov8:/yolov8 \
    meraquetech/tensorrt-yolov8:ultralytics

    cd /yolov8
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
    <!-- wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt -->
    python3 gen_wts.py -w yolov8n.pt -o yolov8n.wts -t detect

    xhost +local:root
    docker run -it --rm --net=host \
        --runtime nvidia \
        --gpus all \
        --privileged \
        -e DISPLAY=$DISPLAY \
        -e XAUTHORITY=/root/.Xauthority \
        -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v $HOME/.Xauthority:/root/.Xauthority:ro \
        -v ./yolov8/images:/workspace/yolov8/build/images:ro \
        -v ./yolov8/weights:/workspace/yolov8/build/weights:ro \
        --device /dev/video0 \
        --device /dev/video1 \
        meraquetech/race_nav:yolov8-trt-x86
    
    

    Serialize ->
    ./yolov8_det -s ./weights/yolov8n.wts yolov8n.engine n

    Serialize Engine Using Docker container ->
    docker run --rm --net=host \
        --runtime nvidia \
        --gpus all \
        --privileged \
        -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v ./yolov8/images:/workspace/yolov8/build/images:ro \
        -v ./yolov8/weights:/workspace/yolov8/build/weights:ro \
        -v ./yolov8/weights:/output \
        meraquetech/race_nav:yolov8-trt-x86 \
        bash -c "cd /workspace/yolov8/build && ./yolov8_det -s ./weights/yolov8n.wts yolov8n.engine n && cp yolov8n.engine /output/"


    DeSerialize ->
    ./yolov8_det -d yolov8n.engine ./images g


```




### Detection (Webcam / Video Stream)
```
// Run detection on a webcam or video device
// Arguments: [.engine] [cam_id: 0/1/...] [c/g] [show: true/false]

No imshow ->
./yolov8_det_video ./weights/yolov8n.engine 0 g false

imshow ->
./yolov8_det_video ./weights/yolov8n.engine 0 g true

./yolov8_det_video yolov8n.engine 0 g true    // webcam 0, GPU postprocess, showwindow
./yolov8_det_video yolov8n.engine 1 c false   // webcam 1, CPU postprocess, headless (no window)




// Press ESC to exit when show=true
// Press Ctrl+C to stop when running headless
```


