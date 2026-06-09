# YOLOv8
```
    

    docker run -it --rm --gpus all --privileged \
    --name yolo_export \
    -v ~/yolo_models:/workspace/models \
    -v ./yolov8:/yolov8
    meraquetech/tensorrt-yolov8:ultralytics

    cd /yolov8
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
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