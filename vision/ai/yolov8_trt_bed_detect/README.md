# yolov8 for hospitalbed detection 
# for building yolov8_trt in X86 and arrch64 system

# updating pre requirements
# nvidia-smi
# nvcc -V
# dpkg -l | grep TensorRT
# ZED
pip3 install ultralytics

# Steaps for building wts file
cp [.pt file] ~/sar/docker/arm64/packages/ros2_src/yolov8_trt_bed_detect/
cd ~/sar/docker/arm64/packages/ros2_src/yolov8_trt_bed_detect/
python3 gen_wts.py -w [.pt file name]
example -> python3 gen_wts.py -w yolov8s_512_augment_data.pt 
output -> yolov8s_512_augment_data.wts

# Steaps for building engine file
cd ~/sar/docker/arm64/packages/ros2_src/
colcon build --packages-select yolov8_trt_bed_detect
source install/setup.bash
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect -s [.wts] [.engine] [n/s/m/l/x] // serialize model to plan file
example -> ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect -s yolov8s_512_augment_data.wts yolov8s_512_augment_data.engine s


# for running the node
cd ~/sar/docker/arm64/packages/ros2_src/
source install/setup.bash
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect -d [.engine] ./ [c/g] -conf [conf_score] -kh [kh] -kv [kv] -d_thresh [d_thresh] --font_dis [font_distance]
// deserialize and run inference, Live Zed2 data will be processed in cpu or gpu <c/g>.
# c/g --> CPU/GPU
# -conf ---> how much confidence socre you want to put
# -kh ---> the absolute distance value for horizontal position
# -kv ---> the absolute distance value for vertical position
# -d_thresh ---> the distance threshold of the detected object
# -font_dis ---> font waypoint custom distance adjustment

# Example 1->
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect -d yolov8s_512_augment_data.engine ./ g -conf 0.8 -kh 0.6 -kv 0.5 -d_thresh 1.1 -font_dis 2.0 //gpu postprocess

# Example 2->
ros2 run yolov8_trt_bed_detect yolov8_trt_bed_detect  -d can_ggn_st_johns_hospital_bed_512_yolov8s.engine ./ g -conf 0.86 -kh 0.6 -kv 0.9 -d_thresh 1.0 -font_dis 4.0

# Run from script file
# pre requirements
<!-- make sure the Zed2 wire is connected -->
./engine_file_build.sh -pt_file [.pt] -wts_file [.wts] -engine_file [.engine] -model_type [n/s/m/l/x] 

example -->
./engine_file_build.sh -pt_file can_ggn_st_johns_os_hospital_bed_512_yolov8s_no_earlystop.pt -wts_file can_ggn_st_johns_os_hospital_bed_512_yolov8s_no_earlystop.wts -engine_file can_ggn_st_johns_os_hospital_bed_512_yolov8s_no_earlystop.engine -model_type s

# for moving the robot
cd ~/sar/docker/arm64/packages/ros2_src/yolov8_trt_bed_detect
python3 move_bot.py

# Steps to move the robot
# run Oneline navigation2
# run move_bot.py for basic waypoint navigation
# run bed detection algorithm 
# The output robot will move towards its destination, 
# even though the generated path by the navigation2 stack might be unsafe. This is because nav2 does not have any understanding of the 3D world. To address this, we will incorporate a 3D voxel grid from the camera to provide environmental information in three dimensions.

