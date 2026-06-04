cmake_minimum_required(VERSION 3.5)
PROJECT(yolov8_trt_bed_detect_opt)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

IF(NOT WIN32)
    SET(SPECIAL_OS_LIBS "pthread" "X11")
    add_definitions(-Wno-write-strings)
ENDIF()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)

set(SRC_FILES
    src/main.cpp
    src/yolo.cpp
)

cuda_add_executable(${PROJECT_NAME} ${SRC_FILES})
add_definitions(-O3 -D_MWAITXINTRIN_H_INCLUDED -Wno-deprecated-declarations)

target_link_libraries(${PROJECT_NAME}
    nvinfer
    nvonnxparser
    ${SPECIAL_OS_LIBS}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBRARIES}
)
ament_target_dependencies(${PROJECT_NAME} ament_cmake rclcpp)

install(TARGETS ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME})
ament_package()
