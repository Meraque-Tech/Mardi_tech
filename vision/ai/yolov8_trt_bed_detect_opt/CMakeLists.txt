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

# Locate NvInfer.h — try common L4T/JetPack install paths
find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS
        /usr/include
        /usr/include/aarch64-linux-gnu
        /usr/local/cuda/include
    REQUIRED
)
message(STATUS "TRT include dir: ${TRT_INCLUDE_DIR}")

link_directories(/usr/local/cuda/lib64)
link_directories(/usr/lib/aarch64-linux-gnu)
link_directories(/usr/lib)

set(SRC_FILES
    src/main.cpp
    src/yolo.cpp
)

cuda_add_executable(${PROJECT_NAME} ${SRC_FILES})
add_definitions(-O3 -D_MWAITXINTRIN_H_INCLUDED -Wno-deprecated-declarations)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
    /usr/local/cuda/include
)

target_link_libraries(${PROJECT_NAME}
    nvinfer
    nvonnxparser
    ${SPECIAL_OS_LIBS}
    cuda
    cudart
    ${OpenCV_LIBRARIES}
)
ament_target_dependencies(${PROJECT_NAME} ament_cmake rclcpp)

install(TARGETS ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME})
ament_package()
