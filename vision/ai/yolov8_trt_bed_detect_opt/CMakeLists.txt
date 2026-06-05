cmake_minimum_required(VERSION 3.5)
PROJECT(yolov8_trt_bed_detect_opt)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

SET(SPECIAL_OS_LIBS "")

option(ENABLE_INT8_CALIBRATOR "Enable int8 calibrator class for quantification calibration, requires opencv dnn" OFF)

IF (ENABLE_INT8_CALIBRATOR)
    ADD_DEFINITIONS(-DENABLE_INT8_CALIBRATOR)
ENDIF()

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.2" CACHE PATH "" FORCE)
set(CUDA_CUDART_LIBRARY "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcudart.so" CACHE FILEPATH "" FORCE)

find_package(ZED 4 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)

set(dependencies
ament_cmake
rclcpp
std_msgs
tf2
geometry_msgs
tf2_ros
visualization_msgs
nav_msgs
std_srvs
)


IF(NOT WIN32)
    SET(SPECIAL_OS_LIBS "pthread")
    add_definitions(-Wno-write-strings)
ENDIF()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)

# tensorrt
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)


FILE(GLOB_RECURSE SRC_FILES src/*.c*)
FILE(GLOB_RECURSE HDR_FILES include/*.h*)

# Exclude OpenGL viewer — not needed for headless inference
list(FILTER SRC_FILES EXCLUDE REGEX ".*GLViewer.*")
list(FILTER HDR_FILES EXCLUDE REGEX ".*GLViewer.*")

include_directories(
  include
)

cuda_add_executable(${PROJECT_NAME} ${HDR_FILES} ${SRC_FILES})
add_definitions(-O3 -D_MWAITXINTRIN_H_INCLUDED -Wno-deprecated-declarations)

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

SET(TRT_LIBS nvinfer nvonnxparser)

target_link_libraries(${PROJECT_NAME}
                        ${TRT_LIBS}
                        ${SPECIAL_OS_LIBS}
                        ${ZED_LIBS}
                        ${OpenCV_LIBRARIES})
ament_target_dependencies(${PROJECT_NAME} ${dependencies})


if(INSTALL_SAMPLES)
    LIST(APPEND SAMPLE_LIST ${PROJECT_NAME})
    SET(SAMPLE_LIST "${SAMPLE_LIST}" PARENT_SCOPE)
endif()

install(TARGETS
    ${PROJECT_NAME}
	DESTINATION lib/${PROJECT_NAME})
ament_package()