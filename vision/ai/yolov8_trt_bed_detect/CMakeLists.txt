cmake_minimum_required(VERSION 3.10)

project(yolov8_trt_bed_detect)

add_definitions(-w)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

set(dependencies
  ament_cmake
  rclcpp
  std_msgs
  std_srvs
)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 53)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

link_directories(${CUDA_LIBRARY_DIRS})

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)
  include_directories(/usr/local/cuda/targets/x86_64-linux/include/)
  link_directories(/usr/local/cuda/targets/x86_64-linux/lib/)
endif()

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(yolov8_trt_bed_detect ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})
ament_target_dependencies(yolov8_trt_bed_detect ${dependencies})
target_link_libraries(yolov8_trt_bed_detect nvinfer cudart myplugins ${OpenCV_LIBS} pthread)

install(TARGETS
  yolov8_trt_bed_detect
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS
  myplugins
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config web_server
  DESTINATION share/${PROJECT_NAME})

set_target_properties(yolov8_trt_bed_detect PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
