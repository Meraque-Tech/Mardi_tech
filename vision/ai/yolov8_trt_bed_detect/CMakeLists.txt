cmake_minimum_required(VERSION 3.10)

project(yolov8_trt_bed_detect)

add_definitions(-w)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

set(_include_dirs "include;/usr/include;/usr/include/opencv4")

add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set the C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

# find_package(OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
# find_package(PCL 1.2 REQUIRED)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(std_srvs REQUIRED)
# find_package(lytbot2_msg_srv REQUIRED)


set(dependencies
ament_cmake
rclcpp
std_msgs
tf2
geometry_msgs
tf2_ros
visualization_msgs
nav2_msgs
std_srvs
)



# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})
# list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)


add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)



include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})


option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()


# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  # cuda
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)

  # tensorrt
  include_directories(/usr/local/cuda/targets/x86_64-linux/include/)
  link_directories(/usr/local/cuda/targets/x86_64-linux/lib/)
#  include_directories(/home/lindsay/TensorRT-7.2.3.4/include)
#  link_directories(/home/lindsay/TensorRT-7.2.3.4/lib)
endif()

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(yolov8_trt_bed_detect ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})
ament_target_dependencies(yolov8_trt_bed_detect ${dependencies})
target_link_libraries(yolov8_trt_bed_detect nvinfer)
target_link_libraries(yolov8_trt_bed_detect cudart)
target_link_libraries(yolov8_trt_bed_detect myplugins)
# target_link_libraries(yolov8_trt_bed_detect ${OpenCV_LIBS})
target_link_libraries(yolov8_trt_bed_detect ${OpenCV_LIBS} ${ZED_LIBS})
# target_link_libraries (yolov8_trt_bed_detect ${PCL_LIBRARIES})
target_link_libraries(yolov8_trt_bed_detect -Wall -Wextra -Wpedantic )

install(TARGETS
  yolov8_trt_bed_detect
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()