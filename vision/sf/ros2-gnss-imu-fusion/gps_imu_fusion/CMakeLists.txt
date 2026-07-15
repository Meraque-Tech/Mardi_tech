cmake_minimum_required(VERSION 3.18)
project(gps_imu_fusion)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# GeographicLib 모듈 경로 설정
# 시스템 기본 경로 사용, 필요시 환경 변수로 재정의 가능
if(DEFINED ENV{GEOGRAPHICLIB_CMAKE_DIR})
  list(APPEND CMAKE_MODULE_PATH $ENV{GEOGRAPHICLIB_CMAKE_DIR})
else()
  list(APPEND CMAKE_MODULE_PATH "/usr/share/cmake/geographiclib")
endif()
find_package(GeographicLib REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

if(GeographicLib_FOUND)
  include_directories(${GeographicLib_INCLUDE_DIRS})
endif()

add_executable(ekf_fusion_node
  src/ekf_fusion_main.cpp
  src/ekf_fusion_node.cpp
  src/kai_ekf_core.cpp
)

ament_target_dependencies(ekf_fusion_node
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

if(GeographicLib_FOUND)
  target_link_libraries(ekf_fusion_node ${GeographicLib_LIBRARIES})
endif()

target_link_libraries(ekf_fusion_node 
  ${EIGEN3_LIBRARIES}
)

install(TARGETS
  ekf_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/gps_vis/gps_to_cartesian.py
  scripts/gps_vis/gps_online_vis_node.py
  scripts/gps_vis/gps_offline_vis_node.py
  scripts/gps_vis/fixed_gps_publisher.py
  scripts/cone_frame_transformer.py
  scripts/velocity_magnitude_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  Eigen3
  GeographicLib
)

ament_package()
