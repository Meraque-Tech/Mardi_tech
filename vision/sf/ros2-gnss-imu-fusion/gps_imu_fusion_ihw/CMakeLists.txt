cmake_minimum_required(VERSION 3.14)
project(gps_imu_fusion_ihw)

# ────────────────────────────────────────────────────
# Compile options
# ────────────────────────────────────────────────────
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ────────────────────────────────────────────────────
# Dependencies
# ────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)            # ★ 추가
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

# ────────────────────────────────────────────────────
# Include dirs (export)
# ────────────────────────────────────────────────────
set(PUBLIC_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
include_directories(${PUBLIC_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})

# ────────────────────────────────────────────────────
# Library: kalman_filter
# ────────────────────────────────────────────────────
add_library(kalman_filter
  src/kalman_filter.cpp
)
target_include_directories(kalman_filter PUBLIC ${PUBLIC_INCLUDE_DIR})
ament_target_dependencies(kalman_filter Eigen3)

# ────────────────────────────────────────────────────
# Executable: sensor_fusion_node
# ────────────────────────────────────────────────────
add_executable(sensor_fusion_node
  src/sensor_fusion_node.cpp
)
target_include_directories(sensor_fusion_node PUBLIC ${PUBLIC_INCLUDE_DIR})
ament_target_dependencies(sensor_fusion_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros Eigen3
)
target_link_libraries(sensor_fusion_node kalman_filter)

# ────────────────────────────────────────────────────
# Executable: global_yaw_integrator_node
# ────────────────────────────────────────────────────
add_executable(global_yaw_integrator_node
  src/global_yaw_integrator_node.cpp)
  
target_include_directories(global_yaw_integrator_node PUBLIC ${PUBLIC_INCLUDE_DIR})
ament_target_dependencies(global_yaw_integrator_node
  rclcpp sensor_msgs geometry_msgs std_msgs)

# ────────────────────────────────────────────────────
# Executable: global_yaw_complementary_node
# ────────────────────────────────────────────────────
add_executable(global_yaw_complementary_node
    src/global_yaw_complementary_node.cpp)
ament_target_dependencies(global_yaw_complementary_node
    rclcpp geometry_msgs sensor_msgs std_msgs nav_msgs tf2)

# ────────────────────────────────────────────────────
# Executable: current_speed_node
# ────────────────────────────────────────────────────
add_executable(current_speed_node 
    src/current_speed_node.cpp)
ament_target_dependencies(current_speed_node
    rclcpp geometry_msgs sensor_msgs std_msgs)

# ────────────────────────────────────────────────────
# Installation
# ────────────────────────────────────────────────────
# 1. libs & executables
install(TARGETS
  kalman_filter
  sensor_fusion_node
  global_yaw_integrator_node  # calculating global yaw - initial yaw from only GPS(using /local_xy), integrating yaw from IMU(w_z)
  global_yaw_complementary_node  # calculating global yaw - using complementary filter (GPS + IMU)
  current_speed_node  # calculating current speed - using GPS and IMU
  DESTINATION lib/${PROJECT_NAME}
)

# 2. headers (keep sub‑folder structure)
install(DIRECTORY include/gps_imu_fusion_ihw
        DESTINATION include/)

# 3. config & launch
install(DIRECTORY config
        DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}/)

# ────────────────────────────────────────────────────
# Linters / tests
# ────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
