cmake_minimum_required(VERSION 3.8)
project(prime_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CustomTwist.msg"
  "msg/Encoder.msg"
  "msg/Waypoint.msg"
  "msg/Task.msg"
  "msg/Mission.msg"
  "msg/MissionCommand.msg"
  "msg/RtkDual.msg"
  "msg/RtkDualStatus.msg"
  "msg/RtkStatus.msg"
  "msg/ImuFusionStatus.msg"
  "msg/Cluster.msg"
  "msg/ClusterArray.msg"
  "msg/MapToOdomStatus.msg"
  "msg/TopicRateStatus.msg"
  "msg/TopicRateStatusArray.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs nav_msgs
  ADD_LINTER_TESTS
 )

 ament_export_dependencies(rosidl_default_runtime)
 ament_package()
