cmake_minimum_required(VERSION 3.14)  # Need 3.14+ for FetchContent_MakeAvailable
project(rtk_localization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(prime_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)




# Download and build GeographicLib if not found
include(FetchContent)


include_directories(
  include 
  ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# --- Include dirs ---
include_directories(
  include
)


set(TARGET_DEPENDENCIES
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  visualization_msgs
  std_srvs
  pcl_conversions
  PCL
  tf2_geometry_msgs
  tf2
  prime_msgs
)

# First try to find system GeographicLib
find_library(SYSTEM_GEOGRAPHIC_LIB NAMES Geographic geographic GeographicLib geographiclib)
if(SYSTEM_GEOGRAPHIC_LIB)
    message(STATUS "Found system GeographicLib: ${SYSTEM_GEOGRAPHIC_LIB}")
    set(GeographicLib_LIBRARIES ${SYSTEM_GEOGRAPHIC_LIB})
    
    # Find include directory
    find_path(GEOGRAPHIC_INCLUDE_DIR NAMES GeographicLib/Geodesic.hpp)
    if(GEOGRAPHIC_INCLUDE_DIR)
        set(GeographicLib_INCLUDE_DIRS ${GEOGRAPHIC_INCLUDE_DIR})
    else()
        # Common include paths
        set(GeographicLib_INCLUDE_DIRS
            /usr/include
            /usr/local/include
            /opt/local/include
        )
    endif()
else()
    # Download GeographicLib
    message(STATUS "Downloading GeographicLib from GitHub...")
    FetchContent_Declare(
        GeographicLib
        GIT_REPOSITORY https://github.com/GeographicLib/GeographicLib.git
        GIT_TAG v2.1.2  # Use specific version
    )
    
    FetchContent_MakeAvailable(GeographicLib)
    
    # Set variables for consistency
    set(GeographicLib_INCLUDE_DIRS
        ${geographiclib_SOURCE_DIR}/include
        ${geographiclib_BINARY_DIR}/include
    )
    set(GeographicLib_LIBRARIES GeographicLib::GeographicLib)
endif()


#****************************************************************
#****************************************************************


################################################################
################################################################


# Create executable
add_executable(rtk_odom src/rtk_odom.cpp src/gps_utils.cpp)
ament_target_dependencies(rtk_odom ${TARGET_DEPENDENCIES})
target_link_libraries(rtk_odom ${GeographicLib_LIBRARIES} ${PCL_LIBRARIES})
# Include directories
target_include_directories(rtk_odom PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GeographicLib_INCLUDE_DIRS}
)

################################################################
################################################################


# Create executable
add_executable(lio_odom_georef src/lio_odom_georef.cpp src/gps_utils.cpp)
ament_target_dependencies(lio_odom_georef ${TARGET_DEPENDENCIES})
target_link_libraries(lio_odom_georef ${GeographicLib_LIBRARIES} ${PCL_LIBRARIES})
# Include directories
target_include_directories(lio_odom_georef PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GeographicLib_INCLUDE_DIRS}
)

################################################################
################################################################

add_executable(rtk_imu_sf 
  src/rtk_imu_sf.cpp 
  src/rtk_utils.cpp 
  src/gps_utils.cpp)
ament_target_dependencies(rtk_imu_sf ${TARGET_DEPENDENCIES})
target_link_libraries(rtk_imu_sf ${GeographicLib_LIBRARIES} ${PCL_LIBRARIES})
# Include directories
target_include_directories(rtk_imu_sf PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GeographicLib_INCLUDE_DIRS}
)

################################################################
################################################################

################################################################
################################################################

add_executable(rtk_mid_raw 
  src/rtk_mid_raw.cpp 
  src/rtk_utils.cpp 
  src/gps_utils.cpp)
ament_target_dependencies(rtk_mid_raw ${TARGET_DEPENDENCIES})
target_link_libraries(rtk_mid_raw ${GeographicLib_LIBRARIES} ${PCL_LIBRARIES})
# Include directories
target_include_directories(rtk_mid_raw PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GeographicLib_INCLUDE_DIRS}
)

################################################################
################################################################

################################################################
################################################################

add_executable(ekf_imu_gps_node 
  src/ekf_imu_gps_node.cpp 
  src/rtk_utils.cpp 
  src/gps_utils.cpp)
ament_target_dependencies(ekf_imu_gps_node ${TARGET_DEPENDENCIES})
target_link_libraries(ekf_imu_gps_node ${GeographicLib_LIBRARIES} ${PCL_LIBRARIES})
# Include directories
target_include_directories(ekf_imu_gps_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GeographicLib_INCLUDE_DIRS}
)

################################################################
################################################################


add_executable(rtk_lio_calibration 
  src/rtk_lio_calibration.cpp 
  src/rtk_utils.cpp 
  src/gps_utils.cpp)
ament_target_dependencies(rtk_lio_calibration ${TARGET_DEPENDENCIES})
target_link_libraries(rtk_lio_calibration ${GeographicLib_LIBRARIES} ${PCL_LIBRARIES})
# Include directories
target_include_directories(rtk_lio_calibration PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GeographicLib_INCLUDE_DIRS}
)

################################################################
################################################################

add_executable(rtk_mid 
  src/rtk_mid.cpp 
  src/rtk_utils.cpp 
  src/gps_utils.cpp)
ament_target_dependencies(rtk_mid ${TARGET_DEPENDENCIES})
target_link_libraries(rtk_mid ${GeographicLib_LIBRARIES} ${PCL_LIBRARIES})
# Include directories
target_include_directories(rtk_mid PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GeographicLib_INCLUDE_DIRS}
)

################################################################
################################################################

add_executable(gnss_pvt_enu_odom
  src/gnss_pvt_enu_odom.cpp
  src/gps_utils.cpp)
ament_target_dependencies(gnss_pvt_enu_odom ${TARGET_DEPENDENCIES} std_msgs)
target_link_libraries(gnss_pvt_enu_odom ${GeographicLib_LIBRARIES} ${PCL_LIBRARIES} nlohmann_json::nlohmann_json)
# Include directories
target_include_directories(gnss_pvt_enu_odom PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GeographicLib_INCLUDE_DIRS}
)

################################################################
################################################################

add_executable(gnss_imu_eskf_node
  src/gnss_imu_eskf_node.cpp
  src/gps_utils.cpp)
ament_target_dependencies(gnss_imu_eskf_node ${TARGET_DEPENDENCIES} std_msgs sensor_msgs)
target_link_libraries(gnss_imu_eskf_node ${GeographicLib_LIBRARIES} ${PCL_LIBRARIES} nlohmann_json::nlohmann_json)
# Include directories
target_include_directories(gnss_imu_eskf_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GeographicLib_INCLUDE_DIRS}
)

################################################################
################################################################


# Install
install(TARGETS
  rtk_odom
  lio_odom_georef
  rtk_imu_sf
  rtk_mid_raw
  ekf_imu_gps_node
  rtk_lio_calibration
  rtk_mid
  gnss_pvt_enu_odom
  gnss_imu_eskf_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)


ament_export_include_directories(include)
ament_package()