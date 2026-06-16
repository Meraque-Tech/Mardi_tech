cmake_minimum_required(VERSION 3.10)
project(data_logger)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# GCC 8 on Ubuntu 18.04 (Jetson L4T) needs explicit stdc++fs linkage
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    set(STDFS_LIB stdc++fs)
endif()

# --- controller ---
add_executable(data_logger_ctrl
    main.cpp
)
target_link_libraries(data_logger_ctrl ${STDFS_LIB})

# --- frame_logger ---
add_executable(frame_logger
    frame_logger/frame_logger.cpp
)
target_link_libraries(frame_logger ${OpenCV_LIBS} ${STDFS_LIB})

# --- video_logger ---
add_executable(video_logger
    video_logger/video_logger.cpp
)
target_link_libraries(video_logger ${OpenCV_LIBS} ${STDFS_LIB})
