cmake_minimum_required(VERSION 3.10)
project(data_logger)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# --- controller ---
add_executable(data_logger_ctrl
    main.cpp
)

# --- frame_logger ---
add_executable(frame_logger
    frame_logger/frame_logger.cpp
)
target_link_libraries(frame_logger ${OpenCV_LIBS})

# --- video_logger ---
add_executable(video_logger
    video_logger/video_logger.cpp
)
target_link_libraries(video_logger ${OpenCV_LIBS})
