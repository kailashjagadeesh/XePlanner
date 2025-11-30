cmake_minimum_required(VERSION 3.10)
project(XePlanner)

set(CMAKE_CXX_STANDARD 17)

find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
    PATHS /usr/include /usr/local/mujoco/include
)

find_library(MUJOCO_LIBRARY mujoco
    PATHS /usr/lib /usr/local/mujoco/lib
)

find_package(glfw3 REQUIRED)

include_directories(${MUJOCO_INCLUDE_DIR})

add_executable(XePlanner 
    src/main.cpp 
    src/utils.cpp
    src/ecbs.cpp
    src/rrt_connect.cpp
)

add_executable(RRTPlanner 
    src/main_rrt.cpp 
    src/rrt_connect.cpp 
    src/utils.cpp
)

# Dynamic link
target_link_libraries(XePlanner ${MUJOCO_LIBRARY} glfw)

target_link_libraries(RRTPlanner ${MUJOCO_LIBRARY} glfw)
