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

add_executable(XePlanner src/main.cpp)

# Dynamic link
target_link_libraries(XePlanner ${MUJOCO_LIBRARY} glfw)
