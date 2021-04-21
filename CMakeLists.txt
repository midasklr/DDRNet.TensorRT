cmake_minimum_required(VERSION 2.6)

project(DDRNet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/home/midas/TensorRT-7.0.0.11/include/)
link_directories(/home/midas/TensorRT-7.0.0.11/lib/)

find_package(OpenCV 3.4.8 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ddrnet ${PROJECT_SOURCE_DIR}/ddrnet.cpp)
target_link_libraries(ddrnet nvinfer)
target_link_libraries(ddrnet cudart)
target_link_libraries(ddrnet ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

