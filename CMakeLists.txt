cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(ptGraph CUDA)

set(CMAKE_CUDA_STANDARD 14)
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_executable(ptGraph main.cu gpu_kernels.cu gpu_kernels.cuh range.hpp bfs-sync.hpp tools.hpp common.cuh common.cu bfs.cuh bfs.cu cc.cuh cc.cu sssp.cuh sssp.cu GraphMeta.cu GraphMeta.cuh CalculateOpt.cu CalculateOpt.cuh constants.cuh TimeRecord.cu TimeRecord.cuh ArgumentParser.cu ArgumentParser.cuh)

set_target_properties(
        ptGraph
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(ptGraph -lcublas -lcurand)