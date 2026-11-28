# Standalone check: does af::matmul read a reused result buffer (see probe.cpp)?
# Not part of the CyxWiz build:
#   cmake -S . -B build -DArrayFire_DIR=<ArrayFire>/cmake
#   cmake --build build --config Release
cmake_minimum_required(VERSION 3.20)
project(matmul_reused_buffer_probe CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(ArrayFire REQUIRED)
add_executable(matmul_reused_buffer_probe probe.cpp)
target_link_libraries(matmul_reused_buffer_probe PRIVATE ArrayFire::af)
