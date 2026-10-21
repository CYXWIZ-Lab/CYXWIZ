# Standalone reproducer for an ArrayFire oneAPI backend bug (see probe.cpp).
# Not part of the CyxWiz build:
#   cmake -S . -B build -DArrayFire_DIR=<ArrayFire>/cmake
#   cmake --build build --config Release
cmake_minimum_required(VERSION 3.20)
project(oneapi_slice_index_probe CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(ArrayFire REQUIRED)
add_executable(oneapi_slice_index_probe probe.cpp)
target_link_libraries(oneapi_slice_index_probe PRIVATE ArrayFire::af)
