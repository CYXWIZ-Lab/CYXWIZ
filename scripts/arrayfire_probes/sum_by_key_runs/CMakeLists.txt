# Standalone check of sumByKey on sorted keys with runs (see probe.cpp).
# Not part of the CyxWiz build:
#   cmake -S . -B build -DArrayFire_DIR=<ArrayFire>/cmake
#   cmake --build build --config Release
cmake_minimum_required(VERSION 3.20)
project(sum_by_key_runs_probe CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(ArrayFire REQUIRED)
add_executable(sum_by_key_runs_probe probe.cpp)
target_link_libraries(sum_by_key_runs_probe PRIVATE ArrayFire::af)
