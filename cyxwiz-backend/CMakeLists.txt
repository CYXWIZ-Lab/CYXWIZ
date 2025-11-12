cmake_minimum_required(VERSION 3.20)

project(cyxwiz-backend VERSION 0.1.0)

# Find ArrayFire (needs to be installed separately)
# Download from: https://arrayfire.com/download
find_package(ArrayFire QUIET)

if(NOT ArrayFire_FOUND)
    message(WARNING "ArrayFire not found. Backend will build without GPU acceleration.")
    message(STATUS "Download ArrayFire from: https://arrayfire.com/download")
    set(CYXWIZ_HAS_ARRAYFIRE FALSE)
else()
    message(STATUS "ArrayFire found: ${ArrayFire_DIR}")
    set(CYXWIZ_HAS_ARRAYFIRE TRUE)
endif()

# Find other packages
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Source files
set(BACKEND_SOURCES
    src/core/engine.cpp
    src/core/tensor.cpp
    src/core/device.cpp
    src/core/memory_manager.cpp
    src/core/cyxwiz_c.cpp
    src/algorithms/optimizer.cpp
    src/algorithms/loss.cpp
    src/algorithms/activation.cpp
    src/algorithms/layer.cpp
    src/algorithms/model.cpp
)

# Header files
set(BACKEND_HEADERS
    include/cyxwiz/engine.h
    include/cyxwiz/tensor.h
    include/cyxwiz/device.h
    include/cyxwiz/memory_manager.h
    include/cyxwiz/optimizer.h
    include/cyxwiz/loss.h
    include/cyxwiz/activation.h
    include/cyxwiz/layer.h
    include/cyxwiz/model.h
    include/cyxwiz/cyxwiz.h
    include/cyxwiz/cyxwiz_c.h
)

# Create shared library (DLL/SO)
add_library(cyxwiz-backend SHARED
    ${BACKEND_SOURCES}
    ${BACKEND_HEADERS}
)

# Generate export header for DLL symbols
include(GenerateExportHeader)
generate_export_header(cyxwiz-backend
    BASE_NAME CYXWIZ
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/cyxwiz/cyxwiz_export.h
)

# Include directories
target_include_directories(cyxwiz-backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(cyxwiz-backend PUBLIC
    fmt::fmt
    spdlog::spdlog
    cyxwiz-protocol
)

if(CYXWIZ_HAS_ARRAYFIRE)
    target_link_libraries(cyxwiz-backend PUBLIC ArrayFire::af)
    target_compile_definitions(cyxwiz-backend PUBLIC CYXWIZ_HAS_ARRAYFIRE)

    if(CYXWIZ_ENABLE_CUDA)
        target_link_libraries(cyxwiz-backend PUBLIC ArrayFire::afcuda)
        target_compile_definitions(cyxwiz-backend PUBLIC CYXWIZ_ENABLE_CUDA)
    endif()

    if(CYXWIZ_ENABLE_OPENCL)
        target_link_libraries(cyxwiz-backend PUBLIC ArrayFire::afopencl)
        target_compile_definitions(cyxwiz-backend PUBLIC CYXWIZ_ENABLE_OPENCL)
    endif()
endif()

# Set properties
set_target_properties(cyxwiz-backend PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

# Python bindings
pybind11_add_module(pycyxwiz
    python/bindings.cpp
)

target_link_libraries(pycyxwiz PRIVATE
    cyxwiz-backend
)

# Install rules
install(TARGETS cyxwiz-backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/cyxwiz
    DESTINATION include
)

install(TARGETS pycyxwiz
    LIBRARY DESTINATION python
)
