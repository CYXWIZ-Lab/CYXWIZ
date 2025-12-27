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

# pybind11 is optional - controlled by parent CYXWIZ_HAS_PYTHON
if(CYXWIZ_HAS_PYTHON)
    find_package(pybind11 CONFIG QUIET)
endif()

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
    src/algorithms/layers/linear.cpp
    src/algorithms/activations/relu.cpp
    src/algorithms/activations/sigmoid.cpp
    src/algorithms/activations/tanh.cpp
    src/algorithms/sequential.cpp
    src/algorithms/scheduler.cpp
    src/algorithms/clustering.cpp
    src/algorithms/data_transform.cpp
    src/algorithms/dataloader.cpp
    src/algorithms/model_evaluation.cpp
    src/algorithms/dimensionality_reduction.cpp
    src/algorithms/feature_importance.cpp
    src/algorithms/model_interpretability.cpp
    src/algorithms/linear_algebra.cpp
    src/algorithms/signal_processing.cpp
    src/algorithms/optimization.cpp
    src/algorithms/time_series.cpp
    src/algorithms/text_processing.cpp
    src/algorithms/utilities.cpp
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
    include/cyxwiz/layers/linear.h
    include/cyxwiz/activations/relu.h
    include/cyxwiz/activations/sigmoid.h
    include/cyxwiz/activations/tanh.h
    include/cyxwiz/sequential.h
    include/cyxwiz/scheduler.h
    include/cyxwiz/clustering.h
    include/cyxwiz/data_transform.h
    include/cyxwiz/dataloader.h
    include/cyxwiz/model_evaluation.h
    include/cyxwiz/dimensionality_reduction.h
    include/cyxwiz/feature_importance.h
    include/cyxwiz/model_interpretability.h
    include/cyxwiz/linear_algebra.h
    include/cyxwiz/signal_processing.h
    include/cyxwiz/optimization.h
    include/cyxwiz/time_series.h
    include/cyxwiz/text_processing.h
    include/cyxwiz/utilities.h
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
        # Find CUDA Toolkit for cuda_runtime.h and cudart library
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
            target_link_libraries(cyxwiz-backend PUBLIC
                ArrayFire::afcuda
                CUDA::cudart  # CUDA runtime for cudaMemGetInfo()
            )
            target_compile_definitions(cyxwiz-backend PUBLIC CYXWIZ_ENABLE_CUDA)
        else()
            message(WARNING "CUDA Toolkit not found. Install from: https://developer.nvidia.com/cuda-downloads")
            message(WARNING "Building without CUDA memory query support")
        endif()
    endif()

    if(CYXWIZ_ENABLE_OPENCL)
        find_package(OpenCL REQUIRED)
        target_link_libraries(cyxwiz-backend PUBLIC
            ArrayFire::afopencl
            OpenCL::OpenCL  # OpenCL library for clGetDeviceInfo()
        )
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

# Python bindings (optional)
if(CYXWIZ_HAS_PYTHON AND pybind11_FOUND)
    pybind11_add_module(pycyxwiz
        python/bindings.cpp
    )

    target_link_libraries(pycyxwiz PRIVATE
        cyxwiz-backend
    )
endif()

# Install rules
install(TARGETS cyxwiz-backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/cyxwiz
    DESTINATION include
)

if(CYXWIZ_HAS_PYTHON AND TARGET pycyxwiz)
    install(TARGETS pycyxwiz
        LIBRARY DESTINATION python
    )
endif()
