cmake_minimum_required(VERSION 3.20)

project(cyxwiz-backend VERSION 0.1.0)

# Add cmake module path for custom Find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

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

# Find libsndfile (for audio I/O)
find_package(SndFile CONFIG QUIET)
if(SndFile_FOUND)
    message(STATUS "libsndfile found - audio processing enabled")
    set(CYXWIZ_HAS_SNDFILE TRUE)
else()
    message(WARNING "libsndfile not found - audio processing disabled")
    set(CYXWIZ_HAS_SNDFILE FALSE)
endif()

# Find FFTW3 (for FFT in audio/signal processing)
find_package(FFTW3 CONFIG QUIET)
if(FFTW3_FOUND)
    message(STATUS "FFTW3 found - FFT-based features enabled")
    set(CYXWIZ_HAS_FFTW TRUE)
else()
    message(WARNING "FFTW3 not found - FFT-based features disabled")
    set(CYXWIZ_HAS_FFTW FALSE)
endif()

# Find DuckDB (optional - for SQL-based data loading)
find_package(duckdb CONFIG QUIET)
if(duckdb_FOUND)
    message(STATUS "DuckDB found - SQL data loading enabled")
    set(CYXWIZ_HAS_DUCKDB TRUE)
else()
    message(WARNING "DuckDB not found - SQL data loading disabled")
    set(CYXWIZ_HAS_DUCKDB FALSE)
endif()

# Source files
set(BACKEND_SOURCES
    src/core/engine.cpp
    src/core/tensor.cpp
    src/core/tensor_utils.cpp
    src/core/tensor_shape.cpp
    src/core/tensor_reductions.cpp
    src/core/tensor_elementwise.cpp
    src/core/tensor_comparison.cpp
    src/core/tensor_logical.cpp
    src/core/tensor_linalg.cpp
    src/core/tensor_broadcast.cpp
    src/core/tensor_indexing.cpp
    src/core/tensor_concat.cpp
    src/core/device.cpp
    src/core/memory_manager.cpp
    src/core/cyxwiz_c.cpp
    src/algorithms/optimizer.cpp
    src/algorithms/loss.cpp
    src/algorithms/activation.cpp
    src/algorithms/layer.cpp
    src/algorithms/model.cpp
    src/algorithms/layers/linear.cpp
    src/algorithms/layers/layer_utils.cpp
    src/algorithms/activations/relu.cpp
    src/algorithms/activations/sigmoid.cpp
    src/algorithms/activations/tanh.cpp
    src/algorithms/sequential.cpp
    src/algorithms/scheduler.cpp
    src/algorithms/clustering.cpp
    src/algorithms/data_transform.cpp
    src/algorithms/model_evaluation.cpp
    src/algorithms/dimensionality_reduction.cpp
    src/algorithms/feature_importance.cpp
    src/algorithms/model_interpretability.cpp
    src/algorithms/linear_algebra.cpp
    src/algorithms/signal_processing.cpp
    src/algorithms/optimization.cpp
    src/algorithms/time_series.cpp
    src/algorithms/text_processing.cpp
    src/algorithms/tokenizer.cpp
    src/algorithms/audio_processing.cpp
    src/algorithms/utilities.cpp
    src/algorithms/rl_interface.cpp
    src/algorithms/dataloader.cpp
    src/core/data_loader.cpp
    # Distributed training
    src/algorithms/distributed/process_group.cpp
    src/algorithms/distributed/cpu_backend.cpp
    src/algorithms/distributed/ddp.cpp
    src/algorithms/distributed/distributed_sampler.cpp
    src/algorithms/distributed/distributed_trainer.cpp
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
    include/cyxwiz/rl_interface.h
    include/cyxwiz/data_loader.h
    include/cyxwiz/dataloader.h
    # Distributed training
    include/cyxwiz/distributed/process_group.h
    include/cyxwiz/distributed/cpu_backend.h
    include/cyxwiz/distributed/nccl_backend.h
    include/cyxwiz/distributed/ddp.h
    include/cyxwiz/distributed/distributed_sampler.h
    include/cyxwiz/distributed/distributed_trainer.h
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

# Platform-specific libraries for distributed training
if(WIN32)
    target_link_libraries(cyxwiz-backend PRIVATE ws2_32)
endif()

if(CYXWIZ_HAS_ARRAYFIRE)
    target_link_libraries(cyxwiz-backend PUBLIC ArrayFire::af)
    target_compile_definitions(cyxwiz-backend PUBLIC CYXWIZ_HAS_ARRAYFIRE)

    if(CYXWIZ_ENABLE_CUDA)
        # Keep CUDA backend support runtime-selected via ArrayFire.
        # Do not link CUDA runtime directly to avoid requiring cudart on non-NVIDIA systems.
        target_compile_definitions(cyxwiz-backend PUBLIC CYXWIZ_ENABLE_CUDA)
        message(STATUS "CUDA backend support enabled (runtime-selected via ArrayFire)")

        # NCCL for distributed GPU training (optional).
        # This is the only path that requires direct CUDA Toolkit linkage.
        option(CYXWIZ_ENABLE_NCCL "Enable NCCL for distributed GPU training" ON)
        if(CYXWIZ_ENABLE_NCCL)
            find_package(CUDAToolkit QUIET)
            if(CUDAToolkit_FOUND)
                message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
                find_package(NCCL QUIET)
                if(NCCL_FOUND)
                    message(STATUS "NCCL found: ${NCCL_LIBRARIES}")
                    if(NCCL_VERSION)
                        message(STATUS "  NCCL version: ${NCCL_VERSION}")
                    endif()
                    target_sources(cyxwiz-backend PRIVATE
                        src/algorithms/distributed/nccl_backend.cpp
                    )
                    target_link_libraries(cyxwiz-backend PUBLIC NCCL::nccl CUDA::cudart)
                    target_compile_definitions(cyxwiz-backend PUBLIC CYXWIZ_HAS_NCCL)
                    set(CYXWIZ_HAS_NCCL TRUE)
                else()
                    message(WARNING "NCCL not found - distributed GPU training will use CPU fallback")
                    message(STATUS "  NCCL can be installed from: https://developer.nvidia.com/nccl")
                    set(CYXWIZ_HAS_NCCL FALSE)
                endif()
            else()
                message(WARNING "CUDA Toolkit not found - NCCL backend disabled")
                set(CYXWIZ_HAS_NCCL FALSE)
            endif()
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

# Link libsndfile and FFTW3 if available (for audio processing)
if(CYXWIZ_HAS_SNDFILE)
    target_link_libraries(cyxwiz-backend PRIVATE SndFile::sndfile)
    target_compile_definitions(cyxwiz-backend PRIVATE CYXWIZ_HAS_SNDFILE)
endif()

if(CYXWIZ_HAS_FFTW)
    if(TARGET FFTW3::fftw3)
        target_link_libraries(cyxwiz-backend PRIVATE FFTW3::fftw3)
    else()
        target_link_libraries(cyxwiz-backend PRIVATE fftw3)
    endif()
    target_compile_definitions(cyxwiz-backend PRIVATE CYXWIZ_HAS_FFTW)
endif()

# Link DuckDB if available
if(CYXWIZ_HAS_DUCKDB)
    find_package(Threads REQUIRED)
    # DuckDB may provide either 'duckdb' or 'duckdb_static' target
    if(TARGET duckdb)
        target_link_libraries(cyxwiz-backend PRIVATE duckdb Threads::Threads)
    elseif(TARGET duckdb_static)
        target_link_libraries(cyxwiz-backend PRIVATE duckdb_static Threads::Threads)
    else()
        message(WARNING "DuckDB found but no valid target available")
    endif()
    target_compile_definitions(cyxwiz-backend PRIVATE CYXWIZ_HAS_DUCKDB)
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
        python/bindings_tensor.cpp
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
