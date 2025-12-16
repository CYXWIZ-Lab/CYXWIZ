cmake_minimum_required(VERSION 3.20)

project(CyxWiz
    VERSION 0.1.0
    DESCRIPTION "CyxWiz - Decentralized ML Compute Platform"
    LANGUAGES CXX C
)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add custom cmake modules path (for FindONNXRuntime, etc.)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build options
option(CYXWIZ_BUILD_ENGINE "Build CyxWiz Engine (Desktop Client)" ON)
option(CYXWIZ_BUILD_SERVER_NODE "Build CyxWiz Server Node" ON)
option(CYXWIZ_BUILD_CENTRAL_SERVER "Build CyxWiz Central Server" ON)
option(CYXWIZ_BUILD_TESTS "Build tests" ON)
option(CYXWIZ_BUILD_DOCS "Build documentation" OFF)
option(CYXWIZ_ENABLE_CUDA "Enable CUDA support for ArrayFire" OFF)
option(CYXWIZ_ENABLE_OPENCL "Enable OpenCL support for ArrayFire" ON)
option(CYXWIZ_ENABLE_ONNX "Enable ONNX Runtime support" ON)
option(CYXWIZ_ENABLE_GGUF "Enable GGUF/llama.cpp support for LLMs" ON)
option(CYXWIZ_ENABLE_PYTORCH "Enable PyTorch/LibTorch support" ON)
option(CYXWIZ_ANDROID_BUILD "Build for Android" OFF)

# Platform detection
if(WIN32)
    set(CYXWIZ_PLATFORM "Windows")
    add_compile_definitions(CYXWIZ_PLATFORM_WINDOWS)
elseif(APPLE)
    set(CYXWIZ_PLATFORM "macOS")
    add_compile_definitions(CYXWIZ_PLATFORM_MACOS)
elseif(ANDROID)
    set(CYXWIZ_PLATFORM "Android")
    add_compile_definitions(CYXWIZ_PLATFORM_ANDROID)
    set(CYXWIZ_ANDROID_BUILD ON)
elseif(UNIX)
    set(CYXWIZ_PLATFORM "Linux")
    add_compile_definitions(CYXWIZ_PLATFORM_LINUX)
endif()

message(STATUS "Building CyxWiz for ${CYXWIZ_PLATFORM}")

# Build type specific flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_definitions(CYXWIZ_DEBUG)
    add_compile_definitions(CYXWIZ_ENABLE_LOGGING)
    add_compile_definitions(CYXWIZ_ENABLE_PROFILING)
    message(STATUS "Debug build - Logging and profiling enabled")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_definitions(CYXWIZ_RELEASE)
    add_compile_definitions(NDEBUG)
    message(STATUS "Release build - Optimizations enabled")
endif()

# Compiler-specific flags
if(MSVC)
    add_compile_options(/W4 /WX- /MP)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3)
    endif()
endif()

# Find vcpkg packages
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ONNX Runtime (optional)
# Cross-platform support:
#   Windows: vcpkg install onnxruntime-gpu (includes CUDA/TensorRT providers)
#   Linux:   vcpkg install onnxruntime or install from https://github.com/microsoft/onnxruntime/releases
#   macOS:   vcpkg install onnxruntime or brew install onnxruntime
if(CYXWIZ_ENABLE_ONNX)
    # First try vcpkg CONFIG mode
    find_package(onnxruntime CONFIG QUIET)
    if(onnxruntime_FOUND)
        message(STATUS "ONNX Runtime found (CONFIG) - ONNX support enabled")
        set(CYXWIZ_HAS_ONNX ON CACHE BOOL "ONNX Runtime available" FORCE)
    else()
        # Fallback to custom FindONNXRuntime module (for vcpkg onnxruntime-gpu which lacks cmake config)
        find_package(ONNXRuntime QUIET)
        if(ONNXRuntime_FOUND)
            message(STATUS "ONNX Runtime found (MODULE) - ONNX support enabled")
            set(CYXWIZ_HAS_ONNX ON CACHE BOOL "ONNX Runtime available" FORCE)
            if(ONNXRUNTIME_HAS_CUDA)
                message(STATUS "  CUDA provider available")
            endif()
        else()
            message(WARNING "ONNX Runtime not found - ONNX support disabled")
            message(STATUS "Install via:")
            message(STATUS "  Windows: vcpkg install onnxruntime-gpu")
            message(STATUS "  Linux:   vcpkg install onnxruntime")
            message(STATUS "  macOS:   vcpkg install onnxruntime or brew install onnxruntime")
            set(CYXWIZ_HAS_ONNX OFF CACHE BOOL "ONNX Runtime available" FORCE)
        endif()
    endif()

    # ONNX protobuf definitions for export (separate from runtime)
    find_package(ONNX CONFIG QUIET)
    if(ONNX_FOUND)
        message(STATUS "ONNX protobuf found - ONNX export enabled")
        set(CYXWIZ_HAS_ONNX_EXPORT ON CACHE BOOL "ONNX export available" FORCE)
    else()
        set(CYXWIZ_HAS_ONNX_EXPORT OFF CACHE BOOL "ONNX export available" FORCE)
    endif()
else()
    set(CYXWIZ_HAS_ONNX OFF CACHE BOOL "ONNX Runtime available" FORCE)
    set(CYXWIZ_HAS_ONNX_EXPORT OFF CACHE BOOL "ONNX export available" FORCE)
endif()

# llama.cpp for GGUF model support (LLMs)
# Cross-platform support:
#   Windows: vcpkg install llama-cpp[cuda] (CUDA backend)
#   Linux:   vcpkg install llama-cpp (CPU) or llama-cpp[cuda]
#   macOS:   vcpkg install llama-cpp[metal] (Metal backend)
if(CYXWIZ_ENABLE_GGUF)
    # First try vcpkg CONFIG mode
    find_package(llama CONFIG QUIET)
    if(llama_FOUND)
        message(STATUS "llama.cpp found (CONFIG) - GGUF support enabled")
        set(CYXWIZ_HAS_GGUF ON CACHE BOOL "llama.cpp available" FORCE)
    else()
        # Fallback to custom FindLlamaCpp module
        find_package(LlamaCpp QUIET)
        if(LlamaCpp_FOUND)
            message(STATUS "llama.cpp found (MODULE) - GGUF support enabled")
            set(CYXWIZ_HAS_GGUF ON CACHE BOOL "llama.cpp available" FORCE)
            if(LLAMACPP_HAS_CUDA)
                message(STATUS "  CUDA backend available")
            endif()
            if(LLAMACPP_HAS_METAL)
                message(STATUS "  Metal backend available")
            endif()
            if(LLAMACPP_HAS_VULKAN)
                message(STATUS "  Vulkan backend available")
            endif()
        else()
            message(WARNING "llama.cpp not found - GGUF support disabled")
            message(STATUS "Install via:")
            message(STATUS "  Windows: vcpkg install llama-cpp[cuda]")
            message(STATUS "  Linux:   vcpkg install llama-cpp")
            message(STATUS "  macOS:   vcpkg install llama-cpp[metal]")
            set(CYXWIZ_HAS_GGUF OFF CACHE BOOL "llama.cpp available" FORCE)
        endif()
    endif()
else()
    set(CYXWIZ_HAS_GGUF OFF CACHE BOOL "llama.cpp available" FORCE)
endif()

# LibTorch for PyTorch/TorchScript model support
# Cross-platform support:
#   Windows: Download from pytorch.org, set TORCH_DIR environment variable
#   Linux:   Download from pytorch.org or conda install pytorch
#   macOS:   Download from pytorch.org
# Note: LibTorch provides TorchConfig.cmake, so CONFIG mode is preferred
if(CYXWIZ_ENABLE_PYTORCH)
    # First try CMake CONFIG mode (official LibTorch provides TorchConfig.cmake)
    find_package(Torch CONFIG QUIET HINTS
        $ENV{TORCH_DIR}
        ${TORCH_DIR}
    )
    if(Torch_FOUND)
        message(STATUS "LibTorch found (CONFIG) - PyTorch support enabled")
        set(CYXWIZ_HAS_PYTORCH ON CACHE BOOL "LibTorch available" FORCE)
        if(TORCH_CUDA_LIBRARIES)
            message(STATUS "  CUDA support available")
        endif()
    else()
        # Fallback to custom FindTorch module
        find_package(Torch QUIET)
        if(Torch_FOUND)
            message(STATUS "LibTorch found (MODULE) - PyTorch support enabled")
            set(CYXWIZ_HAS_PYTORCH ON CACHE BOOL "LibTorch available" FORCE)
            if(TORCH_HAS_CUDA)
                message(STATUS "  CUDA support available")
            endif()
        else()
            message(WARNING "LibTorch not found - PyTorch support disabled")
            message(STATUS "Download from: https://pytorch.org/get-started/locally/")
            message(STATUS "Set TORCH_DIR environment variable or CMAKE_PREFIX_PATH to LibTorch path")
            set(CYXWIZ_HAS_PYTORCH OFF CACHE BOOL "LibTorch available" FORCE)
        endif()
    endif()
else()
    set(CYXWIZ_HAS_PYTORCH OFF CACHE BOOL "LibTorch available" FORCE)
endif()

# ImGui is handled separately (not all features in vcpkg)
find_package(imgui CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})

# Add subdirectories
add_subdirectory(cyxwiz-protocol)
add_subdirectory(cyxwiz-backend)

# CUDA test programs
if(CYXWIZ_ENABLE_CUDA)
    # Full backend test
    add_executable(test_cuda_backend test_cuda_backend.cpp)
    target_link_libraries(test_cuda_backend PRIVATE cyxwiz-backend)
    target_include_directories(test_cuda_backend PRIVATE cyxwiz-backend/include)

    # Simple direct CUDA test
    find_package(CUDAToolkit REQUIRED)
    add_executable(test_cuda_simple test_cuda_simple.cpp)
    target_link_libraries(test_cuda_simple PRIVATE CUDA::cudart)
endif()

# MNIST ONNX test program
# Cross-platform: Windows, Linux, macOS
if(CYXWIZ_HAS_ONNX)
    add_executable(test_mnist_onnx test_mnist_onnx.cpp)
    target_link_libraries(test_mnist_onnx PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(test_mnist_onnx PRIVATE CYXWIZ_HAS_ONNX)

    if(WIN32)
        target_compile_definitions(test_mnist_onnx PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
        )
    endif()

    # Copy ONNX Runtime provider DLLs for CUDA support (Windows)
    if(WIN32)
        # Copy CUDA provider if available
        set(ONNX_CUDA_DLL "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin/onnxruntime_providers_cuda.dll")
        if(EXISTS "${ONNX_CUDA_DLL}")
            add_custom_command(TARGET test_mnist_onnx POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNX_CUDA_DLL}"
                    "$<TARGET_FILE_DIR:test_mnist_onnx>"
                COMMENT "Copying ONNX Runtime CUDA provider DLL"
            )
        endif()

        # Copy TensorRT provider if available
        set(ONNX_TRT_DLL "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin/onnxruntime_providers_tensorrt.dll")
        if(EXISTS "${ONNX_TRT_DLL}")
            add_custom_command(TARGET test_mnist_onnx POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNX_TRT_DLL}"
                    "$<TARGET_FILE_DIR:test_mnist_onnx>"
                COMMENT "Copying ONNX Runtime TensorRT provider DLL"
            )
        endif()
    endif()

    message(STATUS "MNIST ONNX test enabled")
endif()

# Platform-specific builds
if(NOT CYXWIZ_ANDROID_BUILD)
    if(CYXWIZ_BUILD_ENGINE)
        add_subdirectory(cyxwiz-engine)
    endif()

    if(CYXWIZ_BUILD_SERVER_NODE)
        add_subdirectory(cyxwiz-server-node)
    endif()
endif()

# Central server is built with Rust (see cyxwiz-central-server/Cargo.toml)
if(CYXWIZ_BUILD_CENTRAL_SERVER AND NOT CYXWIZ_ANDROID_BUILD)
    message(STATUS "Central Server will be built using Cargo (Rust)")
    message(STATUS "Run: cd cyxwiz-central-server && cargo build --release")
endif()

# Testing
if(CYXWIZ_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 CONFIG REQUIRED)
    add_subdirectory(tests)
endif()

# Documentation
if(CYXWIZ_BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_subdirectory(docs)
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "========== CyxWiz Configuration ==========")
message(STATUS "Platform: ${CYXWIZ_PLATFORM}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  Engine (Desktop Client): ${CYXWIZ_BUILD_ENGINE}")
message(STATUS "  Server Node: ${CYXWIZ_BUILD_SERVER_NODE}")
message(STATUS "  Central Server: ${CYXWIZ_BUILD_CENTRAL_SERVER}")
message(STATUS "  Tests: ${CYXWIZ_BUILD_TESTS}")
message(STATUS "")
message(STATUS "Compute Backends:")
message(STATUS "  CUDA: ${CYXWIZ_ENABLE_CUDA}")
message(STATUS "  OpenCL: ${CYXWIZ_ENABLE_OPENCL}")
message(STATUS "  ONNX Runtime: ${CYXWIZ_HAS_ONNX}")
message(STATUS "  ONNX Export: ${CYXWIZ_HAS_ONNX_EXPORT}")
message(STATUS "  GGUF/llama.cpp: ${CYXWIZ_HAS_GGUF}")
message(STATUS "  PyTorch/LibTorch: ${CYXWIZ_HAS_PYTORCH}")
message(STATUS "  Android Build: ${CYXWIZ_ANDROID_BUILD}")
message(STATUS "==========================================")
message(STATUS "")
