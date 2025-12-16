#[=======================================================================[.rst:
FindTorch
---------

Find the LibTorch (PyTorch C++ API) library.

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines the following :ref:`IMPORTED` targets:

``Torch::Torch``
  The LibTorch library, if found.

Result Variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``Torch_FOUND``
  True if LibTorch is found.
``TORCH_INCLUDE_DIRS``
  Include directories for LibTorch headers.
``TORCH_LIBRARIES``
  Libraries to link against.

Cache Variables
^^^^^^^^^^^^^^^

``TORCH_INCLUDE_DIR``
  The directory containing LibTorch headers.
``TORCH_LIBRARY``
  The path to the LibTorch library.

Cross-Platform Support
^^^^^^^^^^^^^^^^^^^^^^

This module supports:
- Windows: Looks for torch.lib in libtorch/lib
- Linux: Looks for libtorch.so
- macOS: Looks for libtorch.dylib

LibTorch can be downloaded from: https://pytorch.org/get-started/locally/

Set TORCH_DIR environment variable or CMAKE_PREFIX_PATH to LibTorch path.

#]=======================================================================]

# First, try CMake CONFIG mode (official LibTorch provides TorchConfig.cmake)
# This is the preferred method when LibTorch is properly installed
if(NOT Torch_FOUND)
    find_package(Torch CONFIG QUIET HINTS
        $ENV{TORCH_DIR}
        ${TORCH_DIR}
        ${CMAKE_PREFIX_PATH}
    )
    if(Torch_FOUND)
        # Torch CONFIG mode found - targets already defined
        set(TORCH_FOUND TRUE)
        message(STATUS "LibTorch found via CONFIG mode")
        return()
    endif()
endif()

# Fallback: Manual search for LibTorch
message(STATUS "LibTorch CONFIG not found, trying manual search...")

# Look for the header file
find_path(TORCH_INCLUDE_DIR
    NAMES torch/torch.h
    PATHS
        ${TORCH_DIR}
        $ENV{TORCH_DIR}
        ${CMAKE_PREFIX_PATH}
        /opt/libtorch
        /usr/local/libtorch
        C:/libtorch
    PATH_SUFFIXES
        include
        include/torch/csrc/api/include
)

# Look for the library
if(WIN32)
    # Windows: Find import libraries
    find_library(TORCH_LIBRARY
        NAMES torch torch_cpu c10
        PATHS
            ${TORCH_DIR}
            $ENV{TORCH_DIR}
            ${CMAKE_PREFIX_PATH}
            C:/libtorch
        PATH_SUFFIXES
            lib
    )

    # Also find torch_cuda for GPU support
    find_library(TORCH_CUDA_LIBRARY
        NAMES torch_cuda
        PATHS
            ${TORCH_DIR}
            $ENV{TORCH_DIR}
            ${CMAKE_PREFIX_PATH}
            C:/libtorch
        PATH_SUFFIXES
            lib
    )

    # Find c10 library (core)
    find_library(C10_LIBRARY
        NAMES c10
        PATHS
            ${TORCH_DIR}
            $ENV{TORCH_DIR}
            ${CMAKE_PREFIX_PATH}
            C:/libtorch
        PATH_SUFFIXES
            lib
    )

elseif(APPLE)
    # macOS: dynamic library
    find_library(TORCH_LIBRARY
        NAMES torch torch_cpu libtorch
        PATHS
            ${TORCH_DIR}
            $ENV{TORCH_DIR}
            ${CMAKE_PREFIX_PATH}
            /opt/libtorch
            /usr/local/libtorch
            /opt/homebrew
        PATH_SUFFIXES
            lib
            lib64
    )

    find_library(C10_LIBRARY
        NAMES c10
        PATHS
            ${TORCH_DIR}
            $ENV{TORCH_DIR}
            ${CMAKE_PREFIX_PATH}
            /opt/libtorch
            /usr/local/libtorch
        PATH_SUFFIXES
            lib
    )

else()
    # Linux: shared library
    find_library(TORCH_LIBRARY
        NAMES torch torch_cpu libtorch
        PATHS
            ${TORCH_DIR}
            $ENV{TORCH_DIR}
            ${CMAKE_PREFIX_PATH}
            /opt/libtorch
            /usr/local/libtorch
            /usr/local
            /usr
        PATH_SUFFIXES
            lib
            lib64
    )

    find_library(C10_LIBRARY
        NAMES c10
        PATHS
            ${TORCH_DIR}
            $ENV{TORCH_DIR}
            ${CMAKE_PREFIX_PATH}
            /opt/libtorch
            /usr/local/libtorch
        PATH_SUFFIXES
            lib
            lib64
    )

    # Optional: CUDA library
    find_library(TORCH_CUDA_LIBRARY
        NAMES torch_cuda
        PATHS
            ${TORCH_DIR}
            $ENV{TORCH_DIR}
            ${CMAKE_PREFIX_PATH}
            /opt/libtorch
        PATH_SUFFIXES
            lib
            lib64
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Torch
    REQUIRED_VARS
        TORCH_LIBRARY
        TORCH_INCLUDE_DIR
)

if(Torch_FOUND AND NOT TARGET Torch::Torch)
    # Create imported target
    add_library(Torch::Torch INTERFACE IMPORTED)

    # Collect all libraries
    set(_torch_libs ${TORCH_LIBRARY})
    if(C10_LIBRARY)
        list(APPEND _torch_libs ${C10_LIBRARY})
    endif()
    if(TORCH_CUDA_LIBRARY)
        list(APPEND _torch_libs ${TORCH_CUDA_LIBRARY})
        set(TORCH_HAS_CUDA TRUE)
    endif()

    set_target_properties(Torch::Torch PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${_torch_libs}"
    )

    # On Windows, need to set some compile definitions
    if(WIN32)
        set_property(TARGET Torch::Torch APPEND PROPERTY
            INTERFACE_COMPILE_DEFINITIONS "NOMINMAX")
    endif()

    # Set CXX ABI flag (important for Linux)
    if(UNIX AND NOT APPLE)
        # LibTorch pre-built uses old ABI by default
        set_property(TARGET Torch::Torch APPEND PROPERTY
            INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()

# Set output variables
set(TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIR})
set(TORCH_LIBRARIES ${TORCH_LIBRARY})
if(C10_LIBRARY)
    list(APPEND TORCH_LIBRARIES ${C10_LIBRARY})
endif()
if(TORCH_CUDA_LIBRARY)
    list(APPEND TORCH_LIBRARIES ${TORCH_CUDA_LIBRARY})
endif()

mark_as_advanced(
    TORCH_INCLUDE_DIR
    TORCH_LIBRARY
    TORCH_CUDA_LIBRARY
    C10_LIBRARY
)
