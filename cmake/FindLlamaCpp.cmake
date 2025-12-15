#[=======================================================================[.rst:
FindLlamaCpp
------------

Find the llama.cpp library for GGUF model support.

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines the following :ref:`IMPORTED` targets:

``llama-cpp::llama``
  The llama.cpp library, if found.

Result Variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``LLAMACPP_FOUND``
  True if llama.cpp is found.
``LLAMACPP_INCLUDE_DIRS``
  Include directories for llama.cpp headers.
``LLAMACPP_LIBRARIES``
  Libraries to link against.
``LLAMACPP_VERSION``
  The version of llama.cpp found (if available).

Cache Variables
^^^^^^^^^^^^^^^

``LLAMACPP_INCLUDE_DIR``
  The directory containing llama.cpp headers.
``LLAMACPP_LIBRARY``
  The path to the llama.cpp library.

GPU Backend Support
^^^^^^^^^^^^^^^^^^^

This module detects available GPU backends:
``LLAMACPP_HAS_CUDA``
  True if CUDA backend is available (NVIDIA GPUs).
``LLAMACPP_HAS_METAL``
  True if Metal backend is available (Apple Silicon).
``LLAMACPP_HAS_VULKAN``
  True if Vulkan backend is available.

Cross-Platform Support
^^^^^^^^^^^^^^^^^^^^^^

This module supports:
- Windows: Looks for llama.lib and llama.dll
- Linux: Looks for libllama.so
- macOS: Looks for libllama.dylib

#]=======================================================================]

# Look for the header file
find_path(LLAMACPP_INCLUDE_DIR
    NAMES llama.h
    PATHS
        ${LLAMACPP_ROOT}
        $ENV{LLAMACPP_ROOT}
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES
        include
        include/llama
        include/llama-cpp
        include/llama.cpp
)

# Look for the library
if(WIN32)
    # Windows: static import library
    find_library(LLAMACPP_LIBRARY
        NAMES llama llama-cpp
        PATHS
            ${LLAMACPP_ROOT}
            $ENV{LLAMACPP_ROOT}
            ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES
            lib
            lib/x64
    )

    # Also find the DLL for runtime
    find_file(LLAMACPP_DLL
        NAMES llama.dll
        PATHS
            ${LLAMACPP_ROOT}
            $ENV{LLAMACPP_ROOT}
            ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES
            bin
            bin/x64
    )

    # Also find ggml library (dependency)
    find_library(GGML_LIBRARY
        NAMES ggml
        PATHS
            ${LLAMACPP_ROOT}
            $ENV{LLAMACPP_ROOT}
            ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES
            lib
            lib/x64
    )

elseif(APPLE)
    # macOS: dynamic library
    find_library(LLAMACPP_LIBRARY
        NAMES llama libllama
        PATHS
            ${LLAMACPP_ROOT}
            $ENV{LLAMACPP_ROOT}
            ${CMAKE_PREFIX_PATH}
            /usr/local
            /opt/homebrew
        PATH_SUFFIXES
            lib
            lib64
    )

    # ggml dependency
    find_library(GGML_LIBRARY
        NAMES ggml libggml
        PATHS
            ${LLAMACPP_ROOT}
            $ENV{LLAMACPP_ROOT}
            ${CMAKE_PREFIX_PATH}
            /usr/local
            /opt/homebrew
        PATH_SUFFIXES
            lib
            lib64
    )
else()
    # Linux: shared library
    find_library(LLAMACPP_LIBRARY
        NAMES llama libllama
        PATHS
            ${LLAMACPP_ROOT}
            $ENV{LLAMACPP_ROOT}
            ${CMAKE_PREFIX_PATH}
            /usr/local
            /usr
        PATH_SUFFIXES
            lib
            lib64
            lib/x86_64-linux-gnu
    )

    # ggml dependency
    find_library(GGML_LIBRARY
        NAMES ggml libggml
        PATHS
            ${LLAMACPP_ROOT}
            $ENV{LLAMACPP_ROOT}
            ${CMAKE_PREFIX_PATH}
            /usr/local
            /usr
        PATH_SUFFIXES
            lib
            lib64
            lib/x86_64-linux-gnu
    )
endif()

# Try to detect GPU backend support by checking for backend-specific symbols or libraries
set(LLAMACPP_HAS_CUDA FALSE)
set(LLAMACPP_HAS_METAL FALSE)
set(LLAMACPP_HAS_VULKAN FALSE)

# Check for CUDA backend (look for ggml-cuda library)
if(WIN32)
    find_library(GGML_CUDA_LIBRARY
        NAMES ggml-cuda
        PATHS ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES lib
    )
else()
    find_library(GGML_CUDA_LIBRARY
        NAMES ggml-cuda libggml-cuda
        PATHS ${CMAKE_PREFIX_PATH} /usr/local /usr
        PATH_SUFFIXES lib lib64
    )
endif()
if(GGML_CUDA_LIBRARY)
    set(LLAMACPP_HAS_CUDA TRUE)
endif()

# Check for Metal backend (macOS only)
if(APPLE)
    find_library(GGML_METAL_LIBRARY
        NAMES ggml-metal libggml-metal
        PATHS ${CMAKE_PREFIX_PATH} /usr/local /opt/homebrew
        PATH_SUFFIXES lib lib64
    )
    if(GGML_METAL_LIBRARY)
        set(LLAMACPP_HAS_METAL TRUE)
    endif()
endif()

# Check for Vulkan backend
find_library(GGML_VULKAN_LIBRARY
    NAMES ggml-vulkan libggml-vulkan
    PATHS ${CMAKE_PREFIX_PATH} /usr/local /usr
    PATH_SUFFIXES lib lib64
)
if(GGML_VULKAN_LIBRARY)
    set(LLAMACPP_HAS_VULKAN TRUE)
endif()

# Try to get version from header
if(LLAMACPP_INCLUDE_DIR AND EXISTS "${LLAMACPP_INCLUDE_DIR}/llama.h")
    file(STRINGS "${LLAMACPP_INCLUDE_DIR}/llama.h" _llama_version_line
         REGEX "^#define[ \t]+LLAMA_BUILD_NUMBER[ \t]+[0-9]+")
    if(_llama_version_line)
        string(REGEX REPLACE "^#define[ \t]+LLAMA_BUILD_NUMBER[ \t]+([0-9]+).*" "\\1"
               LLAMACPP_BUILD_NUMBER "${_llama_version_line}")
        set(LLAMACPP_VERSION "${LLAMACPP_BUILD_NUMBER}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LlamaCpp
    REQUIRED_VARS
        LLAMACPP_LIBRARY
        LLAMACPP_INCLUDE_DIR
    VERSION_VAR
        LLAMACPP_VERSION
)

if(LlamaCpp_FOUND AND NOT TARGET llama-cpp::llama)
    # Create imported target
    if(WIN32)
        add_library(llama-cpp::llama SHARED IMPORTED)
        set_target_properties(llama-cpp::llama PROPERTIES
            IMPORTED_IMPLIB "${LLAMACPP_LIBRARY}"
            IMPORTED_LOCATION "${LLAMACPP_DLL}"
            INTERFACE_INCLUDE_DIRECTORIES "${LLAMACPP_INCLUDE_DIR}"
        )
    else()
        add_library(llama-cpp::llama SHARED IMPORTED)
        set_target_properties(llama-cpp::llama PROPERTIES
            IMPORTED_LOCATION "${LLAMACPP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${LLAMACPP_INCLUDE_DIR}"
        )
    endif()

    # Link ggml as a dependency if found
    if(GGML_LIBRARY AND NOT TARGET llama-cpp::ggml)
        add_library(llama-cpp::ggml SHARED IMPORTED)
        if(WIN32)
            set_target_properties(llama-cpp::ggml PROPERTIES
                IMPORTED_IMPLIB "${GGML_LIBRARY}"
            )
        else()
            set_target_properties(llama-cpp::ggml PROPERTIES
                IMPORTED_LOCATION "${GGML_LIBRARY}"
            )
        endif()
        set_property(TARGET llama-cpp::llama APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES llama-cpp::ggml)
    endif()

    # Add CUDA backend target if found
    if(LLAMACPP_HAS_CUDA AND NOT TARGET llama-cpp::cuda)
        add_library(llama-cpp::cuda SHARED IMPORTED)
        if(WIN32)
            set_target_properties(llama-cpp::cuda PROPERTIES
                IMPORTED_IMPLIB "${GGML_CUDA_LIBRARY}"
            )
        else()
            set_target_properties(llama-cpp::cuda PROPERTIES
                IMPORTED_LOCATION "${GGML_CUDA_LIBRARY}"
            )
        endif()
    endif()

    # Add Metal backend target if found (macOS)
    if(LLAMACPP_HAS_METAL AND NOT TARGET llama-cpp::metal)
        add_library(llama-cpp::metal SHARED IMPORTED)
        set_target_properties(llama-cpp::metal PROPERTIES
            IMPORTED_LOCATION "${GGML_METAL_LIBRARY}"
        )
    endif()

    # Add Vulkan backend target if found
    if(LLAMACPP_HAS_VULKAN AND NOT TARGET llama-cpp::vulkan)
        add_library(llama-cpp::vulkan SHARED IMPORTED)
        if(WIN32)
            set_target_properties(llama-cpp::vulkan PROPERTIES
                IMPORTED_IMPLIB "${GGML_VULKAN_LIBRARY}"
            )
        else()
            set_target_properties(llama-cpp::vulkan PROPERTIES
                IMPORTED_LOCATION "${GGML_VULKAN_LIBRARY}"
            )
        endif()
    endif()
endif()

# Set output variables
set(LLAMACPP_INCLUDE_DIRS ${LLAMACPP_INCLUDE_DIR})
set(LLAMACPP_LIBRARIES ${LLAMACPP_LIBRARY})

mark_as_advanced(
    LLAMACPP_INCLUDE_DIR
    LLAMACPP_LIBRARY
    LLAMACPP_DLL
    GGML_LIBRARY
    GGML_CUDA_LIBRARY
    GGML_METAL_LIBRARY
    GGML_VULKAN_LIBRARY
)
