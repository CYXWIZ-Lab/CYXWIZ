#[=======================================================================[.rst:
FindONNXRuntime
---------------

Find the ONNX Runtime library.

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines the following :ref:`IMPORTED` targets:

``onnxruntime::onnxruntime``
  The ONNX Runtime library, if found.

Result Variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``ONNXRUNTIME_FOUND``
  True if ONNX Runtime is found.
``ONNXRUNTIME_INCLUDE_DIRS``
  Include directories for ONNX Runtime headers.
``ONNXRUNTIME_LIBRARIES``
  Libraries to link against.
``ONNXRUNTIME_VERSION``
  The version of ONNX Runtime found (if available).

Cache Variables
^^^^^^^^^^^^^^^

``ONNXRUNTIME_INCLUDE_DIR``
  The directory containing ONNX Runtime headers.
``ONNXRUNTIME_LIBRARY``
  The path to the ONNX Runtime library.

Cross-Platform Support
^^^^^^^^^^^^^^^^^^^^^^

This module supports:
- Windows: Looks for onnxruntime.lib and onnxruntime.dll
- Linux: Looks for libonnxruntime.so
- macOS: Looks for libonnxruntime.dylib

GPU Provider Support
^^^^^^^^^^^^^^^^^^^^

When ONNX Runtime GPU package is installed, additional provider libraries
may be available:
- CUDA: onnxruntime_providers_cuda
- TensorRT: onnxruntime_providers_tensorrt
- DirectML (Windows): onnxruntime_providers_dml

#]=======================================================================]

# Look for the header file
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}
        $ENV{ONNXRUNTIME_ROOT}
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES
        include
        include/onnxruntime
)

# Look for the library
if(WIN32)
    # Windows: static import library
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS
            ${ONNXRUNTIME_ROOT}
            $ENV{ONNXRUNTIME_ROOT}
            ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES
            lib
            lib/x64
    )

    # Also find the DLL for runtime
    find_file(ONNXRUNTIME_DLL
        NAMES onnxruntime.dll
        PATHS
            ${ONNXRUNTIME_ROOT}
            $ENV{ONNXRUNTIME_ROOT}
            ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES
            bin
            bin/x64
    )

    # Optional: GPU provider libraries
    find_library(ONNXRUNTIME_PROVIDERS_CUDA_LIBRARY
        NAMES onnxruntime_providers_cuda
        PATHS ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES lib
    )

    find_library(ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY
        NAMES onnxruntime_providers_shared
        PATHS ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES lib
    )

elseif(APPLE)
    # macOS: dynamic library
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime libonnxruntime
        PATHS
            ${ONNXRUNTIME_ROOT}
            $ENV{ONNXRUNTIME_ROOT}
            ${CMAKE_PREFIX_PATH}
            /usr/local
            /opt/homebrew
        PATH_SUFFIXES
            lib
            lib64
    )
else()
    # Linux: shared library
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime libonnxruntime
        PATHS
            ${ONNXRUNTIME_ROOT}
            $ENV{ONNXRUNTIME_ROOT}
            ${CMAKE_PREFIX_PATH}
            /usr/local
            /usr
        PATH_SUFFIXES
            lib
            lib64
            lib/x86_64-linux-gnu
    )

    # Optional: GPU provider libraries
    find_library(ONNXRUNTIME_PROVIDERS_CUDA_LIBRARY
        NAMES onnxruntime_providers_cuda
        PATHS ${CMAKE_PREFIX_PATH} /usr/local /usr
        PATH_SUFFIXES lib lib64
    )
endif()

# Try to get version from header
if(ONNXRUNTIME_INCLUDE_DIR AND EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
    file(STRINGS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h" _ort_version_line
         REGEX "^#define[ \t]+ORT_API_VERSION[ \t]+[0-9]+")
    if(_ort_version_line)
        string(REGEX REPLACE "^#define[ \t]+ORT_API_VERSION[ \t]+([0-9]+).*" "\\1"
               ONNXRUNTIME_API_VERSION "${_ort_version_line}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS
        ONNXRUNTIME_LIBRARY
        ONNXRUNTIME_INCLUDE_DIR
    VERSION_VAR
        ONNXRUNTIME_API_VERSION
)

if(ONNXRuntime_FOUND AND NOT TARGET onnxruntime::onnxruntime)
    # Create imported target
    if(WIN32)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
            IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
    else()
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
    endif()

    # Add CUDA provider target if found
    if(ONNXRUNTIME_PROVIDERS_CUDA_LIBRARY)
        add_library(onnxruntime::providers_cuda SHARED IMPORTED)
        if(WIN32)
            set_target_properties(onnxruntime::providers_cuda PROPERTIES
                IMPORTED_IMPLIB "${ONNXRUNTIME_PROVIDERS_CUDA_LIBRARY}"
            )
        else()
            set_target_properties(onnxruntime::providers_cuda PROPERTIES
                IMPORTED_LOCATION "${ONNXRUNTIME_PROVIDERS_CUDA_LIBRARY}"
            )
        endif()
        set(ONNXRUNTIME_HAS_CUDA TRUE)
    endif()

    # Add shared provider target if found (Windows)
    if(ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY)
        add_library(onnxruntime::providers_shared SHARED IMPORTED)
        set_target_properties(onnxruntime::providers_shared PROPERTIES
            IMPORTED_IMPLIB "${ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY}"
        )
    endif()
endif()

# Set output variables
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})

mark_as_advanced(
    ONNXRUNTIME_INCLUDE_DIR
    ONNXRUNTIME_LIBRARY
    ONNXRUNTIME_DLL
    ONNXRUNTIME_PROVIDERS_CUDA_LIBRARY
    ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY
)
