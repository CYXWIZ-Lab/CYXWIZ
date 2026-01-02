# FindNCCL.cmake
# ---------------
# Find NVIDIA Collective Communications Library (NCCL)
#
# This module finds the NCCL library for high-performance multi-GPU and
# multi-node collective communication primitives.
#
# NCCL is typically installed with CUDA or can be installed separately.
# Download from: https://developer.nvidia.com/nccl
#
# This module defines:
#   NCCL_FOUND        - True if NCCL was found
#   NCCL_INCLUDE_DIRS - Include directories for NCCL
#   NCCL_LIBRARIES    - Libraries to link against
#   NCCL_VERSION      - Version string (if available)
#
# This module creates the following imported target:
#   NCCL::nccl        - The NCCL library
#
# Hints:
#   NCCL_ROOT         - Root directory of NCCL installation
#   NCCL_DIR          - Alias for NCCL_ROOT
#   CUDA_TOOLKIT_ROOT_DIR - NCCL is often installed with CUDA

include(FindPackageHandleStandardArgs)

# Look for NCCL in common locations
set(_NCCL_SEARCH_PATHS
    ${NCCL_ROOT}
    ${NCCL_DIR}
    $ENV{NCCL_ROOT}
    $ENV{NCCL_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}
    $ENV{CUDA_PATH}
    $ENV{CUDA_HOME}
    /usr/local/cuda
    /usr/local
    /usr
    /opt/nccl
)

# Find include directory
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS ${_NCCL_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "NCCL include directory"
)

# Find library
# NCCL library naming:
#   - Linux: libnccl.so, libnccl.so.2, libnccl_static.a
#   - Windows: nccl.lib, nccl.dll (less common, NCCL is primarily Linux)
if(WIN32)
    set(_NCCL_LIB_NAMES nccl nccl64)
else()
    set(_NCCL_LIB_NAMES nccl)
endif()

find_library(NCCL_LIBRARY
    NAMES ${_NCCL_LIB_NAMES}
    PATHS ${_NCCL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x64
    DOC "NCCL library"
)

# Extract version from nccl.h
if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _NCCL_VERSION_LINES
        REGEX "#define NCCL_(MAJOR|MINOR|PATCH)")

    foreach(_line ${_NCCL_VERSION_LINES})
        if(_line MATCHES "#define NCCL_MAJOR[ \t]+([0-9]+)")
            set(NCCL_VERSION_MAJOR "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "#define NCCL_MINOR[ \t]+([0-9]+)")
            set(NCCL_VERSION_MINOR "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "#define NCCL_PATCH[ \t]+([0-9]+)")
            set(NCCL_VERSION_PATCH "${CMAKE_MATCH_1}")
        endif()
    endforeach()

    if(NCCL_VERSION_MAJOR AND NCCL_VERSION_MINOR AND NCCL_VERSION_PATCH)
        set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
    elseif(NCCL_VERSION_MAJOR AND NCCL_VERSION_MINOR)
        set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}")
    endif()
endif()

# Handle standard find_package arguments
find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
    VERSION_VAR NCCL_VERSION
)

# Set output variables
if(NCCL_FOUND)
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})

    # Create imported target
    if(NOT TARGET NCCL::nccl)
        add_library(NCCL::nccl UNKNOWN IMPORTED)
        set_target_properties(NCCL::nccl PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
    endif()

    # Log what we found
    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
    if(NCCL_VERSION)
        message(STATUS "  NCCL version: ${NCCL_VERSION}")
    endif()
endif()

# Hide internal variables
mark_as_advanced(
    NCCL_INCLUDE_DIR
    NCCL_LIBRARY
)
