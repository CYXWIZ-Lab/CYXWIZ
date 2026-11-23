# The MuJoCo SDK for the plugin: the official release archive at one pinned
# version, checked against its published SHA-256, so every machine builds
# against the same MuJoCo the Python side pins (mujoco==CYXWIZ_MUJOCO_VERSION).
#
# Defines the imported target mujoco::mujoco and, for packaging:
#   CYXWIZ_MUJOCO_RUNTIME   the shared library to ship next to the plugin
#   CYXWIZ_MUJOCO_SDK_DIR   the extracted release (LICENSE, THIRD_PARTY_NOTICES.txt, model/)
#
# MUJOCO_DIR (cache path) points at an already extracted release of the same
# version instead, for offline builds and for macOS (the macOS release is a
# .dmg, which CMake cannot extract). The folder has include/mujoco/ and lib/
# with the shared library; on macOS copy the headers and
# libmujoco.<version>.dylib out of mujoco.framework into that layout.

include_guard(GLOBAL)
include(FetchContent)

set(CYXWIZ_MUJOCO_VERSION "3.17.0")
set(MUJOCO_DIR "" CACHE PATH "Extracted MuJoCo ${CYXWIZ_MUJOCO_VERSION} release to use instead of downloading it")

if(MUJOCO_DIR)
    set(CYXWIZ_MUJOCO_SDK_DIR "${MUJOCO_DIR}")
else()
    set(_mujoco_release "https://github.com/google-deepmind/mujoco/releases/download/${CYXWIZ_MUJOCO_VERSION}")
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_mujoco_archive "mujoco-${CYXWIZ_MUJOCO_VERSION}-windows-x86_64.zip")
        set(_mujoco_sha256 "84d92054e767d0b24b7a8a5c7f3e34e883eb8e4bf7e8eefd7604dd09d999a517")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(_mujoco_archive "mujoco-${CYXWIZ_MUJOCO_VERSION}-linux-x86_64.tar.gz")
        set(_mujoco_sha256 "b3246fc698ca27e05ce36047fe2b6866fe23c9e95a26b3a84dd48f011b85e3e4")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_mujoco_archive "mujoco-${CYXWIZ_MUJOCO_VERSION}-linux-aarch64.tar.gz")
        set(_mujoco_sha256 "55467a37b74e7d943899768aee820e6754a66811ae5ffa17d2768f0c40fc983a")
    else()
        message(FATAL_ERROR
            "No MuJoCo ${CYXWIZ_MUJOCO_VERSION} download for ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}.\n"
            "Extract the release from ${_mujoco_release} and set -DMUJOCO_DIR=<folder>.")
    endif()
    FetchContent_Declare(cyxwiz_mujoco_sdk
        URL "${_mujoco_release}/${_mujoco_archive}"
        URL_HASH SHA256=${_mujoco_sha256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        # The release has no CMake project to add; only its files are used.
        SOURCE_SUBDIR cmake-project-not-used)
    FetchContent_MakeAvailable(cyxwiz_mujoco_sdk)
    set(CYXWIZ_MUJOCO_SDK_DIR "${cyxwiz_mujoco_sdk_SOURCE_DIR}")
endif()

if(NOT EXISTS "${CYXWIZ_MUJOCO_SDK_DIR}/include/mujoco/mujoco.h")
    message(FATAL_ERROR "MuJoCo SDK at '${CYXWIZ_MUJOCO_SDK_DIR}' has no include/mujoco/mujoco.h")
endif()
file(STRINGS "${CYXWIZ_MUJOCO_SDK_DIR}/include/mujoco/mujoco.h" _mujoco_header_version
     REGEX "^#define mjVERSION_HEADER [0-9]+")
string(REGEX REPLACE "^#define mjVERSION_HEADER ([0-9]+).*" "\\1" _mujoco_header_version "${_mujoco_header_version}")
string(REPLACE "." ";" _mujoco_parts "${CYXWIZ_MUJOCO_VERSION}")
list(GET _mujoco_parts 0 _mujoco_major)
list(GET _mujoco_parts 1 _mujoco_minor)
list(GET _mujoco_parts 2 _mujoco_patch)
math(EXPR _mujoco_expected "${_mujoco_major} * 1000000 + ${_mujoco_minor} * 1000 + ${_mujoco_patch}")
if(NOT _mujoco_header_version EQUAL _mujoco_expected)
    message(FATAL_ERROR "MuJoCo SDK at '${CYXWIZ_MUJOCO_SDK_DIR}' is header version ${_mujoco_header_version}, "
                        "expected ${CYXWIZ_MUJOCO_VERSION} (${_mujoco_expected})")
endif()

# The release's include/ also carries Dear ImGui and ImPlot headers; only
# include/mujoco is staged, so the Engine's own ImGui stays the one compiled.
set(_mujoco_include "${CMAKE_BINARY_DIR}/mujoco-sdk-${CYXWIZ_MUJOCO_VERSION}/include")
file(COPY "${CYXWIZ_MUJOCO_SDK_DIR}/include/mujoco" DESTINATION "${_mujoco_include}")

if(WIN32)
    set(CYXWIZ_MUJOCO_RUNTIME "${CYXWIZ_MUJOCO_SDK_DIR}/bin/mujoco.dll")
    set(_mujoco_implib "${CYXWIZ_MUJOCO_SDK_DIR}/lib/mujoco.lib")
elseif(APPLE)
    set(CYXWIZ_MUJOCO_RUNTIME "${CYXWIZ_MUJOCO_SDK_DIR}/lib/libmujoco.${CYXWIZ_MUJOCO_VERSION}.dylib")
else()
    set(CYXWIZ_MUJOCO_RUNTIME "${CYXWIZ_MUJOCO_SDK_DIR}/lib/libmujoco.so.${CYXWIZ_MUJOCO_VERSION}")
endif()
if(NOT EXISTS "${CYXWIZ_MUJOCO_RUNTIME}")
    message(FATAL_ERROR "MuJoCo SDK at '${CYXWIZ_MUJOCO_SDK_DIR}' has no ${CYXWIZ_MUJOCO_RUNTIME}")
endif()

add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${CYXWIZ_MUJOCO_RUNTIME}"
    INTERFACE_INCLUDE_DIRECTORIES "${_mujoco_include}")
if(WIN32)
    set_target_properties(mujoco::mujoco PROPERTIES IMPORTED_IMPLIB "${_mujoco_implib}")
endif()
message(STATUS "MuJoCo ${CYXWIZ_MUJOCO_VERSION}: ${CYXWIZ_MUJOCO_SDK_DIR}")
