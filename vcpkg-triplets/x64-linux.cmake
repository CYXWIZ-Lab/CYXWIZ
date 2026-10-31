# x64-linux for build.sh: vcpkg's own triplet (vcpkg/triplets/x64-linux.cmake)
# plus fixes for new GCC. CI builds with its own triplet settings and does not
# use this folder.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# GCC 15 and newer no longer include <cstdint> through other standard headers,
# and some pinned ports (onnxruntime 1.23, ...) use uint32_t without it;
# forcing the include is harmless on older GCC and clang.
set(VCPKG_C_FLAGS "")  # vcpkg needs both or neither
set(VCPKG_CXX_FLAGS "-include cstdint")

# GCC 16 defaults to C++20. Abseil bakes the standard it was built with into
# absl/base/options.h (std::*_ordering under C++20), and ports that build with
# Abseil as C++17 (onnxruntime 1.23, ...) then fail with "'partial_ordering'
# has not been declared in 'std'". Build Abseil as C++17, as older compilers do.
if(PORT STREQUAL "abseil")
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS -DCMAKE_CXX_STANDARD=17)
endif()
