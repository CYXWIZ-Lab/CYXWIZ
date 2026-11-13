// The lookup the backend uses to reach ArrayFire's per-backend functions
// (afcl_*, afcu_*) in libraries the unified loader already opened, by stem,
// whatever versioned file name and folder they were loaded from.
#include <catch2/catch_test_macros.hpp>

#include "../../cyxwiz-backend/src/core/loaded_library.h"

#include <arrayfire.h>

namespace {

// A library every test process has loaded, under a versioned file name on
// macOS (libSystem.B.dylib) and Linux (libc.so.6), as ArrayFire's are.
#ifdef _WIN32
constexpr const char* kLoadedStem = "kernel32";
constexpr const char* kLoadedSymbol = "GetModuleHandleA";
#elif defined(__APPLE__)
constexpr const char* kLoadedStem = "System";
constexpr const char* kLoadedSymbol = "malloc";
#else
constexpr const char* kLoadedStem = "c";
constexpr const char* kLoadedSymbol = "malloc";
#endif

} // namespace

TEST_CASE("A loaded library is found by its stem", "[loaded_library]") {
    CHECK(cyxwiz::detail::IsLibraryLoaded(kLoadedStem));
    CHECK(cyxwiz::detail::FindLoadedSymbol(kLoadedStem, kLoadedSymbol) != nullptr);
}

TEST_CASE("A missing library or symbol is not found", "[loaded_library]") {
    CHECK_FALSE(cyxwiz::detail::IsLibraryLoaded("cyxwiz-no-such-library"));
    CHECK(cyxwiz::detail::FindLoadedSymbol("cyxwiz-no-such-library", kLoadedSymbol) == nullptr);
    CHECK(cyxwiz::detail::FindLoadedSymbol(kLoadedStem, "cyxwiz_no_such_symbol") == nullptr);
}

TEST_CASE("The active ArrayFire OpenCL backend's functions are found", "[loaded_library]") {
    if ((af::getAvailableBackends() & AF_BACKEND_OPENCL) == 0) {
        SKIP("ArrayFire's OpenCL backend is not installed here");
    }
    const af::Backend previous = af::getActiveBackend();
    af::setBackend(AF_BACKEND_OPENCL);
    CHECK(cyxwiz::detail::IsLibraryLoaded("afopencl"));
    CHECK(cyxwiz::detail::FindLoadedSymbol("afopencl", "afcl_get_device_id") != nullptr);
    af::setBackend(previous);
}
