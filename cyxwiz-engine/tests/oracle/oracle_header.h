#pragma once

// The first line of every oracle test (TOFIX228 1.1, ORA-01): which test and
// which commit it was built from.
//
//   oracle: test_computation_truth_spatial_layers, commit 1a2b... (with uncommitted changes)
//
// No backend dependency: a host-only oracle (no device) includes only this.
// The commit comes from the cyxwiz-build-info library, which the CMake
// function cyxwiz_oracle_test() links.

#include <cstdio>

extern "C" const char* cyxwiz_build_commit();
extern "C" int cyxwiz_build_modified();

namespace cyxwiz::test {

inline void PrintOracleHeader(const char* name) {
    std::printf("oracle: %s, commit %s%s\n", name, cyxwiz_build_commit(),
                cyxwiz_build_modified() ? " (with uncommitted changes)" : "");
    std::fflush(stdout);
}

// For an oracle that computes on the host only (no ArrayFire device).
inline void BeginHostOracleTest(const char* name) {
    PrintOracleHeader(name);
    std::printf("device: host (this test uses no ArrayFire device)\n");
    std::fflush(stdout);
}

}  // namespace cyxwiz::test
