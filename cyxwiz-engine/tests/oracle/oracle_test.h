#pragma once

// The start of every oracle test that computes on an ArrayFire device
// (Engine 2.0 core Phase 1, TOFIX228 1.1; ORA-01, ORA-04): the test and the
// commit it was built from, then exactly the device CYXWIZ_TEST_ARRAYFIRE_BACKEND
// asks for, or a failure.
//
//   oracle: test_computation_truth_spatial_layers, commit 1a2b...
//   device: cuda 0 (float64)
//
// A test that goes over its devices itself calls PrintOracleHeader, prints a
// `device:` line for each device, and runs only the requested one when the
// variable is set.

#include "oracle_header.h"

#include "../computation_truth/test_device_selection.h"

namespace cyxwiz::test {

inline bool BeginOracleTest(const char* name) {
    PrintOracleHeader(name);
    return SelectTestDeviceFromEnvironment();
}

}  // namespace cyxwiz::test
