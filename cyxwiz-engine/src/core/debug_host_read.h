#pragma once

// A debugger view's bounded read of device values to the host, through the
// shared boundary so the copy is attributed (no direct af::array::host).

#include "algorithms/arrayfire_host_materialization.h"

#include <arrayfire.h>

namespace cyxwiz {

// `out` holds values.elements() floats, in ArrayFire's column-major order.
inline void ReadDebugValues(const af::array& values, float* out, const char* where, const char* what) {
    MaterializeArrayFireToHost(values.as(f32), out, ArrayFireHostSyncCategory::DebugSampleDump, where,
                               "arrayfire_column_major", "bounded_debug_readback", what);
}

}  // namespace cyxwiz
