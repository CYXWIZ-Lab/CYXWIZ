#pragma once

// A matrix times a column vector on the active ArrayFire backend (ORA-14).
//
// ArrayFire 3.10's OpenCL backend (NVIDIA OpenCL) returns NaN in some rows of a
// float64 matrix-vector product: a [120, 2] design matrix times its [2, 1]
// coefficients came back with 6 to 118 NaN rows, varying from run to run,
// while the same product as an element-wise multiply and a row sum is exact.
// On OpenCL the product is formed that way; every other backend calls
// af::matmul.

#include <arrayfire.h>

namespace cyxwiz {

// m [rows, cols] times v [cols, 1] -> [rows, 1]; with transpose_m, m' v -> [cols, 1].
inline af::array DeviceMatVec(const af::array& m, const af::array& v, bool transpose_m = false) {
    if (af::getActiveBackend() != AF_BACKEND_OPENCL)
        return af::matmul(m, v, transpose_m ? AF_MAT_TRANS : AF_MAT_NONE, AF_MAT_NONE);
    const unsigned rows = static_cast<unsigned>(m.dims(0));
    const unsigned cols = static_cast<unsigned>(m.dims(1));
    if (transpose_m) return af::sum(m * af::tile(af::flat(v), 1, cols), 0).T();
    return af::sum(m * af::tile(af::flat(v).T(), rows), 1);
}

} // namespace cyxwiz
