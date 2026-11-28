#pragma once

// Every ArrayFire matrix product goes through DeviceMatMul (TOFIX242 2.2,
// ORA-14).
//
// ArrayFire 3.10's OpenCL backend forms a product whose result has ONE column
// with its BLAS's matrix-vector routine (CLBlast, built into afopencl.dll):
// y = A x + beta y with beta 0. That routine reads y anyway: the result buffer,
// handed out by ArrayFire's memory pool with whatever its last user left in it,
// so 0 x NaN = NaN (and stale infinities) come out in the product. The BLAS
// rule (beta 0: y is not read) is kept by cuBLAS (CUDA) and MKL (CPU). Seen as:
// a model whose weights went NaN, freed, made the next model's Dense 1 output
// NaN; earlier as random NaN rows of a float64 regression (ORA-14).
//
// On OpenCL a one-column product takes a zero second column (a zero second row
// for a transposed right side): the matrix-matrix routine, which is right; the
// first column is the product. Every other product, and every other backend,
// calls af::matmul as it is. The probe: scripts/arrayfire_probes/
// matmul_reused_buffer. The source scan (tests/unit/
// test_arrayfire_fallback_source_scan.cpp) refuses a direct af::matmul /
// matmulNT / matmulTN / matmulTT anywhere else.

#include <arrayfire.h>

namespace cyxwiz {

inline af::array DeviceMatMul(const af::array& lhs, const af::array& rhs, af::matProp lhs_op = AF_MAT_NONE,
                              af::matProp rhs_op = AF_MAT_NONE) {
    const bool rhs_transposed = rhs_op == AF_MAT_TRANS || rhs_op == AF_MAT_CTRANS;
    const dim_t result_columns = rhs_transposed ? rhs.dims(0) : rhs.dims(1);
    if (result_columns != 1 || af::getActiveBackend() != AF_BACKEND_OPENCL)
        return af::matmul(lhs, rhs, lhs_op, rhs_op);
    const af::dim4 d = rhs.dims();
    const af::array widened = rhs_transposed ? af::join(0, rhs, af::constant(0, 1, d[1], d[2], d[3], rhs.type()))
                                             : af::join(1, rhs, af::constant(0, d[0], 1, d[2], d[3], rhs.type()));
    const af::array product = af::matmul(lhs, widened, lhs_op, rhs_op);
    return product(af::span, 0, af::span, af::span);
}

// As af::matmulNT / matmulTN / matmulTT.
inline af::array DeviceMatMulNT(const af::array& lhs, const af::array& rhs) {
    return DeviceMatMul(lhs, rhs, AF_MAT_NONE, AF_MAT_TRANS);
}
inline af::array DeviceMatMulTN(const af::array& lhs, const af::array& rhs) {
    return DeviceMatMul(lhs, rhs, AF_MAT_TRANS, AF_MAT_NONE);
}
inline af::array DeviceMatMulTT(const af::array& lhs, const af::array& rhs) {
    return DeviceMatMul(lhs, rhs, AF_MAT_TRANS, AF_MAT_TRANS);
}

}  // namespace cyxwiz
