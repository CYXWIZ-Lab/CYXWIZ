#pragma once

// Least squares on the device by the SVD, as LAPACK's gelsd (numpy's and
// torch's default): x = V diag(1 / s) U^T b over the singular values above
// max(m, n) * eps * s_max (eps of the array's precision), so a matrix whose
// columns depend on each other gets the shortest of its best x, where a QR
// route breaks down. Shared by both LinearAlgebra::LeastSquares overloads.

#include <algorithm>
#include <limits>

#include <arrayfire.h>

namespace cyxwiz::detail {

inline af::array LeastSquaresSvd(const af::array& a, const af::array& b) {
    af::array u, s, vt;
    af::svd(u, s, vt, a);
    const dim_t k = s.dims(0);
    const double eps = a.type() == f64 ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon();
    const double cutoff = static_cast<double>(std::max(a.dims(0), a.dims(1))) * eps * af::max<double>(s);
    const af::array kept = (s > cutoff).as(s.type());
    const af::array inverse = kept / (s + (1.0 - kept));  // 1 / s where kept, 0 elsewhere
    const af::array ub = af::matmul(u(af::span, af::seq(0, static_cast<double>(k - 1))), b, AF_MAT_TRANS, AF_MAT_NONE);
    const af::array scaled = ub * af::tile(inverse, 1, static_cast<unsigned>(b.dims(1)));
    return af::matmul(vt(af::seq(0, static_cast<double>(k - 1)), af::span), scaled, AF_MAT_TRANS, AF_MAT_NONE);
}

}  // namespace cyxwiz::detail
