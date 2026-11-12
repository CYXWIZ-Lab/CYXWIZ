#pragma once

// The SVD of a matrix on the device, for LinearAlgebra and its Tensor
// overloads. af::svd builds the full m × m U, which a tall table of rows does
// not fit on a device (160,000 rows: 200 GB); above kTallRows rows and twice
// the columns the decomposition comes from the n × n Gram matrix A^T A
// instead: its eigenvectors are V, the roots of its eigenvalues the singular
// values, and U = A V / s. A singular value then carries an error of about
// s_max * sqrt(eps), so small ones (a rank decision) are judged at that.
//
// Least squares by the same SVD, as LAPACK's gelsd (numpy's and torch's
// default): x = V diag(1 / s) U^T b over the singular values above
// max(m, n) * eps * s_max (sqrt(eps) on the Gram route), so a matrix whose
// columns depend on each other gets the shortest of its best x.

#include <algorithm>
#include <cmath>
#include <limits>

#include <arrayfire.h>

namespace cyxwiz::detail {

constexpr dim_t kTallRows = 4096;

inline bool UsesGram(const af::array& a) { return a.dims(0) > kTallRows && a.dims(0) > 2 * a.dims(1); }

// The relative size below which a singular value is noise.
inline double SingularNoise(const af::array& a) {
    const double eps = a.type() == f64 ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon();
    return UsesGram(a) ? std::sqrt(eps) : eps;
}

// s descending, vt (k × n) and, when `want_u`, u (m × k): k = min(m, n).
inline void ThinSvd(const af::array& a, af::array& u, af::array& s, af::array& vt, bool want_u) {
    if (!UsesGram(a)) {
        af::svd(u, s, vt, a);
        const dim_t k = s.dims(0);
        u = want_u ? u(af::span, af::seq(0, static_cast<double>(k - 1))) : af::array();
        vt = vt(af::seq(0, static_cast<double>(k - 1)), af::span);
        return;
    }
    const af::array gram = af::matmul(a, a, AF_MAT_TRANS, AF_MAT_NONE);
    af::array gu, gs;
    af::svd(gu, gs, vt, gram);  // symmetric PSD: gs = s^2, the rows of vt are V's columns
    s = af::sqrt(af::max(gs, 0.0));
    if (!want_u) {
        u = af::array();
        return;
    }
    const af::array kept = (s > 0.0).as(s.type());
    const af::array inverse = kept / (s + (1.0 - kept));
    u = af::matmul(a, vt, AF_MAT_NONE, AF_MAT_TRANS) * af::tile(af::transpose(inverse), static_cast<unsigned>(a.dims(0)));
}

inline af::array LeastSquaresSvd(const af::array& a, const af::array& b) {
    af::array u, s, vt;
    ThinSvd(a, u, s, vt, true);
    // LAPACK gelsd's cutoff, eps x max(m, n); on the Gram route (the m x m U is
    // never formed) a small singular value carries sqrt(eps) of the largest,
    // times the columns, as Rank counts them. Not eps x m there: at 1M rows in
    // float32 that would drop every direction under 12 % of the largest.
    const double noise = UsesGram(a) ? SingularNoise(a) * static_cast<double>(a.dims(1))
                                     : static_cast<double>(std::max(a.dims(0), a.dims(1))) * SingularNoise(a);
    const double cutoff = noise * af::max<double>(s);
    const af::array kept = (s > cutoff).as(s.type());
    const af::array inverse = kept / (s + (1.0 - kept));  // 1 / s where kept, 0 elsewhere
    const af::array ub = af::matmul(u, b, AF_MAT_TRANS, AF_MAT_NONE);
    const af::array scaled = ub * af::tile(inverse, 1, static_cast<unsigned>(b.dims(1)));
    return af::matmul(vt, scaled, AF_MAT_TRANS, AF_MAT_NONE);
}

}  // namespace cyxwiz::detail
