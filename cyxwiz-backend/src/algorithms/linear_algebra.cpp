// Host-matrix linear algebra (the analysis panels' API) on ArrayFire: the CPU
// option is ArrayFire's CPU backend and there are no hand-written CPU kernels
// (TOFIX140). Matrices cross the host boundary
// once on the way in and once on the way out; float64 where the active device
// has it, else float32.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/linear_algebra.h"
#include "arrayfire_backend_utils.h"
#include "arrayfire_host_values.h"
#include "svd_device.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <arrayfire.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz {

namespace {

using Matrix = std::vector<std::vector<double>>;

bool IsRectangularMatrix(const Matrix& A) {
    if (A.empty() || A[0].empty()) return false;
    for (const auto& row : A) {
        if (row.size() != A[0].size()) return false;
    }
    return true;
}

// Eigenvector of a 2x2 matrix for eigenvalue lambda (closed form, unit length).
std::vector<std::complex<double>> Eigenvector2x2(const Matrix& A, const std::complex<double>& lambda) {
    const std::complex<double> a(A[0][0], 0.0), b(A[0][1], 0.0), c(A[1][0], 0.0), d(A[1][1], 0.0);
    std::vector<std::complex<double>> vector;
    if (std::abs(b) >= std::abs(c) && std::abs(b) > 1e-12) {
        vector = {b, lambda - a};
    } else if (std::abs(c) > 1e-12) {
        vector = {lambda - d, c};
    } else {
        vector = {1.0, 0.0};
    }
    const double norm = std::sqrt(std::norm(vector[0]) + std::norm(vector[1]));
    if (norm <= 1e-12) return {1.0, 0.0};
    return {vector[0] / norm, vector[1] / norm};
}

// Relative pivot / singular-value threshold for the compute precision.
double SingularTolerance() {
    return DeviceHasDouble() ? 1e-12 : 1e-6;
}

template <typename T>
af::array PackColumnMajor(const Matrix& A) {
    const size_t rows = A.size(), cols = A[0].size();
    std::vector<T> flat(rows * cols);
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) flat[c * rows + r] = static_cast<T>(A[r][c]);
    }
    return af::array(static_cast<dim_t>(rows), static_cast<dim_t>(cols), flat.data());
}

af::array ToDevice(const Matrix& A) {
    return DeviceHasDouble() ? PackColumnMajor<double>(A) : PackColumnMajor<float>(A);
}

Matrix ToHostMatrix(const af::array& source, const char* operation) {
    const size_t rows = static_cast<size_t>(source.dims(0)), cols = static_cast<size_t>(source.dims(1));
    const std::vector<double> values = DeviceToHostValues(source, operation);
    Matrix result(rows, std::vector<double>(cols));
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) result[r][c] = values[c * rows + r];
    }
    return result;
}

MatrixResult MatrixFromDevice(const af::array& source, const char* operation) {
    MatrixResult result;
    result.matrix = ToHostMatrix(source, operation);
    result.rows = static_cast<int>(source.dims(0));
    result.cols = static_cast<int>(source.dims(1));
    result.success = true;
    return result;
}

// Singular values (descending) of a.
std::vector<double> SingularValues(const af::array& a, const char* operation) {
    af::array u, s, vt;
    detail::ThinSvd(a, u, s, vt, false);
    return DeviceToHostValues(s, operation);
}

// True when a square matrix is singular or numerically singular: its smallest
// singular value is below the relative tolerance. (An LU pivot test does not
// work everywhere: OpenCL's getrf throws on an exactly singular matrix.)
bool NearlySingular(const af::array& a, const char* operation) {
    const std::vector<double> singular = SingularValues(a, operation);
    return singular.front() == 0.0 || singular.back() <= SingularTolerance() * singular.front();
}

// Cholesky factor of a, or false when a is not positive definite. The status
// alone is not enough (CUDA reports 0 for an indefinite matrix), so the
// factor must also be finite, have a positive diagonal and reproduce a.
bool CholeskyLower(const af::array& a, af::array& lower) {
    if (af::cholesky(lower, a, /*is_upper=*/false) != 0) return false;
    if (!af::allTrue<bool>(af::isInf(lower) == 0 && af::isNaN(lower) == 0)) return false;
    if (!af::allTrue<bool>(af::diag(lower) > 0.0)) return false;
    const af::array residual = af::matmul(lower, lower, AF_MAT_NONE, AF_MAT_TRANS) - a;
    const double scale = std::sqrt(af::sum<double>(a * a));
    return std::sqrt(af::sum<double>(residual * residual)) <= std::sqrt(SingularTolerance()) * (1.0 + scale);
}

template <typename Result>
Result Failure(std::string message) {
    Result result;
    result.error_message = std::move(message);
    return result;
}

template <typename Result>
bool RequireMatrix(const Matrix& A, Result& result) {
    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return false;
    }
    if (!IsRectangularMatrix(A)) {
        result.error_message = "Input matrix must be rectangular and non-empty";
        return false;
    }
    return true;
}

}  // namespace

bool LinearAlgebra::IsSquare(const std::vector<std::vector<double>>& A) {
    if (A.empty()) return false;
    return A.size() == A[0].size();
}

void LinearAlgebra::GetDimensions(const std::vector<std::vector<double>>& A, int& rows, int& cols) {
    rows = static_cast<int>(A.size());
    cols = A.empty() ? 0 : static_cast<int>(A[0].size());
}

bool LinearAlgebra::ValidateDimensions(const std::vector<std::vector<double>>& A, int expected_rows, int expected_cols) {
    if (A.empty()) return expected_rows == 0;
    if (static_cast<int>(A.size()) != expected_rows) return false;
    if (static_cast<int>(A[0].size()) != expected_cols) return false;
    return true;
}

// ============================================================================
// Basic Operations
// ============================================================================

MatrixResult LinearAlgebra::Add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    MatrixResult result;
    if (!RequireMatrix(A, result) || !RequireMatrix(B, result)) return result;
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        result.error_message = "Matrix dimensions must match for addition";
        return result;
    }
    try {
        return MatrixFromDevice(ToDevice(A) + ToDevice(B), "LinearAlgebra::Add");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::Add", e));
    }
}

MatrixResult LinearAlgebra::Subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    MatrixResult result;
    if (!RequireMatrix(A, result) || !RequireMatrix(B, result)) return result;
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        result.error_message = "Matrix dimensions must match for subtraction";
        return result;
    }
    try {
        return MatrixFromDevice(ToDevice(A) - ToDevice(B), "LinearAlgebra::Subtract");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::Subtract", e));
    }
}

MatrixResult LinearAlgebra::Multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    MatrixResult result;
    if (!RequireMatrix(A, result) || !RequireMatrix(B, result)) return result;
    if (A[0].size() != B.size()) {
        result.error_message = "Matrix dimensions incompatible for multiplication (A cols must equal B rows)";
        return result;
    }
    try {
        return MatrixFromDevice(af::matmul(ToDevice(A), ToDevice(B)), "LinearAlgebra::Multiply");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::Multiply", e));
    }
}

MatrixResult LinearAlgebra::ScalarMultiply(const std::vector<std::vector<double>>& A, double scalar) {
    MatrixResult result;
    if (!RequireMatrix(A, result)) return result;
    try {
        return MatrixFromDevice(ToDevice(A) * scalar, "LinearAlgebra::ScalarMultiply");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::ScalarMultiply", e));
    }
}

MatrixResult LinearAlgebra::Transpose(const std::vector<std::vector<double>>& A) {
    MatrixResult result;
    if (!RequireMatrix(A, result)) return result;
    try {
        return MatrixFromDevice(af::transpose(ToDevice(A)), "LinearAlgebra::Transpose");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::Transpose", e));
    }
}

MatrixResult LinearAlgebra::Inverse(const std::vector<std::vector<double>>& A) {
    MatrixResult result;
    if (!RequireMatrix(A, result)) return result;
    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for inversion";
        return result;
    }
    try {
        const af::array a = ToDevice(A);
        if (NearlySingular(a, "LinearAlgebra::Inverse")) {
            result.error_message = "Matrix is singular or nearly singular";
            return result;
        }
        return MatrixFromDevice(af::inverse(a), "LinearAlgebra::Inverse");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::Inverse", e));
    }
}

// ============================================================================
// Scalar Properties
// ============================================================================

ScalarResult LinearAlgebra::Determinant(const std::vector<std::vector<double>>& A) {
    ScalarResult result;
    if (!RequireMatrix(A, result)) return result;
    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for determinant";
        return result;
    }
    try {
        // A (numerically) singular matrix has determinant 0; checked first
        // because OpenCL's getrf throws on an exactly singular matrix.
        const af::array a = ToDevice(A);
        result.value = NearlySingular(a, "LinearAlgebra::Determinant") ? 0.0 : af::det<double>(a);
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Failure<ScalarResult>(DeviceError("LinearAlgebra::Determinant", e));
    }
}

ScalarResult LinearAlgebra::Trace(const std::vector<std::vector<double>>& A) {
    ScalarResult result;
    if (!RequireMatrix(A, result)) return result;
    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for trace";
        return result;
    }
    try {
        result.value = af::sum<double>(af::diag(ToDevice(A)));
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Failure<ScalarResult>(DeviceError("LinearAlgebra::Trace", e));
    }
}

namespace {

// Singular values carry the compute precision's rounding: a float32 device
// leaves ~1e-7 relative where float64 has zeros, so the relative tolerance is
// at least that precision's epsilon times max(m, n) (torch.linalg.matrix_rank's
// rtol). On the Gram route a tall matrix takes, a small singular value carries
// sqrt(eps) of the largest, times the columns (not the rows: the m x m U is
// never formed).
ScalarResult RankOf(const af::array& a, double tolerance) {
    ScalarResult result;
    const std::vector<double> singular = SingularValues(a, "LinearAlgebra::Rank");
    const double largest = singular.empty() ? 0.0 : singular.front();
    const double size = detail::UsesGram(a) ? static_cast<double>(a.dims(1)) : static_cast<double>(std::max(a.dims(0), a.dims(1)));
    const double threshold = std::max(tolerance, detail::SingularNoise(a)) * size * largest;
    result.value = static_cast<double>(std::count_if(singular.begin(), singular.end(), [&](double s) { return s > threshold; }));
    result.success = true;
    return result;
}

}  // namespace

ScalarResult LinearAlgebra::Rank(const std::vector<std::vector<double>>& A, double tolerance) {
    ScalarResult result;
    if (A.empty()) {
        result.value = 0;
        result.success = true;
        return result;
    }
    if (!RequireMatrix(A, result)) return result;
    try {
        return RankOf(ToDevice(A), tolerance);
    } catch (const af::exception& e) {
        return Failure<ScalarResult>(DeviceError("LinearAlgebra::Rank", e));
    }
}

ScalarResult LinearAlgebra::FrobeniusNorm(const std::vector<std::vector<double>>& A) {
    ScalarResult result;
    if (A.empty()) {
        result.value = 0.0;
        result.success = true;
        return result;
    }
    if (!RequireMatrix(A, result)) return result;
    try {
        const af::array a = ToDevice(A);
        result.value = std::sqrt(af::sum<double>(a * a));
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Failure<ScalarResult>(DeviceError("LinearAlgebra::FrobeniusNorm", e));
    }
}

ScalarResult LinearAlgebra::ConditionNumber(const std::vector<std::vector<double>>& A) {
    ScalarResult result;
    if (!RequireMatrix(A, result)) return result;
    try {
        const std::vector<double> singular = SingularValues(ToDevice(A), "LinearAlgebra::ConditionNumber");
        if (singular.empty()) {
            result.error_message = "No singular values computed";
            return result;
        }
        const double smallest = singular.back();
        result.value = smallest < 1e-15 ? std::numeric_limits<double>::infinity() : singular.front() / smallest;
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Failure<ScalarResult>(DeviceError("LinearAlgebra::ConditionNumber", e));
    }
}

// ============================================================================
// Decompositions
// ============================================================================

// Symmetric matrices: ArrayFire has no eigensolver, but A + cI with c above
// the spectral radius (c = ||A||_F + 1) is symmetric positive definite, whose
// SVD is its eigendecomposition: eigenvalues s - c (descending), eigenvectors
// the columns of U. Nonsymmetric 2x2: closed form (complex pairs included).
EigenResult LinearAlgebra::Eigen(const std::vector<std::vector<double>>& A) {
    EigenResult result;
    if (!RequireMatrix(A, result)) return result;
    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for eigendecomposition";
        return result;
    }
    const int n = static_cast<int>(A.size());
    result.n = n;

    if (!IsSymmetric(A)) {
        if (n != 2) {
            result.error_message =
                "Eigendecomposition supports symmetric matrices and nonsymmetric 2x2 matrices only";
            return result;
        }
        const double trace = A[0][0] + A[1][1];
        const double determinant = A[0][0] * A[1][1] - A[0][1] * A[1][0];
        const std::complex<double> root = std::sqrt(std::complex<double>(trace * trace - 4.0 * determinant, 0.0));
        const std::complex<double> lambda0 = (trace + root) / 2.0;
        const std::complex<double> lambda1 = (trace - root) / 2.0;
        if (std::abs(lambda0 - lambda1) <= 1e-12) {
            result.error_message = "Nonsymmetric 2x2 eigendecomposition does not support repeated eigenvalues";
            return result;
        }
        result.eigenvalues = {lambda0, lambda1};
        const auto vector0 = Eigenvector2x2(A, lambda0);
        const auto vector1 = Eigenvector2x2(A, lambda1);
        result.eigenvectors = {{vector0[0], vector1[0]}, {vector0[1], vector1[1]}};
        result.success = true;
        return result;
    }

    try {
        const af::array a = ToDevice(A);
        const double shift = std::sqrt(af::sum<double>(a * a)) + 1.0;
        af::array u, s, vt;
        af::svd(u, s, vt, a + shift * af::identity(n, n, a.type()));
        const std::vector<double> values = DeviceToHostValues(s, "LinearAlgebra::Eigen");
        const Matrix vectors = ToHostMatrix(u, "LinearAlgebra::Eigen");
        result.eigenvalues.resize(n);
        result.eigenvectors.assign(n, std::vector<std::complex<double>>(n));
        for (int col = 0; col < n; ++col) {
            result.eigenvalues[col] = values[col] - shift;
            for (int row = 0; row < n; ++row) result.eigenvectors[row][col] = vectors[row][col];
        }
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Failure<EigenResult>(DeviceError("LinearAlgebra::Eigen", e));
    }
}

SVDResult LinearAlgebra::SVD(const std::vector<std::vector<double>>& A, bool full_matrices) {
    SVDResult result;
    if (!RequireMatrix(A, result)) return result;
    GetDimensions(A, result.m, result.n);
    result.k = std::min(result.m, result.n);
    try {
        af::array u, s, vt;
        const af::array a = ToDevice(A);
        if (full_matrices) {
            if (detail::UsesGram(a)) {
                return Failure<SVDResult>("A has " + std::to_string(result.m) + " rows: its full U (" + std::to_string(result.m) + " x " +
                                          std::to_string(result.m) + ") does not fit; use the reduced SVD");
            }
            af::svd(u, s, vt, a);  // full: U m x m, Vt n x n
        } else {
            detail::ThinSvd(a, u, s, vt, true);
        }
        result.U = ToHostMatrix(u, "LinearAlgebra::SVD");
        result.S = DeviceToHostValues(s, "LinearAlgebra::SVD");
        result.Vt = ToHostMatrix(vt, "LinearAlgebra::SVD");
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        auto failed = Failure<SVDResult>(DeviceError("LinearAlgebra::SVD", e));
        failed.m = result.m;
        failed.n = result.n;
        failed.k = result.k;
        return failed;
    }
}

// A_k = U_k diag(S_k) Vt_k from the top k singular triplets.
MatrixResult LinearAlgebra::LowRankApproximation(const std::vector<std::vector<double>>& A, int k) {
    MatrixResult result;
    if (!RequireMatrix(A, result)) return result;
    if (k <= 0 || k > static_cast<int>(std::min(A.size(), A[0].size()))) {
        result.error_message = "k must be between 1 and min(m,n)";
        return result;
    }
    try {
        af::array u, s, vt;
        detail::ThinSvd(ToDevice(A), u, s, vt, true);
        const af::seq top(0, k - 1);
        const af::array scaled =  // column j times s_j
            u(af::span, top) * af::tile(af::transpose(s(top)), static_cast<unsigned>(A.size()));
        return MatrixFromDevice(af::matmul(scaled, vt(top, af::span)), "LinearAlgebra::LowRankApproximation");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::LowRankApproximation", e));
    }
}

// ============================================================================
// Tables of rows
// ============================================================================

namespace {

// `values` (one per column) as a 1 x n row on the device, or `fallback` everywhere when empty.
af::array RowOf(const std::vector<double>& values, dim_t n, double fallback) {
    if (values.empty()) return af::constant(fallback, 1, n, DeviceHasDouble() ? f64 : f32);
    return ToDevice(Matrix{values});
}

}  // namespace

namespace {

// A DenseView on the device in the compute type (float64 as it is; float32 converted).
af::array ViewToDevice(const DenseView& A) {
    if (DeviceHasDouble()) return af::array(A.rows, A.cols, A.data);
    const size_t count = static_cast<size_t>(A.rows) * static_cast<size_t>(A.cols);
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = static_cast<float>(A.data[i]);
    return af::array(A.rows, A.cols, values.data());
}

std::string ViewRefusal(const DenseView& A) {
    if (A.rows <= 0 || A.cols <= 0 || !A.data) return "The matrix is empty";
    return "";
}

ColumnSummaryResult SummaryOf(const af::array& a) {
    ColumnSummaryResult result;
    const af::array mean = af::mean(a, 0);
    const af::array centred = a - af::tile(mean, static_cast<unsigned>(a.dims(0)));
    const af::array sd = af::sqrt(af::mean(centred * centred, 0));
    result.mean = DeviceToHostValues(mean, "LinearAlgebra::ColumnSummary");
    result.sd = DeviceToHostValues(sd, "LinearAlgebra::ColumnSummary");
    result.cross = ToHostMatrix(af::matmul(centred, centred, AF_MAT_TRANS, AF_MAT_NONE), "LinearAlgebra::ColumnSummary");
    result.rows = static_cast<int>(a.dims(0));
    result.precision = a.type() == f64 ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon();
    result.success = true;
    return result;
}

af::array AffineOf(af::array a, const std::vector<double>& shift, const std::vector<double>& divisor, const Matrix& W) {
    const unsigned n = static_cast<unsigned>(a.dims(0));
    if (!shift.empty()) a -= af::tile(RowOf(shift, a.dims(1), 0.0), n);
    if (!divisor.empty()) a /= af::tile(RowOf(divisor, a.dims(1), 1.0), n);
    return af::matmul(a, ToDevice(W));
}

std::string AffineRefusal(size_t d, const std::vector<double>& shift, const std::vector<double>& divisor, const Matrix& W) {
    if ((!shift.empty() && shift.size() != d) || (!divisor.empty() && divisor.size() != d) || W.size() != d)
        return "shift, divisor and the rows of W must have one entry per column of A";
    return "";
}

}  // namespace

ColumnSummaryResult LinearAlgebra::ColumnSummary(const std::vector<std::vector<double>>& A) {
    ColumnSummaryResult result;
    if (!RequireMatrix(A, result)) return result;
    try {
        return SummaryOf(ToDevice(A));
    } catch (const af::exception& e) {
        return Failure<ColumnSummaryResult>(DeviceError("LinearAlgebra::ColumnSummary", e));
    }
}

ColumnSummaryResult LinearAlgebra::ColumnSummary(const DenseView& A) {
    if (const std::string refusal = ViewRefusal(A); !refusal.empty()) return Failure<ColumnSummaryResult>(refusal);
    try {
        return SummaryOf(ViewToDevice(A));
    } catch (const af::exception& e) {
        return Failure<ColumnSummaryResult>(DeviceError("LinearAlgebra::ColumnSummary", e));
    }
}

SVDResult LinearAlgebra::LeadingEigenvectors(const std::vector<std::vector<double>>& S, int k, int iterations) {
    SVDResult result;
    if (!RequireMatrix(S, result)) return result;
    const int d = static_cast<int>(S.size());
    if (S[0].size() != S.size()) return Failure<SVDResult>("S must be square");
    if (k < 1 || k > d) return Failure<SVDResult>("k must be between 1 and the size of S");
    try {
        const af::array s = ToDevice(S);
        const int l = std::min(d, k + 10);  // oversampled, as scikit-learn's randomized_svd
        af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, 0);
        af::array q = af::randn(af::dim4(d, l), s.type(), engine);
        // Orthonormal columns by Cholesky QR (l x l only; a ridge keeps a rank-deficient block solvable).
        const auto orthonormal = [&](const af::array& z) {
            af::array g = af::matmul(z, z, AF_MAT_TRANS, AF_MAT_NONE);
            g += af::identity(l, l, g.type()) * (1e-12 * af::sum<double>(af::diag(g)) + 1e-300);
            af::array r;
            if (af::cholesky(r, g, /*is_upper=*/true) != 0) throw af::exception("Cholesky QR failed");
            return af::matmul(z, af::inverse(r));
        };
        q = orthonormal(q);
        for (int it = 0; it < iterations; ++it) q = orthonormal(af::matmul(s, q));
        // Rayleigh-Ritz: the small l x l problem gives the vectors within the subspace.
        const af::array within = af::matmul(q, af::matmul(s, q), AF_MAT_TRANS, AF_MAT_NONE);
        af::array u, sv, vt;
        af::svd(u, sv, vt, within);
        const af::seq top(0, k - 1);
        const af::array vectors = af::matmul(q, u(af::span, top));  // d x k
        result.S = DeviceToHostValues(sv(top), "LinearAlgebra::LeadingEigenvectors");
        result.Vt = ToHostMatrix(af::transpose(vectors), "LinearAlgebra::LeadingEigenvectors");
        result.m = result.n = d;
        result.k = k;
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Failure<SVDResult>(DeviceError("LinearAlgebra::LeadingEigenvectors", e));
    }
}

MatrixResult LinearAlgebra::AffineProjection(const std::vector<std::vector<double>>& A, const std::vector<double>& shift,
                                             const std::vector<double>& divisor, const std::vector<std::vector<double>>& W) {
    MatrixResult result;
    if (!RequireMatrix(A, result) || !RequireMatrix(W, result)) return result;
    if (const std::string refusal = AffineRefusal(A[0].size(), shift, divisor, W); !refusal.empty()) return Failure<MatrixResult>(refusal);
    try {
        return MatrixFromDevice(AffineOf(ToDevice(A), shift, divisor, W), "LinearAlgebra::AffineProjection");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::AffineProjection", e));
    }
}

ScalarResult LinearAlgebra::Rank(const DenseView& A, double tolerance) {
    if (const std::string refusal = ViewRefusal(A); !refusal.empty()) return Failure<ScalarResult>(refusal);
    try {
        return RankOf(ViewToDevice(A), tolerance);
    } catch (const af::exception& e) {
        return Failure<ScalarResult>(DeviceError("LinearAlgebra::Rank", e));
    }
}

MatrixResult LinearAlgebra::LeastSquares(const DenseView& A, const DenseView& b) {
    if (const std::string refusal = ViewRefusal(A); !refusal.empty()) return Failure<MatrixResult>(refusal);
    if (const std::string refusal = ViewRefusal(b); !refusal.empty()) return Failure<MatrixResult>("b: " + refusal);
    if (A.rows != b.rows) return Failure<MatrixResult>("A and b must have same number of rows");
    try {
        return MatrixFromDevice(detail::LeastSquaresSvd(ViewToDevice(A), ViewToDevice(b)), "LinearAlgebra::LeastSquares");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::LeastSquares", e));
    }
}

bool LinearAlgebra::AffineProjection(const DenseView& A, const std::vector<double>& shift, const std::vector<double>& divisor,
                                     const std::vector<std::vector<double>>& W, std::vector<double>& out, std::string& error) {
    if (const std::string refusal = ViewRefusal(A); !refusal.empty()) return error = refusal, false;
    MatrixResult check;
    if (!RequireMatrix(W, check)) return error = check.error_message, false;
    if (const std::string refusal = AffineRefusal(static_cast<size_t>(A.cols), shift, divisor, W); !refusal.empty()) return error = refusal, false;
    try {
        out = DeviceToHostValues(AffineOf(ViewToDevice(A), shift, divisor, W), "LinearAlgebra::AffineProjection");
        return true;
    } catch (const af::exception& e) {
        error = DeviceError("LinearAlgebra::AffineProjection", e);
        return false;
    }
}

// ============================================================================
// Sparse (CSR)
// ============================================================================

namespace {

std::string CsrRefusal(const CsrView& X) {
    if (X.rows <= 0 || X.cols <= 0 || !X.row_offsets) return "The sparse matrix is empty";
    if (X.row_offsets[0] != 0) return "The sparse matrix's first row offset is not 0";
    for (int r = 0; r < X.rows; ++r)
        if (X.row_offsets[r + 1] < X.row_offsets[r]) return "The sparse matrix's row offsets go down";
    if (X.row_offsets[X.rows] > 0 && (!X.column_indices || !X.values)) return "The sparse matrix has entries but no values";
    return "";
}

}  // namespace

MatrixResult LinearAlgebra::SparseGram(const CsrView& X) {
    if (const std::string refusal = CsrRefusal(X); !refusal.empty()) return Failure<MatrixResult>(refusal);
    try {
        // X^T X = X^T [X's columns, a block at a time]: the device's sparse
        // product of the transposed CSR with each dense block of columns, in
        // float32 as the features are kept (a dense float64 product of the
        // rows costs a hundred times more on a gaming GPU).
        const dim_t nnz = X.row_offsets[X.rows];
        af::array gram = af::constant(0.0, X.cols, X.cols, f32);
        if (nnz == 0) return MatrixFromDevice(gram, "LinearAlgebra::SparseGram");
        const af::array x = af::sparse(X.rows, X.cols, af::array(nnz, X.values), af::array(static_cast<dim_t>(X.rows + 1), X.row_offsets),
                                       af::array(nnz, X.column_indices), AF_STORAGE_CSR);
        const int width = std::max(1, std::min(X.cols, 32000000 / std::max(1, X.rows)));  // ~128 MB a block
        std::vector<int> place;
        std::vector<float> values;
        for (int first = 0; first < X.cols; first += width) {
            const int last = std::min(X.cols, first + width);
            place.clear();
            values.clear();
            for (int r = 0; r < X.rows; ++r)
                for (int k = X.row_offsets[r]; k < X.row_offsets[r + 1]; ++k) {
                    const int c = X.column_indices[k];
                    if (c < first || c >= last) continue;
                    place.push_back(r + (c - first) * X.rows);
                    values.push_back(X.values[k]);
                }
            af::array block = af::constant(0.0f, static_cast<dim_t>(X.rows) * (last - first), f32);
            if (!place.empty())
                block(af::array(static_cast<dim_t>(place.size()), place.data())) = af::array(static_cast<dim_t>(values.size()), values.data());
            block = af::moddims(block, X.rows, last - first);
            gram(af::span, af::seq(first, last - 1)) = af::matmul(x, block, AF_MAT_TRANS, AF_MAT_NONE);
        }
        return MatrixFromDevice(gram, "LinearAlgebra::SparseGram");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::SparseGram", e));
    }
}

MatrixResult LinearAlgebra::SparseMultiply(const CsrView& X, const std::vector<std::vector<double>>& B) {
    if (const std::string refusal = CsrRefusal(X); !refusal.empty()) return Failure<MatrixResult>(refusal);
    MatrixResult check;
    if (!RequireMatrix(B, check)) return check;
    if (static_cast<int>(B.size()) != X.cols)
        return Failure<MatrixResult>("B must have as many rows as X has columns");
    try {
        // The CSR in float32 (the features' own type), as the device's sparse product takes it.
        const dim_t nnz = X.row_offsets[X.rows];
        const af::array b = ToDevice(B).as(f32);
        if (nnz == 0) return MatrixFromDevice(af::constant(0.0, X.rows, b.dims(1), f32), "LinearAlgebra::SparseMultiply");
        const af::array x = af::sparse(X.rows, X.cols, af::array(nnz, X.values), af::array(static_cast<dim_t>(X.rows + 1), X.row_offsets),
                                       af::array(nnz, X.column_indices), AF_STORAGE_CSR);
        return MatrixFromDevice(af::matmul(x, b), "LinearAlgebra::SparseMultiply");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::SparseMultiply", e));
    }
}

// Reduced QR (PyTorch's default): Q m x k with orthonormal columns, R k x n,
// k = min(m, n).
QRResult LinearAlgebra::QR(const std::vector<std::vector<double>>& A) {
    QRResult result;
    if (!RequireMatrix(A, result)) return result;
    GetDimensions(A, result.m, result.n);
    try {
        const int k = std::min(result.m, result.n);
        af::array q, r, tau;
        af::qr(q, r, tau, ToDevice(A));
        result.Q = ToHostMatrix(q(af::span, af::seq(0, k - 1)), "LinearAlgebra::QR");
        result.R = ToHostMatrix(r(af::seq(0, k - 1), af::span), "LinearAlgebra::QR");
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Failure<QRResult>(DeviceError("LinearAlgebra::QR", e));
    }
}

CholeskyResult LinearAlgebra::Cholesky(const std::vector<std::vector<double>>& A) {
    CholeskyResult result;
    if (!RequireMatrix(A, result)) return result;
    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for Cholesky decomposition";
        return result;
    }
    result.n = static_cast<int>(A.size());
    try {
        af::array lower;
        if (!CholeskyLower(ToDevice(A), lower)) {
            result.error_message = "Matrix is not positive definite";
            return result;
        }
        result.L = ToHostMatrix(lower, "LinearAlgebra::Cholesky");
        result.is_positive_definite = true;
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        auto failed = Failure<CholeskyResult>(DeviceError("LinearAlgebra::Cholesky", e));
        failed.n = result.n;
        return failed;
    }
}

// P A = L U with partial pivoting; P[i] is the row of A that lands in row i.
LUResult LinearAlgebra::LU(const std::vector<std::vector<double>>& A) {
    LUResult result;
    if (!RequireMatrix(A, result)) return result;
    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for LU decomposition";
        return result;
    }
    result.n = static_cast<int>(A.size());
    try {
        af::array lower, upper, pivot;
        af::lu(lower, upper, pivot, ToDevice(A));  // pivot: permutation indices
        result.L = ToHostMatrix(lower, "LinearAlgebra::LU");
        result.U = ToHostMatrix(upper, "LinearAlgebra::LU");
        af::array permutation = pivot.as(s32);
        permutation.eval();
        result.P.resize(static_cast<size_t>(result.n));
        MaterializeArrayFireToHost(permutation, result.P.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                                   "LinearAlgebra::LU", "permutation_indices");
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        auto failed = Failure<LUResult>(DeviceError("LinearAlgebra::LU", e));
        failed.n = result.n;
        return failed;
    }
}

// ============================================================================
// Linear Systems
// ============================================================================

MatrixResult LinearAlgebra::Solve(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
    MatrixResult result;
    if (!RequireMatrix(A, result) || !RequireMatrix(b, result)) return result;
    if (!IsSquare(A)) {
        result.error_message = "Matrix A must be square for Solve";
        return result;
    }
    if (b.size() != A.size()) {
        result.error_message = "Dimensions mismatch: A rows must equal b rows";
        return result;
    }
    try {
        const af::array a = ToDevice(A);
        if (NearlySingular(a, "LinearAlgebra::Solve")) {
            result.error_message = "Matrix is singular or nearly singular";
            return result;
        }
        return MatrixFromDevice(af::solve(a, ToDevice(b)), "LinearAlgebra::Solve");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::Solve", e));
    }
}

MatrixResult LinearAlgebra::LeastSquares(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
    MatrixResult result;
    if (!RequireMatrix(A, result) || !RequireMatrix(b, result)) return result;
    if (A.size() != b.size()) {
        result.error_message = "A and b must have same number of rows";
        return result;
    }
    try {
        // The shortest of the best x when A's columns depend on each other (detail::LeastSquaresSvd).
        return MatrixFromDevice(detail::LeastSquaresSvd(ToDevice(A), ToDevice(b)), "LinearAlgebra::LeastSquares");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::LeastSquares", e));
    }
}

// ============================================================================
// Matrix Properties (an ArrayFire error throws: a bool cannot carry it)
// ============================================================================

bool LinearAlgebra::IsSymmetric(const std::vector<std::vector<double>>& A, double tolerance) {
    if (!IsRectangularMatrix(A) || !IsSquare(A)) return false;
    try {
        const af::array a = ToDevice(A);
        return af::allTrue<bool>(af::abs(a - af::transpose(a)) <= tolerance);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("LinearAlgebra::IsSymmetric", e));
    }
}

bool LinearAlgebra::IsPositiveDefinite(const std::vector<std::vector<double>>& A) {
    return Cholesky(A).is_positive_definite;
}

// Orthonormal columns: A^T A = I (a square A is then orthogonal; a reduced
// QR's Q qualifies too).
bool LinearAlgebra::IsOrthogonal(const std::vector<std::vector<double>>& A, double tolerance) {
    if (!IsRectangularMatrix(A) || A.size() < A[0].size()) return false;
    try {
        const af::array a = ToDevice(A);
        const af::array gram = af::matmul(a, a, AF_MAT_TRANS, AF_MAT_NONE);
        const af::array eye = af::identity(gram.dims(0), gram.dims(1), gram.type());
        // float32 devices cannot reach the float64 default tolerance.
        const double effective = DeviceHasDouble() ? tolerance : std::max(tolerance, 1e-5);
        return af::allTrue<bool>(af::abs(gram - eye) <= effective);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("LinearAlgebra::IsOrthogonal", e));
    }
}

} // namespace cyxwiz
