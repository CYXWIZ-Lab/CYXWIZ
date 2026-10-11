// Host-matrix linear algebra (the analysis panels' API) on ArrayFire: the CPU
// option is ArrayFire's CPU backend, there are no hand-written CPU kernels, and
// a build without ArrayFire refuses (TOFIX140). Matrices cross the host boundary
// once on the way in and once on the way out; float64 where the active device
// has it, else float32.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/linear_algebra.h"
#include "arrayfire_backend_utils.h"
#include "arrayfire_host_materialization.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

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

#ifdef CYXWIZ_HAS_ARRAYFIRE

bool DoubleOnDevice() {
    return af::isDoubleAvailable(af::getDevice());
}

// Relative pivot / singular-value threshold for the compute precision.
double SingularTolerance() {
    return DoubleOnDevice() ? 1e-12 : 1e-6;
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
    return DoubleOnDevice() ? PackColumnMajor<double>(A) : PackColumnMajor<float>(A);
}

// Every element of `source`, column-major, as double.
std::vector<double> ToHostValues(const af::array& source, const char* operation) {
    std::vector<double> values(static_cast<size_t>(source.elements()));
    if (values.empty()) return values;
    if (source.type() == f64) {
        af::array data = source;
        data.eval();
        MaterializeArrayFireToHost(data, values.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                                   operation, "arrayfire_column_major");
        return values;
    }
    af::array data = source.as(f32);
    data.eval();
    std::vector<float> narrow(values.size());
    MaterializeArrayFireToHost(data, narrow.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                               operation, "arrayfire_column_major");
    std::copy(narrow.begin(), narrow.end(), values.begin());
    return values;
}

Matrix ToHostMatrix(const af::array& source, const char* operation) {
    const size_t rows = static_cast<size_t>(source.dims(0)), cols = static_cast<size_t>(source.dims(1));
    const std::vector<double> values = ToHostValues(source, operation);
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
    af::svd(u, s, vt, a);
    return ToHostValues(s, operation);
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

std::string DeviceError(const char* operation, const af::exception& e) {
    return std::string(operation) + " failed on the ArrayFire device: " + e.what();
}

#else

template <typename Result>
Result NoArrayFire(const char* operation) {
    Result result;
    result.error_message = std::string(operation) + " runs on ArrayFire, and this build has no ArrayFire";
    return result;
}

#endif

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
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return MatrixFromDevice(ToDevice(A) + ToDevice(B), "LinearAlgebra::Add");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::Add", e));
    }
#else
    return NoArrayFire<MatrixResult>("LinearAlgebra::Add");
#endif
}

MatrixResult LinearAlgebra::Subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    MatrixResult result;
    if (!RequireMatrix(A, result) || !RequireMatrix(B, result)) return result;
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        result.error_message = "Matrix dimensions must match for subtraction";
        return result;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return MatrixFromDevice(ToDevice(A) - ToDevice(B), "LinearAlgebra::Subtract");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::Subtract", e));
    }
#else
    return NoArrayFire<MatrixResult>("LinearAlgebra::Subtract");
#endif
}

MatrixResult LinearAlgebra::Multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    MatrixResult result;
    if (!RequireMatrix(A, result) || !RequireMatrix(B, result)) return result;
    if (A[0].size() != B.size()) {
        result.error_message = "Matrix dimensions incompatible for multiplication (A cols must equal B rows)";
        return result;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return MatrixFromDevice(af::matmul(ToDevice(A), ToDevice(B)), "LinearAlgebra::Multiply");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::Multiply", e));
    }
#else
    return NoArrayFire<MatrixResult>("LinearAlgebra::Multiply");
#endif
}

MatrixResult LinearAlgebra::ScalarMultiply(const std::vector<std::vector<double>>& A, double scalar) {
    MatrixResult result;
    if (!RequireMatrix(A, result)) return result;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return MatrixFromDevice(ToDevice(A) * scalar, "LinearAlgebra::ScalarMultiply");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::ScalarMultiply", e));
    }
#else
    (void)scalar;
    return NoArrayFire<MatrixResult>("LinearAlgebra::ScalarMultiply");
#endif
}

MatrixResult LinearAlgebra::Transpose(const std::vector<std::vector<double>>& A) {
    MatrixResult result;
    if (!RequireMatrix(A, result)) return result;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return MatrixFromDevice(af::transpose(ToDevice(A)), "LinearAlgebra::Transpose");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::Transpose", e));
    }
#else
    return NoArrayFire<MatrixResult>("LinearAlgebra::Transpose");
#endif
}

MatrixResult LinearAlgebra::Inverse(const std::vector<std::vector<double>>& A) {
    MatrixResult result;
    if (!RequireMatrix(A, result)) return result;
    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for inversion";
        return result;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
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
#else
    return NoArrayFire<MatrixResult>("LinearAlgebra::Inverse");
#endif
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
#ifdef CYXWIZ_HAS_ARRAYFIRE
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
#else
    return NoArrayFire<ScalarResult>("LinearAlgebra::Determinant");
#endif
}

ScalarResult LinearAlgebra::Trace(const std::vector<std::vector<double>>& A) {
    ScalarResult result;
    if (!RequireMatrix(A, result)) return result;
    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for trace";
        return result;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        result.value = af::sum<double>(af::diag(ToDevice(A)));
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Failure<ScalarResult>(DeviceError("LinearAlgebra::Trace", e));
    }
#else
    return NoArrayFire<ScalarResult>("LinearAlgebra::Trace");
#endif
}

ScalarResult LinearAlgebra::Rank(const std::vector<std::vector<double>>& A, double tolerance) {
    ScalarResult result;
    if (A.empty()) {
        result.value = 0;
        result.success = true;
        return result;
    }
    if (!RequireMatrix(A, result)) return result;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const std::vector<double> singular = SingularValues(ToDevice(A), "LinearAlgebra::Rank");
        const double largest = singular.empty() ? 0.0 : singular.front();
        const double threshold = tolerance * static_cast<double>(std::max(A.size(), A[0].size())) * largest;
        result.value = static_cast<double>(
            std::count_if(singular.begin(), singular.end(), [&](double s) { return s > threshold; }));
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Failure<ScalarResult>(DeviceError("LinearAlgebra::Rank", e));
    }
#else
    (void)tolerance;
    return NoArrayFire<ScalarResult>("LinearAlgebra::Rank");
#endif
}

ScalarResult LinearAlgebra::FrobeniusNorm(const std::vector<std::vector<double>>& A) {
    ScalarResult result;
    if (A.empty()) {
        result.value = 0.0;
        result.success = true;
        return result;
    }
    if (!RequireMatrix(A, result)) return result;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array a = ToDevice(A);
        result.value = std::sqrt(af::sum<double>(a * a));
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Failure<ScalarResult>(DeviceError("LinearAlgebra::FrobeniusNorm", e));
    }
#else
    return NoArrayFire<ScalarResult>("LinearAlgebra::FrobeniusNorm");
#endif
}

ScalarResult LinearAlgebra::ConditionNumber(const std::vector<std::vector<double>>& A) {
    ScalarResult result;
    if (!RequireMatrix(A, result)) return result;
#ifdef CYXWIZ_HAS_ARRAYFIRE
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
#else
    return NoArrayFire<ScalarResult>("LinearAlgebra::ConditionNumber");
#endif
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

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array a = ToDevice(A);
        const double shift = std::sqrt(af::sum<double>(a * a)) + 1.0;
        af::array u, s, vt;
        af::svd(u, s, vt, a + shift * af::identity(n, n, a.type()));
        const std::vector<double> values = ToHostValues(s, "LinearAlgebra::Eigen");
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
#else
    return NoArrayFire<EigenResult>("LinearAlgebra::Eigen");
#endif
}

SVDResult LinearAlgebra::SVD(const std::vector<std::vector<double>>& A, bool full_matrices) {
    SVDResult result;
    if (!RequireMatrix(A, result)) return result;
    GetDimensions(A, result.m, result.n);
    result.k = std::min(result.m, result.n);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array u, s, vt;
        af::svd(u, s, vt, ToDevice(A));  // full: U m x m, Vt n x n
        if (!full_matrices) {
            u = u(af::span, af::seq(0, result.k - 1));
            vt = vt(af::seq(0, result.k - 1), af::span);
        }
        result.U = ToHostMatrix(u, "LinearAlgebra::SVD");
        result.S = ToHostValues(s, "LinearAlgebra::SVD");
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
#else
    (void)full_matrices;
    return NoArrayFire<SVDResult>("LinearAlgebra::SVD");
#endif
}

// A_k = U_k diag(S_k) Vt_k from the top k singular triplets.
MatrixResult LinearAlgebra::LowRankApproximation(const std::vector<std::vector<double>>& A, int k) {
    MatrixResult result;
    if (!RequireMatrix(A, result)) return result;
    if (k <= 0 || k > static_cast<int>(std::min(A.size(), A[0].size()))) {
        result.error_message = "k must be between 1 and min(m,n)";
        return result;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array u, s, vt;
        af::svd(u, s, vt, ToDevice(A));
        const af::seq top(0, k - 1);
        const af::array scaled =  // column j times s_j
            u(af::span, top) * af::tile(af::transpose(s(top)), static_cast<unsigned>(A.size()));
        return MatrixFromDevice(af::matmul(scaled, vt(top, af::span)), "LinearAlgebra::LowRankApproximation");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::LowRankApproximation", e));
    }
#else
    return NoArrayFire<MatrixResult>("LinearAlgebra::LowRankApproximation");
#endif
}

// Reduced QR (PyTorch's default): Q m x k with orthonormal columns, R k x n,
// k = min(m, n).
QRResult LinearAlgebra::QR(const std::vector<std::vector<double>>& A) {
    QRResult result;
    if (!RequireMatrix(A, result)) return result;
    GetDimensions(A, result.m, result.n);
#ifdef CYXWIZ_HAS_ARRAYFIRE
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
#else
    return NoArrayFire<QRResult>("LinearAlgebra::QR");
#endif
}

CholeskyResult LinearAlgebra::Cholesky(const std::vector<std::vector<double>>& A) {
    CholeskyResult result;
    if (!RequireMatrix(A, result)) return result;
    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for Cholesky decomposition";
        return result;
    }
    result.n = static_cast<int>(A.size());
#ifdef CYXWIZ_HAS_ARRAYFIRE
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
#else
    return NoArrayFire<CholeskyResult>("LinearAlgebra::Cholesky");
#endif
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
#ifdef CYXWIZ_HAS_ARRAYFIRE
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
#else
    return NoArrayFire<LUResult>("LinearAlgebra::LU");
#endif
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
#ifdef CYXWIZ_HAS_ARRAYFIRE
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
#else
    return NoArrayFire<MatrixResult>("LinearAlgebra::Solve");
#endif
}

MatrixResult LinearAlgebra::LeastSquares(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
    MatrixResult result;
    if (!RequireMatrix(A, result) || !RequireMatrix(b, result)) return result;
    if (A.size() != b.size()) {
        result.error_message = "A and b must have same number of rows";
        return result;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // af::solve takes the QR least-squares route for a non-square A.
        return MatrixFromDevice(af::solve(ToDevice(A), ToDevice(b)), "LinearAlgebra::LeastSquares");
    } catch (const af::exception& e) {
        return Failure<MatrixResult>(DeviceError("LinearAlgebra::LeastSquares", e));
    }
#else
    return NoArrayFire<MatrixResult>("LinearAlgebra::LeastSquares");
#endif
}

// ============================================================================
// Matrix Properties (an ArrayFire error throws: a bool cannot carry it)
// ============================================================================

bool LinearAlgebra::IsSymmetric(const std::vector<std::vector<double>>& A, double tolerance) {
    if (!IsRectangularMatrix(A) || !IsSquare(A)) return false;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array a = ToDevice(A);
        return af::allTrue<bool>(af::abs(a - af::transpose(a)) <= tolerance);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("LinearAlgebra::IsSymmetric", e));
    }
#else
    (void)tolerance;
    throw std::runtime_error(NoArrayFire<ScalarResult>("LinearAlgebra::IsSymmetric").error_message);
#endif
}

bool LinearAlgebra::IsPositiveDefinite(const std::vector<std::vector<double>>& A) {
    return Cholesky(A).is_positive_definite;
}

// Orthonormal columns: A^T A = I (a square A is then orthogonal; a reduced
// QR's Q qualifies too).
bool LinearAlgebra::IsOrthogonal(const std::vector<std::vector<double>>& A, double tolerance) {
    if (!IsRectangularMatrix(A) || A.size() < A[0].size()) return false;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array a = ToDevice(A);
        const af::array gram = af::matmul(a, a, AF_MAT_TRANS, AF_MAT_NONE);
        const af::array eye = af::identity(gram.dims(0), gram.dims(1), gram.type());
        // float32 devices cannot reach the float64 default tolerance.
        const double effective = DoubleOnDevice() ? tolerance : std::max(tolerance, 1e-5);
        return af::allTrue<bool>(af::abs(gram - eye) <= effective);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("LinearAlgebra::IsOrthogonal", e));
    }
#else
    (void)tolerance;
    throw std::runtime_error(NoArrayFire<ScalarResult>("LinearAlgebra::IsOrthogonal").error_message);
#endif
}

} // namespace cyxwiz
