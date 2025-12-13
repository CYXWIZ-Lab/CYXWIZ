// Windows compatibility
#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/linear_algebra.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Undefine Windows macros that conflict with std::min/max
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz {

// ============================================================================
// Helper Functions
// ============================================================================

static bool s_gpu_checked = false;
static bool s_use_gpu = false;

static bool CheckGPUAvailable() {
    if (s_gpu_checked) return s_use_gpu;
    s_gpu_checked = true;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::Backend backend = af::getActiveBackend();
        s_use_gpu = (backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL);
        if (s_use_gpu) {
            spdlog::info("[LinearAlgebra] GPU acceleration enabled");
        }
    } catch (const af::exception& e) {
        spdlog::warn("[LinearAlgebra] GPU check failed: {}", e.what());
        s_use_gpu = false;
    }
#endif
    return s_use_gpu;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
static af::array VectorToAfArray(const std::vector<std::vector<double>>& mat) {
    if (mat.empty()) return af::array();

    size_t rows = mat.size();
    size_t cols = mat[0].size();
    std::vector<double> flat;
    flat.reserve(rows * cols);

    // ArrayFire uses column-major order
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) {
            flat.push_back(mat[r][c]);
        }
    }

    return af::array(static_cast<dim_t>(rows), static_cast<dim_t>(cols), flat.data());
}

static std::vector<std::vector<double>> AfArrayToVector(const af::array& arr) {
    int rows = static_cast<int>(arr.dims(0));
    int cols = static_cast<int>(arr.dims(1));

    std::vector<double> flat(rows * cols);
    arr.host(flat.data());

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    // ArrayFire uses column-major order
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            result[r][c] = flat[c * rows + r];
        }
    }

    return result;
}
#endif

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

    if (A.empty() || B.empty()) {
        result.error_message = "Input matrices cannot be empty";
        return result;
    }

    int rowsA, colsA, rowsB, colsB;
    GetDimensions(A, rowsA, colsA);
    GetDimensions(B, rowsB, colsB);

    if (rowsA != rowsB || colsA != colsB) {
        result.error_message = "Matrix dimensions must match for addition";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array aB = VectorToAfArray(B);
            af::array aC = aA + aB;
            result.matrix = AfArrayToVector(aC);
            result.rows = rowsA;
            result.cols = colsA;
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU Add failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    result.matrix.resize(rowsA, std::vector<double>(colsA));
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            result.matrix[i][j] = A[i][j] + B[i][j];
        }
    }
    result.rows = rowsA;
    result.cols = colsA;
    result.success = true;
    return result;
}

MatrixResult LinearAlgebra::Subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    MatrixResult result;

    if (A.empty() || B.empty()) {
        result.error_message = "Input matrices cannot be empty";
        return result;
    }

    int rowsA, colsA, rowsB, colsB;
    GetDimensions(A, rowsA, colsA);
    GetDimensions(B, rowsB, colsB);

    if (rowsA != rowsB || colsA != colsB) {
        result.error_message = "Matrix dimensions must match for subtraction";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array aB = VectorToAfArray(B);
            af::array aC = aA - aB;
            result.matrix = AfArrayToVector(aC);
            result.rows = rowsA;
            result.cols = colsA;
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU Subtract failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    result.matrix.resize(rowsA, std::vector<double>(colsA));
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            result.matrix[i][j] = A[i][j] - B[i][j];
        }
    }
    result.rows = rowsA;
    result.cols = colsA;
    result.success = true;
    return result;
}

MatrixResult LinearAlgebra::Multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    MatrixResult result;

    if (A.empty() || B.empty()) {
        result.error_message = "Input matrices cannot be empty";
        return result;
    }

    int rowsA, colsA, rowsB, colsB;
    GetDimensions(A, rowsA, colsA);
    GetDimensions(B, rowsB, colsB);

    if (colsA != rowsB) {
        result.error_message = "Matrix A columns must equal Matrix B rows for multiplication";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array aB = VectorToAfArray(B);
            af::array aC = af::matmul(aA, aB);
            result.matrix = AfArrayToVector(aC);
            result.rows = rowsA;
            result.cols = colsB;
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU Multiply failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback - naive O(n^3) multiplication
    result.matrix.resize(rowsA, std::vector<double>(colsB, 0.0));
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result.matrix[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    result.rows = rowsA;
    result.cols = colsB;
    result.success = true;
    return result;
}

MatrixResult LinearAlgebra::ScalarMultiply(const std::vector<std::vector<double>>& A, double scalar) {
    MatrixResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    int rows, cols;
    GetDimensions(A, rows, cols);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array aC = scalar * aA;
            result.matrix = AfArrayToVector(aC);
            result.rows = rows;
            result.cols = cols;
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU ScalarMultiply failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    result.matrix.resize(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.matrix[i][j] = scalar * A[i][j];
        }
    }
    result.rows = rows;
    result.cols = cols;
    result.success = true;
    return result;
}

MatrixResult LinearAlgebra::Transpose(const std::vector<std::vector<double>>& A) {
    MatrixResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    int rows, cols;
    GetDimensions(A, rows, cols);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array aT = af::transpose(aA);
            result.matrix = AfArrayToVector(aT);
            result.rows = cols;
            result.cols = rows;
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU Transpose failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    result.matrix.resize(cols, std::vector<double>(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.matrix[j][i] = A[i][j];
        }
    }
    result.rows = cols;
    result.cols = rows;
    result.success = true;
    return result;
}

MatrixResult LinearAlgebra::Inverse(const std::vector<std::vector<double>>& A) {
    MatrixResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for inversion";
        return result;
    }

    int n = static_cast<int>(A.size());

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array aInv = af::inverse(aA);
            result.matrix = AfArrayToVector(aInv);
            result.rows = n;
            result.cols = n;
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU Inverse failed (matrix may be singular): {}", e.what());
            result.error_message = "Matrix inversion failed - matrix may be singular";
            return result;
        }
    }
#endif

    // CPU fallback using Gauss-Jordan elimination
    std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n));

    // Create augmented matrix [A | I]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i][j] = A[i][j];
            aug[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        // Find pivot
        int maxRow = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(aug[row][col]) > std::abs(aug[maxRow][col])) {
                maxRow = row;
            }
        }
        std::swap(aug[col], aug[maxRow]);

        if (std::abs(aug[col][col]) < 1e-12) {
            result.error_message = "Matrix is singular or nearly singular";
            return result;
        }

        // Scale pivot row
        double pivot = aug[col][col];
        for (int j = 0; j < 2 * n; ++j) {
            aug[col][j] /= pivot;
        }

        // Eliminate column
        for (int row = 0; row < n; ++row) {
            if (row != col) {
                double factor = aug[row][col];
                for (int j = 0; j < 2 * n; ++j) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse from right half
    result.matrix.resize(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result.matrix[i][j] = aug[i][j + n];
        }
    }
    result.rows = n;
    result.cols = n;
    result.success = true;
    return result;
}

// ============================================================================
// Scalar Properties
// ============================================================================

ScalarResult LinearAlgebra::Determinant(const std::vector<std::vector<double>>& A) {
    ScalarResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for determinant";
        return result;
    }

    int n = static_cast<int>(A.size());

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            double det = af::det<double>(aA);
            result.value = det;
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU Determinant failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback using LU decomposition
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> U = A;
    int swaps = 0;

    for (int col = 0; col < n; ++col) {
        // Find pivot
        int maxRow = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(U[row][col]) > std::abs(U[maxRow][col])) {
                maxRow = row;
            }
        }

        if (maxRow != col) {
            std::swap(U[col], U[maxRow]);
            swaps++;
        }

        if (std::abs(U[col][col]) < 1e-12) {
            result.value = 0.0;
            result.success = true;
            return result;
        }

        for (int row = col + 1; row < n; ++row) {
            double factor = U[row][col] / U[col][col];
            for (int j = col; j < n; ++j) {
                U[row][j] -= factor * U[col][j];
            }
        }
    }

    // Determinant is product of diagonal of U, with sign from swaps
    double det = (swaps % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < n; ++i) {
        det *= U[i][i];
    }

    result.value = det;
    result.success = true;
    return result;
}

ScalarResult LinearAlgebra::Trace(const std::vector<std::vector<double>>& A) {
    ScalarResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for trace";
        return result;
    }

    int n = static_cast<int>(A.size());
    double trace = 0.0;
    for (int i = 0; i < n; ++i) {
        trace += A[i][i];
    }

    result.value = trace;
    result.success = true;
    return result;
}

ScalarResult LinearAlgebra::Rank(const std::vector<std::vector<double>>& A, double tolerance) {
    ScalarResult result;

    if (A.empty()) {
        result.value = 0;
        result.success = true;
        return result;
    }

    // Use SVD to compute rank (count singular values > tolerance)
    SVDResult svd = SVD(A, false);
    if (!svd.success) {
        result.error_message = "SVD failed: " + svd.error_message;
        return result;
    }

    int rank = 0;
    double maxSV = svd.S.empty() ? 0.0 : svd.S[0];
    double thresh = tolerance * std::max(svd.m, svd.n) * maxSV;

    for (double s : svd.S) {
        if (s > thresh) {
            rank++;
        }
    }

    result.value = static_cast<double>(rank);
    result.success = true;
    return result;
}

ScalarResult LinearAlgebra::FrobeniusNorm(const std::vector<std::vector<double>>& A) {
    ScalarResult result;

    if (A.empty()) {
        result.value = 0.0;
        result.success = true;
        return result;
    }

    double sum = 0.0;
    for (const auto& row : A) {
        for (double val : row) {
            sum += val * val;
        }
    }

    result.value = std::sqrt(sum);
    result.success = true;
    return result;
}

ScalarResult LinearAlgebra::ConditionNumber(const std::vector<std::vector<double>>& A) {
    ScalarResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    SVDResult svd = SVD(A, false);
    if (!svd.success) {
        result.error_message = "SVD failed: " + svd.error_message;
        return result;
    }

    if (svd.S.empty()) {
        result.error_message = "No singular values computed";
        return result;
    }

    double maxSV = svd.S.front();
    double minSV = svd.S.back();

    if (minSV < 1e-15) {
        result.value = std::numeric_limits<double>::infinity();
    } else {
        result.value = maxSV / minSV;
    }

    result.success = true;
    return result;
}

// ============================================================================
// Decompositions
// ============================================================================

EigenResult LinearAlgebra::Eigen(const std::vector<std::vector<double>>& A) {
    EigenResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for eigendecomposition";
        return result;
    }

    int n = static_cast<int>(A.size());
    result.n = n;

    // Note: ArrayFire v3 does not have built-in eigendecomposition (af::eigen removed)
    // Using CPU implementation via QR iteration algorithm
    // For future ArrayFire versions, GPU acceleration can be added back

    // CPU fallback: Power iteration for dominant eigenvalue (simplified)
    // For a full implementation, use QR algorithm or Jacobi method
    result.error_message = "CPU eigendecomposition not fully implemented - requires ArrayFire";

    // Simple power iteration for largest eigenvalue
    std::vector<double> v(n, 1.0);
    double lambda = 0.0;

    for (int iter = 0; iter < 1000; ++iter) {
        // Compute Av
        std::vector<double> Av(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Av[i] += A[i][j] * v[j];
            }
        }

        // Find max absolute value
        double maxVal = 0.0;
        for (double x : Av) {
            maxVal = std::max(maxVal, std::abs(x));
        }

        if (maxVal < 1e-12) break;

        // Normalize
        double newLambda = maxVal;
        for (int i = 0; i < n; ++i) {
            v[i] = Av[i] / maxVal;
        }

        if (std::abs(newLambda - lambda) < 1e-10) {
            lambda = newLambda;
            break;
        }
        lambda = newLambda;
    }

    result.eigenvalues.push_back(std::complex<double>(lambda, 0.0));
    result.eigenvectors.resize(n, std::vector<std::complex<double>>(1));
    for (int i = 0; i < n; ++i) {
        result.eigenvectors[i][0] = std::complex<double>(v[i], 0.0);
    }

    result.success = true;
    result.error_message = "Note: Only dominant eigenvalue computed (CPU fallback)";
    return result;
}

SVDResult LinearAlgebra::SVD(const std::vector<std::vector<double>>& A, bool full_matrices) {
    SVDResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    int rows, cols;
    GetDimensions(A, rows, cols);
    result.m = rows;
    result.n = cols;
    result.k = std::min(rows, cols);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array U, S, Vt;
            af::svd(U, S, Vt, aA);

            result.U = AfArrayToVector(U);
            result.Vt = AfArrayToVector(af::transpose(Vt));  // ArrayFire returns V, we want V^T

            // Extract singular values
            std::vector<float> svals(result.k);
            S.host(svals.data());
            result.S.resize(result.k);
            for (int i = 0; i < result.k; ++i) {
                result.S[i] = static_cast<double>(svals[i]);
            }

            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU SVD failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback: Simplified SVD using power iteration
    // For production, use LAPACK or similar
    result.error_message = "CPU SVD not fully implemented - requires ArrayFire for full SVD";
    result.S.resize(result.k, 0.0);
    result.success = false;
    return result;
}

QRResult LinearAlgebra::QR(const std::vector<std::vector<double>>& A) {
    QRResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    int rows, cols;
    GetDimensions(A, rows, cols);
    result.m = rows;
    result.n = cols;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array Q, R;
            af::qr(Q, R, aA);

            result.Q = AfArrayToVector(Q);
            result.R = AfArrayToVector(R);
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU QR failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback: Gram-Schmidt orthogonalization
    int k = std::min(rows, cols);
    result.Q.resize(rows, std::vector<double>(k, 0.0));
    result.R.resize(k, std::vector<double>(cols, 0.0));

    for (int j = 0; j < k; ++j) {
        // Copy column j of A to v
        std::vector<double> v(rows);
        for (int i = 0; i < rows; ++i) {
            v[i] = A[i][j];
        }

        // Orthogonalize against previous columns
        for (int i = 0; i < j; ++i) {
            double dot = 0.0;
            for (int r = 0; r < rows; ++r) {
                dot += result.Q[r][i] * A[r][j];
            }
            result.R[i][j] = dot;
            for (int r = 0; r < rows; ++r) {
                v[r] -= dot * result.Q[r][i];
            }
        }

        // Normalize
        double norm = 0.0;
        for (int r = 0; r < rows; ++r) {
            norm += v[r] * v[r];
        }
        norm = std::sqrt(norm);

        result.R[j][j] = norm;
        if (norm > 1e-12) {
            for (int r = 0; r < rows; ++r) {
                result.Q[r][j] = v[r] / norm;
            }
        }
    }

    result.success = true;
    return result;
}

CholeskyResult LinearAlgebra::Cholesky(const std::vector<std::vector<double>>& A) {
    CholeskyResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for Cholesky decomposition";
        return result;
    }

    int n = static_cast<int>(A.size());
    result.n = n;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array L;
            af::cholesky(L, aA, true);  // true = lower triangular

            result.L = AfArrayToVector(L);
            result.is_positive_definite = true;
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU Cholesky failed (matrix may not be positive definite): {}", e.what());
            result.error_message = "Cholesky decomposition failed - matrix may not be positive definite";
            result.is_positive_definite = false;
            return result;
        }
    }
#endif

    // CPU fallback: Cholesky-Banachiewicz algorithm
    result.L.resize(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;

            if (i == j) {
                for (int k = 0; k < j; ++k) {
                    sum += result.L[j][k] * result.L[j][k];
                }
                double val = A[j][j] - sum;
                if (val <= 0) {
                    result.error_message = "Matrix is not positive definite";
                    result.is_positive_definite = false;
                    return result;
                }
                result.L[j][j] = std::sqrt(val);
            } else {
                for (int k = 0; k < j; ++k) {
                    sum += result.L[i][k] * result.L[j][k];
                }
                result.L[i][j] = (A[i][j] - sum) / result.L[j][j];
            }
        }
    }

    result.is_positive_definite = true;
    result.success = true;
    return result;
}

LUResult LinearAlgebra::LU(const std::vector<std::vector<double>>& A) {
    LUResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }

    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for LU decomposition";
        return result;
    }

    int n = static_cast<int>(A.size());
    result.n = n;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array L, U, P;
            af::lu(L, U, P, aA);

            result.L = AfArrayToVector(L);
            result.U = AfArrayToVector(U);

            // Extract permutation indices from P matrix
            std::vector<int> perm(n);
            std::vector<float> Pdata(n * n);
            P.host(Pdata.data());
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (Pdata[j * n + i] > 0.5) {
                        perm[i] = j;
                        break;
                    }
                }
            }
            result.P = perm;

            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU LU failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback: Doolittle's method with partial pivoting
    result.L.resize(n, std::vector<double>(n, 0.0));
    result.U = A;
    result.P.resize(n);
    std::iota(result.P.begin(), result.P.end(), 0);  // P = [0, 1, 2, ..., n-1]

    for (int k = 0; k < n; ++k) {
        // Find pivot
        int maxRow = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(result.U[i][k]) > std::abs(result.U[maxRow][k])) {
                maxRow = i;
            }
        }

        if (maxRow != k) {
            std::swap(result.U[k], result.U[maxRow]);
            std::swap(result.P[k], result.P[maxRow]);
            std::swap(result.L[k], result.L[maxRow]);
        }

        result.L[k][k] = 1.0;

        for (int i = k + 1; i < n; ++i) {
            if (std::abs(result.U[k][k]) < 1e-12) {
                result.error_message = "Matrix is singular";
                return result;
            }
            result.L[i][k] = result.U[i][k] / result.U[k][k];
            for (int j = k; j < n; ++j) {
                result.U[i][j] -= result.L[i][k] * result.U[k][j];
            }
        }
    }

    result.success = true;
    return result;
}

// ============================================================================
// Linear Systems
// ============================================================================

MatrixResult LinearAlgebra::Solve(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
    MatrixResult result;

    if (A.empty() || b.empty()) {
        result.error_message = "Input matrices cannot be empty";
        return result;
    }

    if (!IsSquare(A)) {
        result.error_message = "Matrix A must be square for Solve";
        return result;
    }

    int n = static_cast<int>(A.size());
    int bRows, bCols;
    GetDimensions(b, bRows, bCols);

    if (bRows != n) {
        result.error_message = "Dimensions mismatch: A rows must equal b rows";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array ab = VectorToAfArray(b);
            af::array x = af::solve(aA, ab);
            result.matrix = AfArrayToVector(x);
            result.rows = n;
            result.cols = bCols;
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU Solve failed: {}", e.what());
            result.error_message = "Linear system solve failed - matrix may be singular";
            return result;
        }
    }
#endif

    // CPU fallback: Use LU decomposition
    LUResult lu = LU(A);
    if (!lu.success) {
        result.error_message = "LU decomposition failed: " + lu.error_message;
        return result;
    }

    // Apply permutation to b
    std::vector<std::vector<double>> Pb(n, std::vector<double>(bCols));
    for (int i = 0; i < n; ++i) {
        Pb[i] = b[lu.P[i]];
    }

    // Forward substitution: L * y = Pb
    std::vector<std::vector<double>> y(n, std::vector<double>(bCols));
    for (int c = 0; c < bCols; ++c) {
        for (int i = 0; i < n; ++i) {
            y[i][c] = Pb[i][c];
            for (int j = 0; j < i; ++j) {
                y[i][c] -= lu.L[i][j] * y[j][c];
            }
        }
    }

    // Back substitution: U * x = y
    result.matrix.resize(n, std::vector<double>(bCols));
    for (int c = 0; c < bCols; ++c) {
        for (int i = n - 1; i >= 0; --i) {
            result.matrix[i][c] = y[i][c];
            for (int j = i + 1; j < n; ++j) {
                result.matrix[i][c] -= lu.U[i][j] * result.matrix[j][c];
            }
            result.matrix[i][c] /= lu.U[i][i];
        }
    }

    result.rows = n;
    result.cols = bCols;
    result.success = true;
    return result;
}

MatrixResult LinearAlgebra::LeastSquares(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
    MatrixResult result;

    if (A.empty() || b.empty()) {
        result.error_message = "Input matrices cannot be empty";
        return result;
    }

    int rowsA, colsA, rowsB, colsB;
    GetDimensions(A, rowsA, colsA);
    GetDimensions(b, rowsB, colsB);

    if (rowsA != rowsB) {
        result.error_message = "A and b must have same number of rows";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable()) {
        try {
            af::array aA = VectorToAfArray(A);
            af::array ab = VectorToAfArray(b);
            af::array x = af::solve(aA, ab, AF_MAT_NONE);  // Least squares for non-square
            result.matrix = AfArrayToVector(x);
            result.rows = colsA;
            result.cols = colsB;
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[LinearAlgebra] GPU LeastSquares failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback: Normal equations (A^T * A) * x = A^T * b
    auto At = Transpose(A);
    if (!At.success) {
        result.error_message = "Transpose failed: " + At.error_message;
        return result;
    }

    auto AtA = Multiply(At.matrix, A);
    if (!AtA.success) {
        result.error_message = "Matrix multiplication failed: " + AtA.error_message;
        return result;
    }

    auto Atb = Multiply(At.matrix, b);
    if (!Atb.success) {
        result.error_message = "Matrix multiplication failed: " + Atb.error_message;
        return result;
    }

    return Solve(AtA.matrix, Atb.matrix);
}

// ============================================================================
// Matrix Properties
// ============================================================================

bool LinearAlgebra::IsSymmetric(const std::vector<std::vector<double>>& A, double tolerance) {
    if (!IsSquare(A)) return false;

    int n = static_cast<int>(A.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(A[i][j] - A[j][i]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

bool LinearAlgebra::IsPositiveDefinite(const std::vector<std::vector<double>>& A) {
    CholeskyResult chol = Cholesky(A);
    return chol.is_positive_definite;
}

bool LinearAlgebra::IsOrthogonal(const std::vector<std::vector<double>>& A, double tolerance) {
    if (!IsSquare(A)) return false;

    int n = static_cast<int>(A.size());
    auto At = Transpose(A);
    auto AtA = Multiply(At.matrix, A);

    if (!AtA.success) return false;

    // Check if AtA is identity
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(AtA.matrix[i][j] - expected) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

MatrixResult LinearAlgebra::Identity(int n) {
    MatrixResult result;
    if (n <= 0) {
        result.error_message = "Size must be positive";
        return result;
    }

    result.matrix.resize(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        result.matrix[i][i] = 1.0;
    }
    result.rows = n;
    result.cols = n;
    result.success = true;
    return result;
}

MatrixResult LinearAlgebra::Identity(int rows, int cols) {
    MatrixResult result;
    if (rows <= 0 || cols <= 0) {
        result.error_message = "Dimensions must be positive";
        return result;
    }

    result.matrix.resize(rows, std::vector<double>(cols, 0.0));
    int diag_len = std::min(rows, cols);
    for (int i = 0; i < diag_len; ++i) {
        result.matrix[i][i] = 1.0;
    }
    result.rows = rows;
    result.cols = cols;
    result.success = true;
    return result;
}

MatrixResult LinearAlgebra::Zeros(int n) {
    return Zeros(n, n);
}

MatrixResult LinearAlgebra::Zeros(int rows, int cols) {
    MatrixResult result;
    if (rows <= 0 || cols <= 0) {
        result.error_message = "Dimensions must be positive";
        return result;
    }

    result.matrix.resize(rows, std::vector<double>(cols, 0.0));
    result.rows = rows;
    result.cols = cols;
    result.success = true;
    return result;
}

MatrixResult LinearAlgebra::Ones(int n) {
    return Ones(n, n);
}

MatrixResult LinearAlgebra::Ones(int rows, int cols) {
    MatrixResult result;
    if (rows <= 0 || cols <= 0) {
        result.error_message = "Dimensions must be positive";
        return result;
    }

    result.matrix.resize(rows, std::vector<double>(cols, 1.0));
    result.rows = rows;
    result.cols = cols;
    result.success = true;
    return result;
}

MatrixResult LinearAlgebra::Diagonal(const std::vector<double>& diag) {
    MatrixResult result;
    if (diag.empty()) {
        result.error_message = "Diagonal cannot be empty";
        return result;
    }

    int n = static_cast<int>(diag.size());
    result.matrix.resize(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        result.matrix[i][i] = diag[i];
    }
    result.rows = n;
    result.cols = n;
    result.success = true;
    return result;
}

std::vector<double> LinearAlgebra::GetDiagonal(const std::vector<std::vector<double>>& A) {
    if (A.empty()) return {};

    int n = std::min(static_cast<int>(A.size()), static_cast<int>(A[0].size()));
    std::vector<double> diag(n);
    for (int i = 0; i < n; ++i) {
        diag[i] = A[i][i];
    }
    return diag;
}

MatrixResult LinearAlgebra::LowRankApproximation(const std::vector<std::vector<double>>& A, int k) {
    MatrixResult result;

    SVDResult svd = SVD(A, false);
    if (!svd.success) {
        result.error_message = "SVD failed: " + svd.error_message;
        return result;
    }

    if (k <= 0 || k > svd.k) {
        result.error_message = "k must be between 1 and min(m,n)";
        return result;
    }

    // Truncate to k components: A_k = U_k * S_k * V_k^T
    int m = svd.m;
    int n = svd.n;

    // Compute U_k * S_k
    std::vector<std::vector<double>> US(m, std::vector<double>(k));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            US[i][j] = svd.U[i][j] * svd.S[j];
        }
    }

    // Compute (U_k * S_k) * V_k^T
    result.matrix.resize(m, std::vector<double>(n, 0.0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int l = 0; l < k; ++l) {
                result.matrix[i][j] += US[i][l] * svd.Vt[l][j];
            }
        }
    }

    result.rows = m;
    result.cols = n;
    result.success = true;
    return result;
}

} // namespace cyxwiz
