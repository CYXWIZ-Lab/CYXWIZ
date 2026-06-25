// Windows compatibility
#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/linear_algebra.h"
#include "arrayfire_backend_utils.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>

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
    af::array materialized = arr;
    materialized.eval();
    materialized.host(flat.data());

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    // ArrayFire uses column-major order
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            result[r][c] = flat[c * rows + r];
        }
    }

    return result;
}

static std::string BuildMatrixContext(
    const char* matrix_name,
    int rows,
    int cols)
{
    return std::string(matrix_name ? matrix_name : "matrix") +
           "=[" + std::to_string(rows) + "x" + std::to_string(cols) + "]";
}

static std::string BuildMatrixContext(
    const char* left_name,
    int left_rows,
    int left_cols,
    const char* right_name,
    int right_rows,
    int right_cols)
{
    return BuildMatrixContext(left_name, left_rows, left_cols) +
           "; " +
           BuildMatrixContext(right_name, right_rows, right_cols);
}

static void LogLinearAlgebraFallbackOnce(
    const char* operation_name,
    const char* error_message,
    const std::string& matrix_context)
{
    const BackendFallbackReason reason = ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(matrix_context);
    if (ShouldLogArrayFireBackendFallbackOnce(operation_name, reason, context)) {
        spdlog::warn("{}",
            BuildArrayFireBackendFallbackMessage(
                operation_name,
                reason,
                reason != BackendFallbackReason::CudaJitParamOverflow,
                error_message,
                context));
    }
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

static bool IsRectangularMatrix(const std::vector<std::vector<double>>& A) {
    if (A.empty()) {
        return false;
    }
    const size_t cols = A[0].size();
    if (cols == 0) {
        return false;
    }
    for (const auto& row : A) {
        if (row.size() != cols) {
            return false;
        }
    }
    return true;
}

static std::vector<std::vector<double>> SymmetricAtA(const std::vector<std::vector<double>>& A,
                                                     int rows, int cols) {
    std::vector<std::vector<double>> result(cols, std::vector<double>(cols, 0.0));
    for (int i = 0; i < cols; ++i) {
        for (int j = i; j < cols; ++j) {
            double sum = 0.0;
            for (int r = 0; r < rows; ++r) {
                sum += A[r][i] * A[r][j];
            }
            result[i][j] = sum;
            result[j][i] = sum;
        }
    }
    return result;
}

static bool JacobiEigenSymmetric(std::vector<std::vector<double>> matrix,
                                 std::vector<double>& eigenvalues,
                                 std::vector<std::vector<double>>& eigenvectors) {
    const int n = static_cast<int>(matrix.size());
    if (n <= 0) {
        return false;
    }

    eigenvectors.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        eigenvectors[i][i] = 1.0;
    }

    const int max_iterations = std::max(50, 100 * n * n);
    const double tolerance = 1e-12;
    for (int iter = 0; iter < max_iterations; ++iter) {
        int p = 0;
        int q = 1;
        double max_offdiag = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double value = std::abs(matrix[i][j]);
                if (value > max_offdiag) {
                    max_offdiag = value;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_offdiag < tolerance) {
            break;
        }

        const double app = matrix[p][p];
        const double aqq = matrix[q][q];
        const double apq = matrix[p][q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double sign = tau >= 0.0 ? 1.0 : -1.0;
        const double t = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) {
                continue;
            }
            const double akp = matrix[k][p];
            const double akq = matrix[k][q];
            matrix[k][p] = c * akp - s * akq;
            matrix[p][k] = matrix[k][p];
            matrix[k][q] = s * akp + c * akq;
            matrix[q][k] = matrix[k][q];
        }

        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for (int k = 0; k < n; ++k) {
            const double vkp = eigenvectors[k][p];
            const double vkq = eigenvectors[k][q];
            eigenvectors[k][p] = c * vkp - s * vkq;
            eigenvectors[k][q] = s * vkp + c * vkq;
        }
    }

    eigenvalues.resize(n);
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = matrix[i][i];
    }
    return true;
}

static std::vector<std::complex<double>> Eigenvector2x2(
    const std::vector<std::vector<double>>& A,
    const std::complex<double>& lambda) {
    const std::complex<double> a(A[0][0], 0.0);
    const std::complex<double> b(A[0][1], 0.0);
    const std::complex<double> c(A[1][0], 0.0);
    const std::complex<double> d(A[1][1], 0.0);

    std::vector<std::complex<double>> vector;
    if (std::abs(b) >= std::abs(c) && std::abs(b) > 1e-12) {
        vector = {b, lambda - a};
    } else if (std::abs(c) > 1e-12) {
        vector = {lambda - d, c};
    } else {
        vector = {std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0)};
    }

    double norm = 0.0;
    for (const auto& value : vector) {
        norm += std::norm(value);
    }
    norm = std::sqrt(norm);
    if (norm <= 1e-12) {
        return {std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0)};
    }
    for (auto& value : vector) {
        value /= norm;
    }
    return vector;
}

static bool CompleteOrthonormalColumns(std::vector<std::vector<double>>& matrix,
                                       int rows,
                                       int cols,
                                       double tolerance = 1e-12) {
    for (int col = 0; col < cols; ++col) {
        std::vector<double> candidate(rows, 0.0);
        for (int row = 0; row < rows; ++row) {
            candidate[row] = matrix[row][col];
        }

        auto orthogonalize = [&]() {
            for (int prev = 0; prev < col; ++prev) {
                double dot = 0.0;
                for (int row = 0; row < rows; ++row) {
                    dot += candidate[row] * matrix[row][prev];
                }
                for (int row = 0; row < rows; ++row) {
                    candidate[row] -= dot * matrix[row][prev];
                }
            }
        };
        auto norm = [&]() {
            double value = 0.0;
            for (double entry : candidate) {
                value += entry * entry;
            }
            return std::sqrt(value);
        };

        orthogonalize();
        double length = norm();
        if (length <= tolerance) {
            bool found = false;
            for (int basis = 0; basis < rows && !found; ++basis) {
                std::fill(candidate.begin(), candidate.end(), 0.0);
                candidate[basis] = 1.0;
                orthogonalize();
                length = norm();
                found = length > tolerance;
            }
            if (!found) {
                return false;
            }
        }

        for (int row = 0; row < rows; ++row) {
            matrix[row][col] = candidate[row] / length;
        }
    }
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::Add",
                e.what(),
                BuildMatrixContext("A", rowsA, colsA, "B", rowsB, colsB));
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::Subtract",
                e.what(),
                BuildMatrixContext("A", rowsA, colsA, "B", rowsB, colsB));
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::Multiply",
                e.what(),
                BuildMatrixContext("A", rowsA, colsA, "B", rowsB, colsB));
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::ScalarMultiply",
                e.what(),
                BuildMatrixContext("A", rows, cols));
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::Transpose",
                e.what(),
                BuildMatrixContext("A", rows, cols));
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::Inverse",
                e.what(),
                BuildMatrixContext("A", n, n));
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::Determinant",
                e.what(),
                BuildMatrixContext("A", n, n));
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

    if (!IsRectangularMatrix(A)) {
        result.error_message = "Input matrix must be rectangular and non-empty";
        return result;
    }

    if (!IsSquare(A)) {
        result.error_message = "Matrix must be square for eigendecomposition";
        return result;
    }

    int n = static_cast<int>(A.size());
    result.n = n;

    // Note: ArrayFire v3 does not have built-in eigendecomposition (af::eigen removed)
    // For future ArrayFire versions, GPU acceleration can be added back

    if (IsSymmetric(A)) {
        std::vector<double> eigenvalues;
        std::vector<std::vector<double>> eigenvectors;
        if (!JacobiEigenSymmetric(A, eigenvalues, eigenvectors)) {
            result.error_message = "CPU symmetric eigensolver failed";
            return result;
        }

        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int left, int right) {
            return eigenvalues[left] > eigenvalues[right];
        });

        result.eigenvalues.resize(n);
        result.eigenvectors.assign(n, std::vector<std::complex<double>>(n));
        for (int out_col = 0; out_col < n; ++out_col) {
            const int source_col = order[out_col];
            result.eigenvalues[out_col] = std::complex<double>(eigenvalues[source_col], 0.0);
            for (int row = 0; row < n; ++row) {
                result.eigenvectors[row][out_col] =
                    std::complex<double>(eigenvectors[row][source_col], 0.0);
            }
        }

        result.success = true;
        return result;
    }

    if (n == 2) {
        const double a = A[0][0];
        const double b = A[0][1];
        const double c = A[1][0];
        const double d = A[1][1];
        const double trace = a + d;
        const double determinant = a * d - b * c;
        const double discriminant = trace * trace - 4.0 * determinant;

        std::complex<double> sqrt_discriminant;
        if (discriminant >= 0.0) {
            sqrt_discriminant = std::complex<double>(std::sqrt(discriminant), 0.0);
        } else {
            sqrt_discriminant = std::complex<double>(0.0, std::sqrt(-discriminant));
        }

        const std::complex<double> lambda0 =
            (std::complex<double>(trace, 0.0) + sqrt_discriminant) / 2.0;
        const std::complex<double> lambda1 =
            (std::complex<double>(trace, 0.0) - sqrt_discriminant) / 2.0;

        if (std::abs(lambda0 - lambda1) <= 1e-12) {
            result.error_message =
                "CPU nonsymmetric 2x2 eigendecomposition does not support repeated eigenvalues";
            return result;
        }

        result.eigenvalues = {lambda0, lambda1};
        result.eigenvectors.assign(2, std::vector<std::complex<double>>(2));
        const auto vector0 = Eigenvector2x2(A, lambda0);
        const auto vector1 = Eigenvector2x2(A, lambda1);
        for (int row = 0; row < 2; ++row) {
            result.eigenvectors[row][0] = vector0[row];
            result.eigenvectors[row][1] = vector1[row];
        }

        result.success = true;
        return result;
    }

    result.error_message =
        "CPU eigendecomposition supports symmetric matrices and nonsymmetric 2x2 matrices only";
    return result;
}

SVDResult LinearAlgebra::SVD(const std::vector<std::vector<double>>& A, bool full_matrices) {
    SVDResult result;

    if (A.empty()) {
        result.error_message = "Input matrix cannot be empty";
        return result;
    }
    if (!IsRectangularMatrix(A)) {
        result.error_message = "Input matrix must be rectangular and non-empty";
        return result;
    }

    int rows, cols;
    GetDimensions(A, rows, cols);
    result.m = rows;
    result.n = cols;
    result.k = std::min(rows, cols);

    const std::vector<std::vector<double>> ata = SymmetricAtA(A, rows, cols);
    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> eigenvectors;
    if (!JacobiEigenSymmetric(ata, eigenvalues, eigenvectors)) {
        result.error_message = "CPU SVD symmetric eigensolver failed";
        return result;
    }

    std::vector<int> order(cols);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int left, int right) {
        return eigenvalues[left] > eigenvalues[right];
    });

    const double zero_tolerance = 1e-12;
    const int u_cols = full_matrices ? rows : result.k;
    const int vt_rows = full_matrices ? cols : result.k;
    result.S.assign(result.k, 0.0);
    result.U.assign(rows, std::vector<double>(u_cols, 0.0));
    result.Vt.assign(vt_rows, std::vector<double>(cols, 0.0));

    for (int component = 0; component < vt_rows; ++component) {
        const int source = order[component];
        const double lambda = std::max(0.0, eigenvalues[source]);
        const double sigma = std::sqrt(lambda);
        if (component < result.k) {
            result.S[component] = sigma;
        }

        for (int col = 0; col < cols; ++col) {
            result.Vt[component][col] = eigenvectors[col][source];
        }

        if (component >= result.k) {
            continue;
        }
        if (sigma <= zero_tolerance) {
            continue;
        }

        for (int row = 0; row < rows; ++row) {
            double projection = 0.0;
            for (int col = 0; col < cols; ++col) {
                projection += A[row][col] * result.Vt[component][col];
            }
            result.U[row][component] = projection / sigma;
        }
    }

    if (full_matrices && !CompleteOrthonormalColumns(result.U, rows, u_cols, zero_tolerance)) {
        result.error_message = "CPU SVD failed to complete orthonormal U basis";
        result.U.clear();
        result.S.clear();
        result.Vt.clear();
        return result;
    }

    result.success = true;
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::QR",
                e.what(),
                BuildMatrixContext("A", rows, cols));
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::Cholesky",
                e.what(),
                BuildMatrixContext("A", n, n));
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
            P.eval();
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::LU",
                e.what(),
                BuildMatrixContext("A", n, n));
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::Solve",
                e.what(),
                BuildMatrixContext("A", n, n, "b", bRows, bCols));
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
            LogLinearAlgebraFallbackOnce(
                "LinearAlgebra::LeastSquares",
                e.what(),
                BuildMatrixContext("A", rowsA, colsA, "b", rowsB, colsB));
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

} // namespace cyxwiz
