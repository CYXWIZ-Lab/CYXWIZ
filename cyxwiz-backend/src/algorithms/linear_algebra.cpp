// Windows compatibility
#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/linear_algebra.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdint>
#include <limits>

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

static cyxwiz::DataType AfTypeToTensorType(af::dtype dtype) {
    switch (dtype) {
        case af::dtype::f32: return cyxwiz::DataType::Float32;
        case af::dtype::f64: return cyxwiz::DataType::Float64;
        case af::dtype::s32: return cyxwiz::DataType::Int32;
        case af::dtype::s64: return cyxwiz::DataType::Int64;
        case af::dtype::u8:  return cyxwiz::DataType::UInt8;
        default: return cyxwiz::DataType::Float64;
    }
}

static cyxwiz::Tensor AfArrayToTensorWithShape(const af::array& arr, const std::vector<size_t>& shape) {
    cyxwiz::Tensor out(shape, AfTypeToTensorType(arr.type()));
    if (out.NumBytes() > 0) {
        arr.host(out.Data());
    }
    return out;
}
#endif

static double TensorValueAsDouble(const Tensor& t, size_t idx) {
    switch (t.GetDataType()) {
        case DataType::Float64:
            return static_cast<const double*>(t.Data())[idx];
        case DataType::Float32:
            return static_cast<double>(static_cast<const float*>(t.Data())[idx]);
        case DataType::Int32:
            return static_cast<double>(static_cast<const int32_t*>(t.Data())[idx]);
        case DataType::Int64:
            return static_cast<double>(static_cast<const int64_t*>(t.Data())[idx]);
        case DataType::UInt8:
            return static_cast<double>(static_cast<const uint8_t*>(t.Data())[idx]);
        default:
            return 0.0;
    }
}

static std::vector<std::vector<double>> Tensor2DToMatrix(const Tensor& t) {
    const auto& shape = t.Shape();
    if (shape.size() != 2) {
        return {};
    }

    const size_t rows = shape[0];
    const size_t cols = shape[1];
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols));

    if (rows == 0 || cols == 0) {
        return out;
    }

    if (t.GetDataType() == DataType::Float64) {
        const double* src = static_cast<const double*>(t.Data());
        for (size_t r = 0; r < rows; ++r) {
            std::memcpy(out[r].data(), src + r * cols, cols * sizeof(double));
        }
        return out;
    }

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            out[r][c] = TensorValueAsDouble(t, r * cols + c);
        }
    }
    return out;
}

static std::vector<std::vector<double>> TensorVectorOr2DToMatrix(const Tensor& t) {
    const auto& shape = t.Shape();
    if (shape.size() == 2) {
        return Tensor2DToMatrix(t);
    }
    if (shape.size() == 1) {
        const size_t n = shape[0];
        std::vector<std::vector<double>> out(n, std::vector<double>(1));
        for (size_t i = 0; i < n; ++i) {
            out[i][0] = TensorValueAsDouble(t, i);
        }
        return out;
    }
    return {};
}

static Tensor MatrixToTensor(const std::vector<std::vector<double>>& mat, bool squeeze_single_col) {
    if (mat.empty()) {
        if (squeeze_single_col) {
            return Tensor({0}, DataType::Float64);
        }
        return Tensor({0, 0}, DataType::Float64);
    }

    const size_t rows = mat.size();
    const size_t cols = mat[0].size();

    if (squeeze_single_col && cols == 1) {
        Tensor out({rows}, DataType::Float64);
        double* dst = static_cast<double*>(out.Data());
        for (size_t r = 0; r < rows; ++r) {
            dst[r] = mat[r][0];
        }
        return out;
    }

    Tensor out({rows, cols}, DataType::Float64);
    double* dst = static_cast<double*>(out.Data());
    for (size_t r = 0; r < rows; ++r) {
        std::memcpy(dst + r * cols, mat[r].data(), cols * sizeof(double));
    }
    return out;
}

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

// ============================================================================
// Tensor-First Operations
// ============================================================================

TensorResult LinearAlgebra::Multiply(const Tensor& A, const Tensor& B) {
    TensorResult result;

    const auto& shapeA = A.Shape();
    const auto& shapeB = B.Shape();
    if (shapeA.size() != 2 || shapeB.size() != 2) {
        result.error_message = "A and B must be 2D tensors for matrix multiplication";
        return result;
    }

    const size_t colsA = shapeA[1];
    const size_t rowsB = shapeB[0];
    if (colsA != rowsB) {
        result.error_message = "Matrix A columns must equal Matrix B rows for multiplication";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArrayRowMajor2D();
        af::array aB = B.GetArrayRowMajor2D();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }
        if (aB.type() != af::dtype::f32 && aB.type() != af::dtype::f64) {
            aB = aB.as(af::dtype::f64);
        }

        af::array aC = af::matmul(aA, aB);
        result.tensor = Tensor::FromArrayRowMajor2D(aC);
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        spdlog::warn("[LinearAlgebra] Tensor Multiply AF path failed, fallback to CPU: {}", e.what());
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    auto B_mat = Tensor2DToMatrix(B);
    auto cpu_result = Multiply(A_mat, B_mat);
    if (!cpu_result.success) {
        result.error_message = cpu_result.error_message;
        return result;
    }

    result.tensor = MatrixToTensor(cpu_result.matrix, false);
    result.success = true;
    return result;
}

TensorResult LinearAlgebra::Transpose(const Tensor& A) {
    TensorResult result;

    const auto& shape = A.Shape();
    if (shape.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }

    const size_t rows = shape[0];
    const size_t cols = shape[1];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArray();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }

        af::array aT = af::transpose(aA);
        result.tensor = AfArrayToTensorWithShape(aT, {cols, rows});
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        spdlog::warn("[LinearAlgebra] Tensor Transpose AF path failed, fallback to CPU: {}", e.what());
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    auto cpu_result = Transpose(A_mat);
    if (!cpu_result.success) {
        result.error_message = cpu_result.error_message;
        return result;
    }

    result.tensor = MatrixToTensor(cpu_result.matrix, false);
    result.success = true;
    return result;
}

TensorResult LinearAlgebra::Inverse(const Tensor& A) {
    TensorResult result;

    const auto& shape = A.Shape();
    if (shape.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }
    if (shape[0] != shape[1]) {
        result.error_message = "Matrix must be square for inversion";
        return result;
    }

    const size_t n = shape[0];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArray();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }

        af::array aInv = af::inverse(aA);
        result.tensor = AfArrayToTensorWithShape(aInv, {n, n});
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        spdlog::warn("[LinearAlgebra] Tensor Inverse AF path failed, fallback to CPU: {}", e.what());
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    auto cpu_result = Inverse(A_mat);
    if (!cpu_result.success) {
        result.error_message = cpu_result.error_message;
        return result;
    }

    result.tensor = MatrixToTensor(cpu_result.matrix, false);
    result.success = true;
    return result;
}

ScalarResult LinearAlgebra::FrobeniusNorm(const Tensor& A) {
    ScalarResult result;

    const auto& shape = A.Shape();
    if (shape.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArray();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }
        af::array sq = aA * aA;
        result.value = std::sqrt(af::sum<double>(af::flat(sq)));
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        spdlog::warn("[LinearAlgebra] Tensor FrobeniusNorm AF path failed, fallback to CPU: {}", e.what());
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    return FrobeniusNorm(A_mat);
}

TensorResult LinearAlgebra::Solve(const Tensor& A, const Tensor& b) {
    TensorResult result;

    const auto& shapeA = A.Shape();
    const auto& shapeB = b.Shape();

    if (shapeA.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }
    if (shapeA[0] != shapeA[1]) {
        result.error_message = "Matrix A must be square for Solve";
        return result;
    }
    if (shapeB.size() != 1 && shapeB.size() != 2) {
        result.error_message = "b must be a 1D or 2D tensor";
        return result;
    }

    const size_t n = shapeA[0];
    const bool b_was_vector = (shapeB.size() == 1);
    const size_t b_rows = b_was_vector ? shapeB[0] : shapeB[0];
    const size_t b_cols = b_was_vector ? 1 : shapeB[1];
    if (b_rows != n) {
        result.error_message = "Dimensions mismatch: A rows must equal b rows";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArray();
        af::array aB = b.GetArray();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }
        if (aB.type() != af::dtype::f32 && aB.type() != af::dtype::f64) {
            aB = aB.as(af::dtype::f64);
        }
        if (b_was_vector) {
            aB = af::moddims(aB, static_cast<dim_t>(n), static_cast<dim_t>(1));
        }

        af::array x = af::solve(aA, aB);
        if (b_was_vector) {
            af::array x_vec = af::moddims(x, static_cast<dim_t>(n));
            result.tensor = AfArrayToTensorWithShape(x_vec, {n});
        } else {
            result.tensor = AfArrayToTensorWithShape(x, {n, b_cols});
        }
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        spdlog::warn("[LinearAlgebra] Tensor Solve AF path failed, fallback to CPU: {}", e.what());
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    auto b_mat = TensorVectorOr2DToMatrix(b);
    auto cpu_result = Solve(A_mat, b_mat);
    if (!cpu_result.success) {
        result.error_message = cpu_result.error_message;
        return result;
    }

    result.tensor = MatrixToTensor(cpu_result.matrix, b_was_vector);
    result.success = true;
    return result;
}

TensorResult LinearAlgebra::LeastSquares(const Tensor& A, const Tensor& b) {
    TensorResult result;

    const auto& shapeA = A.Shape();
    const auto& shapeB = b.Shape();

    if (shapeA.size() != 2) {
        result.error_message = "A must be a 2D tensor";
        return result;
    }
    if (shapeB.size() != 1 && shapeB.size() != 2) {
        result.error_message = "b must be a 1D or 2D tensor";
        return result;
    }

    const size_t rowsA = shapeA[0];
    const size_t colsA = shapeA[1];
    const bool b_was_vector = (shapeB.size() == 1);
    const size_t rowsB = b_was_vector ? shapeB[0] : shapeB[0];
    const size_t colsB = b_was_vector ? 1 : shapeB[1];
    if (rowsA != rowsB) {
        result.error_message = "A and b must have same number of rows";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array aA = A.GetArray();
        af::array aB = b.GetArray();
        if (aA.type() != af::dtype::f32 && aA.type() != af::dtype::f64) {
            aA = aA.as(af::dtype::f64);
        }
        if (aB.type() != af::dtype::f32 && aB.type() != af::dtype::f64) {
            aB = aB.as(af::dtype::f64);
        }
        if (b_was_vector) {
            aB = af::moddims(aB, static_cast<dim_t>(rowsB), static_cast<dim_t>(1));
        }

        af::array x = af::solve(aA, aB, AF_MAT_NONE);
        if (b_was_vector) {
            af::array x_vec = af::moddims(x, static_cast<dim_t>(colsA));
            result.tensor = AfArrayToTensorWithShape(x_vec, {colsA});
        } else {
            result.tensor = AfArrayToTensorWithShape(x, {colsA, colsB});
        }
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        spdlog::warn("[LinearAlgebra] Tensor LeastSquares AF path failed, fallback to CPU: {}", e.what());
    }
#endif

    auto A_mat = Tensor2DToMatrix(A);
    auto b_mat = TensorVectorOr2DToMatrix(b);
    auto cpu_result = LeastSquares(A_mat, b_mat);
    if (!cpu_result.success) {
        result.error_message = cpu_result.error_message;
        return result;
    }

    result.tensor = MatrixToTensor(cpu_result.matrix, b_was_vector);
    result.success = true;
    return result;
}

} // namespace cyxwiz
