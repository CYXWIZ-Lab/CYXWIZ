// Windows compatibility
#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/linear_algebra.h"
#include <algorithm>
#include <vector>

// Undefine Windows macros that conflict with std::min/max
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz {

// ============================================================================
// Utility Functions (host matrix constructors; no compute)
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

} // namespace cyxwiz
