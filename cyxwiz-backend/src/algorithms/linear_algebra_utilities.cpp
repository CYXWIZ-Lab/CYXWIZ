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
