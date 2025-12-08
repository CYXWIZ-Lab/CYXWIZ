#pragma once

#include "api_export.h"
#include <vector>
#include <string>
#include <complex>

namespace cyxwiz {

// ============================================================================
// Result Structures
// ============================================================================

struct CYXWIZ_API MatrixResult {
    std::vector<std::vector<double>> matrix;
    int rows = 0;
    int cols = 0;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API ScalarResult {
    double value = 0.0;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API EigenResult {
    std::vector<std::complex<double>> eigenvalues;
    std::vector<std::vector<std::complex<double>>> eigenvectors;
    int n = 0;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API SVDResult {
    std::vector<std::vector<double>> U;      // Left singular vectors (m x m or m x k)
    std::vector<double> S;                    // Singular values
    std::vector<std::vector<double>> Vt;     // Right singular vectors transposed (n x n or k x n)
    int m = 0;  // Rows of original matrix
    int n = 0;  // Cols of original matrix
    int k = 0;  // Number of singular values (min(m,n))
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API QRResult {
    std::vector<std::vector<double>> Q;      // Orthogonal matrix (m x m or m x k)
    std::vector<std::vector<double>> R;      // Upper triangular (k x n or m x n)
    int m = 0;
    int n = 0;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API CholeskyResult {
    std::vector<std::vector<double>> L;      // Lower triangular (A = L * L^T)
    int n = 0;
    bool is_positive_definite = false;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API LUResult {
    std::vector<std::vector<double>> L;      // Lower triangular
    std::vector<std::vector<double>> U;      // Upper triangular
    std::vector<int> P;                       // Permutation indices
    int n = 0;
    bool success = false;
    std::string error_message;
};

// ============================================================================
// Linear Algebra Class
// ============================================================================

class CYXWIZ_API LinearAlgebra {
public:
    // ==================== Basic Operations ====================

    // Matrix addition: C = A + B
    static MatrixResult Add(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B
    );

    // Matrix subtraction: C = A - B
    static MatrixResult Subtract(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B
    );

    // Matrix multiplication: C = A * B
    static MatrixResult Multiply(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B
    );

    // Scalar multiplication: C = scalar * A
    static MatrixResult ScalarMultiply(
        const std::vector<std::vector<double>>& A,
        double scalar
    );

    // Matrix transpose: B = A^T
    static MatrixResult Transpose(
        const std::vector<std::vector<double>>& A
    );

    // Matrix inverse: B = A^(-1)
    static MatrixResult Inverse(
        const std::vector<std::vector<double>>& A
    );

    // ==================== Scalar Properties ====================

    // Determinant of square matrix
    static ScalarResult Determinant(
        const std::vector<std::vector<double>>& A
    );

    // Trace (sum of diagonal elements)
    static ScalarResult Trace(
        const std::vector<std::vector<double>>& A
    );

    // Matrix rank
    static ScalarResult Rank(
        const std::vector<std::vector<double>>& A,
        double tolerance = 1e-10
    );

    // Frobenius norm
    static ScalarResult FrobeniusNorm(
        const std::vector<std::vector<double>>& A
    );

    // Condition number (ratio of largest to smallest singular value)
    static ScalarResult ConditionNumber(
        const std::vector<std::vector<double>>& A
    );

    // ==================== Decompositions ====================

    // Eigenvalue decomposition (for square matrices)
    static EigenResult Eigen(
        const std::vector<std::vector<double>>& A
    );

    // Singular Value Decomposition: A = U * S * V^T
    static SVDResult SVD(
        const std::vector<std::vector<double>>& A,
        bool full_matrices = false  // If false, return thin SVD
    );

    // QR decomposition: A = Q * R
    static QRResult QR(
        const std::vector<std::vector<double>>& A
    );

    // Cholesky decomposition: A = L * L^T (for positive definite matrices)
    static CholeskyResult Cholesky(
        const std::vector<std::vector<double>>& A
    );

    // LU decomposition with partial pivoting: P * A = L * U
    static LUResult LU(
        const std::vector<std::vector<double>>& A
    );

    // ==================== Linear Systems ====================

    // Solve linear system: A * x = b
    static MatrixResult Solve(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& b
    );

    // Least squares solution: minimize ||A * x - b||^2
    static MatrixResult LeastSquares(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& b
    );

    // ==================== Matrix Properties ====================

    // Check if matrix is symmetric
    static bool IsSymmetric(
        const std::vector<std::vector<double>>& A,
        double tolerance = 1e-10
    );

    // Check if matrix is positive definite
    static bool IsPositiveDefinite(
        const std::vector<std::vector<double>>& A
    );

    // Check if matrix is orthogonal (A^T * A = I)
    static bool IsOrthogonal(
        const std::vector<std::vector<double>>& A,
        double tolerance = 1e-10
    );

    // ==================== Utility ====================

    // Create identity matrix
    static MatrixResult Identity(int n);

    // Create zero matrix
    static MatrixResult Zeros(int rows, int cols);

    // Create matrix filled with ones
    static MatrixResult Ones(int rows, int cols);

    // Create diagonal matrix from vector
    static MatrixResult Diagonal(const std::vector<double>& diag);

    // Extract diagonal from matrix
    static std::vector<double> GetDiagonal(
        const std::vector<std::vector<double>>& A
    );

    // Low-rank approximation using SVD (keep top k singular values)
    static MatrixResult LowRankApproximation(
        const std::vector<std::vector<double>>& A,
        int k
    );

private:
    // Helper: Check matrix dimensions
    static bool ValidateDimensions(
        const std::vector<std::vector<double>>& A,
        int expected_rows,
        int expected_cols
    );

    // Helper: Check if square matrix
    static bool IsSquare(const std::vector<std::vector<double>>& A);

    // Helper: Get matrix dimensions
    static void GetDimensions(
        const std::vector<std::vector<double>>& A,
        int& rows,
        int& cols
    );
};

} // namespace cyxwiz
