// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

#include "api_export.h"
#include "tensor.h"
#include <cstdint>
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

struct CYXWIZ_API TensorResult {
    Tensor tensor;
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

// A sparse matrix in compressed rows (CSR) the caller owns: row r's entries
// are column_indices / values [row_offsets[r], row_offsets[r + 1]). A view,
// so nothing is copied (TF-IDF and Count features are kept this way).
struct CYXWIZ_API CsrView {
    int rows = 0;
    int cols = 0;
    const int* row_offsets = nullptr;     // rows + 1
    const int* column_indices = nullptr;  // nnz
    const float* values = nullptr;        // nnz
};

// A dense matrix the caller owns, column-major (column j is data[j * rows]
// .. data[(j + 1) * rows - 1]): a table's number columns read once into one
// buffer go to the device as they are, nothing repacked.
struct CYXWIZ_API DenseView {
    int64_t rows = 0;
    int cols = 0;
    const double* data = nullptr;
};

struct CYXWIZ_API ColumnSummaryResult {
    std::vector<double> mean;                 // per column
    std::vector<double> sd;                   // population (divides by n)
    std::vector<std::vector<double>> cross;   // (A - mean)^T (A - mean), cols x cols
    int rows = 0;
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

    // Matrix rank: singular values above tolerance * max(rows, cols) * the largest;
    // the tolerance is at least the compute precision's epsilon.
    static ScalarResult Rank(
        const std::vector<std::vector<double>>& A,
        double tolerance = 1e-10
    );
    static ScalarResult Rank(const DenseView& A, double tolerance = 1e-10);

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
    static MatrixResult LeastSquares(const DenseView& A, const DenseView& b);

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

    // Create identity matrix (square)
    static MatrixResult Identity(int n);

    // Create identity matrix (non-square) - 1s on main diagonal
    static MatrixResult Identity(int rows, int cols);

    // Create zero matrix (square)
    static MatrixResult Zeros(int n);

    // Create zero matrix
    static MatrixResult Zeros(int rows, int cols);

    // Create matrix filled with ones (square)
    static MatrixResult Ones(int n);

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

    // ==================== Tables of rows ====================
    // One upload and one download each, with no n x d temporary on the host.

    // Each column's mean and population standard deviation (divides by n),
    // and the columns' cross products about their means (divided by n - 1:
    // the covariance), from one upload of A.
    static ColumnSummaryResult ColumnSummary(const std::vector<std::vector<double>>& A);
    static ColumnSummaryResult ColumnSummary(const DenseView& A);

    // The k leading eigenvectors (as rows) and eigenvalues of a symmetric
    // positive semi-definite S (d x d), by a randomized subspace iteration
    // on the device (scikit-learn's randomized TruncatedSVD on S; a fixed
    // seed): for a few of many directions, where a full SVD of S is slow.
    static SVDResult LeadingEigenvectors(const std::vector<std::vector<double>>& S, int k, int iterations = 8);

    // ((A - 1 shift) / divisor) W: each column shifted and divided by its
    // entries (empty: 0 and 1), then times W (cols x k).
    static MatrixResult AffineProjection(const std::vector<std::vector<double>>& A, const std::vector<double>& shift,
                                         const std::vector<double>& divisor, const std::vector<std::vector<double>>& W);
    // The same of a DenseView, into `out` column-major (rows x W's columns).
    static bool AffineProjection(const DenseView& A, const std::vector<double>& shift, const std::vector<double>& divisor,
                                 const std::vector<std::vector<double>>& W, std::vector<double>& out, std::string& error);

    // ==================== Sparse (CSR) ====================

    // X^T X (cols x cols) of a CSR matrix, on the device, a block of rows at
    // a time (X is never made dense whole).
    static MatrixResult SparseGram(const CsrView& X);

    // X B for a CSR X (rows x cols) and a dense B (cols x k), on the device.
    static MatrixResult SparseMultiply(const CsrView& X, const std::vector<std::vector<double>>& B);

    // ==================== Tensor-First Operations ====================

    // Tensor matrix multiplication: C = A * B
    static TensorResult Multiply(
        const Tensor& A,
        const Tensor& B
    );

    // Tensor transpose: B = A^T
    static TensorResult Transpose(
        const Tensor& A
    );

    // Tensor inverse: B = A^(-1)
    static TensorResult Inverse(
        const Tensor& A
    );

    // Tensor Frobenius norm
    static ScalarResult FrobeniusNorm(
        const Tensor& A
    );

    // Tensor linear solve: A * x = b
    static TensorResult Solve(
        const Tensor& A,
        const Tensor& b
    );

    // Tensor least squares: minimize ||A * x - b||^2
    static TensorResult LeastSquares(
        const Tensor& A,
        const Tensor& b
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
