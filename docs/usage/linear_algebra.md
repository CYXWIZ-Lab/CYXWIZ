# Linear algebra tools

The Matrix Calculator, SVD, QR, Cholesky and Eigen panels compute on the device
selected in the Devices tab (CUDA, OpenCL, or ArrayFire's CPU backend), the same
device training uses. There is no separate CPU code path: on a CPU selection the
work runs on ArrayFire's CPU backend. A build without ArrayFire reports that the
operation needs ArrayFire.

## Precision

Results are computed in float64 when the device supports it (NVIDIA CUDA and
OpenCL, the CPU backend), otherwise in float32. Every operation is checked
against PyTorch `torch.linalg` in float64 to 1e-8.

## What each tool returns

| Tool | Result |
| --- | --- |
| Matrix Calculator | Add, subtract, multiply, scalar multiply, transpose, inverse, determinant, trace, rank, Frobenius norm, condition number (largest / smallest singular value) |
| SVD | Singular values (descending), U and Vᵀ; "full matrices" gives square U (m×m) and Vᵀ (n×n), otherwise the thin form (m×k, k×n). Low-rank approximation keeps the top k singular triplets |
| QR | Reduced QR, as PyTorch: Q is m×k with orthonormal columns, R is k×n upper triangular, k = min(m, n) |
| Cholesky | Lower-triangular L with A = L·Lᵀ; a matrix that is not positive definite is refused |
| Eigen | Symmetric matrices of any size (real eigenvalues, descending, orthonormal eigenvectors) and nonsymmetric 2×2 matrices (complex pairs included) |

## Errors and limits

- Inverse and Solve refuse a singular or nearly singular matrix (smallest
  singular value below 1e-12 of the largest; 1e-6 in float32). Its determinant
  is reported as 0.
- Eigen refuses nonsymmetric matrices larger than 2×2: ArrayFire has no general
  eigensolver.
- The Cholesky panel's "Symmetric" / "Pos. Def." status updates when you edit
  the matrix; verification products (L·Lᵀ, Q·R) are computed with the result.
