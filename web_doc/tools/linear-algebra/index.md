# Linear Algebra Tools

GPU-accelerated linear algebra operations powered by ArrayFire for high-performance numerical computing.

## Overview

The Linear Algebra tools provide:
- **Matrix Operations** - Multiplication, inversion, decomposition
- **Eigenvalue Problems** - Eigenvalues, eigenvectors, SVD
- **Solvers** - Linear systems, least squares
- **Utilities** - Norms, ranks, conditions

## Tools Reference

### Matrix Operations Panel

```
+------------------------------------------------------------------+
|  Matrix Operations                                         [x]    |
+------------------------------------------------------------------+
|  MATRIX A                           MATRIX B                      |
|  Source: [tensor_A       v]         Source: [tensor_B       v]    |
|  Shape: (1000, 500)                 Shape: (500, 200)             |
|                                                                   |
|  OPERATION                                                        |
|  (o) Multiply (A @ B)    ( ) Element-wise Multiply                |
|  ( ) Add                 ( ) Subtract                             |
|  ( ) Transpose           ( ) Inverse                              |
|  ( ) Kronecker Product   ( ) Hadamard Product                     |
|                                                                   |
|  [ Execute ]                                                      |
|                                                                   |
|  RESULT                                                           |
|  +-----------------------------------------------------------+   |
|  | Output Shape      | (1000, 200)                            |   |
|  | Computation Time  | 2.34 ms (GPU)                          |   |
|  | Memory Used       | 800 KB                                 |   |
|  | Backend           | CUDA (RTX 4060)                        |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  PREVIEW (5x5 corner)                                             |
|  [[ 1.234  2.345  3.456  4.567  5.678]                            |
|   [ 2.345  3.456  4.567  5.678  6.789]                            |
|   [ 3.456  4.567  5.678  6.789  7.890]                            |
|   [ 4.567  5.678  6.789  7.890  8.901]                            |
|   [ 5.678  6.789  7.890  8.901  9.012]]                           |
|                                                                   |
|  [Save Result]  [Copy to Console]  [Visualize]                    |
+------------------------------------------------------------------+
```

### Matrix Decomposition Panel

```
+------------------------------------------------------------------+
|  Matrix Decomposition                                      [x]    |
+------------------------------------------------------------------+
|  Input Matrix: [covariance_matrix v]                              |
|  Shape: (100, 100)                                                |
|                                                                   |
|  DECOMPOSITION TYPE                                               |
|  (o) SVD (Singular Value Decomposition)                           |
|  ( ) Eigendecomposition                                           |
|  ( ) LU Decomposition                                             |
|  ( ) QR Decomposition                                             |
|  ( ) Cholesky Decomposition                                       |
|                                                                   |
|  OPTIONS                                                          |
|  [x] Full matrices  [ ] Compute only singular values              |
|                                                                   |
|  [ Decompose ]                                                    |
|                                                                   |
|  SVD RESULTS: A = U * S * V^T                                     |
|  +-----------------------------------------------------------+   |
|  | Component | Shape        | Description                    |   |
|  +-----------------------------------------------------------+   |
|  | U         | (100, 100)   | Left singular vectors         |   |
|  | S         | (100,)       | Singular values               |   |
|  | V^T       | (100, 100)   | Right singular vectors        |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  SINGULAR VALUES (Top 10)                                         |
|  +-----------------------------------------------------------+   |
|  | Index | Value     | Cumulative Variance                  |   |
|  +-----------------------------------------------------------+   |
|  | 0     | 45.678    | 52.3%                                |   |
|  | 1     | 23.456    | 67.8%                                |   |
|  | 2     | 12.345    | 78.2%                                |   |
|  | 3     | 8.901     | 84.5%                                |   |
|  | 4     | 5.678     | 88.7%                                |   |
|  | ...   | ...       | ...                                  |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  EXPLAINED VARIANCE PLOT                                          |
|  [Bar chart showing cumulative variance by component]             |
|                                                                   |
|  [Save Components]  [Reconstruct with k components]               |
+------------------------------------------------------------------+
```

### Linear System Solver

```
+------------------------------------------------------------------+
|  Linear System Solver                                      [x]    |
+------------------------------------------------------------------+
|  Solve: Ax = b                                                    |
|                                                                   |
|  Matrix A: [coefficient_matrix v]  Shape: (1000, 1000)            |
|  Vector b: [target_vector      v]  Shape: (1000,)                 |
|                                                                   |
|  SOLVER METHOD                                                    |
|  ( ) Direct (LU)         - Exact, O(n^3)                          |
|  (o) Least Squares       - Over/underdetermined                   |
|  ( ) Conjugate Gradient  - Large sparse systems                   |
|  ( ) GMRES               - Non-symmetric systems                  |
|                                                                   |
|  [ Solve ]                                                        |
|                                                                   |
|  SOLUTION                                                         |
|  +-----------------------------------------------------------+   |
|  | Solution Shape     | (1000,)                               |   |
|  | Solve Time         | 45.6 ms                               |   |
|  | Residual Norm      | 1.23e-10                              |   |
|  | Method Used        | Least Squares (QR)                    |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  SOLUTION VECTOR (First 10)                                       |
|  x = [1.234, 2.345, 3.456, 4.567, 5.678, 6.789, ...]              |
|                                                                   |
|  VERIFICATION                                                     |
|  ||Ax - b|| = 1.23e-10 (residual norm)                            |
|  ||x|| = 234.56 (solution norm)                                   |
|                                                                   |
|  [Save Solution]  [Verify]  [Sensitivity Analysis]                |
+------------------------------------------------------------------+
```

### Eigenvalue Analysis

```
+------------------------------------------------------------------+
|  Eigenvalue Analysis                                       [x]    |
+------------------------------------------------------------------+
|  Input Matrix: [square_matrix v]                                  |
|  Shape: (500, 500)                                                |
|                                                                   |
|  ANALYSIS TYPE                                                    |
|  (o) Full Eigendecomposition                                      |
|  ( ) Top-k Eigenvalues Only                                       |
|  ( ) Symmetric/Hermitian (optimized)                              |
|                                                                   |
|  Number of eigenvalues: [10    ] (for top-k)                      |
|                                                                   |
|  [ Compute Eigenvalues ]                                          |
|                                                                   |
|  EIGENVALUE RESULTS                                               |
|  +-----------------------------------------------------------+   |
|  | Index | Eigenvalue (Real) | Eigenvalue (Imag) | Magnitude |   |
|  +-----------------------------------------------------------+   |
|  | 0     | 123.456           | 0.000             | 123.456   |   |
|  | 1     | 89.012            | 0.000             | 89.012    |   |
|  | 2     | 45.678            | 12.345            | 47.317    |   |
|  | 3     | 45.678            | -12.345           | 47.317    |   |
|  | 4     | 23.456            | 0.000             | 23.456    |   |
|  | ...   | ...               | ...               | ...       |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  MATRIX PROPERTIES                                                |
|  +-----------------------------------------------------------+   |
|  | Property           | Value                                 |   |
|  +-----------------------------------------------------------+   |
|  | Trace              | 281.602 (sum of eigenvalues)          |   |
|  | Determinant        | 1.23e+45                              |   |
|  | Spectral Radius    | 123.456                               |   |
|  | Condition Number   | 5.26                                  |   |
|  | Rank               | 500 (full rank)                       |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Plot Spectrum]  [Save Eigenvectors]  [Power Iteration]          |
+------------------------------------------------------------------+
```

### Matrix Properties Panel

```
+------------------------------------------------------------------+
|  Matrix Properties                                         [x]    |
+------------------------------------------------------------------+
|  Input Matrix: [data_matrix v]                                    |
|  Shape: (1000, 500)                                               |
|                                                                   |
|  [ Analyze Properties ]                                           |
|                                                                   |
|  BASIC PROPERTIES                                                 |
|  +-----------------------------------------------------------+   |
|  | Property           | Value                                 |   |
|  +-----------------------------------------------------------+   |
|  | Dimensions         | 1000 x 500                            |   |
|  | Elements           | 500,000                               |   |
|  | Data Type          | float32                               |   |
|  | Memory Size        | 2.0 MB                                |   |
|  | Sparsity           | 15.3% zeros                           |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  NORMS                                                            |
|  +-----------------------------------------------------------+   |
|  | Norm Type          | Value                                 |   |
|  +-----------------------------------------------------------+   |
|  | Frobenius Norm     | 1234.567                              |   |
|  | L1 Norm (max col)  | 456.789                               |   |
|  | L-inf Norm (max row)| 234.567                              |   |
|  | Spectral Norm      | 123.456                               |   |
|  | Nuclear Norm       | 2345.678                              |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  NUMERICAL PROPERTIES                                             |
|  +-----------------------------------------------------------+   |
|  | Rank               | 498 (numerical)                       |   |
|  | Condition Number   | 1.23e+4                               |   |
|  | Numerical Stability| Moderate (condition > 1e3)            |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Export Report]  [Condition Analysis]                            |
+------------------------------------------------------------------+
```

## ArrayFire Operations

CyxWiz uses ArrayFire for GPU-accelerated linear algebra:

### Matrix Operations

```cpp
// C++ with ArrayFire (internal implementation)
#include <arrayfire.h>

af::array A = af::randu(1000, 500);
af::array B = af::randu(500, 200);

// Matrix multiplication
af::array C = af::matmul(A, B);

// Transpose
af::array At = af::transpose(A);

// Element-wise
af::array D = A * A;  // Hadamard product
```

### Decompositions

```cpp
// SVD
af::array U, S, Vt;
af::svd(U, S, Vt, A);

// LU decomposition
af::array L, U, P;
af::lu(L, U, P, A);

// QR decomposition
af::array Q, R;
af::qr(Q, R, A);

// Cholesky (for positive definite)
af::array L = af::cholesky(A);

// Eigenvalues
af::array eigenvalues, eigenvectors;
af::eigen(eigenvalues, eigenvectors, A);
```

### Solvers

```cpp
// Solve Ax = b
af::array x = af::solve(A, b);

// Least squares
af::array x = af::solve(A, b, AF_MAT_NONE);

// LU solve (pre-factored)
af::array x = af::solveLU(L, U, P, b);
```

## Scripting Functions

### Matrix Operations

```python
import pycyxwiz.linalg as la

# Matrix multiplication
C = la.matmul(A, B)
C = A @ B  # Operator overload

# Transpose
At = la.transpose(A)
At = A.T

# Inverse
A_inv = la.inv(A)

# Determinant
det = la.det(A)

# Trace
tr = la.trace(A)
```

### Decompositions

```python
# SVD
U, S, Vt = la.svd(A)
U, S, Vt = la.svd(A, full_matrices=False)  # Economy

# Eigendecomposition
eigenvalues, eigenvectors = la.eig(A)
eigenvalues = la.eigvals(A)  # Values only

# LU
P, L, U = la.lu(A)

# QR
Q, R = la.qr(A)

# Cholesky
L = la.cholesky(A)
```

### Solvers

```python
# Direct solve
x = la.solve(A, b)

# Least squares
x, residuals, rank, s = la.lstsq(A, b)

# Pseudo-inverse
A_pinv = la.pinv(A)
x = A_pinv @ b
```

### Norms and Properties

```python
# Norms
fro_norm = la.norm(A, 'fro')  # Frobenius
spectral_norm = la.norm(A, 2)  # Spectral
nuclear_norm = la.norm(A, 'nuc')  # Nuclear

# Properties
rank = la.matrix_rank(A)
cond = la.cond(A)
```

## Integration with Node Editor

### Linear Algebra Nodes

| Node | Inputs | Outputs | GPU Accelerated |
|------|--------|---------|-----------------|
| **MatMul** | A, B | C = A @ B | Yes |
| **Transpose** | A | A^T | Yes |
| **Inverse** | A | A^-1 | Yes |
| **SVD** | A | U, S, V | Yes |
| **Solve** | A, b | x | Yes |
| **Norm** | A | scalar | Yes |

### Example Pipeline

```
[Data Matrix] -> [SVD] -> [Truncate] -> [Reconstruct] -> [Low-rank Approx]
                  |
                  v
           [Singular Values] -> [Plot]
```

## Performance Comparison

### Backend Selection

| Backend | Hardware | Performance | Use Case |
|---------|----------|-------------|----------|
| **CUDA** | NVIDIA GPU | Fastest | Large matrices |
| **OpenCL** | AMD/Intel GPU | Fast | Cross-platform |
| **CPU** | Any CPU | Baseline | Small matrices |

### Benchmarks (1000x1000 matrix multiplication)

| Backend | Time | Speedup |
|---------|------|---------|
| CPU (single-threaded) | 2,500 ms | 1x |
| CPU (multi-threaded) | 450 ms | 5.6x |
| OpenCL (Intel UHD) | 120 ms | 20.8x |
| CUDA (RTX 3060) | 15 ms | 166x |
| CUDA (RTX 4090) | 5 ms | 500x |

## Best Practices

### Numerical Stability

1. **Check condition number** before solving systems
2. **Use appropriate decomposition** for the problem
3. **Prefer SVD** for rank-deficient matrices
4. **Use Cholesky** for positive definite matrices (2x faster than LU)

### Memory Management

1. **Use in-place operations** when possible
2. **Release intermediate results** promptly
3. **Batch operations** to minimize GPU transfers
4. **Use appropriate precision** (float32 vs float64)

---

**Next**: [Signal Processing Tools](../signal-processing/index.md) | [Optimization Tools](../optimization/index.md)
