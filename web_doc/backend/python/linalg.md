# Linear Algebra (`cx.linalg`)

The `linalg` submodule provides MATLAB-style linear algebra functions for matrix operations, decompositions, and solving linear systems.

## Overview

```python
import pycyxwiz as cx

# Matrix creation
A = cx.linalg.eye(3)
B = cx.linalg.zeros(3, 4)

# Decompositions
U, S, Vt = cx.linalg.svd(A)

# Solving systems
x = cx.linalg.solve(A, b)
```

## Matrix Creation

### `eye(n)` / `eye(rows, cols)`

Create identity matrix.

```python
# Square identity
I = cx.linalg.eye(3)
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]

# Rectangular (1s on diagonal)
I = cx.linalg.eye(3, 4)
# [[1, 0, 0, 0],
#  [0, 1, 0, 0],
#  [0, 0, 1, 0]]
```

**Parameters:**
- `n` (int): Size for square matrix
- `rows`, `cols` (int): Dimensions for rectangular matrix

**Returns:** 2D list of floats

---

### `zeros(n)` / `zeros(rows, cols)`

Create zero matrix.

```python
Z = cx.linalg.zeros(3)      # 3x3 zeros
Z = cx.linalg.zeros(2, 4)   # 2x4 zeros
```

**Parameters:**
- `n` (int): Size for square matrix
- `rows`, `cols` (int): Dimensions for rectangular matrix

**Returns:** 2D list of floats

---

### `ones(n)` / `ones(rows, cols)`

Create matrix of ones.

```python
O = cx.linalg.ones(3)       # 3x3 ones
O = cx.linalg.ones(2, 4)    # 2x4 ones
```

**Parameters:**
- `n` (int): Size for square matrix
- `rows`, `cols` (int): Dimensions for rectangular matrix

**Returns:** 2D list of floats

---

### `diag(d)`

Create diagonal matrix from vector.

```python
D = cx.linalg.diag([1, 2, 3])
# [[1, 0, 0],
#  [0, 2, 0],
#  [0, 0, 3]]
```

**Parameters:**
- `d` (list): Diagonal elements

**Returns:** 2D list (square matrix)

## Matrix Decompositions

### `svd(A, full_matrices=False)`

Singular Value Decomposition: `A = U @ diag(S) @ Vt`

```python
A = [[1, 2], [3, 4], [5, 6]]
U, S, Vt = cx.linalg.svd(A)

# U: Left singular vectors (3x2 for full_matrices=False)
# S: Singular values (list of 2 values)
# Vt: Right singular vectors transposed (2x2)
```

**Parameters:**
- `A` (2D list): Input matrix
- `full_matrices` (bool): If True, return full U and Vt matrices. Default: False

**Returns:** Tuple `(U, S, Vt)`
- `U`: 2D list, left singular vectors
- `S`: 1D list, singular values (descending order)
- `Vt`: 2D list, right singular vectors (transposed)

**Example:**
```python
import numpy as np

A = [[1, 2], [3, 4]]
U, S, Vt = cx.linalg.svd(A)

# Verify reconstruction
U_np = np.array(U)
S_np = np.diag(S)
Vt_np = np.array(Vt)
A_reconstructed = U_np @ S_np @ Vt_np
# Should match A
```

---

### `eig(A)`

Eigenvalue decomposition: `A @ v = lambda * v`

```python
A = [[1, 2], [2, 1]]
eigenvalues, eigenvectors = cx.linalg.eig(A)

# eigenvalues: [-1, 3]
# eigenvectors: [[0.707, 0.707], [-0.707, 0.707]]
```

**Parameters:**
- `A` (2D list): Square input matrix

**Returns:** Tuple `(eigenvalues, eigenvectors)`
- `eigenvalues`: 1D list of eigenvalues
- `eigenvectors`: 2D list, columns are eigenvectors

---

### `qr(A)`

QR decomposition: `A = Q @ R`

```python
A = [[1, 2], [3, 4], [5, 6]]
Q, R = cx.linalg.qr(A)

# Q: Orthogonal matrix (3x2 for thin QR)
# R: Upper triangular (2x2)
```

**Parameters:**
- `A` (2D list): Input matrix (m x n)

**Returns:** Tuple `(Q, R)`
- `Q`: 2D list, orthogonal matrix
- `R`: 2D list, upper triangular matrix

---

### `chol(A)`

Cholesky decomposition: `A = L @ L.T` (A must be positive definite)

```python
A = [[4, 2], [2, 5]]
L = cx.linalg.chol(A)

# L @ L.T = A
```

**Parameters:**
- `A` (2D list): Symmetric positive definite matrix

**Returns:** 2D list, lower triangular matrix L

**Raises:** RuntimeError if A is not positive definite

---

### `lu(A)`

LU decomposition with partial pivoting: `P @ A = L @ U`

```python
A = [[1, 2], [3, 4]]
L, U, P = cx.linalg.lu(A)

# P @ A = L @ U
```

**Parameters:**
- `A` (2D list): Square input matrix

**Returns:** Tuple `(L, U, P)`
- `L`: 2D list, lower triangular with ones on diagonal
- `U`: 2D list, upper triangular
- `P`: 2D list, permutation matrix

## Matrix Properties

### `det(A)`

Compute determinant.

```python
A = [[1, 2], [3, 4]]
d = cx.linalg.det(A)  # -2.0
```

**Parameters:**
- `A` (2D list): Square matrix

**Returns:** float, determinant value

---

### `rank(A, tol=1e-10)`

Compute matrix rank.

```python
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Rank 2 (rows are linearly dependent)
r = cx.linalg.rank(A)  # 2
```

**Parameters:**
- `A` (2D list): Input matrix
- `tol` (float): Tolerance for zero singular values. Default: 1e-10

**Returns:** int, matrix rank

---

### `trace(A)`

Compute trace (sum of diagonal elements).

```python
A = [[1, 2], [3, 4]]
t = cx.linalg.trace(A)  # 5.0 (1 + 4)
```

**Parameters:**
- `A` (2D list): Square matrix

**Returns:** float, trace value

---

### `norm(A)`

Compute Frobenius norm: `sqrt(sum(a_ij^2))`

```python
A = [[1, 2], [3, 4]]
n = cx.linalg.norm(A)  # 5.477
```

**Parameters:**
- `A` (2D list): Input matrix

**Returns:** float, Frobenius norm

---

### `cond(A)`

Compute condition number (ratio of largest to smallest singular value).

```python
A = [[1, 2], [3, 4]]
c = cx.linalg.cond(A)  # ~14.93

# High condition number = ill-conditioned matrix
```

**Parameters:**
- `A` (2D list): Input matrix

**Returns:** float, condition number

## Matrix Operations

### `inv(A)`

Compute matrix inverse.

```python
A = [[1, 2], [3, 4]]
A_inv = cx.linalg.inv(A)
# [[-2, 1], [1.5, -0.5]]

# Verify: A @ A_inv = I
```

**Parameters:**
- `A` (2D list): Square invertible matrix

**Returns:** 2D list, inverse matrix

**Raises:** RuntimeError if A is singular

---

### `transpose(A)`

Compute matrix transpose.

```python
A = [[1, 2, 3], [4, 5, 6]]
A_T = cx.linalg.transpose(A)
# [[1, 4], [2, 5], [3, 6]]
```

**Parameters:**
- `A` (2D list): Input matrix

**Returns:** 2D list, transposed matrix

---

### `solve(A, b)`

Solve linear system `Ax = b`.

```python
A = [[2, 1], [1, 3]]
b = [[5], [6]]
x = cx.linalg.solve(A, b)
# x such that A @ x = b
```

**Parameters:**
- `A` (2D list): Coefficient matrix (n x n)
- `b` (2D list): Right-hand side (n x m)

**Returns:** 2D list, solution x

**Raises:** RuntimeError if A is singular

---

### `lstsq(A, b)`

Least squares solution to `Ax = b` (for overdetermined systems).

```python
# Overdetermined system (more equations than unknowns)
A = [[1, 1], [2, 1], [1, 2]]
b = [[1], [2], [3]]
x = cx.linalg.lstsq(A, b)
# Minimizes ||Ax - b||^2
```

**Parameters:**
- `A` (2D list): Coefficient matrix (m x n, m >= n)
- `b` (2D list): Right-hand side (m x k)

**Returns:** 2D list, least squares solution

---

### `matmul(A, B)`

Matrix multiplication.

```python
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = cx.linalg.matmul(A, B)
# [[19, 22], [43, 50]]
```

**Parameters:**
- `A` (2D list): Left matrix (m x n)
- `B` (2D list): Right matrix (n x p)

**Returns:** 2D list, product matrix (m x p)

## Complete Example

```python
import pycyxwiz as cx
import numpy as np

# Create test matrix
A = [[4, 12, -16],
     [12, 37, -43],
     [-16, -43, 98]]

print("Matrix A:")
for row in A:
    print(row)

# Properties
print(f"\nDeterminant: {cx.linalg.det(A)}")
print(f"Rank: {cx.linalg.rank(A)}")
print(f"Trace: {cx.linalg.trace(A)}")
print(f"Norm: {cx.linalg.norm(A):.4f}")
print(f"Condition: {cx.linalg.cond(A):.4f}")

# Cholesky (A is positive definite)
L = cx.linalg.chol(A)
print("\nCholesky L:")
for row in L:
    print([f"{x:.4f}" for x in row])

# SVD
U, S, Vt = cx.linalg.svd(A)
print(f"\nSVD singular values: {S}")

# Solve system
b = [[1], [2], [3]]
x = cx.linalg.solve(A, b)
print(f"\nSolution to Ax = b: {x}")

# Verify
A_np = np.array(A)
x_np = np.array(x)
residual = A_np @ x_np - np.array(b)
print(f"Residual norm: {np.linalg.norm(residual):.2e}")
```

## Error Handling

```python
# Singular matrix
try:
    A = [[1, 2], [2, 4]]  # Rank 1, singular
    inv = cx.linalg.inv(A)
except RuntimeError as e:
    print(f"Error: {e}")  # Matrix is singular

# Non-positive definite
try:
    A = [[1, 2], [2, 1]]  # Not positive definite
    L = cx.linalg.chol(A)
except RuntimeError as e:
    print(f"Error: {e}")  # Matrix is not positive definite
```

---

**Next**: [Signal Processing](signal.md) | [Back to Index](index.md)
