---
title: Linear Algebra window
description: The Tools > Linear Algebra window for matrix work, and the PCA, Truncated SVD, Whitening, Linear Solve and Linear Projection nodes.
type: reference
area: general
audience: user
status: experimental
since: 1.0.8
owner: main-window
last_reviewed: 2026-11-25
related:
  - optimization-and-calculus.md
  - ../user-guide/data-studio/dimensionality-reduction.md
  - ../concepts/steps-learn-from-train-rows.md
redirect_from:
  - ../usage/linear_algebra.md
---

**Tools > Linear Algebra** is one window for matrix work: you type a
matrix, paste one or read one from a table, and the answer is worked out as
you edit. Five graph nodes do the same work on every run of a graph: PCA,
Truncated SVD, Whitening, Linear Solve and Linear Projection. They learn
from the Train rows.

The window and the nodes share one backend on the device selected in
**Preferences > Devices** (CUDA, OpenCL, or ArrayFire's CPU backend). There
is no separate CPU code path: on a CPU selection the work runs on
ArrayFire's CPU backend.

The menu items **Matrix Calculator**, **Eigenvalue Decomposition**, **SVD
(Singular Value Decomposition)**, **QR Decomposition** and **Cholesky
Decomposition** open this window on that operation.

## The window

The operations are in four groups along the top. Hover over one to see
what it does.

| Group | Operations | Second input |
| --- | --- | --- |
| Combine | A + B, A - B, A × B, k × A | Matrix B, or the number k |
| Describe | Transpose, Inverse, Determinant, Rank, Norm (Frobenius), Condition (largest / smallest singular value) | None |
| Solve | A x = b (a square A that is not singular), Least squares (any A) | Right-hand side b |
| Factor | Eigen, SVD, QR, Cholesky, LU | None |

### Matrix A

- **Typed**: a grid of up to 64 × 64, with - / + for rows and columns.
  Below it are quick fills: **Identity**, **Zero**, **Symmetrize**,
  **Random**, **Transpose** and **Example** (a matrix that suits the
  operation).
- **Paste**: rows on lines, values split by tabs, commas, semicolons or
  spaces. A first line or column of names is left out. Every row must have
  the same count, up to 64 × 64.
- **From a table**: any table of the session (a loaded Data Input, a node's
  result), with the number columns you tick. **Matrix** takes either the
  rows themselves, or the columns' correlation or covariance (n - 1),
  worked out on the device. Rows with a missing value in a ticked column
  are left out, and the note under the table says how many. A table is read
  up to 1,000,000 values (rows × columns), counted before it is read. A
  larger table is refused, with a pointer to its correlation or covariance,
  or to sampling it in Prepare. **Read again** reads the table again after
  a run changed it.

B and b are typed, or come **From a table** the same way. When A and b come
from two different tables they line up row by row. The row counts must
match, and a missing value is refused there, because a row cannot be left
out of one side only.

### The answer

The answer shows:

- a sentence (for example "Least squares: x = (3.0630, -2.0064) brings A x
  closest to b. A x misses b by 14.0493.");
- its figures (rank, determinant, condition, the residual);
- bars for the singular values or eigenvalues, with each one's share;
- one tab per result matrix (U, Vᵀ, Q, R, L, ...);
- a check that the factors give A back ("U S V^T gives A back: largest
  difference 1.8e-15").

The SVD's **Low rank** tab keeps the top *k* directions you choose. Values
below 1e-10 of the matrix's scale are shown as 0, as rounding noise.

If an operation cannot run, the window says why and names the operation
that can. For example, a singular A has no inverse and points to Least
squares, and a non-symmetric A has no Cholesky factor and points to LU.

Under the answer:

- **Copy** copies the sentence and the open tab as tab-separated rows.
- **Export CSV...** saves the open tab.
- **Use as A** makes a result matrix the next A.

### Into the graph

An answer read from tables can become a node of the open graph. The node
then does the same work on every run, fitted on the Train rows. A button
appears under the answer when it applies:

| Answer | Button | The node |
| --- | --- | --- |
| SVD of a table's rows | **Add as a step: Truncated SVD** | `n_components` = the Low rank *k*, the ticked columns |
| SVD or Eigen of a covariance | **Add as a step: PCA**, or **Add as a step: Whitening** | PCA centred; Whitening by its PCA directions |
| SVD or Eigen of a correlation | **Add as a step: PCA** | PCA with **scale** on |
| A x = b or Least squares, b from a table | **Keep as a node: Linear Solve** | A branch reading A's table; b's table goes into its **b** input when it is another table |
| A × B, B from another table | **Add as a step: Linear Projection** | B's table goes into its **Matrix** input |

Linear Projection is offered only when B uses every number column of its
table, because the node takes them all as its outputs.

A step goes in the data path after the table's node. It keeps the table's
columns with its own beside them (**keep_columns** on), so the nodes after
it still find theirs. A branch reads the table and leaves the data path as
it is. The message under the buttons says where the node went. Ctrl+Z on
the canvas undoes it.

## The nodes

PCA, Truncated SVD and Whitening learn from the Train rows when a Data
Split marks them, as scikit-learn's `fit(X_train).transform(X)`. Without a
Split they learn from every row. They keep what they learned as fitted
state, like the scalers. In **Properties > Fitted preprocessing state**:

- **Mode**: *fit_transform* or *transform_only*.
- **Save fitted state**: write the state to a `.cyxstate.json` artifact.
- **State artifact path**: the artifact *transform_only* maps new rows
  with.

Each of the three reads the feature columns (empty: every number column
but the label). The output is Float32 columns alone, with the label as `y`
and the split carried, or the input with Float64 columns beside it
(**keep_columns**, null on rows missing a feature). Each direction is
signed so its largest weight is positive.

- **PCA**: centred (and with **scale**, standardised) directions of the
  covariance, `pc_0` ... See
  [Dimensionality reduction](../user-guide/data-studio/dimensionality-reduction.md#pca).
- **Truncated SVD**: the singular value decomposition of the Train rows,
  uncentred, as scikit-learn's `TruncatedSVD`: each row becomes x V_k,
  `svd_0` ... Uncentred suits word counts and TF-IDF (latent semantic
  analysis). It reads a TF-IDF or Count step's sparse columns as they are,
  without forming the dense table.
- **Whitening**: centred, rotated onto the covariance's directions, and
  divided by each direction's standard deviation.
  - **method** *pca* (scikit-learn's `PCA(whiten=True)`) outputs
    `white_0` ... for the **n_components** strongest directions (0: every
    one).
  - **method** *zca* rotates back onto the columns (`<column>_white`), the
    whitened data closest to the input.
  - **epsilon** is added to each variance before its root. A direction with
    no variance needs it above 0.
- **Linear Solve**: A is the **A columns** of the Data input (empty: every
  number column but b's), and b is the **b columns** of the **b** input or
  of the same table. **method**:
  - *auto*: the one solution for a square A of full rank, else least
    squares;
  - *solve*: refuses any other A;
  - *least_squares*: the shortest of the best x.

  The output has one row per column of A (`column`, then `x_<b column>`),
  fitted on the Train rows when a Split marks them. A second table on the
  **b** pin is a parameter of the step, not training rows: it lines up row
  by row with the Data input.
- **Linear Projection**: multiplies every row by the matrix on its
  **Matrix** input. The matrix's number columns are the outputs (with
  **prefix**). Its rows weigh the data's columns in order, or as its text
  column names them, for example a matrix exported from this window or
  another run's loadings. The Matrix table is a parameter, not training
  rows.

The example `examples/cyxgraph/linear_algebra/sensor_calibration.cyxgraph`
finds a meter's calibration weights from three sensors with Linear Solve
(least squares); its README explains it. The data is made up, so no
download is needed.

## Precision

Results are computed in float64 when the device supports it (NVIDIA CUDA
and OpenCL, the CPU backend), otherwise in float32. Every operation is
checked against PyTorch `torch.linalg` and NumPy in float64 to 1e-8. On a
GPU without float64, such as an Intel Iris Xe, it is checked at float32
precision (5e-5 relative). The nodes are checked against scikit-learn.

- **Rank** counts the singular values above the compute precision's
  epsilon times max(m, n), as `torch.linalg.matrix_rank` does, so float32
  rounding is not counted as rank.
- **A tall matrix** (more than 4,096 rows and twice as many rows as
  columns) takes its SVD through the Gram matrix AᵀA, so the m × m U is
  never formed. Its rank threshold is sqrt(epsilon) times the columns.
- **Least squares** drops singular values below LAPACK `gelsd`'s cutoff
  (epsilon times max(m, n) of the largest), so a rank-deficient A gets the
  shortest x. On the Gram route the cutoff is the rank threshold, so Rank
  and Least squares agree on which directions count.

## Errors and limits

- Inverse and A x = b refuse a singular matrix (rank below full) and point
  to Least squares. Its determinant is reported as 0.
- Eigen takes symmetric matrices of any size (real eigenvalues, descending,
  orthonormal eigenvectors), and others up to 2 × 2 (complex pairs
  included). ArrayFire has no general eigensolver.
- Cholesky needs a symmetric positive definite A. LU reorders rows (A with
  its rows reordered = L U).

## Related

- [Optimization and Calculus window](optimization-and-calculus.md)
- [Dimensionality reduction](../user-guide/data-studio/dimensionality-reduction.md)
- [Steps learn from the Train rows](../concepts/steps-learn-from-train-rows.md)
