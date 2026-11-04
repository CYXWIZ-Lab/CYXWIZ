"""Truth for Tools > Linear Algebra (core/linear_algebra_presentation, TOFIX129 D9): numpy's
answers for the window's operations on the panels' own examples and a few more.

    py -3.12 generate_linear_algebra_presentation_fixtures.py
"""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "linear_algebra_presentation.json"


def mat(a):
    return [[float(v) for v in row] for row in np.atleast_2d(a)]


cases = []


def add(name, op, a, b=None, k=None, low_rank=None, **expected):
    case = {"name": name, "op": op, "a": mat(a)}
    if b is not None:
        case["b"] = mat(b)
    if k is not None:
        case["k"] = k
    if low_rank is not None:
        case["low_rank"] = low_rank
    case["expected"] = expected
    cases.append(case)


A12 = np.arange(1, 13, dtype=float).reshape(4, 3)
P = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)
E = np.array([[4, -2, 1], [-2, 4, -2], [1, -2, 4]], dtype=float)
QR_A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41], [-1, 1, 0]], dtype=float)
SPD = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)
INDEFINITE = np.array([[1, 2], [2, 1]], dtype=float)
SINGULAR = np.array([[1, 2], [2, 4]], dtype=float)
NONSYM = np.array([[1, 2, 0], [0, 1, 3], [4, 0, 1]], dtype=float)
LU_A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)

add("add", "add", P, B, matrix=mat(P + B))
add("subtract", "subtract", P, B, matrix=mat(P - B))
add("multiply", "multiply", P, B, matrix=mat(P @ B))
add("multiply_mismatch", "multiply", A12, P, error="A has 3 columns and B has 2 rows")
add("scale", "scale", P, k=2.5, matrix=mat(2.5 * P))
add("transpose", "transpose", A12, matrix=mat(A12.T))
add("inverse", "inverse", P, matrix=mat(np.linalg.inv(P)), condition=float(np.linalg.cond(P)))
add("inverse_singular", "inverse", SINGULAR, error="A is singular (rank 1 of 2)")
add("determinant", "determinant", E, det=float(np.linalg.det(E)), trace=float(np.trace(E)))
add("determinant_singular", "determinant", SINGULAR, det=0.0)
add("rank", "rank", A12, rank=int(np.linalg.matrix_rank(A12)))
add("norm", "norm", A12, norm=float(np.linalg.norm(A12)))
add("condition", "condition", P, condition=float(np.linalg.cond(P)))
add("solve", "solve", P, np.array([[5], [6]]), matrix=mat(np.linalg.solve(P, [[5], [6]])), condition=float(np.linalg.cond(P)))
add("solve_singular", "solve", SINGULAR, np.array([[5], [6]]), error="A is singular (rank 1 of 2)")
x, residual, rank, _ = np.linalg.lstsq(A12, np.array([[1], [2], [2], [4]], dtype=float), rcond=None)
add("least_squares_rank_deficient", "least_squares", A12, np.array([[1], [2], [2], [4]]), matrix=mat(x),
    miss=float(np.linalg.norm(A12 @ x - np.array([[1], [2], [2], [4]]))), rank=int(rank))
w = np.linalg.eigvalsh(E)[::-1]
add("eigen_symmetric", "eigen", E, values=[float(v) for v in w], det=float(np.prod(w)), trace=float(np.sum(w)))
add("eigen_nonsymmetric_3x3", "eigen", NONSYM, error="Eigen takes symmetric matrices of any size")
s = np.linalg.svd(A12, compute_uv=False)
U, S, Vt = np.linalg.svd(A12, full_matrices=False)
add("svd", "svd", A12, low_rank=1, values=[float(v) for v in s], rank=int(np.linalg.matrix_rank(A12)),
    low_rank_matrix=mat(U[:, :1] * S[:1] @ Vt[:1]), share=[float(v) for v in s**2 / np.sum(s**2)])
add("qr", "qr", QR_A, abs_diag_r=[float(v) for v in np.abs(np.diag(np.linalg.qr(QR_A)[1]))], rank=3)
add("cholesky", "cholesky", SPD, matrix=mat(np.linalg.cholesky(SPD)))
add("cholesky_indefinite", "cholesky", INDEFINITE, error="A is symmetric but not positive definite: its eigenvalues are 3.0000 · -1.0000")
add("cholesky_nonsymmetric", "cholesky", NONSYM, error="A is not symmetric")
add("lu", "lu", LU_A, det=float(np.linalg.det(LU_A)))

OUT.write_text(json.dumps({"numpy_version": np.__version__, "cases": cases}, indent=1))
print(f"wrote {OUT} ({len(cases)} cases)")
