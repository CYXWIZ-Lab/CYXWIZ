"""PyTorch (torch.linalg, float64) fixtures for the host LinearAlgebra API (TOFIX140, linear algebra on the device).

Each case names an operation, its input matrices and PyTorch's answer. Decompositions whose signs are not
unique (SVD, QR, symmetric eigenvectors) carry the sign-free facts the test checks: singular values,
eigenvalues, |diag R|; the test checks reconstruction and orthonormality itself.

    py -3.12 generate_linear_algebra_fixtures.py
"""

import json
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "linear_algebra_pytorch.json"
torch.set_default_dtype(torch.float64)


def mat(t):
    return [[round(float(v), 15) for v in row] for row in t.tolist()]


def vec(t):
    return [round(float(v), 15) for v in t.tolist()]


def random(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(rows, cols, generator=g) * 2.0 - 1.0


def main():
    square = random(4, 4, 1) + 2.0 * torch.eye(4)          # nonsymmetric, well conditioned
    other = random(4, 4, 2)
    symmetric = (lambda m: m + m.T)(random(5, 5, 3))
    spd = (lambda m: m @ m.T + 4.0 * torch.eye(4))(random(4, 4, 4))
    tall = random(6, 3, 5)
    wide = random(3, 5, 6)
    rhs = random(4, 2, 7)
    tall_rhs = random(6, 2, 8)
    singular = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    rank2 = random(5, 2, 9) @ random(2, 4, 10)
    complex_pair = torch.tensor([[1.0, -2.0], [3.0, 0.5]])  # eigenvalues 0.75 +- 2.44i
    real_pair = torch.tensor([[4.0, 1.0], [2.0, 3.0]])      # eigenvalues 5, 2

    cases = []

    def add(op, inputs, **expected):
        cases.append({"op": op, "inputs": {k: mat(v) for k, v in inputs.items()}, **expected})

    add("add", {"A": square, "B": other}, matrix=mat(square + other))
    add("subtract", {"A": square, "B": other}, matrix=mat(square - other))
    add("multiply", {"A": tall, "B": wide}, matrix=mat(tall @ wide))
    add("scalar_multiply", {"A": wide}, scalar=-2.5, matrix=mat(-2.5 * wide))
    add("transpose", {"A": tall}, matrix=mat(tall.T))
    add("inverse", {"A": square}, matrix=mat(torch.linalg.inv(square)))
    add("inverse_singular", {"A": singular})
    for name, a in [("square", square), ("spd", spd), ("singular", singular)]:
        add("determinant", {"A": a}, value=float(torch.linalg.det(a)), label=name)
    add("trace", {"A": square}, value=float(torch.trace(square)))
    for name, a in [("square", square), ("tall", tall), ("rank2", rank2), ("singular", singular)]:
        add("rank", {"A": a}, value=int(torch.linalg.matrix_rank(a)), label=name)
        add("singular_values", {"A": a}, values=vec(torch.linalg.svdvals(a)), label=name)
    add("frobenius", {"A": tall}, value=float(torch.linalg.matrix_norm(tall, "fro")))
    for name, a in [("square", square), ("tall", tall), ("wide", wide)]:
        add("condition", {"A": a}, value=float(torch.linalg.cond(a)), label=name)
    add("eigen_symmetric", {"A": symmetric}, values=vec(torch.linalg.eigvalsh(symmetric).flip(0)))
    add("eigen_symmetric", {"A": spd}, values=vec(torch.linalg.eigvalsh(spd).flip(0)))
    for name, a in [("complex_pair", complex_pair), ("real_pair", real_pair)]:
        ev = torch.linalg.eigvals(a)
        ev = ev[torch.argsort(ev.real, descending=True)]
        add("eigen_2x2", {"A": a}, real=vec(ev.real), imag=vec(ev.imag), label=name)
    for name, a in [("square", square), ("tall", tall), ("wide", wide)]:
        r = torch.linalg.qr(a, mode="reduced").R
        add("qr", {"A": a}, abs_diag_r=vec(torch.diagonal(r).abs()), label=name)
    add("cholesky", {"A": spd}, matrix=mat(torch.linalg.cholesky(spd)))
    add("cholesky_not_pd", {"A": symmetric})
    p, l, u = torch.linalg.lu(square)  # A = P L U, so row i of P^T A is A[perm[i]]
    add("lu", {"A": square}, L=mat(l), U=mat(u), perm=[int(i) for i in torch.argmax(p, dim=0).tolist()])
    add("solve", {"A": square, "B": rhs}, matrix=mat(torch.linalg.solve(square, rhs)))
    add("solve_singular", {"A": singular, "B": random(3, 1, 11)})
    add("least_squares", {"A": tall, "B": tall_rhs}, matrix=mat(torch.linalg.lstsq(tall, tall_rhs).solution))
    uu, ss, vv = torch.linalg.svd(rank2 + 0.01 * random(5, 4, 12), full_matrices=False)
    noisy = uu @ torch.diag(ss) @ vv
    add("low_rank", {"A": noisy}, k=2, matrix=mat(uu[:, :2] @ torch.diag(ss[:2]) @ vv[:2]))
    for name, a, expected in [("symmetric", symmetric, True), ("square", square, False)]:
        add("is_symmetric", {"A": a}, flag=expected, label=name)
    q = torch.linalg.qr(tall, mode="reduced").Q
    for name, a, expected in [("reduced_q", q, True), ("square", square, False)]:
        add("is_orthogonal", {"A": a}, flag=expected, label=name)

    OUT.write_text(json.dumps({"schema_version": 1, "torch_version": torch.__version__, "cases": cases}, indent=1))
    print(f"wrote {OUT} ({len(cases)} cases, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
