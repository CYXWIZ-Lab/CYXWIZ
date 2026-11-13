#!/usr/bin/env python3
"""Calibrating sensors with pycyxwiz.linalg: least squares, rank, SVD, eigen.

What it shows: the operations behind Tools > Linear Algebra and the Linear
Solve / PCA nodes, from Python, on the device the Engine computes on. Three
sensors and a reference meter are made up for this example (seeded, no
dataset needed): the meter reads 2.0 s1 - 0.5 s2 + 1.5 s3 + 0.3. Least
squares finds those weights back from the readings; a fourth "sensor" that is
only s1 + s2 makes A rank-deficient, and least squares then gives the
shortest of the best weights; the SVD of the centred readings and the
eigenvalues of their covariance give the same directions (PCA). Every answer
is checked against numpy.linalg (docs/usage/linear_algebra.md).

Run it with the Python pycyxwiz is built for, naming the folder that holds
pycyxwiz (and, if different, the folder of its DLLs); it needs numpy:

    python examples/python/linear_algebra_calibration.py build/windows-release/bin/Release

Expected result:

    weights: s1 2.0006, s2 -0.4993, s3 1.4998, offset 0.3000 (made with 2, -0.5, 1.5, 0.3)
    A: 200 x 4, rank 4, condition 1.4
    with s4 = s1 + s2: rank 4 of 5; shortest weights s1 1.5001, s2 -0.9997, s3 1.4998, s4 0.5004, offset 0.3000
    singular values of the centred readings: 14.06, 13.49, 11.19
    PCA from the covariance: variances 0.9937, 0.9150, 0.6291 (= singular values squared / 199)
    QR gives A back: 8.9e-16; Cholesky of the covariance gives it back: 1.1e-16
    every answer matches numpy.linalg

(the "gives A back" figures are rounding: about 1e-15 in float64, 1e-7 on a
GPU without float64).

Two things to notice: with s4 the readings hold the same information, so the
fit is as good (the same residual) but the weight of s1 + s2 is shared out
between s1, s2 and s4; and the SVD and the covariance's eigenvectors are the
same directions, which is why the PCA node can take either route.
"""
import os
import sys

folder = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
dll_folder = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else folder
for d in (folder, dll_folder, r"C:\Program Files\ArrayFire\v3\lib"):
    if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(d)
os.environ["PATH"] = os.pathsep.join((folder, dll_folder)) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, folder)

import numpy as np  # noqa: E402

import pycyxwiz  # noqa: E402

la = pycyxwiz.linalg

# The readings: three sensors and the meter they should explain.
rng = np.random.default_rng(7)
n = 200
sensors = rng.normal(0.0, 1.0, size=(n, 3))
meter = sensors @ np.array([2.0, -0.5, 1.5]) + 0.3 + rng.normal(0.0, 0.01, size=n)
A = np.column_stack([sensors, np.ones(n)])  # the ones column fits the offset


def close(ours, theirs, what, tol=1e-6):
    """Same up to the device's precision (float32 GPUs: 1e-4 relative)."""
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    scale = max(1.0, float(np.max(np.abs(theirs))))
    if np.max(np.abs(ours - theirs)) > tol * scale and np.max(np.abs(ours - theirs)) > 1e-4 * scale:
        raise SystemExit(f"{what}: pycyxwiz {ours} but numpy {theirs}")


# Least squares: the weights that bring A x closest to the meter.
x = la.lstsq(A, meter)
close(x, np.linalg.lstsq(A, meter, rcond=None)[0], "lstsq")
print(f"weights: s1 {x[0]:.4f}, s2 {x[1]:.4f}, s3 {x[2]:.4f}, offset {x[3]:.4f} (made with 2, -0.5, 1.5, 0.3)")
rank, cond = la.rank(A), la.cond(A)
close(rank, np.linalg.matrix_rank(A), "rank")
close(cond, np.linalg.cond(A), "cond")
print(f"A: {A.shape[0]} x {A.shape[1]}, rank {int(rank)}, condition {cond:.1f}")

# A fourth sensor with nothing new: A loses a rank, least squares still answers.
A4 = np.column_stack([sensors, sensors[:, 0] + sensors[:, 1], np.ones(n)])
x4 = la.lstsq(A4, meter)
close(x4, np.linalg.lstsq(A4, meter, rcond=None)[0], "lstsq, rank-deficient")
close(np.linalg.norm(A4 @ x4 - meter), np.linalg.norm(A @ x - meter), "the same residual")
print(f"with s4 = s1 + s2: rank {int(la.rank(A4))} of {A4.shape[1]}; shortest weights "
      f"s1 {x4[0]:.4f}, s2 {x4[1]:.4f}, s3 {x4[2]:.4f}, s4 {x4[3]:.4f}, offset {x4[4]:.4f}")

# The SVD of the centred readings and the eigenvalues of their covariance: one set of directions.
centred = sensors - sensors.mean(axis=0)
U, S, Vt = la.svd(centred)
close(S, np.linalg.svd(centred, compute_uv=False), "svd")
values, vectors = la.eig(np.cov(sensors, rowvar=False))
values = np.sort(np.real(values))[::-1]
close(values, S**2 / (n - 1), "eigenvalues = singular values squared / (n - 1)")
print("singular values of the centred readings: " + ", ".join(f"{s:.2f}" for s in S))
print("PCA from the covariance: variances " + ", ".join(f"{v:.4f}" for v in values) + f" (= singular values squared / {n - 1})")

# Factors that give their matrix back.
Q, R = la.qr(A)
L = la.chol(np.cov(sensors, rowvar=False))
qr_back = np.max(np.abs(np.asarray(Q) @ np.asarray(R) - A))
chol_back = np.max(np.abs(np.asarray(L) @ np.asarray(L).T - np.cov(sensors, rowvar=False)))
close(qr_back, 0.0, "Q R = A")
close(chol_back, 0.0, "L L^T = covariance")
print(f"QR gives A back: {qr_back:.1e}; Cholesky of the covariance gives it back: {chol_back:.1e}")
print("every answer matches numpy.linalg")
