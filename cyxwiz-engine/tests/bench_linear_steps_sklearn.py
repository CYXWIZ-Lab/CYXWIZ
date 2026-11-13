"""The shapes of bench_linear_steps.cpp through scikit-learn / numpy (TOFIX129 D9):
fit on the 80 % Train rows, transform every row. The yardstick for the Engine's
times; random data of the same shapes, not the same values.

    py -3.14 bench_linear_steps_sklearn.py
"""
import time

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD


def timed(what, fn):
    t0 = time.perf_counter()
    fn()
    print(f"  {what:<34} {time.perf_counter() - t0:8.2f} s")


def dense(rows, columns):
    print(f"dense: {rows} rows x {columns} columns")
    rng = np.random.default_rng(7)
    x = rng.standard_normal((rows, 5)) @ rng.standard_normal((5, columns)) + 0.3 * rng.standard_normal((rows, columns))
    train = x[np.arange(rows) % 10 < 8]
    k = min(10, columns)
    timed("PCA (10)", lambda: PCA(k).fit(train).transform(x))
    timed("Truncated SVD (10)", lambda: TruncatedSVD(k, algorithm="arpack").fit(train).transform(x))
    timed("Whitening PCA (10)", lambda: PCA(k, whiten=True).fit(train).transform(x))

    def zca():
        mu = train.mean(0)
        lam, v = np.linalg.eigh(np.cov(train, rowvar=False))
        return (x - mu) @ (v @ np.diag(1 / np.sqrt(lam)) @ v.T)

    timed("Whitening ZCA", zca)
    w = rng.standard_normal((columns, k))
    timed("Linear Projection (10)", lambda: x @ w)
    timed("Linear Solve (least squares, 1 b)", lambda: np.linalg.lstsq(train[:, 1:], train[:, :1], rcond=None))


def sparse(rows, vocabulary, per_row, k):
    print(f"sparse: {rows} rows x {vocabulary} features, {per_row} entries a row")
    rng = np.random.default_rng(11)
    x = sp.random(rows, vocabulary, density=per_row / vocabulary, format="csr", random_state=11, dtype=np.float32)
    train = x[np.arange(rows) % 10 < 8]
    timed(f"Truncated SVD ({k}) sparse", lambda: TruncatedSVD(k, algorithm="arpack").fit(train).transform(x))


dense(200000, 50)
dense(1000000, 20)
sparse(100000, 2000, 50, 50)
