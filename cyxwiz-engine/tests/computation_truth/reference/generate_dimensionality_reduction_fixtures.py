"""PyTorch (float64) fixtures for DimensionalityReduction (TOFIX140, PCA / t-SNE / UMAP on the device).

PCA follows scikit-learn PCA (svd_flip on V). t-SNE follows scikit-learn TSNE(method="exact",
init="pca"): _binary_search_perplexity, _joint_probabilities, _kl_divergence and _gradient_descent,
in float64 throughout. UMAP follows umap-learn: nearest neighbours with the point itself,
smooth_knn_dist, compute_membership_strengths, the fuzzy union and find_ab_params (scipy curve_fit,
as umap-learn calls it), a PCA start, and the layout updates in expectation over each epoch.

    py -3.12 generate_dimensionality_reduction_fixtures.py
"""

import json
import math
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "dimensionality_reduction_pytorch.json"
torch.set_default_dtype(torch.float64)
EPS = torch.finfo(torch.float64).eps


def mat(t):
    return [[round(float(v), 15) for v in row] for row in t.tolist()]


def vec(t):
    return [round(float(v), 15) for v in t.tolist()]


def blobs(seed, centers, sizes, scale):
    g = torch.Generator().manual_seed(seed)
    points, labels = [], []
    for c, (center, size) in enumerate(zip(centers, sizes)):
        points.append(torch.tensor(center) + torch.randn(size, len(center), generator=g) * scale)
        labels += [c] * size
    x, y = torch.cat(points), torch.tensor(labels)
    order = torch.randperm(len(y), generator=g)
    return x[order], y[order]


def sq_dist(a, b):
    """Exact squared euclidean distances (torch.cdist takes a matmul shortcut above 25 rows,
    which leaves ~1e-8 self-distances)."""
    return (a[:, None, :] - b[None, :, :]).pow(2).sum(-1)


def pca(x, k, center=True, scale=False):
    n = x.shape[0]
    centered = x - x.mean(0)
    divisor = torch.ones(x.shape[1])
    if scale:
        divisor = centered.pow(2).mean(0).sqrt()
        divisor[divisor < 1e-10] = 1.0
    z = (centered if center else x) / divisor
    cov = z.T @ z / (n - 1)
    values, vectors = torch.linalg.eigh(cov)
    values, vectors = values.flip(0)[:k], vectors.flip(1)[:, :k].T
    peak = vectors.abs().argmax(1)
    signs = torch.sign(vectors[torch.arange(k), peak])
    vectors = vectors * signs[:, None]
    total = float(torch.trace(cov))
    return {"components": vectors, "transformed": z @ vectors.T, "variance": values, "total": total}


# ------------------------------------------------------------------ t-SNE

def binary_search_perplexity(sq, perplexity):
    n = sq.shape[0]
    desired = math.log(perplexity)
    p = torch.zeros(n, n)
    for i in range(n):
        beta, beta_min, beta_max = 1.0, -math.inf, math.inf
        for _ in range(100):
            row = torch.exp(-sq[i] * beta)
            row[i] = 0.0
            sum_p = float(row.sum())
            if sum_p == 0.0:
                sum_p = 1e-8
            mean_dist = float((sq[i] * row).sum()) / sum_p
            row = row / sum_p
            entropy = math.log(sum_p) + beta * mean_dist
            p[i] = row
            diff = entropy - desired
            if abs(diff) <= 1e-5:
                break
            if diff > 0:
                beta_min = beta
                beta = beta * 2.0 if beta_max == math.inf else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2.0 if beta_min == -math.inf else (beta + beta_min) / 2.0
    return p


def tsne(x, dims, perplexity, learning_rate, n_iter):
    n = x.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    p = binary_search_perplexity(sq_dist(x, x), perplexity)
    p = p + p.T
    p = torch.clamp(p / max(float(p.sum()), EPS), min=EPS) * off
    y = pca(x, dims)["transformed"]
    y = y / y[:, 0].std(unbiased=False) * 1e-4
    dof = max(dims - 1, 1)
    history, final = [], [0.0]

    def objective(y, probabilities):
        w = (1.0 + sq_dist(y, y) / dof).pow(-(dof + 1.0) / 2.0) * off
        q = torch.clamp(w / w.sum(), min=EPS)
        kl = float((probabilities * torch.log(torch.clamp(probabilities, min=EPS) / q)).sum())
        pq = (probabilities - q) * w
        grad = (2.0 * (dof + 1.0) / dof) * (y * pq.sum(1, keepdim=True) - pq @ y)
        return kl, grad

    def descend(y, probabilities, momentum, start, end):
        update, gains = torch.zeros_like(y), torch.ones_like(y)
        best_error, best_iter, last = float("inf"), start, start - 1
        for i in range(start, end):
            kl, grad = objective(y, probabilities)
            grad_norm = float(grad.norm())
            inc = update * grad < 0.0
            gains = torch.clamp(torch.where(inc, gains + 0.2, gains * 0.8), min=0.01)
            update = momentum * update - learning_rate * (grad * gains)
            y = y + update
            last = i
            final[0] = kl
            if (i + 1) % 50 == 0:
                history.append(kl)
                if kl < best_error:
                    best_error, best_iter = kl, i
                elif i - best_iter > 300:
                    break
                if grad_norm <= 1e-7:
                    break
        return y, last

    y, explored = descend(y, p * 12.0, 0.5, 0, min(250, n_iter))
    y, finished = descend(y, p, 0.8, explored + 1, n_iter)
    return {"embeddings": mat(y), "kl_history": history, "final_kl": final[0],
            "iterations_run": max(explored, finished) + 1}


# ------------------------------------------------------------------- UMAP

def find_ab(min_dist):
    import numpy as np
    from scipy.optimize import curve_fit

    xv = np.linspace(0, 3.0, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist)))
    params, _ = curve_fit(lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b)), xv, yv)
    # MINPACK stops near 1e-8; polish to the exact least-squares optimum (Gauss-Newton),
    # which the device fit reaches, and make sure it is still scipy's answer.
    x, y = torch.tensor(xv), torch.tensor(yv)
    a, b = float(params[0]), float(params[1])
    log_x = torch.log(torch.where(x > 0, x, torch.ones_like(x)))
    for _ in range(50):
        power = torch.where(x > 0, x.pow(2 * b), torch.zeros_like(x))
        denom = 1.0 + a * power
        r = 1.0 / denom - y
        j = torch.stack([-power / denom ** 2, -a * power * 2 * log_x / denom ** 2], 1)
        step = torch.linalg.solve(j.T @ j, -(j.T @ r))
        a, b = a + float(step[0]), b + float(step[1])
    assert abs(a - params[0]) < 1e-6 * abs(a) and abs(b - params[1]) < 1e-6 * abs(b), (a, b, params)
    return a, b


def fuzzy_graph(x, k):
    n = x.shape[0]
    dist, index = sq_dist(x, x).sqrt().sort(1)
    dist, index = dist[:, :k], index[:, :k]
    target = math.log2(k)
    mean_all = float(dist.mean())
    graph = torch.zeros(n, n)
    for i in range(n):
        nonzero = dist[i][dist[i] > 0]
        rho = float(nonzero[0]) if len(nonzero) else 0.0
        lo, hi, mid = 0.0, math.inf, 1.0
        for _ in range(64):
            d = dist[i, 1:] - rho
            psum = float(torch.where(d > 0, torch.exp(-d / mid), torch.ones_like(d)).sum())
            if abs(psum - target) < 1e-5:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                mid = mid * 2 if hi == math.inf else (lo + hi) / 2.0
        sigma = mid
        floor = 1e-3 * (float(dist[i].mean()) if rho > 0 else mean_all)
        sigma = max(sigma, floor)
        for j in range(k):
            gap = float(dist[i, j]) - rho
            if int(index[i, j]) == i:
                value = 0.0
            elif gap <= 0 or sigma == 0:
                value = 1.0
            else:
                value = math.exp(-gap / sigma)
            graph[i, index[i, j]] = value
    return graph + graph.T - graph * graph.T


def umap(x, dims, k, min_dist, epochs):
    n = x.shape[0]
    a, b = find_ab(min_dist)
    graph = fuzzy_graph(x, k)
    peak = float(graph.max())
    graph[graph < peak / epochs] = 0.0
    rate = graph / peak
    repulse_scale = 5.0 * rate.sum(1) / n
    y = pca(x, dims)["transformed"]
    y = y * (10.0 / float(y.abs().max()))
    y = 10.0 * (y - y.min(0).values) / (y.max(0).values - y.min(0).values)
    others = ~torch.eye(n, dtype=torch.bool)
    for epoch in range(epochs):
        alpha = 1.0 - epoch / epochs
        d2 = sq_dist(y, y)
        positive = d2 > 0
        safe = torch.where(positive, d2, torch.ones_like(d2))
        attract = torch.where(positive, -2.0 * a * b * safe.pow(b - 1.0) / (a * safe.pow(b) + 1.0), torch.zeros_like(d2))
        repel = 2.0 * b / ((0.001 + d2) * (a * d2.pow(b) + 1.0))
        updated = y.clone()
        for f in range(dims):
            diff = y[:, f, None] - y[None, :, f]
            pull = 2.0 * (rate * torch.clamp(attract * diff, -4, 4)).sum(1)
            push_terms = torch.where(others, torch.where(positive, torch.clamp(repel * diff, -4, 4), torch.full_like(d2, 4.0)),
                                     torch.zeros_like(d2))
            updated[:, f] = y[:, f] + alpha * (pull + repulse_scale * push_terms.sum(1))
        y = updated
    return {"embeddings": mat(y), "a": a, "b": b}


def main():
    cases = []

    def add(op, label, data, **extra):
        cases.append({"op": op, "label": label, "data": mat(data), **extra})

    wide, _ = blobs(1, [[0, 0, 0, 0, 0], [6, 2, -1, 3, 0], [1, 7, 4, -2, 5]], [40, 35, 45], 1.0)
    wide = wide * torch.tensor([1.0, 3.0, 0.5, 2.0, 1.5])
    for label, k, center, scale in [("centered_k3", 3, True, False), ("scaled_k2", 2, True, True),
                                    ("uncentered_k2", 2, False, False), ("all_components", 0, True, False)]:
        r = pca(wide, k if k > 0 else 5, center, scale)
        add("pca", label, wide, n_components=k, center=center, scale=scale, components=mat(r["components"]),
            transformed=mat(r["transformed"]), explained_variance=vec(r["variance"]),
            explained_variance_ratio=vec(r["variance"] / r["total"]),
            singular_values=vec((r["variance"] * (wide.shape[0] - 1)).sqrt()))

    small, truth = blobs(2, [[0, 0, 0, 0], [5, 5, 0, 1], [0, 6, 5, -2]], [20, 22, 18], 0.8)
    # t-SNE is chaotic: relative differences grow ~10x every 10-20 iterations at useful learning
    # rates, so step-for-step cases use a small rate (both stages, 300 iterations) or a short run.
    for label, dims, perplexity, lr, n_iter in [("2d_lr0.05_300", 2, 10, 0.05, 300), ("3d_lr0.05_300", 3, 8, 0.05, 300),
                                                ("2d_lr1_60", 2, 10, 1.0, 60)]:
        add("tsne", label, small, n_dims=dims, perplexity=perplexity, learning_rate=lr, n_iterations=n_iter,
            **tsne(small, dims, perplexity, lr, n_iter))
    add("tsne_separates", "2d_1000", small, truth=truth.tolist())

    # The layout is chaotic too (~100x per 4 epochs at the full rate): step-for-step for 5-6 epochs.
    for label, dims, k, min_dist, epochs in [("2d_k10", 2, 10, 0.1, 5), ("3d_k8", 3, 8, 0.5, 6)]:
        add("umap", label, small, n_dims=dims, n_neighbors=k, min_dist=min_dist, n_epochs=epochs,
            **umap(small, dims, k, min_dist, epochs))
    add("umap_separates", "2d_default", small, truth=truth.tolist())

    add("refuse_tsne", "perplexity", small, message="Perplexity must be at least 1")
    add("refuse_umap", "neighbours", small, message="Neighbours must be between 2")
    add("refuse_pca", "one_sample", small[:1], message="at least 2 samples")

    OUT.write_text(json.dumps({"schema_version": 1, "torch_version": torch.__version__, "cases": cases}, indent=1))
    print(f"wrote {OUT} ({len(cases)} cases, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
