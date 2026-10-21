"""PyTorch (float64) fixtures for Clustering (TOFIX140, clustering on the device).

References follow the libraries the API follows: scikit-learn KMeans, DBSCAN, GaussianMixture,
silhouette_samples / davies_bouldin_score / calinski_harabasz_score (written in torch here), and
scipy.cluster.hierarchy linkage / fcluster / cophenet on torch-computed distances. Random starts
(K-Means, GMM) are checked against the fixed point they must reach on well-separated blobs.

    py -3.12 generate_clustering_fixtures.py
"""

import json
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "clustering_pytorch.json"
torch.set_default_dtype(torch.float64)


def mat(t):
    return [[round(float(v), 15) for v in row] for row in t.tolist()]


def vec(t):
    return [round(float(v), 15) for v in t.tolist()]


def blobs(seed, centers, sizes, scales):
    g = torch.Generator().manual_seed(seed)
    points, labels = [], []
    for c, (center, size, scale) in enumerate(zip(centers, sizes, scales)):
        points.append(torch.tensor(center) + torch.randn(size, len(center), generator=g) * torch.tensor(scale))
        labels += [c] * size
    x, y = torch.cat(points), torch.tensor(labels)
    order = torch.randperm(len(y), generator=g)
    return x[order], y[order]


def distances(a, b, metric):
    # Exact differences: torch.cdist(p=2) takes a matmul shortcut above 25 rows.
    diff = a[:, None, :] - b[None, :, :]
    if metric == "euclidean":
        return diff.pow(2).sum(-1).sqrt()
    if metric == "manhattan":
        return diff.abs().sum(-1)
    an, bn = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
    return 1.0 - an @ bn.T


def dbscan(x, eps, min_samples, metric):
    """scikit-learn's dbscan_inner, on the inclusive eps-neighbourhoods."""
    adjacency = distances(x, x, metric) <= eps
    neighbours = [torch.nonzero(row).flatten().tolist() for row in adjacency]
    core = [len(nb) >= min_samples for nb in neighbours]
    labels = [-1] * len(x)
    label = 0
    for i in range(len(x)):
        if labels[i] != -1 or not core[i]:
            continue
        stack = [i]
        while stack:
            p = stack.pop()
            if labels[p] != -1:
                continue
            labels[p] = label
            if core[p]:
                stack += [v for v in neighbours[p] if labels[v] == -1]
        label += 1
    return labels, core


def silhouette_samples(x, codes, k):
    d = distances(x, x, "euclidean")
    one_hot = torch.nn.functional.one_hot(codes, k).double()
    counts = one_hot.sum(0)
    totals = d @ one_hot
    own = totals.gather(1, codes[:, None]).squeeze(1)
    own_count = counts[codes]
    a = own / (own_count - 1)
    means = totals / counts
    means[torch.arange(len(x)), codes] = float("inf")
    b = means.min(1).values
    s = (b - a) / torch.maximum(a, b)
    s = torch.nan_to_num(s)
    s[own_count == 1] = 0.0
    return s


def davies_bouldin(x, codes, k):
    centroids = torch.stack([x[codes == c].mean(0) for c in range(k)])
    intra = torch.stack([(x[codes == c] - centroids[c]).norm(dim=1).mean() for c in range(k)])
    cd = distances(centroids, centroids, "euclidean")
    if torch.allclose(intra, torch.zeros_like(intra), atol=1e-8) or torch.allclose(cd, torch.zeros_like(cd), atol=1e-8):
        return 0.0
    cd[cd == 0] = float("inf")
    return float(((intra[:, None] + intra[None, :]) / cd).max(1).values.mean())


def calinski_harabasz(x, codes, k):
    mean = x.mean(0)
    extra = intra = 0.0
    for c in range(k):
        member = x[codes == c]
        centre = member.mean(0)
        extra += len(member) * float(((centre - mean) ** 2).sum())
        intra += float(((member - centre) ** 2).sum())
    return 1.0 if intra == 0 else extra * (len(x) - k) / (intra * (k - 1))


def gmm_fixed_point(x, truth, kind, reg=1e-6, tol=1e-14):
    """scikit-learn GaussianMixture EM, started from the true hard assignment."""
    n, d = x.shape
    k = int(truth.max()) + 1
    eps = torch.finfo(torch.float64).eps

    def m_step(resp):
        nk = resp.sum(0) + 10 * eps
        means = resp.T @ x / nk[:, None]
        if kind == "full":
            covs = []
            for c in range(k):
                diff = x - means[c]
                covs.append((resp[:, c, None] * diff).T @ diff / nk[c] + reg * torch.eye(d))
            covs = torch.stack(covs)
        elif kind == "tied":
            cov = (x.T @ x - (nk[:, None] * means).T @ means) / nk.sum() + reg * torch.eye(d)
            covs = cov.expand(k, d, d)
        else:
            var = resp.T @ (x * x) / nk[:, None] - means ** 2 + reg
            if kind == "spherical":
                var = var.mean(1, keepdim=True).expand(k, d)
            covs = torch.diag_embed(var)
        return nk / n, means, covs

    def e_step(weights, means, covs):
        log_prob = torch.stack([torch.distributions.MultivariateNormal(means[c], covs[c]).log_prob(x)
                                for c in range(k)], 1) + weights.log()
        norm = torch.logsumexp(log_prob, 1)
        return (log_prob - norm[:, None]).exp(), float(norm.mean())

    params = m_step(torch.nn.functional.one_hot(truth, k).double())
    bound = -float("inf")
    for _ in range(10000):
        resp, mean_ll = e_step(*params)
        params = m_step(resp)
        if abs(mean_ll - bound) < tol:
            break
        bound = mean_ll
    resp, mean_ll = e_step(*params)
    weights, means, covs = params
    cov_params = {"full": k * d * (d + 1) // 2, "tied": d * (d + 1) // 2, "diag": k * d, "spherical": k}[kind]
    parameters = k * d + cov_params + k - 1
    ll = mean_ll * n
    return {"weights": vec(weights), "means": mat(means), "covariances": [mat(c) for c in covs],
            "log_likelihood": ll, "bic": -2 * ll + parameters * torch.log(torch.tensor(float(n))).item(),
            "aic": -2 * ll + 2 * parameters, "labels": resp.argmax(1).tolist()}


def main():
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage

    x3, y3 = blobs(1, [[0.0, 0.0], [8.0, 1.0], [3.0, 9.0]], [40, 50, 30], [[0.6, 0.4], [0.5, 0.8], [0.7, 0.5]])
    cases = []

    def add(op, label, data, **extra):
        cases.append({"op": op, "label": label, "data": mat(data), **extra})

    # K-Means: the blobs' partition, centroids = blob means.
    centroids = torch.stack([x3[y3 == c].mean(0) for c in range(3)])
    inertia = float(sum(((x3[y3 == c] - centroids[c]) ** 2).sum() for c in range(3)))
    for init in ("kmeans++", "random"):
        add("kmeans", init, x3, k=3, init=init, truth=y3.tolist(), centroids=mat(centroids), inertia=inertia)
    add("elbow", "blobs", x3, k_min=2, k_max=6, inertia_k3=inertia)

    # DBSCAN, three metrics, with scattered noise.
    g = torch.Generator().manual_seed(2)
    noisy = torch.cat([x3, torch.rand(15, 2, generator=g) * torch.tensor([16.0, 18.0]) - torch.tensor([4.0, 4.0])])
    positive = noisy + torch.tensor([6.0, 6.0])
    for metric, data, eps, min_samples in [("euclidean", noisy, 1.0, 5), ("manhattan", noisy, 1.3, 5),
                                           ("cosine", positive, 0.0005, 4)]:
        labels, core = dbscan(data, eps, min_samples, metric)
        add("dbscan", metric, data, eps=eps, min_samples=min_samples, metric=metric, labels=labels, core=core)
    d = distances(noisy, noisy, "euclidean").sort(1).values[:, 4].sort().values
    add("kdistances", "k4", noisy, k=4, distances=vec(d))

    # Hierarchical against scipy on the same distances.
    small, _ = blobs(3, [[0.0, 0.0, 0.0], [5.0, 4.0, 0.0], [0.0, 6.0, 5.0]], [20, 25, 15],
                     [[0.8, 0.6, 0.5], [0.6, 0.9, 0.7], [0.7, 0.5, 0.9]])
    small_positive = small + 8.0
    for method, metric in [("ward", "euclidean"), ("complete", "euclidean"), ("average", "euclidean"),
                           ("single", "euclidean"), ("complete", "manhattan"), ("average", "manhattan"),
                           ("single", "cosine"), ("average", "cosine")]:
        data = small_positive if metric == "cosine" else small
        dm = distances(data, data, metric)
        iu = torch.triu_indices(len(data), len(data), 1)
        condensed = dm[iu[0], iu[1]].numpy()
        z = linkage(condensed, method=method)
        cut = float(0.5 * (z[-3, 2] + z[-2, 2]))
        add("hierarchical", f"{method}_{metric}", data, linkage=method, metric=metric, k=3, linkage_matrix=z.tolist(),
            labels=fcluster(z, 3, "maxclust").tolist(), cophenetic=float(cophenet(z, condensed)[0]),
            cut_height=cut, cut_labels=fcluster(z, cut, "distance").tolist())

    # Gaussian mixture fixed points, every covariance type.
    xg, yg = blobs(4, [[0.0, 0.0], [7.0, 2.0], [2.0, 8.0]], [60, 80, 50], [[1.0, 0.5], [0.6, 1.2], [0.9, 0.9]])
    for kind in ("full", "tied", "diag", "spherical"):
        add("gmm", kind, xg, k=3, covariance_type=kind, truth=yg.tolist(), **gmm_fixed_point(xg, yg, kind))

    # Metrics with arbitrary label values, noise and a singleton cluster.
    raw = torch.tensor([7, 3, 12])[y3]
    raw[:6] = -1
    raw[10] = 99  # singleton
    kept = raw != -1
    values = sorted(set(raw[kept].tolist()))
    codes = torch.tensor([values.index(v) for v in raw[kept].tolist()])
    k = len(values)
    s = silhouette_samples(x3[kept], codes, k)
    per_sample = torch.zeros(len(raw))
    per_sample[kept] = s
    add("evaluate", "labels_noise_singleton", x3, labels=raw.tolist(), silhouette=float(s.mean()),
        per_sample=vec(per_sample), cluster_silhouettes=[float(s[codes == c].mean()) for c in range(k)],
        davies_bouldin=davies_bouldin(x3[kept], codes, k), calinski_harabasz=calinski_harabasz(x3[kept], codes, k),
        n_clusters=k, n_samples=int(kept.sum()))

    # Refusals.
    add("refuse_kmeans", "too_many_clusters", x3[:5], message="between 1 and the number of samples")
    add("refuse_hierarchical", "ward_manhattan", small, message="Ward linkage needs the euclidean metric")
    add("refuse_dbscan", "unknown_metric", small, message="Unknown metric")
    add("refuse_gmm", "unknown_covariance", small, message="Unknown covariance type")
    add("refuse_evaluate", "one_cluster", small, message="between 2 and n - 1 clusters")

    OUT.write_text(json.dumps({"schema_version": 1, "torch_version": torch.__version__, "cases": cases}, indent=1))
    print(f"wrote {OUT} ({len(cases)} cases, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
