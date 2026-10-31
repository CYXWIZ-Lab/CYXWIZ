"""Truth for Data Studio > Explore > Groups and Projection (TOFIX129 D5):
scikit-learn and scipy on the Palmer penguins table, for test_groups_explore.

Writes groups_sklearn.json beside this file:
  table      - the four measurements (null where missing) and species, all 344 rows;
  kmeans     - k = 3, standardised: labels, inertia, the scores, per-group
               silhouette and means, the crosstab with species and adjusted Rand;
  elbow      - K-Means inertia and silhouette for k = 2 .. 8 and the elbow
               (largest second difference);
  dbscan     - eps 0.6, min_samples 5 (euclidean): labels; the k-distance
               curves (the 4th other point) for euclidean, manhattan and cosine
               and their Kneedle knees;
  ward       - the linkage matrix, cophenetic correlation, the labels cut into
               3, silhouettes of the cuts 2 .. 8, and scipy's dendrogram of the
               last 12 merges (icoord, dcoord, leaf sizes);
  gmm        - full covariance BIC and AIC for k = 2 .. 6 (best of 3 starts);
  pca        - explained variance ratio, components, the first two scores;
  ari        - adjusted_rand_score on small labelings, edge cases included.

Run with the oracle venv (numpy, pandas, scipy, scikit-learn):
  python generate_groups_fixtures.py path/to/penguins.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn import metrics as M
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(sys.argv[1])
feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
out = {"columns": feats,
       "table": {c: [None if pd.isna(v) else float(v) for v in df[c]] for c in feats}}
out["table"]["species"] = df["species"].tolist()
d = df.dropna(subset=feats).reset_index(drop=True)
raw = d[feats].values
X = StandardScaler().fit_transform(raw)
out["used"] = len(d)


def knee(v):
    """Kneedle on an ascending curve: the point furthest above the chord, both axes on 0 .. 1."""
    v = np.asarray(v)
    x = np.linspace(0.0, 1.0, len(v))
    y = (v - v[0]) / (v[-1] - v[0])
    return float(v[int(np.argmax(x - y))])


km = KMeans(3, n_init=10, random_state=42).fit(X)
lab = km.labels_
sil = M.silhouette_samples(X, lab)
out["kmeans"] = dict(labels=lab.tolist(), inertia=float(km.inertia_), silhouette=float(M.silhouette_score(X, lab)),
                     davies_bouldin=float(M.davies_bouldin_score(X, lab)),
                     calinski_harabasz=float(M.calinski_harabasz_score(X, lab)),
                     group_silhouette=[float(sil[lab == c].mean()) for c in range(3)],
                     means=[raw[lab == c].mean(axis=0).tolist() for c in range(3)],
                     overall_means=raw.mean(axis=0).tolist(), overall_sds=raw.std(axis=0).tolist(),
                     species=sorted(d["species"].unique().tolist()),
                     crosstab=[[int(((lab == c) & (d["species"] == s)).sum()) for s in sorted(d["species"].unique())]
                               for c in range(3)],
                     adjusted_rand=float(M.adjusted_rand_score(d["species"], lab)))

ks = list(range(2, 9))
inertia, silhouette = [], []
for k in ks:
    m = KMeans(k, n_init=10, random_state=42).fit(X)
    inertia.append(float(m.inertia_))
    silhouette.append(float(M.silhouette_score(X, m.labels_)))
second = [inertia[i - 1] - 2 * inertia[i] + inertia[i + 1] for i in range(1, len(ks) - 1)]
out["elbow"] = dict(ks=ks, inertia=inertia, silhouette=silhouette, elbow=ks[1 + int(np.argmax(second))])

db = DBSCAN(eps=0.6, min_samples=5).fit(X)
kdist = {}
for metric in ["euclidean", "manhattan", "cosine"]:
    curve = np.sort(NearestNeighbors(n_neighbors=5, metric=metric).fit(X).kneighbors(X)[0][:, -1])
    kdist[metric] = dict(curve=curve.tolist(), knee=knee(curve))
out["dbscan"] = dict(eps=0.6, min_samples=5, labels=db.labels_.tolist(), kdistances=kdist)

Z = linkage(X, "ward")
cuts = []
for k in ks:
    cuts.append(float(M.silhouette_score(X, fcluster(Z, k, "maxclust"))))
dn = dendrogram(Z, truncate_mode="lastp", p=12, no_plot=True)
leaf_sizes = [int(s.strip("()")) if s.startswith("(") else 1 for s in dn["ivl"]]
out["ward"] = dict(linkage=Z.tolist(), cophenetic=float(cophenet(Z, pdist(X))[0]),
                   labels=(fcluster(Z, 3, "maxclust") - 1).tolist(), ks=ks, silhouette=cuts,
                   icoord=dn["icoord"], dcoord=dn["dcoord"], leaf_rows=leaf_sizes)

bic, aic = [], []
for k in range(2, 7):
    g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=42).fit(X)
    bic.append(float(g.bic(X)))
    aic.append(float(g.aic(X)))
out["gmm"] = dict(ks=list(range(2, 7)), bic=bic, aic=aic)

pca = PCA().fit(X)
P = pca.transform(X)
out["pca"] = dict(ratio=pca.explained_variance_ratio_.tolist(), components=pca.components_.tolist(),
                  x=P[:, 0].tolist(), y=P[:, 1].tolist())

rng = np.random.default_rng(7)
cases = [([0, 0, 1, 1], [0, 0, 1, 1]), ([0, 0, 1, 1], [1, 1, 0, 0]), ([0, 0, 0, 0], [0, 0, 0, 0]),
         ([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 0, 1, 2], [0, 0, 1, 1]), ([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])]
for _ in range(4):
    n = int(rng.integers(10, 60))
    cases.append((rng.integers(0, 4, n).tolist(), rng.integers(0, 3, n).tolist()))
out["ari"] = [dict(a=[int(v) for v in a], b=[int(v) for v in b], ari=float(M.adjusted_rand_score(a, b))) for a, b in cases]

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groups_sklearn.json")
with open(path, "w") as f:
    json.dump(out, f)
print("wrote", path, "kmeans ari %.4f elbow %d knee %.4f ward coph %.4f gmm bic %s" % (
    out["kmeans"]["adjusted_rand"], out["elbow"]["elbow"], kdist["euclidean"]["knee"], out["ward"]["cophenetic"],
    [round(b, 1) for b in bic]))
