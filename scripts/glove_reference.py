"""Reference GloVe (TOFIX152 P5): the algorithm of the Stanford C code
(cooccur.c + glove.c) in numpy / PyTorch, for the Engine's figures and as
the oracle of its device implementation.

  py -3.12 scripts/glove_reference.py tokens.txt out.vec [--epochs 25] [--dim 100]

tokens.txt: one sentence per line, words split by spaces (lowercased and split
as the Word Embeddings node splits them). Prints the vocabulary size, the
number of non-zero co-occurrences, the loss of each epoch, timings and the
nearest words of a few probes; writes word2vec text vectors (w + w~).
"""
import argparse
import collections
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens")
    ap.add_argument("out")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--x-max", type=float, default=100.0)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    t0 = time.time()
    sentences = [line.split() for line in open(a.tokens, encoding="utf-8")]
    counts = collections.Counter(w for s in sentences for w in s)
    vocab = [w for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if c >= a.min_count]
    index = {w: i for i, w in enumerate(vocab)}
    v = len(vocab)
    # Corpus as ids with a sentence number; words outside the vocabulary
    # dropped (cooccur.c skips them before windowing).
    ids, sent = [], []
    for s_no, s in enumerate(sentences):
        for w in s:
            i = index.get(w)
            if i is not None:
                ids.append(i)
                sent.append(s_no)
    ids = np.asarray(ids, dtype=np.int64)
    sent = np.asarray(sent, dtype=np.int64)

    # Co-occurrences: symmetric window, weight 1 / distance.
    keys, weights = [], []
    for d in range(1, a.window + 1):
        same = sent[:-d] == sent[d:]
        left, right = ids[:-d][same], ids[d:][same]
        for x, y in ((left, right), (right, left)):
            k = x * v + y
            uk, inv = np.unique(k, return_inverse=True)
            keys.append(uk)
            weights.append(np.bincount(inv, minlength=len(uk)) / d)
    k = np.concatenate(keys)
    w = np.concatenate(weights)
    uk, inv = np.unique(k, return_inverse=True)
    x = np.bincount(inv, weights=w, minlength=len(uk))
    rows = torch.from_numpy(uk // v)
    cols = torch.from_numpy(uk % v)
    xij = torch.from_numpy(x).float()
    t1 = time.time()
    print(f"vocabulary {v} words, {len(ids):,} tokens, {len(uk):,} non-zero co-occurrences, counted in {t1 - t0:.1f} s")

    # Training: weighted least squares with AdaGrad (glove.c: initial gradient
    # sums 1, vectors uniform in [-0.5, 0.5) / dim).
    g = torch.Generator().manual_seed(a.seed)
    W = (torch.rand(v, a.dim, generator=g) - 0.5) / a.dim
    C = (torch.rand(v, a.dim, generator=g) - 0.5) / a.dim
    bw = (torch.rand(v, generator=g) - 0.5) / a.dim
    bc = (torch.rand(v, generator=g) - 0.5) / a.dim
    gW, gC = torch.ones(v, a.dim), torch.ones(v, a.dim)
    gbw, gbc = torch.ones(v), torch.ones(v)
    fx = torch.clamp(xij / a.x_max, max=1.0) ** a.alpha
    logx = torch.log(xij)
    n = len(xij)
    for epoch in range(a.epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for s in range(0, n, a.batch):
            b = perm[s:s + a.batch]
            i, j = rows[b], cols[b]
            diff = (W[i] * C[j]).sum(1) + bw[i] + bc[j] - logx[b]
            fdiff = fx[b] * diff
            total += 0.5 * float((fdiff * diff).sum())
            # glove.c scales by the rate first (fdiff *= eta): the step is
            # rate*g / sqrt(history) and the history grows by (rate*g)^2.
            step = a.rate * fdiff
            gw = step[:, None] * C[j]
            gc = step[:, None] * W[i]
            # AdaGrad, summed per row over the batch (rows read before the step).
            for P, G, idx, grad in ((W, gW, i, gw), (C, gC, j, gc), (bw, gbw, i, step), (bc, gbc, j, step)):
                upd = torch.zeros_like(P).index_add_(0, idx, grad)
                sq = torch.zeros_like(P).index_add_(0, idx, grad * grad)
                touched = torch.unique(idx)
                P[touched] -= upd[touched] / torch.sqrt(G[touched])
                G[touched] += sq[touched]
        print(f"epoch {epoch + 1}: cost {total / n:.5f}")
    t2 = time.time()
    print(f"trained {a.epochs} epochs in {t2 - t1:.1f} s")

    vec = (W + C).numpy()
    unit = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    for probe in ("therapy", "anxiety", "depressed", "happy"):
        if probe in index:
            sims = unit @ unit[index[probe]]
            sims[index[probe]] = -2
            top = np.argsort(-sims)[:5]
            print(probe + ":", " ".join(f"{vocab[t]} {sims[t]:.3f}" for t in top))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{v} {a.dim}\n")
        for word, row in zip(vocab, vec):
            f.write(word + " " + " ".join(repr(float(np.float32(z))) for z in row) + "\n")


if __name__ == "__main__":
    main()
