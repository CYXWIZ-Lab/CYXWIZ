"""GloVe references (TOFIX152 P5). Run with Python 3.12: py -3.12 generate_glove_fixtures.py

The algorithm is the Stanford C code's (cooccur.c + glove.c), as in
scripts/glove_reference.py:
1. Counts: numpy co-occurrences of word ids in sentences (a pair d positions
   apart in one sentence adds 1/d both ways), ordered by row then column; the
   test must find the same pairs and sums as WordEmbeddings::CountCooccurrences.
2. Steps: PyTorch float32 runs the same AdaGrad steps as WordEmbeddings::GloVeSteps
   on given vectors, biases and batches (step = rate * f(X) * diff, a row moves
   by its summed steps / sqrt(history) and its history grows by their squares,
   every row read before the step); the test must land on the same state.
3. Quality: this reference trains on a corpus of word groups; the test trains
   ours with the same settings and compares how often a word's nearest
   neighbour is in its own group (the generators differ, so values are not
   compared).
"""

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def count(tokens, sentence_of, v, window):
    ids = np.asarray(tokens, dtype=np.int64)
    sent = np.asarray(sentence_of, dtype=np.int64)
    keys, weights = [], []
    for d in range(1, window + 1):
        if d >= len(ids):
            break
        same = sent[:-d] == sent[d:]
        left, right = ids[:-d][same], ids[d:][same]
        keys += [left * v + right, right * v + left]
        weights += [np.full(len(left), 1.0 / d)] * 2
    k = np.concatenate(keys)
    w = np.concatenate(weights)
    uk, inv = np.unique(k, return_inverse=True)
    x = np.bincount(inv, weights=w, minlength=len(uk))
    return (uk // v).tolist(), (uk % v).tolist(), x.tolist()


def glove_steps(state, batches, x_max, alpha, rate):
    s = {k: torch.tensor(val, dtype=torch.float32) for k, val in state.items()}
    costs = []
    for batch in batches:
        i = torch.tensor(batch["rows"])
        j = torch.tensor(batch["cols"])
        x = torch.tensor(batch["counts"], dtype=torch.float32)
        W, C, bw, bc = s["word"], s["context"], s["word_bias"], s["context_bias"]
        diff = (W[i] * C[j]).sum(1) + bw[i] + bc[j] - torch.log(x)
        fdiff = torch.clamp(x / x_max, max=1.0) ** alpha * diff
        costs.append(float((0.5 * fdiff * diff).sum()))
        step = rate * fdiff
        gw = step[:, None] * C[j]
        gc = step[:, None] * W[i]
        for p, h, idx, grad in (("word", "word_sq", i, gw), ("context", "context_sq", j, gc),
                                ("word_bias", "word_bias_sq", i, step), ("context_bias", "context_bias_sq", j, step)):
            upd = torch.zeros_like(s[p]).index_add_(0, idx, grad)
            sq = torch.zeros_like(s[p]).index_add_(0, idx, grad * grad)
            touched = torch.unique(idx)
            s[p][touched] -= upd[touched] / torch.sqrt(s[h][touched])
            s[h][touched] += sq[touched]
    return {k: val.numpy() for k, val in s.items()}, costs


def initial_state(rng, v, d):
    def table(shape):
        return ((rng.random(shape) - 0.5) / d).astype(np.float32)

    return {"word": table((v, d)), "context": table((v, d)), "word_bias": table(v), "context_bias": table(v),
            "word_sq": np.ones((v, d), np.float32), "context_sq": np.ones((v, d), np.float32),
            "word_bias_sq": np.ones(v, np.float32), "context_bias_sq": np.ones(v, np.float32)}


def counts_case(rng):
    v, window = 9, 3
    sentences = [rng.integers(0, v, size=n).tolist() for n in (7, 1, 5, 12, 2)]
    tokens = [t for s in sentences for t in s]
    sentence_of = [k for k, s in enumerate(sentences) for _ in s]
    rows, cols, values = count(tokens, sentence_of, v, window)
    return {"op": "counts", "vocab_size": v, "window": window, "tokens": tokens, "sentence_of": sentence_of,
            "rows": rows, "cols": cols, "values": values}


def steps_case(rng, v, d, sizes, label):
    state = initial_state(rng, v, d)
    # Larger than glove.c's start so the steps move the vectors visibly.
    state["word"] *= 40
    state["context"] *= 40
    batches = []
    for b in sizes:
        rows = rng.integers(0, v, size=b)
        rows[: min(3, b)] = rows[0]  # a repeated row: summed steps
        counts = np.exp(rng.uniform(-1.0, 6.0, size=b)).astype(np.float32)  # some above x_max
        batches.append({"rows": rows.tolist(), "cols": rng.integers(0, v, size=b).tolist(),
                        "counts": counts.astype(np.float64).tolist()})
    x_max, alpha, rate = 100.0, 0.75, 0.05
    final, costs = glove_steps(state, batches, x_max, alpha, rate)
    flat = {k: val.astype(np.float64).ravel().tolist() for k, val in state.items()}
    expected = {k: val.astype(np.float64).ravel().tolist() for k, val in final.items()}
    return {"op": "steps", "label": label, "vocab_size": v, "dim": d, "x_max": x_max, "alpha": alpha,
            "learning_rate": rate, "state": flat, "batches": batches, "expected": expected, "costs": costs}


def group_corpus(rng, groups=6, per_group=6, sentences=1500, length=8):
    words = [[f"g{g}w{i}" for i in range(per_group)] for g in range(groups)]
    filler = ["the", "and", "of", "a"]
    corpus = []
    for _ in range(sentences):
        group = words[rng.integers(groups)]
        sentence = [group[i] for i in rng.integers(0, per_group, size=length)]
        for _ in range(2):
            sentence.insert(int(rng.integers(0, len(sentence) + 1)), filler[rng.integers(len(filler))])
        corpus.append(sentence)
    return corpus, words


def group_accuracy(vectors, index, groups):
    names = [w for g in groups for w in g]
    group_of = {w: i for i, g in enumerate(groups) for w in g}
    m = np.stack([vectors[index[w]] for w in names])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    sim = m @ m.T
    np.fill_diagonal(sim, -np.inf)
    nearest = sim.argmax(1)
    return float(np.mean([group_of[names[i]] == group_of[names[j]] for i, j in enumerate(nearest)]))


def quality_case(rng):
    corpus, groups = group_corpus(rng)
    settings = {"vector_size": 24, "window": 3, "min_count": 1, "x_max": 10.0, "alpha": 0.75,
                "learning_rate": 0.05, "epochs": 30, "batch_size": 256}
    counts = {}
    for s in corpus:
        for w in s:
            counts[w] = counts.get(w, 0) + 1
    vocab = sorted(counts, key=lambda w: -counts[w])
    index = {w: i for i, w in enumerate(vocab)}
    tokens = [index[w] for s in corpus for w in s]
    sentence_of = [k for k, s in enumerate(corpus) for _ in s]
    v, d = len(vocab), settings["vector_size"]
    rows, cols, values = count(tokens, sentence_of, v, settings["window"])
    state = initial_state(rng, v, d)
    rows, cols, values = np.asarray(rows), np.asarray(cols), np.asarray(values, np.float32)
    for _ in range(settings["epochs"]):
        perm = rng.permutation(len(rows))
        batches = [{"rows": rows[perm[s:s + settings["batch_size"]]].tolist(),
                    "cols": cols[perm[s:s + settings["batch_size"]]].tolist(),
                    "counts": values[perm[s:s + settings["batch_size"]]].tolist()}
                   for s in range(0, len(rows), settings["batch_size"])]
        state, _ = glove_steps(state, batches, settings["x_max"], settings["alpha"], settings["learning_rate"])
    vectors = state["word"] + state["context"]
    return {"op": "quality", "settings": settings, "sentences": corpus, "groups": groups,
            "reference_accuracy": group_accuracy(vectors, index, groups)}


def main():
    rng = np.random.default_rng(1525)
    cases = [counts_case(rng), steps_case(rng, 10, 4, (6, 5, 8), "small"),
             steps_case(rng, 60, 32, (8192, 8192), "trainer sizes"), quality_case(rng)]
    path = HERE.parent / "fixtures" / "glove_reference.json"
    path.write_text(json.dumps({"torch": torch.__version__, "cases": cases}, indent=1))
    print(f"wrote {path}: counts + steps + quality (reference accuracy {cases[-1]['reference_accuracy']:.3f})")


if __name__ == "__main__":
    main()
