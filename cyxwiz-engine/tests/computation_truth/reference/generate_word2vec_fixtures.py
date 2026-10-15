"""word2vec references (TOFIX152). Run with Python 3.12: py -3.12 generate_word2vec_fixtures.py

1. Steps: PyTorch float32 runs the same SGNS steps as WordEmbeddings::Word2VecSteps
   (SGD on the summed loss of a batch, all rows from the same old values; CBOW
   gives every context word the mean's whole gradient, gensim's rule) on given
   matrices and batches; the test must land on the same matrices.
2. Quality: gensim 4 Word2Vec (workers=1) on a corpus of word groups; the test
   trains ours with the same settings and compares how often a word's nearest
   neighbour is in its own group (vectors from different generators differ, so
   values are not compared).
"""

import json
from pathlib import Path

import gensim
import numpy as np
import torch
from gensim.models import Word2Vec

HERE = Path(__file__).resolve().parent


def sgns_steps(w_in, w_out, batches):
    w_in = torch.tensor(w_in, dtype=torch.float32)
    w_out = torch.tensor(w_out, dtype=torch.float32)
    losses = []
    for batch in batches:
        w_out.requires_grad_(True)
        b, c, k = len(batch["outputs"]), batch["context_width"], batch["negative"]
        inputs = torch.tensor(batch["inputs"]).reshape(b, c)
        valid = (inputs >= 0).float()
        count = valid.sum(1).clamp(min=1.0)
        rows = w_in[inputs.clamp(min=0)] * valid.unsqueeze(2)
        h = (rows.sum(1) / count.unsqueeze(1)).detach().requires_grad_(True)
        targets = torch.cat([torch.tensor(batch["outputs"]).reshape(b, 1),
                             torch.tensor(batch["negatives"]).reshape(b, k)], dim=1)
        target_valid = (targets >= 0).float()
        u = w_out[targets.clamp(min=0)]
        score = (u * h.unsqueeze(1)).sum(2)
        label = torch.zeros(b, k + 1)
        label[:, 0] = 1.0
        loss = (torch.where(label > 0.5, torch.nn.functional.softplus(-score),
                            torch.nn.functional.softplus(score)) * target_valid).sum()
        loss.backward()
        with torch.no_grad():
            # Every context word takes the whole dLoss/dh (skip-gram: the one input).
            flat = inputs.reshape(-1)
            grads = (h.grad.unsqueeze(1) * valid.unsqueeze(2)).reshape(b * c, -1)
            w_in = w_in.index_add(0, flat.clamp(min=0), grads, alpha=-batch["alpha"])
            w_out = (w_out - batch["alpha"] * w_out.grad).detach()
        losses.append(float(loss.detach()))
    return w_in.detach().numpy(), w_out.detach().numpy(), losses


def steps_case(rng):
    v, d = 12, 5
    w_in = (rng.normal(0, 0.3, size=(v, d))).astype(np.float32)
    w_out = (rng.normal(0, 0.3, size=(v, d))).astype(np.float32)
    batches = []
    for alpha in (0.05, 0.04, 0.03):  # skip-gram: one input word
        b, k = 8, 3
        inputs = rng.integers(0, v, size=b)
        inputs[:3] = inputs[0]  # a repeated input word: summed update
        negatives = rng.integers(0, v, size=(b, k))
        negatives[1, 2] = -1  # a skipped negative
        batches.append({"inputs": inputs.tolist(), "context_width": 1, "outputs": rng.integers(0, v, size=b).tolist(),
                        "negatives": negatives.ravel().tolist(), "negative": k, "alpha": alpha})
    for alpha in (0.025, 0.02):  # CBOW: up to four context words
        b, c, k = 6, 4, 2
        inputs = rng.integers(0, v, size=(b, c))
        inputs[0, 3] = -1
        inputs[2, 2:] = -1
        batches.append({"inputs": inputs.ravel().tolist(), "context_width": c,
                        "outputs": rng.integers(0, v, size=b).tolist(),
                        "negatives": rng.integers(0, v, size=(b, k)).ravel().tolist(), "negative": k, "alpha": alpha})
    final_in, final_out, losses = sgns_steps(w_in, w_out, batches)
    return {"op": "steps", "vocab_size": v, "dim": d, "input_vectors": w_in.astype(np.float64).ravel().tolist(),
            "output_vectors": w_out.astype(np.float64).ravel().tolist(), "batches": batches,
            "expected_input": final_in.astype(np.float64).ravel().tolist(),
            "expected_output": final_out.astype(np.float64).ravel().tolist(), "losses": losses}


def large_steps_case(rng):
    """The trainer's sizes: 40 words, 24 dims, 512 examples, 5 negatives."""
    v, d, b, k = 40, 24, 512, 5
    w_in = ((rng.random((v, d)) - 0.5) / d).astype(np.float32)
    w_out = (rng.normal(0, 0.2, size=(v, d))).astype(np.float32)
    batches = []
    for alpha in (0.025, 0.0249):
        negatives = rng.integers(0, v, size=(b, k))
        negatives[rng.random((b, k)) < 0.05] = -1
        batches.append({"inputs": rng.integers(0, v, size=b).tolist(), "context_width": 1,
                        "outputs": rng.integers(0, v, size=b).tolist(), "negatives": negatives.ravel().tolist(),
                        "negative": k, "alpha": alpha})
    c = 6
    inputs = rng.integers(0, v, size=(b, c))
    inputs[rng.random((b, c)) < 0.3] = -1
    batches.append({"inputs": inputs.ravel().tolist(), "context_width": c, "outputs": rng.integers(0, v, size=b).tolist(),
                    "negatives": rng.integers(0, v, size=(b, k)).ravel().tolist(), "negative": k, "alpha": 0.0248})
    final_in, final_out, losses = sgns_steps(w_in, w_out, batches)
    return {"op": "steps", "label": "trainer sizes", "vocab_size": v, "dim": d,
            "input_vectors": w_in.astype(np.float64).ravel().tolist(),
            "output_vectors": w_out.astype(np.float64).ravel().tolist(), "batches": batches,
            "expected_input": final_in.astype(np.float64).ravel().tolist(),
            "expected_output": final_out.astype(np.float64).ravel().tolist(), "losses": losses}


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


def group_accuracy(vectors, groups):
    names = [w for g in groups for w in g]
    group_of = {w: i for i, g in enumerate(groups) for w in g}
    m = np.stack([vectors[w] for w in names])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    sim = m @ m.T
    np.fill_diagonal(sim, -np.inf)
    nearest = sim.argmax(1)
    return float(np.mean([group_of[names[i]] == group_of[names[j]] for i, j in enumerate(nearest)]))


def quality_case(rng, sg):
    corpus, groups = group_corpus(rng)
    settings = {"vector_size": 24, "window": 3, "min_count": 1, "negative": 5, "sample": 0.0, "epochs": 20,
                "alpha": 0.025, "min_alpha": 0.0001}
    model = Word2Vec(corpus, sg=sg, workers=1, seed=1, **settings)
    return {"op": "quality", "label": "skip-gram" if sg else "cbow", "cbow": not sg, "settings": settings,
            "sentences": corpus, "groups": groups, "gensim_accuracy": group_accuracy(model.wv, groups)}


def main():
    rng = np.random.default_rng(1520)
    cases = [steps_case(rng), quality_case(rng, 1), quality_case(rng, 0)]
    # Appended so the cases above keep their data.
    cases.insert(1, large_steps_case(np.random.default_rng(1521)))
    path = HERE.parent / "fixtures" / "word2vec_reference.json"
    path.write_text(json.dumps({"torch": torch.__version__, "gensim": gensim.__version__, "cases": cases}, indent=1))
    print(f"wrote {path}: steps + quality (gensim accuracy skip-gram {cases[2]['gensim_accuracy']:.3f}, "
          f"cbow {cases[3]['gensim_accuracy']:.3f})")


if __name__ == "__main__":
    main()
