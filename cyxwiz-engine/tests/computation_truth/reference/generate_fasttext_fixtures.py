"""fastText references (TOFIX152 P6). Run with Python 3.12: py -3.12 generate_fasttext_fixtures.py

1. Hash and pieces: gensim 4's ft_hash_bytes, compute_ngrams_bytes and
   ft_ngram_hashes (fastText's dictionary.cc rules: FNV-1a over signed bytes,
   n-grams of "<word>" by UTF-8 characters).
2. Steps: PyTorch float32 runs the steps of WordEmbeddings::FastTextSteps on
   given rows and batches: an input word's vector is the mean of its row and
   its n-grams' rows (CBOW: of all its input words' rows); the output rows
   take SGD on the summed loss; every input row of the mean takes the whole
   dLoss/dmean (fastText's and gensim's update); all rows from the same old
   values.
3. Vectors: a vocabulary word = mean of its row and its n-gram rows, another
   word = mean of its n-gram rows (gensim's adjust_vectors and get_vector).
4. Quality: gensim 4 FastText (workers=1) on a corpus of word groups; the test
   trains ours with the same settings and compares how often a word's nearest
   neighbour is in its own group, for the words and for unseen variants of them.
5. .bin: a small gensim FastText saved with save_facebook_model
   (fixtures/word_embeddings/small_fasttext.bin); the vectors gensim's
   load_facebook_model gives its words and unseen words, and a vocabulary lined
   up with n-gram vectors for the tokens it lacks.
"""

import json
from pathlib import Path

import gensim
import numpy as np
import torch
from gensim.models import FastText
from gensim.models.fasttext import ft_ngram_hashes, load_facebook_model, save_facebook_model
from gensim.models.fasttext_inner import compute_ngrams_bytes, ft_hash_bytes

HERE = Path(__file__).resolve().parent


def hash_case():
    texts = ["", "a", "<th", "the", "therapy>", "café", "été", "\U0001F60A", "x\U0001F60Ay", "да"]
    return {"op": "hash", "texts": texts, "hashes": [int(ft_hash_bytes(t.encode("utf-8"))) for t in texts]}


def ngrams_case():
    rows = []
    for word in ["therapy", "a", "café", "\U0001F60A", "sleep"]:
        for minn, maxn in ((3, 6), (1, 3), (2, 2)):
            grams = [g.decode("utf-8") for g in compute_ngrams_bytes(word, minn, maxn)]
            rows.append({"word": word, "min_n": minn, "max_n": maxn, "ngrams": grams,
                         "buckets": [int(h) for h in ft_ngram_hashes(word, minn, maxn, 2000000)]})
    return {"op": "ngrams", "rows": rows}


def piece_rows(words, minn, maxn, bucket):
    v = len(words)
    return [[w] + [v + int(h) for h in ft_ngram_hashes(word, minn, maxn, bucket)] for w, word in enumerate(words)]


def fasttext_steps(words, minn, maxn, bucket, w_in, w_out, batches):
    """w_in: [V + bucket, d] (words then n-grams)."""
    rows_of = piece_rows(words, minn, maxn, bucket)
    w_in = torch.tensor(w_in, dtype=torch.float32)
    w_out = torch.tensor(w_out, dtype=torch.float32)
    losses = []
    for batch in batches:
        b, c, k = len(batch["outputs"]), batch["context_width"], batch["negative"]
        inputs = np.array(batch["inputs"]).reshape(b, c)
        example_rows = [[r for word in inputs[e] if word >= 0 for r in rows_of[word]] for e in range(b)]
        h = torch.stack([w_in[rows].mean(0) for rows in example_rows]).detach().requires_grad_(True)
        w_out.requires_grad_(True)
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
            new_in = w_in.clone()
            for e, rows in enumerate(example_rows):
                share = 1.0
                for r in rows:
                    new_in[r] -= batch["alpha"] * share * h.grad[e]
            w_in = new_in
            w_out = (w_out - batch["alpha"] * w_out.grad).detach()
        losses.append(float(loss.detach()))
    return w_in.numpy(), w_out.numpy(), losses


def steps_case(rng, label, words, minn, maxn, bucket, d, batches_spec):
    v = len(words)
    w_in = rng.normal(0, 0.3, size=(v + bucket, d)).astype(np.float32)
    w_out = rng.normal(0, 0.3, size=(v, d)).astype(np.float32)
    batches = []
    for alpha, b, c, k in batches_spec:
        inputs = rng.integers(0, v, size=(b, c))
        if c > 1:
            inputs[rng.random((b, c)) < 0.3] = -1
            inputs[:, 0] = rng.integers(0, v, size=b)  # every example keeps one word
        else:
            inputs[:3, 0] = inputs[0, 0]  # a repeated input word: summed update
        negatives = rng.integers(0, v, size=(b, k))
        negatives[rng.random((b, k)) < 0.1] = -1
        batches.append({"inputs": inputs.ravel().tolist(), "context_width": c,
                        "outputs": rng.integers(0, v, size=b).tolist(), "negatives": negatives.ravel().tolist(),
                        "negative": k, "alpha": alpha})
    final_in, final_out, losses = fasttext_steps(words, minn, maxn, bucket, w_in, w_out, batches)
    f64 = lambda m: m.astype(np.float64).ravel().tolist()
    return {"op": "steps", "label": label, "words": words, "min_n": minn, "max_n": maxn, "bucket": bucket, "dim": d,
            "word_rows": f64(w_in[:v]), "ngram_rows": f64(w_in[v:]), "output_vectors": f64(w_out),
            "batches": batches, "expected_word_rows": f64(final_in[:v]), "expected_ngram_rows": f64(final_in[v:]),
            "expected_output": f64(final_out), "losses": losses}


def vectors_case(rng):
    words = ["therapy", "therapist", "sleep", "sleepy", "cat"]
    minn, maxn, bucket, d = 3, 5, 40, 4
    v = len(words)
    word_rows = rng.normal(0, 0.5, size=(v, d)).astype(np.float32)
    ngram_rows = rng.normal(0, 0.5, size=(bucket, d)).astype(np.float32)
    wanted = ["sleep", "therapies", "cat", "zz", "x", ""]
    expected, sources = [], []
    for word in wanted:
        hashes = ft_ngram_hashes(word, minn, maxn, bucket)
        if word in words:
            vec = (word_rows[words.index(word)] + ngram_rows[hashes].sum(0)) / (len(hashes) + 1)
            sources.append(0)
        elif hashes:
            vec = ngram_rows[hashes].sum(0) / len(hashes)
            sources.append(1)
        else:
            vec = np.zeros(d, dtype=np.float32)
            sources.append(2)
        expected.append(vec.astype(np.float64).tolist())
    return {"op": "vectors", "words": words, "min_n": minn, "max_n": maxn, "bucket": bucket, "dim": d,
            "word_rows": word_rows.astype(np.float64).ravel().tolist(),
            "ngram_rows": ngram_rows.astype(np.float64).ravel().tolist(), "wanted": wanted,
            "expected": expected, "sources": sources}


def group_corpus(rng, groups=6, per_group=6, sentences=1500, length=8):
    stems = ["sleep", "worry", "happy", "anger", "music", "money"]
    words = [[f"{stems[g]}{suffix}" for suffix in ["", "s", "ed", "ing", "er", "y"][:per_group]] for g in range(groups)]
    filler = ["the", "and", "of", "a"]
    corpus = []
    for _ in range(sentences):
        group = words[rng.integers(groups)]
        sentence = [group[i] for i in rng.integers(0, per_group, size=length)]
        for _ in range(2):
            sentence.insert(int(rng.integers(0, len(sentence) + 1)), filler[rng.integers(len(filler))])
        corpus.append(sentence)
    return corpus, words


def group_accuracy(names, vectors, group_of, known_names, known_vectors):
    """How often each name's nearest known word (cosine, not itself) is in its group."""
    m = np.stack(known_vectors)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    hits = 0
    for name, vec in zip(names, vectors):
        sim = m @ (vec / np.linalg.norm(vec))
        for i, other in enumerate(known_names):
            if other == name:
                sim[i] = -np.inf
        hits += group_of[known_names[int(sim.argmax())]] == group_of[name]
    return float(hits / len(names))


def quality_case(rng, sg):
    corpus, groups = group_corpus(rng)
    settings = {"vector_size": 24, "window": 3, "min_count": 1, "negative": 5, "sample": 0.0, "epochs": 20,
                "alpha": 0.025, "min_alpha": 0.0001, "min_n": 3, "max_n": 6, "bucket": 2000}
    model = FastText(corpus, sg=sg, workers=1, seed=1, **settings)
    names = [w for g in groups for w in g]
    group_of = {w: i for i, g in enumerate(groups) for w in g}
    known = [model.wv[w] for w in names]
    unseen = [w + "x" for w in names]  # never in the corpus: vectors from pieces
    for w in unseen:
        group_of[w] = group_of[w[:-1]]
    return {"op": "quality", "label": "skip-gram" if sg else "cbow", "cbow": not sg, "settings": settings,
            "sentences": corpus, "groups": groups, "unseen": unseen,
            "gensim_accuracy": group_accuracy(names, known, group_of, names, known),
            "gensim_unseen_accuracy": group_accuracy(unseen, [model.wv[w] for w in unseen], group_of, names, known)}


def bin_case(rng):
    corpus, _ = group_corpus(rng, sentences=300)
    model = FastText(corpus, sg=1, workers=1, seed=3, vector_size=6, window=3, min_count=1, negative=3, epochs=2,
                     min_n=3, max_n=4, bucket=50)
    path = HERE.parent / "fixtures" / "word_embeddings" / "small_fasttext.bin"
    save_facebook_model(model, str(path))
    loaded = load_facebook_model(str(path)).wv
    words = list(loaded.index_to_key)
    wanted = ["sleep", "worrying", "sleepyx", "Sleep", "zz", ""]
    expected = [loaded[w].astype(np.float64).tolist() if (w in loaded.key_to_index or ft_ngram_hashes(w, 3, 4, 50))
                else [0.0] * 6 for w in wanted]
    sources = [0 if w in loaded.key_to_index else (1 if ft_ngram_hashes(w, 3, 4, 50) else 2) for w in wanted]
    vocabulary = ["[PAD]", "[UNK]", "sleep", "Worry", "sleepyx", "!!", "\U0001F60A"]
    return {"op": "bin", "file": "small_fasttext.bin", "dim": 6, "min_n": 3, "max_n": 4, "bucket": 50,
            "model": "skip-gram", "words": words, "first_vectors": loaded[words[:3]].astype(np.float64).ravel().tolist(),
            "wanted": wanted, "expected": expected, "sources": sources, "vocabulary": vocabulary,
            "aligned_rows": {"2": loaded["sleep"].astype(np.float64).tolist(),
                             "3": loaded["worry"].astype(np.float64).tolist(),
                             "4": loaded["sleepyx"].astype(np.float64).tolist()}}


def main():
    rng = np.random.default_rng(1526)
    small_words = ["the", "therapy", "sleep", "sleepy", "cat", "cats"]
    cases = [hash_case(), ngrams_case(),
             steps_case(rng, "small", small_words, 3, 4, 50, 5, [(0.05, 8, 1, 3), (0.04, 8, 1, 3), (0.03, 6, 3, 2)]),
             steps_case(np.random.default_rng(1527), "trainer sizes",
                        [f"w{i}{'x' * (i % 5)}y" for i in range(40)], 3, 6, 1000, 24,
                        [(0.025, 512, 1, 5), (0.0249, 512, 6, 5)]),
             vectors_case(rng), quality_case(rng, 1), quality_case(rng, 0), bin_case(np.random.default_rng(1528))]
    path = HERE.parent / "fixtures" / "fasttext_reference.json"
    path.write_text(json.dumps({"torch": torch.__version__, "gensim": gensim.__version__, "cases": cases}, indent=1))
    q = [c for c in cases if c["op"] == "quality"]
    print(f"wrote {path}: gensim group accuracy skip-gram {q[0]['gensim_accuracy']:.3f} "
          f"(unseen {q[0]['gensim_unseen_accuracy']:.3f}), cbow {q[1]['gensim_accuracy']:.3f} "
          f"(unseen {q[1]['gensim_unseen_accuracy']:.3f})")


if __name__ == "__main__":
    main()
