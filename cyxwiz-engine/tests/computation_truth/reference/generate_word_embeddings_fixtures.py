"""Word vector files and their expected readings (TOFIX152).

Writes small word2vec / GloVe / fastText files with gensim 4 and records what
gensim's KeyedVectors.load_word2vec_format reads back, plus the expected
vocabulary alignments (a numpy reference of WordEmbeddings::Align's rules).
Run with Python 3.12 (gensim): py -3.12 generate_word_embeddings_fixtures.py
"""

import json
from pathlib import Path

import gensim
import numpy as np
from gensim.models import KeyedVectors

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "fixtures" / "word_embeddings"


def keyed(words, dim, rng):
    kv = KeyedVectors(vector_size=dim)
    kv.add_vectors(words, rng.normal(0, 1, size=(len(words), dim)).astype(np.float32))
    return kv


def reading(path, binary, no_header=False, limit=None):
    kv = KeyedVectors.load_word2vec_format(str(path), binary=binary, no_header=no_header, limit=limit)
    # gensim sizes a header-less file by its line count, so a skipped repeated
    # word leaves an unnamed (None) all-zero slot at the end; it is not a word.
    keep = [i for i, w in enumerate(kv.index_to_key) if w is not None]
    return {"words": [kv.index_to_key[i] for i in keep], "dim": int(kv.vector_size),
            "values": kv.vectors[keep].astype(np.float64).ravel().tolist()}


def align(words, values, dim, vocab, lowercase=True, unknown="mean", pad_id=0, unknown_id=1):
    """WordEmbeddings::Align with missing = zeros."""
    table = np.asarray(values, dtype=np.float32).reshape(len(words), dim)
    exact, lower = {}, {}
    for r, w in enumerate(words):
        exact.setdefault(w, r)
        lower.setdefault(w.lower() if w.isascii() else ascii_lower(w), r)
    out = np.zeros((len(vocab), dim), dtype=np.float32)
    counts = {"exact": 0, "lowercase": 0, "missing": 0}
    missing_words = []
    for i, token in enumerate(vocab):
        if i == pad_id:
            continue
        if i == unknown_id and unknown == "mean":
            out[i] = table.mean(axis=0)
            continue
        if token in exact:
            out[i] = table[exact[token]]
            counts["exact"] += 1
        elif lowercase and ascii_lower(token) in lower:
            out[i] = table[lower[ascii_lower(token)]]
            counts["lowercase"] += 1
        elif i != unknown_id:
            counts["missing"] += 1
            missing_words.append(token)
    return {"values": out.astype(np.float64).ravel().tolist(), "dim": dim, "missing_words": missing_words, **counts}


def ascii_lower(word):
    return "".join(c.lower() if "A" <= c <= "Z" else c for c in word)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(152)
    words = ["the", "paris", "London", "café", "naïve", "king", "queen", "man", "woman", "data"]
    kv = keyed(words, 6, rng)
    kv.save_word2vec_format(str(OUT / "vectors.bin"), binary=True)
    kv.save_word2vec_format(str(OUT / "vectors.vec"), binary=False)
    kv.save_word2vec_format(str(OUT / "glove.txt"), binary=False, write_header=False)
    # A hand-written GloVe file: CRLF line ends, a repeated word (the first
    # vector wins) and tabs-free single spaces as GloVe writes them.
    (OUT / "glove_repeat.txt").write_bytes(
        b"apple 0.5 -1.25 2\r\nbanana 1e-3 0 -0.75\r\napple 9 9 9\r\ncherry -2 0.125 3.5\r\n")

    cases = []
    for name, binary, no_header in [("vectors.bin", True, False), ("vectors.vec", False, False),
                                    ("glove.txt", False, True), ("glove_repeat.txt", False, True)]:
        cases.append({"op": "load", "file": name, "max_words": 0, **reading(OUT / name, binary, no_header)})
    cases.append({"op": "load", "file": "vectors.bin", "max_words": 4, **reading(OUT / "vectors.bin", True, limit=4)})
    cases.append({"op": "load", "file": "vectors.vec", "max_words": 3, **reading(OUT / "vectors.vec", False, limit=3)})

    loaded = reading(OUT / "vectors.vec", False)
    vocab = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "the", "Paris", "london", "café", "CAFÉ", "king", "unicorn", "data"]
    for label, kwargs in [("default", {}), ("no lowercase", {"lowercase": False}),
                          ("unknown as missing", {"unknown": "missing"}), ("no pad or unknown", {"pad_id": -1, "unknown_id": -1})]:
        options = {"lowercase_fallback": kwargs.get("lowercase", True), "unknown": kwargs.get("unknown", "mean"),
                   "pad_id": kwargs.get("pad_id", 0), "unknown_id": kwargs.get("unknown_id", 1)}
        cases.append({"op": "align", "label": label, "file": "vectors.vec", "vocabulary": vocab, "options": options,
                      **align(loaded["words"], loaded["values"], loaded["dim"], vocab, **kwargs)})

    path = HERE.parent / "fixtures" / "word_embeddings_gensim.json"
    path.write_text(json.dumps({"gensim": gensim.__version__, "cases": cases}, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print(f"wrote {path} ({len(cases)} cases, gensim {gensim.__version__})")


if __name__ == "__main__":
    main()
