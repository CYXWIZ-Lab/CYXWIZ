"""How closely two word2vec models agree: for the most frequent words of the
reference, the share of their 10 nearest neighbours that both models find
(TOFIX152; used with tests/benchmarks_word2vec.cpp).

py -3.12 scripts/word2vec_neighbour_overlap.py reference.vec candidate.vec [words]
"""
import sys

from gensim.models import KeyedVectors

reference = KeyedVectors.load_word2vec_format(sys.argv[1])
candidate = KeyedVectors.load_word2vec_format(sys.argv[2])
count = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
words = [w for w in reference.index_to_key[:count] if w in candidate.key_to_index]
shared = 0
for word in words:
    a = {w for w, _ in reference.most_similar(word, topn=10)}
    b = {w for w, _ in candidate.most_similar(word, topn=10)}
    shared += len(a & b)
print(f"{len(words)} words: {shared / (10 * len(words)):.3f} of the 10 nearest neighbours shared")
