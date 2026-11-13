# SPDX-License-Identifier: Apache-2.0
# Part of the CyxWiz plugin SDK examples (see cmake/sdk/NOTICE).
# TOFIX129 D8.
"""pycyxwiz.text against the Python truth.

The cases are cyxwiz-engine/tests/computation_truth/fixtures/text_truth.json,
written by generate_text_fixtures.py from Python's re and str.lower,
scikit-learn's ENGLISH_STOP_WORDS, vaderSentiment and NLTK's PorterStemmer:
words, the stop words, VADER's unrounded scores and the stems are those;
terms are scikit-learn's word n-grams of those words. Run with the Python that
pycyxwiz was built for:

    python test_python_text.py <folder holding pycyxwiz> [<folder of its DLLs>]

Exit code 0 when every check passes.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
check = Checks()

here = os.path.dirname(os.path.abspath(__file__))
fixture = os.path.join(here, "..", "..", "..", "cyxwiz-engine", "tests", "computation_truth", "fixtures", "text_truth.json")
with open(fixture, encoding="utf-8") as f:
    truth = json.load(f)


def ngrams(words, low, high):
    """scikit-learn's _word_ngrams: the shortest first, each in text order."""
    out = []
    for n in range(low, high + 1):
        out += [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return out


stop = truth["stop_words"]
check(cyx.text.english_stop_words() == sorted(stop), "english_stop_words(): the %d stop words, sorted" % len(stop))

bad = [w for w in truth["words"] if cyx.text.words(w["text"]) != w["words"]]
check(not bad, "words(): %d cases%s" % (len(truth["words"]), "" if not bad else "; first wrong: %r" % bad[0]["text"]))

stop_set = set(stop)
wrong = 0
for w in truth["words"]:
    kept = [x for x in w["words"] if x not in stop_set]
    if (cyx.text.terms(w["text"]) != kept
            or cyx.text.terms(w["text"], stop_words=None, ngram_range=(1, 2)) != ngrams(w["words"], 1, 2)
            or cyx.text.terms(w["text"], stop_words="english", ngram_range=(2, 3)) != ngrams(kept, 2, 3)):
        wrong += 1
check(wrong == 0, "terms(): stop words left out, then word n-grams (%d texts, 3 settings; %d wrong)" % (len(truth["words"]), wrong))

for bad_arguments in ({"stop_words": "french"}, {"ngram_range": (2, 1)}, {"ngram_range": (0, 1)}):
    try:
        cyx.text.terms("not good", **bad_arguments)
        check(False, "terms() refuses %r" % bad_arguments)
    except ValueError:
        check(True, "terms() refuses %r" % bad_arguments)

worst = 0.0
for v in truth["vader"]:
    got = cyx.text.vader(v["text"])
    worst = max(worst, *(abs(got[k] - v[k]) for k in ("neg", "neu", "pos", "compound")))
check(worst <= 1e-12, "vader(): %d cases, largest difference %.1e" % (len(truth["vader"]), worst))
scores = cyx.text.vader("not good")
check(any(word == "good" and valence < 0 for word, valence in scores["words"]), "vader()['words']: \"good\" after \"not\" reads negative")

bad = [p for p in truth["porter"] if cyx.text.stem(p["word"]) != p["stem"]]
check(not bad, "stem(): %d words%s" % (len(truth["porter"]), "" if not bad else "; first wrong: %r" % bad[0]["word"]))

check.finish("test_python_text")
