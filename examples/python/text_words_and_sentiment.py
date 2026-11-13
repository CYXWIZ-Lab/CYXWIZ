#!/usr/bin/env python3
"""Words, terms, stems and VADER sentiment with pycyxwiz.text.

What it shows: the rules behind Data Studio's Explore > Text and the text
nodes, from Python. A few short product reviews written for this example
(no dataset needed) are split into words, counted per label with the common
English words left out, turned into word pairs as the TF-IDF node counts them
(the negations kept), stemmed, and scored with VADER. The same calls give the
same answers as Python's re, scikit-learn, NLTK and vaderSentiment
(docs/usage/text.md, python_doc/pycyxwiz.md section 9.1).

Run it with the Python pycyxwiz is built for, naming the folder that holds
pycyxwiz (and, if different, the folder of its DLLs):

    python examples/python/text_words_and_sentiment.py build/windows-release/bin/Release

Expected result:

    good: great 2, battery 2, lasts 1, days 1, really 1, happy 1
    bad:  stopped 2, charger 1, good 1, charging 1, week 1, worth 1
    pairs with "not" (stop words kept): not good, not worth
    stems: charging -> charg, charger -> charger, charged -> charg
    +0.855 positive good  Great battery, it lasts two days. Really happy with it!
    +0.875 positive good  The screen is bright and the battery is great :)
    +0.000 neutral  good  Fast delivery, works exactly as described. I'd buy it again.
    -0.512 negative bad   The charger is not good and it stopped charging after a week.
    -0.723 negative bad   Not worth the money. The sound stopped working, very disappointing.
    +0.232 positive bad   It isn't bad, but I wouldn't buy it again.

Two things to notice: "good" is among the bad reviews' words because "not" is
a stop word (keep the stop words for sentiment features: the pairs line), and
VADER reads "isn't bad" as positive, which is right for that clause, while
the rest of the review is a complaint without a rated word. A lexicon scores
words, not intent.
"""
import os
import sys
from collections import Counter

folder = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
dll_folder = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else folder
for d in (folder, dll_folder, r"C:\Program Files\ArrayFire\v3\lib"):
    if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(d)
os.environ["PATH"] = os.pathsep.join((folder, dll_folder)) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, folder)

import pycyxwiz.text as tx  # noqa: E402

reviews = [
    ("good", "Great battery, it lasts two days. Really happy with it!"),
    ("good", "The screen is bright and the battery is great :)"),
    ("good", "Fast delivery, works exactly as described. I'd buy it again."),
    ("bad", "The charger is not good and it stopped charging after a week."),
    ("bad", "Not worth the money. The sound stopped working, very disappointing."),
    ("bad", "It isn't bad, but I wouldn't buy it again."),
]

# The words per label, the common English words left out (Explore > Text > Words).
for label in ("good", "bad"):
    counts = Counter()
    for lab, text in reviews:
        if lab == label:
            counts.update(tx.terms(text, stop_words="english"))
    print(f"{label + ':':5} " + ", ".join(f"{w} {c}" for w, c in counts.most_common(6)))

# Word pairs as the TF-IDF node counts them. With the stop words left out,
# "not" goes too, so sentiment features keep them (stop_words=None).
pairs = set()
for _, text in reviews:
    pairs.update(p for p in tx.terms(text, stop_words=None, ngram_range=(2, 2)) if p.startswith("not "))
print('pairs with "not" (stop words kept): ' + ", ".join(sorted(pairs)))

# Porter stems: the forms of a word counted as one (Words "by stem").
print("stems: " + ", ".join(f"{w} -> {tx.stem(w)}" for w in ("charging", "charger", "charged")))

# VADER on every review (Explore > Text > Sentiment; the Sentiment Analysis node
# labels a row positive from 0.05 and negative at -0.05 and below).
for label, text in reviews:
    compound = tx.vader(text)["compound"]
    reads = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    print(f"{compound:+.3f} {reads:8} {label:5} {text}")
