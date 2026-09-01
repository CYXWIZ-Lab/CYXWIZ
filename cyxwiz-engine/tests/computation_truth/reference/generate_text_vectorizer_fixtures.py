"""Generate scikit-learn text-vectorizer computation-truth fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


DOCUMENTS = [
    "Good good movie",
    "not good movie",
    "bad ending",
]


CASES = [
    {
        "name": "count_raw_unigram_bigram",
        "operator": "CountVectorizer",
        "parameters": {
            "ngram_range": "1,2",
            "binary": "false",
            "norm": "none",
        },
    },
    {
        "name": "count_binary_bigram_only",
        "operator": "CountVectorizer",
        "parameters": {
            "ngram_range": "2,2",
            "binary": "true",
            "norm": "none",
        },
    },
    {
        "name": "count_l2_unigram",
        "operator": "CountVectorizer",
        "parameters": {
            "ngram_range": "1,1",
            "binary": "false",
            "norm": "l2",
        },
    },
    {
        "name": "count_max_features_uses_corpus_count",
        "operator": "CountVectorizer",
        "documents": [
            "burst burst burst burst",
            "spread token",
            "spread value",
            "spread ending",
        ],
        "parameters": {
            "max_features": "1",
            "ngram_range": "1,1",
            "binary": "false",
            "norm": "none",
        },
    },
    {
        "name": "tfidf_raw_unigram_bigram",
        "operator": "TFIDFVectorizer",
        "parameters": {
            "ngram_range": "1,2",
            "use_idf": "true",
            "smooth_idf": "true",
            "norm": "none",
        },
    },
    {
        "name": "tfidf_raw_without_idf",
        "operator": "TFIDFVectorizer",
        "parameters": {
            "ngram_range": "1,1",
            "use_idf": "false",
            "smooth_idf": "false",
            "norm": "none",
        },
    },
    {
        "name": "tfidf_l2_unigram_through_trigram",
        "operator": "TFIDFVectorizer",
        "parameters": {
            "ngram_range": "1,3",
            "use_idf": "true",
            "smooth_idf": "true",
            "norm": "l2",
        },
    },
    {
        "name": "tfidf_max_features_uses_corpus_count",
        "operator": "TFIDFVectorizer",
        "documents": [
            "burst burst burst burst",
            "spread token",
            "spread value",
            "spread ending",
        ],
        "parameters": {
            "max_features": "1",
            "ngram_range": "1,1",
            "use_idf": "true",
            "smooth_idf": "true",
            "norm": "none",
        },
    },
]


def generate_case(case: dict[str, object]) -> dict[str, object]:
    parameters = case["parameters"]
    assert isinstance(parameters, dict)
    documents = case.get("documents", DOCUMENTS)
    assert isinstance(documents, list)
    ngram_range = tuple(int(value) for value in parameters["ngram_range"].split(","))
    max_features = int(parameters.get("max_features", "128"))

    if case["operator"] == "CountVectorizer":
        vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=ngram_range,
            stop_words=None,
            binary=parameters["binary"] == "true",
            max_features=max_features,
        )
        matrix = vectorizer.fit_transform(documents).astype(float)
        if parameters["norm"] != "none":
            matrix = normalize(matrix, norm=parameters["norm"], copy=False)
        idf = None
    else:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=ngram_range,
            stop_words=None,
            use_idf=parameters["use_idf"] == "true",
            smooth_idf=parameters["smooth_idf"] == "true",
            norm=None if parameters["norm"] == "none" else parameters["norm"],
            max_features=max_features,
        )
        matrix = vectorizer.fit_transform(documents)
        idf = (
            [float(value) for value in vectorizer.idf_]
            if parameters["use_idf"] == "true"
            else [1.0] * len(vectorizer.get_feature_names_out())
        )

    generated = {
        **case,
        "documents": documents,
        "feature_names": vectorizer.get_feature_names_out().tolist(),
        "expected": matrix.toarray().tolist(),
    }
    if idf is not None:
        generated["idf"] = idf
    return generated


def main() -> None:
    fixture = {
        "schema_version": 1,
        "oracle": {"name": "scikit-learn", "version": sklearn.__version__},
        "dtype": "float64",
        "tolerance": 1e-5,
        "notes": (
            "CyxWiz tokenization is constrained to simple lowercase ASCII words "
            "in this fixture. Count l1/l2 cases use sklearn.preprocessing.normalize."
        ),
        "cases": [generate_case(case) for case in CASES],
    }
    output = Path(__file__).resolve().parent.parent / "fixtures" / (
        "text_vectorizers_sklearn.json"
    )
    serialized = json.dumps(fixture, indent=2) + "\n"
    if "--stdout" in sys.argv[1:]:
        print(serialized, end="")
    else:
        output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
