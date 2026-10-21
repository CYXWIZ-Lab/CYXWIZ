"""Benchmark the tracked sentiment TF-IDF feature configurations.

This intentionally uses one fixed sparse sklearn classifier for both cases so
the measurement isolates text-feature quality. It is reference evidence, not a
replacement for a live CyxWiz MLP/device/training-lifecycle run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DEFAULT_DATASET = Path(
    "D:/demo/mrcj/datasets/sentiment analysis/sentiment_mental_health.csv"
)
SEED = 42


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
        for row in csv.DictReader(source):
            text = (row.get("statement") or "").strip()
            label = (row.get("status") or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    if not texts:
        raise RuntimeError("dataset has no non-empty statement/status rows")
    return texts, labels


def load_cyxwiz_stop_words(repository_root: Path) -> list[str]:
    source_path = (
        repository_root
        / "cyxwiz-backend"
        / "src"
        / "algorithms"
        / "text_processing.cpp"
    )
    source = source_path.read_text(encoding="utf-8")
    start = source.index("g_stopwords = {")
    end = source.index("};", start)
    words = sorted(set(re.findall(r'"([a-z]+)"', source[start:end])))
    if "not" not in words or "no" not in words:
        raise RuntimeError("CyxWiz stop-word extraction lost negation terms")
    return words


def benchmark_case(
    name: str,
    vectorizer_parameters: dict[str, object],
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    train_labels: list[str],
    val_labels: list[str],
    test_labels: list[str],
) -> dict[str, object]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        max_features=8000,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float32,
        **vectorizer_parameters,
    )
    vectorizer_start = time.perf_counter()
    train_features = vectorizer.fit_transform(train_texts)
    val_features = vectorizer.transform(val_texts)
    test_features = vectorizer.transform(test_texts)
    vectorizer_seconds = time.perf_counter() - vectorizer_start

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=30,
        tol=1e-4,
        random_state=SEED,
        class_weight="balanced",
        average=True,
    )
    classifier_start = time.perf_counter()
    classifier.fit(train_features, train_labels)
    classifier_seconds = time.perf_counter() - classifier_start

    return {
        "name": name,
        "vectorizer": {
            "max_features": 8000,
            "min_df": vectorizer_parameters["min_df"],
            "ngram_range": list(vectorizer_parameters["ngram_range"]),
            "stop_words": (
                "none" if vectorizer_parameters["stop_words"] is None
                else "cyxwiz_english"
            ),
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
        "matrix": {
            "train_rows": train_features.shape[0],
            "columns": train_features.shape[1],
            "train_nnz": int(train_features.nnz),
        },
        "classifier_iterations": int(classifier.n_iter_),
        "validation_accuracy": float(
            accuracy_score(val_labels, classifier.predict(val_features))
        ),
        "test_accuracy": float(
            accuracy_score(test_labels, classifier.predict(test_features))
        ),
        "timing_seconds": {
            "vectorizer_fit_and_transform": vectorizer_seconds,
            "classifier_fit": classifier_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    arguments = parser.parse_args()
    dataset_path = arguments.dataset.resolve()
    repository_root = Path(__file__).resolve().parents[3]

    texts, labels = load_dataset(dataset_path)
    train_texts, holdout_texts, train_labels, holdout_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        holdout_texts,
        holdout_labels,
        test_size=0.5,
        random_state=SEED,
        stratify=holdout_labels,
    )
    stop_words = load_cyxwiz_stop_words(repository_root)

    cases = [
        benchmark_case(
            "unigram_benchmark",
            {"min_df": 2, "ngram_range": (1, 1), "stop_words": stop_words},
            train_texts,
            val_texts,
            test_texts,
            train_labels,
            val_labels,
            test_labels,
        ),
        benchmark_case(
            "unigram_bigram_candidate",
            {"min_df": 1, "ngram_range": (1, 2), "stop_words": None},
            train_texts,
            val_texts,
            test_texts,
            train_labels,
            val_labels,
            test_labels,
        ),
    ]
    baseline, candidate = cases
    result = {
        "schema_version": 1,
        "benchmark_kind": "reference_text_feature_isolation",
        "engine_graph_training": False,
        "oracle": {"name": "scikit-learn", "version": sklearn.__version__},
        "dataset": {
            "path": str(dataset_path),
            "sha256": file_sha256(dataset_path),
            "rows": len(texts),
            "class_counts": dict(sorted(Counter(labels).items())),
        },
        "split": {
            "seed": SEED,
            "stratified": True,
            "train_rows": len(train_texts),
            "validation_rows": len(val_texts),
            "test_rows": len(test_texts),
        },
        "estimator": {
            "type": "SGDClassifier",
            "loss": "log_loss",
            "alpha": 1e-5,
            "max_iter": 30,
            "class_weight": "balanced",
            "average": True,
        },
        "cases": cases,
        "candidate_delta": {
            "validation_accuracy": (
                candidate["validation_accuracy"] - baseline["validation_accuracy"]
            ),
            "test_accuracy": candidate["test_accuracy"] - baseline["test_accuracy"],
        },
        "limitations": [
            "This isolates text features with one sparse sklearn classifier.",
            "It does not validate CyxWiz MLP device placement or training lifecycle.",
            "A live Engine run is required before replacing the benchmark graph.",
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
