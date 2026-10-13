"""ROC / precision-recall references from scikit-learn (TOFIX140, ROC / PR on the device).

Writes ../fixtures/roc_pr_sklearn.json for test_computation_truth_roc_pr:
    python generate_roc_pr_fixtures.py

Binary: roc_curve(drop_intermediate=False), roc_auc_score, precision_recall_curve,
average_precision_score. Multi-class: the same one-vs-rest per class, macro means.
"""
import json
from pathlib import Path

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "roc_pr_sklearn.json"


def floats(values):
    return [float(v) for v in values]


def binary_case(label, y, s):
    fpr, tpr, roc_thr = roc_curve(y, s, drop_intermediate=False)
    precision, recall, pr_thr = precision_recall_curve(y, s)
    return {
        "op": "binary", "label": label, "y_true": [int(v) for v in y], "y_scores": floats(s),
        "roc": {"fpr": floats(fpr), "tpr": floats(tpr), "thresholds": floats(roc_thr[1:]),
                "auc": float(roc_auc_score(y, s))},
        "pr": {"precision": floats(precision), "recall": floats(recall), "thresholds": floats(pr_thr),
               "ap": float(average_precision_score(y, s))},
    }


def multiclass_case(label, y, scores):
    classes = scores.shape[1]
    aucs, aps = [], []
    for c in range(classes):
        yc = (y == c).astype(int)
        aucs.append(float(roc_auc_score(yc, scores[:, c])))
        aps.append(float(average_precision_score(yc, scores[:, c])))
    return {
        "op": "multiclass", "label": label, "y_true": [int(v) for v in y], "y_scores": [floats(r) for r in scores],
        "class_auc": aucs, "class_ap": aps, "macro_ap": float(np.mean(aps)),
    }


def main():
    rng = np.random.default_rng(140)
    cases = []

    # Ties: scores rounded to two decimals, labels correlated with the score.
    s = np.round(rng.uniform(0, 1, 300), 2)
    y = (rng.uniform(0, 1, 300) < 0.2 + 0.6 * s).astype(int)
    cases.append(binary_case("ties", y, s))

    y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.45, 0.7, 0.9, 0.2, 0.55, 0.65])
    cases.append(binary_case("small", y, s))

    y = np.array([0, 1, 0, 1, 1])
    s = np.full(5, 0.5)
    cases.append(binary_case("all_tied", y, s))

    logits = rng.normal(0, 1, (200, 4))
    y = rng.integers(0, 4, 200)
    logits[np.arange(200), y] += 1.2
    scores = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    cases.append(multiclass_case("softmax4", y, scores))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generator": "scikit-learn " + sklearn_version, "cases": cases}, indent=1))
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
