"""Generate sklearn/PyTorch classification-metric computation-truth fixtures."""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


TOLERANCE = 1.0e-12


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def float_list(values: Any) -> list[float | None]:
    return [finite_or_none(float(value)) for value in values]


def confusion_expected(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    names = class_names or [f"Class {label}" for label in labels]
    return {
        "labels": labels,
        "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": names,
        "total_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float_list(precision),
        "recall": float_list(recall),
        "f1": float_list(f1),
        "support": [int(value) for value in support],
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
    }


def binary_expected(
    y_true: list[int], y_scores: list[float], threshold: float
) -> dict[str, Any]:
    y_pred = [int(score >= threshold) for score in y_scores]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.simplefilter("ignore", UserWarning)
        balanced = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "y_pred": y_pred,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def curve_expected(y_true: list[int], y_scores: list[float]) -> dict[str, Any]:
    fpr, tpr, roc_thresholds = roc_curve(
        y_true, y_scores, drop_intermediate=False
    )
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true, y_scores, drop_intermediate=False
    )
    return {
        "roc": {
            "fpr": float_list(fpr),
            "tpr": float_list(tpr),
            "thresholds": float_list(roc_thresholds),
            "auc": float(roc_auc_score(y_true, y_scores)),
        },
        "pr": {
            "precision": float_list(precision),
            "recall": float_list(recall),
            "thresholds": float_list(pr_thresholds),
            "average_precision": float(average_precision_score(y_true, y_scores)),
        },
    }


def threshold_expected(
    y_true: list[int], y_scores: list[float], criterion: str
) -> float:
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in sorted(set(y_scores)):
        metrics = binary_expected(y_true, y_scores, threshold)
        if criterion == "f1":
            score = metrics["f1"]
        elif criterion == "youden":
            score = metrics["recall"] + metrics["specificity"] - 1.0
        elif criterion == "balanced":
            score = metrics["balanced_accuracy"]
        else:
            raise ValueError(f"unsupported criterion: {criterion}")
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def main() -> None:
    confusion_cases = [
        {
            "name": "multiclass_union_predicted_only_class",
            "y_true": [2, 0, 2, 1, 0, 1, 2, 2],
            "y_pred": [2, 0, 1, 1, 2, 1, 2, 3],
            "class_names": ["zero", "one", "two", "three"],
        },
        {
            "name": "signed_noncontiguous_labels",
            "y_true": [-3, 7, 7, -3, 2],
            "y_pred": [7, 7, 2, -3, 2],
        },
    ]
    for case in confusion_cases:
        case["expected"] = confusion_expected(
            case["y_true"], case["y_pred"], case.get("class_names")
        )

    pytorch_binary_logits = torch.tensor(
        [-2.0, 0.0, 1.5, -0.0, 3.0, -3.0], dtype=torch.float64
    )
    binary_cases = [
        {
            "name": "threshold_ties_and_imbalance",
            "decision_oracle": "score >= threshold",
            "y_true": [0, 1, 0, 1, 1, 0],
            "y_scores": [0.2, 0.8, 0.8, 0.5, 0.5, 0.1],
            "threshold": 0.5,
        },
        {
            "name": "pytorch_binary_logit_zero_boundary",
            "decision_oracle": "torch.ge(logits, 0)",
            "y_true": [0, 1, 1, 0, 1, 0],
            "y_scores": [float(value) for value in pytorch_binary_logits.tolist()],
            "threshold": 0.0,
            "pytorch_y_pred": [
                int(value)
                for value in torch.ge(pytorch_binary_logits, 0.0).to(torch.int64).tolist()
            ],
        },
        {
            "name": "positive_only_balanced_accuracy",
            "decision_oracle": "score >= threshold",
            "y_true": [1, 1, 1],
            "y_scores": [0.2, 0.7, 0.8],
            "threshold": 0.5,
        },
        {
            "name": "negative_only_balanced_accuracy",
            "decision_oracle": "score >= threshold",
            "y_true": [0, 0, 0],
            "y_scores": [0.2, 0.7, 0.1],
            "threshold": 0.5,
        },
    ]
    for case in binary_cases:
        case["expected"] = binary_expected(
            case["y_true"], case["y_scores"], case["threshold"]
        )

    curve_cases = [
        {
            "name": "distinct_scores",
            "y_true": [0, 0, 1, 1],
            "y_scores": [0.1, 0.4, 0.35, 0.8],
        },
        {
            "name": "tied_scores_and_imbalance",
            "y_true": [0, 1, 0, 1, 1, 0],
            "y_scores": [0.2, 0.8, 0.8, 0.5, 0.5, 0.1],
        },
        {
            "name": "pytorch_binary_logits",
            "y_true": binary_cases[1]["y_true"],
            "y_scores": binary_cases[1]["y_scores"],
        },
    ]
    large_y_true = [1 if (index * 11) % 7 < 3 else 0 for index in range(1005)]
    large_y_scores = [float(((index * 37) % 17) - 8) / 4.0 for index in range(1005)]
    curve_cases.append(
        {
            "name": "large_tied_route_boundary_1005",
            "y_true": large_y_true,
            "y_scores": large_y_scores,
        }
    )
    for case in curve_cases:
        case["expected"] = curve_expected(case["y_true"], case["y_scores"])

    multiclass_logits = torch.tensor(
        [
            [2.0, 0.5, -1.0],
            [0.1, 2.5, 0.2],
            [-0.5, 0.3, 1.2],
            [1.0, 1.0, 0.5],
            [0.2, 0.4, 0.4],
            [-2.0, -1.0, -0.5],
            [0.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=torch.float64,
    )
    multiclass_y_true = [0, 1, 2, 1, 2, 2, 0, 1]
    multiclass_scores = torch.softmax(multiclass_logits, dim=1)
    multiclass_y_pred = torch.argmax(multiclass_logits, dim=1)
    class_roc = []
    class_pr = []
    for class_index in range(multiclass_scores.shape[1]):
        binary_true = [int(label == class_index) for label in multiclass_y_true]
        scores = [float(value) for value in multiclass_scores[:, class_index].tolist()]
        expected = curve_expected(binary_true, scores)
        class_roc.append(expected["roc"])
        class_pr.append(expected["pr"])
    multiclass_case = {
        "name": "pytorch_softmax_argmax_three_class",
        "decision_oracle": "torch.softmax + torch.argmax(dim=1)",
        "y_true": multiclass_y_true,
        "logits": multiclass_logits.tolist(),
        "y_scores": multiclass_scores.tolist(),
        "y_pred": [int(value) for value in multiclass_y_pred.tolist()],
        "expected": {
            "confusion": confusion_expected(
                multiclass_y_true,
                [int(value) for value in multiclass_y_pred.tolist()],
            ),
            "roc": {
                "class_fpr": [value["fpr"] for value in class_roc],
                "class_tpr": [value["tpr"] for value in class_roc],
                "class_thresholds": [value["thresholds"] for value in class_roc],
                "class_auc": [value["auc"] for value in class_roc],
                "auc": float(np.mean([value["auc"] for value in class_roc])),
            },
            "pr": {
                "class_precision": [value["precision"] for value in class_pr],
                "class_recall": [value["recall"] for value in class_pr],
                "class_thresholds": [value["thresholds"] for value in class_pr],
                "class_ap": [value["average_precision"] for value in class_pr],
                "average_precision": float(
                    np.mean([value["average_precision"] for value in class_pr])
                ),
            },
        },
    }

    threshold_input = curve_cases[1]
    threshold_case = {
        "name": "ascending_unique_score_candidates_strict_first_tie",
        "y_true": threshold_input["y_true"],
        "y_scores": threshold_input["y_scores"],
        "expected": {
            criterion: threshold_expected(
                threshold_input["y_true"], threshold_input["y_scores"], criterion
            )
            for criterion in ("f1", "youden", "balanced")
        },
    }

    auc_cases = [
        {
            "name": "increasing_x",
            "x": [0.0, 0.25, 1.0],
            "y": [0.0, 0.75, 1.0],
        },
        {
            "name": "decreasing_x",
            "x": [1.0, 0.25, 0.0],
            "y": [1.0, 0.75, 0.0],
        },
        {
            "name": "constant_x",
            "x": [0.5, 0.5, 0.5],
            "y": [0.0, 0.5, 1.0],
        },
    ]
    for case in auc_cases:
        case["expected"] = float(auc(case["x"], case["y"]))

    fixture = {
        "schema_version": 1,
        "oracles": {
            "metrics": {"name": "scikit-learn", "version": sklearn.__version__},
            "decisions": {"name": "PyTorch", "version": torch.__version__},
        },
        "dtype": "float64",
        "zero_division": 0,
        "roc_drop_intermediate": False,
        "roc_infinite_threshold_encoding": "null",
        "pr_average": "non-interpolated-average-precision",
        "tolerance": TOLERANCE,
        "confusion_cases": confusion_cases,
        "binary_cases": binary_cases,
        "curve_cases": curve_cases,
        "multiclass_cases": [multiclass_case],
        "threshold_cases": [threshold_case],
        "auc_cases": auc_cases,
    }
    output = Path(__file__).resolve().parent.parent / "fixtures" / (
        "classification_metrics_sklearn_pytorch.json"
    )
    serialized = json.dumps(fixture, indent=2, allow_nan=False) + "\n"
    if "--stdout" in sys.argv[1:]:
        print(serialized, end="")
    else:
        output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
