"""Generate scikit-learn regression-metric computation-truth fixtures."""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import sklearn
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


CASES = [
    {
        "name": "mixed_nonzero_targets",
        "y_true": [3.0, -0.5, 2.0, 7.0],
        "y_pred": [2.5, 0.0, 2.0, 8.0],
    },
    {
        "name": "perfect_constant_target",
        "y_true": [4.0, 4.0, 4.0],
        "y_pred": [4.0, 4.0, 4.0],
    },
    {
        "name": "imperfect_constant_target",
        "y_true": [4.0, 4.0, 4.0],
        "y_pred": [4.0, 5.0, 3.0],
    },
    {
        "name": "zero_and_signed_targets",
        "y_true": [0.0, 1.0, -2.0, 0.0],
        "y_pred": [1.0, 1.5, -1.0, 0.0],
    },
    {
        "name": "single_sample_undefined_r2",
        "y_true": [2.0],
        "y_pred": [1.5],
    },
]


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def main() -> None:
    cases = []
    for case in CASES:
        y_true = case["y_true"]
        y_pred = case["y_pred"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            r_squared = finite_or_none(float(r2_score(y_true, y_pred)))
        expected = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r_squared": r_squared,
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
            "max_error": float(max_error(y_true, y_pred)),
        }
        cases.append({**case, "expected": expected})

    fixture = {
        "schema_version": 1,
        "oracle": {"name": "scikit-learn", "version": sklearn.__version__},
        "dtype": "float64",
        "mape_units": "relative_ratio",
        "tolerance": 1e-12,
        "cases": cases,
    }
    output = Path(__file__).resolve().parent.parent / "fixtures" / (
        "regression_metrics_sklearn.json"
    )
    serialized = json.dumps(fixture, indent=2) + "\n"
    if "--stdout" in sys.argv[1:]:
        print(serialized, end="")
    else:
        output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
