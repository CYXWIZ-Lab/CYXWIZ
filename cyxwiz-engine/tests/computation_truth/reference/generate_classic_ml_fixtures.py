"""Classical supervised model references from scikit-learn (TOFIX145 Group C, classic ML on the device).

Writes ../fixtures/classic_ml_sklearn.json for test_computation_truth_classic_ml:
    py -3.14 generate_classic_ml_fixtures.py

Classes are 0..k-1 (sklearn's sorted classes_). Each case carries the training rows, labels, held-out rows, the
fitted parameters and predict / predict_proba on both.
"""
import json
from pathlib import Path

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.naive_bayes import GaussianNB


def blobs(rng, centers, spread, per_class):
    x, y = [], []
    for label, center in enumerate(centers):
        x.append(rng.normal(center, spread, size=(per_class, len(center))))
        y.append(np.full(per_class, label))
    return np.vstack(x), np.concatenate(y)


def gaussian_nb_case(label, x, y, x_test, var_smoothing=1e-9):
    model = GaussianNB(var_smoothing=var_smoothing).fit(x, y)
    return {
        "op": "gaussian_nb",
        "label": label,
        "x": x.tolist(),
        "y": y.astype(int).tolist(),
        "x_test": x_test.tolist(),
        "n_classes": int(len(model.classes_)),
        "var_smoothing": var_smoothing,
        "theta": model.theta_.tolist(),
        "var": model.var_.tolist(),
        "class_prior": model.class_prior_.tolist(),
        "epsilon": float(model.epsilon_),
        "predict": model.predict(x).astype(int).tolist(),
        "predict_proba": model.predict_proba(x).tolist(),
        "predict_test": model.predict(x_test).astype(int).tolist(),
        "predict_proba_test": model.predict_proba(x_test).tolist(),
    }


def main():
    rng = np.random.default_rng(145)
    cases = []

    # Three overlapping classes in 4 features.
    x, y = blobs(rng, [[0, 0, 0, 0], [1.2, 0.8, -0.5, 0.3], [-0.7, 1.5, 0.9, -1.0]], 0.9, 30)
    x_test = rng.normal(0.3, 1.2, size=(25, 4))
    cases.append(gaussian_nb_case("three overlapping classes", x, y, x_test))

    # Two well-separated classes, unequal sizes (priors differ).
    a = rng.normal([-3, 2], 0.5, size=(40, 2))
    b = rng.normal([3, -2], 0.7, size=(15, 2))
    x = np.vstack([a, b])
    y = np.concatenate([np.zeros(40, int), np.ones(15, int)])
    cases.append(gaussian_nb_case("two separated classes, unequal priors", x, y, rng.normal(0, 3, size=(20, 2))))

    # A constant column (zero variance in every class): var_smoothing keeps it finite.
    x, y = blobs(rng, [[0, 5], [2, 5], [4, 5]], 0.6, 12)
    x[:, 1] = 5.0
    cases.append(gaussian_nb_case("constant column, larger smoothing", x, y,
                                  np.column_stack([rng.normal(2, 2, 10), np.full(10, 5.0)]), var_smoothing=1e-3))

    out = {"sklearn": sklearn_version, "cases": cases}
    path = Path(__file__).resolve().parent.parent / "fixtures" / "classic_ml_sklearn.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} ({len(cases)} cases, scikit-learn {sklearn_version})")


if __name__ == "__main__":
    main()
