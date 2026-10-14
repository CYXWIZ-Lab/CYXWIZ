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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from scipy.optimize import minimize


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


def knn_case(label, x, y, x_test, n_neighbors, weights, metric):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm="brute").fit(x, y)
    return {
        "op": "knn",
        "label": label,
        "x": x.tolist(),
        "y": y.astype(int).tolist(),
        "x_test": x_test.tolist(),
        "n_classes": int(len(model.classes_)),
        "n_neighbors": n_neighbors,
        "weights": weights,
        "metric": metric,
        "predict": model.predict(x).astype(int).tolist(),
        "predict_proba": model.predict_proba(x).tolist(),
        "predict_test": model.predict(x_test).astype(int).tolist(),
        "predict_proba_test": model.predict_proba(x_test).tolist(),
    }


def logistic_case(label, x, y, x_test, C):
    # Converged far past the default tol so the reference is the optimum itself.
    model = LogisticRegression(C=C, solver="lbfgs", tol=1e-12, max_iter=100000).fit(x, y)
    return {
        "op": "logistic",
        "label": label,
        "x": x.tolist(),
        "y": y.astype(int).tolist(),
        "x_test": x_test.tolist(),
        "n_classes": int(len(model.classes_)),
        "C": C,
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "predict": model.predict(x).astype(int).tolist(),
        "predict_proba": model.predict_proba(x).tolist(),
        "predict_test": model.predict(x_test).astype(int).tolist(),
        "predict_proba_test": model.predict_proba(x_test).tolist(),
    }


def svc_case(label, x, y, x_test, **params):
    # tol far below the default so both solvers land on the same optimum.
    model = SVC(tol=1e-9, decision_function_shape="ovo", **params).fit(x, y)
    dec = model.decision_function(x_test)
    if len(model.classes_) == 2:
        dec = -dec  # sklearn flips the binary sign; libsvm's first class is positive
    return {
        "op": "svc",
        "label": label,
        "x": x.tolist(),
        "y": y.astype(int).tolist(),
        "x_test": x_test.tolist(),
        "n_classes": int(len(model.classes_)),
        "params": {k: v for k, v in params.items()},
        "gamma_value": float(model._gamma),
        "n_support": model.n_support_.tolist(),
        "predict": model.predict(x).astype(int).tolist(),
        "predict_test": model.predict(x_test).astype(int).tolist(),
        "decision_test": np.asarray(dec).reshape(len(x_test), -1).tolist(),
    }


def svr_case(label, x, t, x_test, **params):
    model = SVR(tol=1e-9, **params).fit(x, t)
    case = {
        "op": "svr",
        "label": label,
        "x": x.tolist(),
        "t": t.tolist(),
        "x_test": x_test.tolist(),
        "params": {k: v for k, v in params.items()},
        "gamma_value": float(model._gamma),
        "n_support": int(len(model.support_)),
        "intercept": float(model.intercept_[0]),
        "predict": model.predict(x).tolist(),
        "predict_test": model.predict(x_test).tolist(),
        "reference": "scikit-learn",
    }
    if params.get("kernel") == "linear":
        # libsvm can stop short of the optimum here: polish its (w, b) on the
        # primal 0.5 |w|^2 + C sum max(0, |t - w.x - b| - eps) and keep the
        # polished point when it is strictly better.
        C, eps = params.get("C", 1.0), params.get("epsilon", 0.1)

        def primal(v):
            w, b = v[:-1], v[-1]
            return 0.5 * w @ w + C * np.maximum(np.abs(t - x @ w - b) - eps, 0.0).sum()

        start = np.r_[model.coef_[0], model.intercept_[0]]
        polished = minimize(primal, start, method="Nelder-Mead",
                            options={"xatol": 1e-13, "fatol": 1e-15, "maxiter": 200000, "maxfev": 200000})
        if polished.fun < primal(start) - 1e-9:
            w, b = polished.x[:-1], polished.x[-1]
            case["reference"] = (f"scipy Nelder-Mead polish of scikit-learn's solution: primal {polished.fun:.10f} "
                                 f"against libsvm's {primal(start):.10f}")
            case["intercept"] = float(b)
            case["predict"] = (x @ w + b).tolist()
            case["predict_test"] = (x_test @ w + b).tolist()
    return case


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

    # k-nearest neighbours: each metric and weighting; k = 4 makes vote ties
    # (the first class wins, as in sklearn).
    x, y = blobs(rng, [[0, 0, 0], [1.5, 1.0, -0.5], [-1.0, 1.5, 1.0]], 0.8, 20)
    x_test = rng.normal(0.2, 1.3, size=(30, 3))
    cases.append(knn_case("euclidean uniform k=5", x, y, x_test, 5, "uniform", "euclidean"))
    cases.append(knn_case("euclidean distance k=7", x, y, x_test, 7, "distance", "euclidean"))
    cases.append(knn_case("manhattan uniform k=4 (vote ties)", x, y, x_test, 4, "uniform", "manhattan"))
    cases.append(knn_case("cosine distance k=3", x + 3.0, y, x_test + 3.0, 3, "distance", "cosine"))

    # Logistic regression: binary (one coefficient row) and multinomial, mild and
    # strong L2; features on different scales.
    x, y = blobs(rng, [[0, 0, 0], [1.0, 0.6, -0.4]], 0.9, 40)
    x[:, 2] *= 25.0
    cases.append(logistic_case("binary overlapping C=1", x, y, rng.normal(0.4, 1.2, size=(25, 3)) * [1, 1, 25], 1.0))
    x, y = blobs(rng, [[0, 0, 0, 0], [1.4, 0.7, -0.6, 0.2], [-0.8, 1.3, 0.8, -1.1]], 0.8, 30)
    x_test = rng.normal(0.2, 1.2, size=(30, 4))
    cases.append(logistic_case("multinomial three classes C=1", x, y, x_test, 1.0))
    cases.append(logistic_case("multinomial three classes C=0.05", x, y, x_test, 0.05))

    # Support vector machines: each kernel, gamma scale / auto / number,
    # binary and one-vs-one.
    x, y = blobs(rng, [[0, 0], [1.6, 1.0]], 0.8, 30)
    x_test = rng.normal(0.8, 1.2, size=(25, 2))
    cases.append(svc_case("binary rbf gamma=scale C=1", x, y, x_test, kernel="rbf", C=1.0, gamma="scale"))
    cases.append(svc_case("binary linear C=0.5", x, y, x_test, kernel="linear", C=0.5))
    x, y = blobs(rng, [[0, 0, 0], [1.5, 1.0, -0.5], [-1.0, 1.4, 1.0]], 0.7, 20)
    x_test = rng.normal(0.2, 1.2, size=(30, 3))
    cases.append(svc_case("three classes rbf gamma=0.5 C=10", x, y, x_test, kernel="rbf", C=10.0, gamma=0.5))
    cases.append(svc_case("three classes poly degree 3 gamma=auto coef0=1", x, y, x_test,
                          kernel="poly", C=1.0, degree=3, gamma="auto", coef0=1.0))
    cases.append(svc_case("three classes sigmoid gamma=0.1 coef0=0.5", x, y, x_test,
                          kernel="sigmoid", C=1.0, gamma=0.1, coef0=0.5))
    x = rng.uniform(-3, 3, size=(60, 2))
    t = np.sin(x[:, 0]) + 0.3 * x[:, 1] + rng.normal(0, 0.1, 60)
    x_test = rng.uniform(-3, 3, size=(25, 2))
    cases.append(svr_case("svr rbf eps=0.1 C=1", x, t, x_test, kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"))
    cases.append(svr_case("svr linear eps=0.2 C=10", x, t, x_test, kernel="linear", C=10.0, epsilon=0.2))

    out = {"sklearn": sklearn_version, "cases": cases}
    path = Path(__file__).resolve().parent.parent / "fixtures" / "classic_ml_sklearn.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} ({len(cases)} cases, scikit-learn {sklearn_version})")


if __name__ == "__main__":
    main()
