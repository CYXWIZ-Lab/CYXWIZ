"""Truth for Results > Test > Threshold and Calibration (TOFIX129 D11), for
test_threshold_calibration.

Reads the penguins predictions of tests/fixtures/evaluation_sklearn.json
(binary: sex, P(MALE); multiclass: species, a probability per class) and
writes what scikit-learn 1.8 / scipy 1.17 / numpy 2.3 say:

  sweep        for every distinct P(MALE), the precision / recall / F1 when
               MALE is predicted at score >= it (sklearn.metrics).
  recall_0_9   the highest such threshold with recall >= 0.9.
  cost_1_5     the threshold with the lowest 1 x FP + 5 x FN (lowest score on
               a tie).
  floor        several classes: at floors 0.5 / 0.8 / 0.95, rows whose top
               probability reaches the floor, and their accuracy.
  reliability  10 equal bins of the top probability (several classes) and of
               P(MALE) against "is MALE" (yes / no): counts, mean confidence,
               accuracy, and ECE.
  temperature  fitted on the even rows: T minimizing the mean NLL of
               softmax(log p / T) (scipy minimize_scalar, bounded, on log T);
               the check rows' ECE before and after.
  platt        sklearn.calibration._sigmoid_calibration on the even rows'
               P(MALE): a, b; the odd rows' calibrated values.
  isotonic     IsotonicRegression(out_of_bounds="clip") on the even rows; the
               odd rows' calibrated values.

Run with Python 3.14 (scikit-learn 1.8, scipy 1.17, numpy 2.3):
  python generate_threshold_calibration_fixtures.py
Writes ../fixtures/threshold_calibration_sklearn.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.calibration import _sigmoid_calibration
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score

here = Path(__file__).resolve().parent
src = json.loads((here.parent.parent / "fixtures" / "evaluation_sklearn.json").read_text(encoding="utf-8"))
out = {"source": "tests/fixtures/evaluation_sklearn.json"}

# ---- Binary: sex, P(MALE) -----------------------------------------------------
b = src["binary"]
y = np.array([1 if t == "MALE" else 0 for t in b["truth"]])
s = np.array(b["score_male"], dtype=np.float64)
sweep = []
for t in np.unique(s):
    pred = (s >= t).astype(int)
    sweep.append({"threshold": float(t), "precision": float(precision_score(y, pred, zero_division=0)),
                  "recall": float(recall_score(y, pred)), "f1": float(f1_score(y, pred, zero_division=0))})
out["sweep"] = sweep
ok = [p["threshold"] for p in sweep if p["recall"] >= 0.9]
out["recall_0_9"] = max(ok) if ok else None
costs = []
for p in sweep:
    pred = (s >= p["threshold"]).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    costs.append((1 * fp + 5 * fn, p["threshold"]))
best_cost = min(c for c, _ in costs)
out["cost_1_5"] = min(t for c, t in costs if c == best_cost)


def reliability(conf, correct, bins=10):
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for k in range(bins):
        m = idx == k
        n = int(m.sum())
        c = float(conf[m].mean()) if n else 0.0
        a = float(correct[m].mean()) if n else 0.0
        rows.append({"count": n, "confidence": c, "accuracy": a})
        ece += n / len(conf) * abs(a - c)
    return {"bins": rows, "ece": ece}


fit, check = np.arange(len(y))[0::2], np.arange(len(y))[1::2]
a, b0 = _sigmoid_calibration(s[fit], y[fit])
platt = 1.0 / (1.0 + np.exp(a * s[check] + b0))
iso = IsotonicRegression(out_of_bounds="clip").fit(s[fit], y[fit]).predict(s[check])
out["binary_reliability_all"] = reliability(s, y)
out["platt"] = {"a": float(a), "b": float(b0), "check": platt.tolist(),
                "ece_before": reliability(s[check], y[check])["ece"], "ece_after": reliability(platt, y[check])["ece"]}
out["isotonic"] = {"check": iso.tolist(), "ece_after": reliability(iso, y[check])["ece"]}

# ---- Several classes: species -------------------------------------------------
mc = src["multiclass"]
names = sorted(mc["scores"].keys())
P = np.column_stack([np.array(mc["scores"][n], dtype=np.float64) for n in names])
truth = np.array([names.index(t) for t in mc["truth"]])
top = P.max(axis=1)
pred = P.argmax(axis=1)
out["classes"] = names
out["multiclass_probabilities"] = P.tolist()
out["multiclass_truth"] = truth.tolist()
out["floor"] = [{"floor": f, "answered": int((top >= f).sum()),
                 "accuracy": float((pred[top >= f] == truth[top >= f]).mean()) if (top >= f).any() else 0.0}
                for f in (0.5, 0.8, 0.95)]
out["multiclass_reliability_all"] = reliability(top, (pred == truth).astype(float))
fit, check = np.arange(len(truth))[0::2], np.arange(len(truth))[1::2]
logp = np.log(np.clip(P, 1e-12, None))


def nll(log_t, rows):
    z = logp[rows] / np.exp(log_t)
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(rows)), truth[rows]]))


r = minimize_scalar(lambda u: nll(u, fit), bounds=(np.log(0.05), np.log(20.0)), method="bounded",
                    options={"xatol": 1e-10})
T = float(np.exp(r.x))
z = logp[check] / T
q = np.exp(z - z.max(axis=1, keepdims=True))
q = q / q.sum(axis=1, keepdims=True)
right = (pred[check] == truth[check]).astype(float)
out["temperature"] = {"T": T, "nll_before": nll(0.0, fit), "nll_after": nll(r.x, fit),
                      "ece_before": reliability(top[check], right)["ece"], "ece_after": reliability(q.max(axis=1), right)["ece"]}

path = here.parent / "fixtures" / "threshold_calibration_sklearn.json"
path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print("T", T, "platt", a, b0, "recall>=0.9 at", out["recall_0_9"], "cost at", out["cost_1_5"])
