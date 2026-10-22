"""Truth for Data Studio > Results > Test (TOFIX129 D4): scikit-learn's figures on
penguins predictions, for test_test_results_presentation.

A small MLP is trained on the Palmer penguins table so the rows look like a
real model's output. Writes evaluation_sklearn.json beside this file:
  binary     - sex (labels FEMALE / MALE, one score = P(MALE)): AUC for each
               positive class, the 2 x 2 counts and figures at 0.3 / 0.5 / 0.7,
               and the best-F1 threshold (the highest score on a tie);
  multiclass - species (text labels, a score column per class): accuracy,
               macro / weighted F1, per-class AUC;
  regression - body mass from the other measurements: MAE, RMSE, R-squared.

Run with the oracle venv (numpy, pandas, scikit-learn):
  python generate_evaluation_fixtures.py path/to/penguins.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn import metrics as M
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(sys.argv[1])
feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
d = df.dropna(subset=feats + ["species", "sex"]).reset_index(drop=True)
out = {}

# Binary: sex.
X = d[feats].values
y = d["sex"].str.upper().values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
m = make_pipeline(StandardScaler(), MLPClassifier((16,), max_iter=400, random_state=0)).fit(Xtr, ytr)
p_male = m.predict_proba(Xte)[:, list(m.classes_).index("MALE")]
pred = m.predict(Xte)
male = (yte == "MALE").astype(int)
at = []
for t in [0.3, 0.5, 0.7]:
    p = (p_male >= t).astype(int)
    tn, fp, fn, tp = M.confusion_matrix(male, p, labels=[0, 1]).ravel()
    at.append(dict(threshold=t, tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                   precision=M.precision_score(male, p, zero_division=0), recall=M.recall_score(male, p),
                   specificity=tn / (tn + fp), f1=M.f1_score(male, p), balanced=M.balanced_accuracy_score(male, p),
                   mcc=M.matthews_corrcoef(male, p)))
best_t, best_f1 = None, -1.0
for t in sorted(np.unique(p_male), reverse=True):
    f = M.f1_score(male, (p_male >= t).astype(int))
    if f > best_f1:
        best_t, best_f1 = float(t), f
out["binary"] = dict(truth=list(yte), predicted=list(pred), score_male=[float(v) for v in p_male],
                     auc_male=M.roc_auc_score(male, p_male), auc_female=M.roc_auc_score(1 - male, 1 - p_male),
                     ap_male=M.average_precision_score(male, p_male), at=at, best_threshold=best_t, best_f1=best_f1,
                     accuracy=M.accuracy_score(yte, pred))

# Three classes: species from bill length and body mass (a few mistakes).
X = d[["bill_length_mm", "body_mass_g"]].values
y = d["species"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
m = make_pipeline(StandardScaler(), MLPClassifier((8,), max_iter=1500, random_state=0)).fit(Xtr, ytr)
proba = m.predict_proba(Xte)
pred = m.predict(Xte)
classes = list(m.classes_)
out["multiclass"] = dict(classes=classes, truth=list(yte), predicted=list(pred),
                         scores={c: [float(v) for v in proba[:, i]] for i, c in enumerate(classes)},
                         accuracy=M.accuracy_score(yte, pred), macro_f1=M.f1_score(yte, pred, average="macro"),
                         weighted_f1=M.f1_score(yte, pred, average="weighted"),
                         auc={c: M.roc_auc_score((yte == c).astype(int), proba[:, i]) for i, c in enumerate(classes)})

# Regression: body mass.
X = d[feats[:3]].values
y = d["body_mass_g"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), max_iter=3000, random_state=0,
                                                 learning_rate_init=0.01)).fit(Xtr, ytr)
pr = m.predict(Xte)
out["regression"] = dict(truth=[float(v) for v in yte], predicted=[float(v) for v in pr],
                         mae=M.mean_absolute_error(yte, pr), rmse=M.root_mean_squared_error(yte, pr), r2=M.r2_score(yte, pr))

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_sklearn.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1, default=float)
print("wrote", path, "binary auc %.4f, multiclass acc %.4f, r2 %.4f" % (out["binary"]["auc_male"], out["multiclass"]["accuracy"],
                                                                       out["regression"]["r2"]))
