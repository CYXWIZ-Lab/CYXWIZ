---
redirect_from:
  - ../../../usage/classic_ml.md
related:
  - ../../../user-guide/evaluation/cross-validation-and-learning-curves.md
  - ../../../user-guide/evaluation/threshold-and-calibration.md
  - ../../../concepts/numerical-accuracy.md
---

## Classic models

[Naive Bayes](../analytics/naive-bayes.md),
[KNN Classifier](../analytics/knn-classifier.md),
[Logistic Regression](../analytics/logistic-regression.md),
[SVM Classifier](../analytics/svm-classifier.md) and
[SVM Regressor](../analytics/svm-regressor.md) fit a classical model on a
table. They compute on the device selected in the Devices tab (CUDA, OpenCL,
or ArrayFire's CPU backend); there is no separate CPU code path. Results are
float64 where the device supports it, otherwise float32.

Each model gives the same predictions and probabilities as scikit-learn with
the same parameters (within 1e-8 in float64, 1e-6 for the iteratively solved
logistic coefficients). On a GPU without float64, such as an Intel Iris Xe,
the same checks pass at float32 precision with the same labels.

### Fit a model

1. Connect a Data Input (or any table step) to the node's **Data** pin.
2. Set **Target column** to the label column. Labels may be numbers, text or
   true/false.
3. Leave **Feature columns** empty to use every numeric column except the
   target, or list them, comma-separated.
4. Run the graph. **Predictions** is the input table plus a prediction
   column holding the label itself (`"apple"`, not a class number).
5. **Model** is the fitted model, saved as JSON. With **Save model path**
   empty it goes to the project's artifacts
   (`classic_model/<run>/<node>_<name>.cyxclassic.json`).

Classes are numbered in sorted order (numbers by value, text
alphabetically), as scikit-learn's `classes_`, so ties and probability
columns agree with it.

The Plot node cannot draw these models (they have no trees); plot the
**Predictions** output instead.

### Which model

| Node | How it decides | Same as |
| --- | --- | --- |
| Naive Bayes | Per-class means and variances (each variance plus `var_smoothing` times the largest feature variance), priors from class counts; the class with the highest joint log-likelihood wins | scikit-learn `GaussianNB` |
| Logistic Regression | Mean log loss plus ‖coef‖² / (2·C·n), minimised with L-BFGS (intercepts not penalised). Two classes fit one sigmoid row, more a softmax. When `max_iter` comes first the fit stands and the log warns, as scikit-learn's ConvergenceWarning | scikit-learn `LogisticRegression(solver="lbfgs")` |
| SVM Classifier | Kernel SVM: the dual solved by SMO as libsvm, on the device, without holding an n × n kernel matrix. More than two classes fit one model per pair of classes; the first class with the most votes wins | scikit-learn `SVC` |
| SVM Regressor | Epsilon-insensitive regression: errors inside the ±epsilon tube cost nothing, larger ones cost C per unit. Numeric target only | scikit-learn `SVR` |
| KNN Classifier | The `n_neighbors` nearest training rows vote; `distance` weighting counts each vote as 1/distance, and a training row at distance 0 takes all the weight. A vote tie goes to the first class in sorted order | scikit-learn `KNeighborsClassifier(algorithm="brute")` |

Logistic regression is sensitive to feature scale: a column in the
thousands next to one in single digits slows L-BFGS down. Put a
[Standard Scaler](../preprocessing/standard-scaler.md) before it when the log
reports that `max_iter` was reached.

### Probability columns

Turn on **Probability columns** on Naive Bayes, KNN, Logistic Regression,
the SVM Classifier or the Classic Model Predictor to append one column per
class, `<prediction column>_p_<class>` (for example `prediction_p_apple`),
in class order; each row sums to 1. The
[ROC Curve](../analytics/roc-curve.md) and
[Precision-Recall Curve](../analytics/precision-recall-curve.md) nodes take
one of them as their score column.

They are scikit-learn's `predict_proba`. For the SVM Classifier this is
`SVC(probability=True)`: per pair of classes, a sigmoid (Platt scaling) is
fitted to decision values from a 5-fold cross-validation drawn from
**Random state** as scikit-learn draws it, so the same `random_state` gives
the same probabilities. This fits about five more SVMs per pair of classes,
so leave it off unless you need the column. The predicted label stays the
one-vs-one vote, so for a few rows near a boundary the label and the most
probable class can differ, as in scikit-learn.

### Predict with a fitted model

The [Classic Model Predictor](../analytics/classic-model-predictor.md)
applies any of these models to another table, for example a test file or
new rows without a label.

1. Connect the new table to its **Data** pin. It needs the columns the model
   was fitted on, matched by name (order does not matter; the label column
   is not needed).
2. Give it the model: connect a classic node's **Model** output to its
   **Model** pin to use the model fitted in this run, or leave the pin empty
   and set **Saved model** to a `.cyxclassic.json` file from an earlier run.
   A relative path resolves against the project folder. When both are set,
   the pin wins.
3. Run the graph. **Predictions** is the input table plus the prediction
   column, exactly as the fitting node writes it.

A missing feature column stops the run and names the column; so does a file
that is not a saved classic model. An SVM Classifier saved without
probability sigmoids gives no probability columns and says so; the SVM
Regressor has none.

### Size

KNN keeps its training rows in the saved model and computes distances in row
blocks, so memory stays bounded but time grows with rows × training rows.
The SVM solver computes two kernel rows per iteration, so memory stays
O(rows); time grows faster than linearly with rows, so it suits up to tens of
thousands of rows.

### The saved model

```json
{"format": "cyxwiz_classic_model", "version": 1, "model_type": "NaiveBayesClassifier",
 "model": {"feature_names": [], "class_labels": [], "numeric_labels": false,
           "parameters": {}}}
```

`parameters` holds the learned values: `theta`, `var`, `class_prior`,
`class_count` and `epsilon` for Naive Bayes; `x`, `labels` and the settings
for KNN; `coef`, `intercept`, `n_iter`, `converged` and the settings for
Logistic Regression; `support_vectors`, per class pair `pair_first` /
`pair_second`, `pair_support`, `pair_coef` and `rho`, the resolved `gamma`
and the settings for the SVM Classifier, plus `prob_a`, `prob_b` and
`random_state` when it was fitted with probability columns
(P(pair_first) = 1 / (1 + exp(prob_a · decision + prob_b))). A pair's
decision is Σ coef · K(support vector, row) − rho; positive votes for
`pair_first`. The SVM Regressor saves the same fields for its one problem
plus `epsilon` and its `target` column, and predicts
Σ coef · K(support vector, row) − rho.
