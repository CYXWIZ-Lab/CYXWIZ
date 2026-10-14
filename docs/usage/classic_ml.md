# Classic ML models

The Naive Bayes, KNN Classifier, Logistic Regression, SVM Classifier and SVM
Regressor nodes (Analytics) fit a classical model on a table and compute on the device selected in the Devices
tab (CUDA, OpenCL, or ArrayFire's CPU backend). There is no separate CPU code
path. Results are float64 where the device supports it, otherwise float32.

Each model is checked against scikit-learn 1.8: same parameters, same
predictions and probabilities (within 1e-8 in float64; 1e-6 for the
iteratively solved logistic coefficients). On a GPU without float64 (for example
an Intel Iris Xe, OpenCL or oneAPI) the same checks pass at float32 precision:
5e-5 relative for direct results, 1e-3 for solved ones, with the same labels.

## Using a node

1. Connect a Data Input (or any table step) to the node's **Data** pin.
2. Set **Target column** to the label column. Labels may be numbers, text or
   true/false.
3. Leave **Feature columns** empty to use every numeric column except the
   target, or list them, comma-separated.
4. Run the graph. **Predictions** is the input table plus a prediction column
   (named by **Prediction column**, default `prediction`), holding the label
   itself (`"apple"`, not a class number).
5. **Model** is the fitted model, saved as JSON. With **Save model path**
   empty it goes to the project's artifacts
   (`classic_model/<run>/<node>_<name>.cyxclassic.json`).

Classes are numbered in sorted order (numbers by value, text alphabetically),
as scikit-learn's `classes_`, so ties and probability columns agree with it.

The Plot node cannot draw these models (they have no trees); plot the
Predictions output instead. To predict on other rows with a fitted model, see
[Predicting with a fitted model](#predicting-with-a-fitted-model).

## Models

| Node | Behaviour | Parameters | Same as |
| --- | --- | --- | --- |
| Naive Bayes | Per-class means and population variances; every variance plus var_smoothing × the largest feature variance; priors from class counts. Predicts the class with the highest joint log-likelihood. | `var_smoothing` (default 1e-9) | scikit-learn `GaussianNB` |
| Logistic Regression | Minimises the mean log loss plus ‖coef‖² / (2·C·n) with L-BFGS (intercepts not penalised). Two classes fit one coefficient row (sigmoid); more classes a softmax over all of them. Converged when the largest gradient entry is at most `tol` or a step stops lowering the loss; if `max_iter` comes first the fit stands and the log warns, as scikit-learn's ConvergenceWarning. | `C` (1.0), `max_iter` (100), `tol` (1e-4) | scikit-learn `LogisticRegression(solver="lbfgs")` |
| SVM Classifier | Kernel support vector machine: the dual solved by SMO as libsvm (second-order working set selection), on the device without holding an n × n kernel matrix. More than two classes fit one model per pair of classes; each pair votes and the first class with the most votes wins. With **Probability columns** on, the probabilities are calibrated as scikit-learn's `probability=True` (see below). | `C` (1.0), `kernel` (rbf, linear, poly, sigmoid), `gamma` (scale, auto or a positive number), `degree` (3), `coef0` (0.0), `tol` (1e-3), `random_state` (0) | scikit-learn `SVC` |
| SVM Regressor | Epsilon-insensitive support vector regression: errors inside the ±epsilon tube cost nothing, larger ones cost C per unit. The same device SMO solver as the classifier, on libsvm's two-variables-per-row problem. The target must be numeric; the prediction column is a number. | `C` (1.0), `epsilon` (0.1), and the classifier's kernel settings | scikit-learn `SVR` |
| KNN Classifier | Distances from each row to every training row, the n_neighbors nearest vote. `distance` weighting counts each vote as 1/distance; a training row at distance 0 takes all the weight. A vote tie goes to the first class in sorted order. | `n_neighbors` (5), `weights` (uniform, distance), `metric` (euclidean, manhattan, cosine) | scikit-learn `KNeighborsClassifier(algorithm="brute")` |

## Probability columns

Turn on **Probability columns** (Output) on Naive Bayes, KNN, Logistic
Regression, the SVM Classifier or the Classic Model Predictor to append one
column per class, `<prediction column>_p_<class>` (for example
`prediction_p_apple`), in class order; each row sums to 1. The ROC Curve and
Precision-Recall nodes take one of them as their score column.

They are scikit-learn's `predict_proba`. For the SVM Classifier this is
`SVC(probability=True)`: per pair of classes, a sigmoid (Platt scaling) is
fitted to decision values from a 5-fold cross-validation, and the pairs are
combined into class probabilities. The folds are drawn from **Random state**
exactly as scikit-learn draws them, so the same random_state gives the same
probabilities as scikit-learn. This fits about five more SVMs per pair of
classes, so it is off unless the column is wanted. The predicted label stays the
one-vs-one vote (scikit-learn's `predict`), so for a few rows near a boundary the
label and the most probable class can differ, as in scikit-learn.

A model saved with probability sigmoids (`prob_a`, `prob_b`) gives the
predictor its probability columns; an SVM Classifier saved without them is
refused with the reason, and the SVM Regressor has none.

## Predicting with a fitted model

The **Classic Model Predictor** node (Analytics) applies any of these models to
another table, for example a test file or new rows that have no label yet.

1. Connect the new table to its **Data** pin. It needs the columns the model was
   fitted on, matched by name (their order does not matter; the label column is
   not needed).
2. Give it the model, one of two ways:
   - connect a classic node's **Model** output to its **Model** pin, to use the
     model fitted in this run; or
   - leave the Model pin empty and set **Saved model** to a `.cyxclassic.json`
     file (Browse), to use a model saved by an earlier run. A path relative to
     the project resolves against the project folder.
   When both are set, the Model pin wins.
3. Run the graph. **Predictions** is the input table plus the prediction column,
   exactly as the fitting node writes it: the same class labels (text or
   numbers), or a number for the SVM Regressor.

A missing feature column stops the run and names the column; so does a file
that is not a saved classic model.

## Size

KNN keeps its training rows: the saved model holds them, and prediction
computes distances in row blocks, so memory stays bounded but time grows with
rows × training rows.

The SVM solver computes two kernel rows per iteration instead of holding the
kernel matrix, so memory stays O(rows); time grows faster than linearly with
rows, so it suits up to tens of thousands of rows.

## The saved model

```json
{"format": "cyxwiz_classic_model", "version": 1, "model_type": "NaiveBayesClassifier",
 "model": {"feature_names": [...], "class_labels": [...], "numeric_labels": false,
           "parameters": {...}}}
```

`parameters` holds the learned values: `theta`, `var`, `class_prior`,
`class_count`, `epsilon` for Naive Bayes; `x`, `labels` and the settings for
KNN; `coef`, `intercept`, `n_iter`, `converged` and the settings for Logistic
Regression; `support_vectors`, per class pair `pair_first` / `pair_second`,
`pair_support` (rows of `support_vectors`), `pair_coef` and `rho`, the resolved
`gamma` and the settings for the SVM Classifier, plus `prob_a`, `prob_b` and
`random_state` when it was fitted with probability columns (P(pair_first) =
1 / (1 + exp(prob_a · decision + prob_b))). A pair's decision is
Σ coef · K(support vector, row) − rho; positive votes for `pair_first`. The SVM
Regressor saves the same fields for its one problem plus `epsilon`, its
`target` column and no class labels; its prediction is
Σ coef · K(support vector, row) − rho.

Logistic regression is sensitive to feature scale: a column in the thousands
next to one in single digits slows L-BFGS down. Put a Standard Scaler before it
when the log reports that `max_iter` was reached.
