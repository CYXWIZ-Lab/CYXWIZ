# Classic ML classifiers

The Naive Bayes, KNN Classifier and Logistic Regression nodes (Analytics) fit
a classical model on a table and compute on the device selected in the Devices
tab (CUDA, OpenCL, or ArrayFire's CPU backend). There is no separate CPU code
path. Results are float64 where the device supports it, otherwise float32.

Each model is checked against scikit-learn 1.8: same parameters, same
predictions and probabilities (within 1e-8 in float64; 1e-6 for the
iteratively solved logistic coefficients).

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
Predictions output instead.

## Models

| Node | Behaviour | Parameters | Same as |
| --- | --- | --- | --- |
| Naive Bayes | Per-class means and population variances; every variance plus var_smoothing × the largest feature variance; priors from class counts. Predicts the class with the highest joint log-likelihood. | `var_smoothing` (default 1e-9) | scikit-learn `GaussianNB` |
| Logistic Regression | Minimises the mean log loss plus ‖coef‖² / (2·C·n) with L-BFGS (intercepts not penalised). Two classes fit one coefficient row (sigmoid); more classes a softmax over all of them. Converged when the largest gradient entry is at most `tol` or a step stops lowering the loss; if `max_iter` comes first the fit stands and the log warns, as scikit-learn's ConvergenceWarning. | `C` (1.0), `max_iter` (100), `tol` (1e-4) | scikit-learn `LogisticRegression(solver="lbfgs")` |
| KNN Classifier | Distances from each row to every training row, the n_neighbors nearest vote. `distance` weighting counts each vote as 1/distance; a training row at distance 0 takes all the weight. A vote tie goes to the first class in sorted order. | `n_neighbors` (5), `weights` (uniform, distance), `metric` (euclidean, manhattan, cosine) | scikit-learn `KNeighborsClassifier(algorithm="brute")` |

## Size

KNN keeps its training rows: the saved model holds them, and prediction
computes distances in row blocks, so memory stays bounded but time grows with
rows × training rows.

## The saved model

```json
{"format": "cyxwiz_classic_model", "version": 1, "model_type": "NaiveBayesClassifier",
 "model": {"feature_names": [...], "class_labels": [...], "numeric_labels": false,
           "parameters": {...}}}
```

`parameters` holds the learned values: `theta`, `var`, `class_prior`,
`class_count`, `epsilon` for Naive Bayes; `x`, `labels` and the settings for
KNN; `coef`, `intercept`, `n_iter`, `converged` and the settings for Logistic
Regression.

Logistic regression is sensitive to feature scale: a column in the thousands
next to one in single digits slows L-BFGS down. Put a Standard Scaler before it
when the log reports that `max_iter` was reached.
