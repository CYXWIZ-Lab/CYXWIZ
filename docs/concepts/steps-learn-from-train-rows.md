---
title: Steps learn from the Train rows
description: Why data steps in a training graph fit on the Train rows only, which steps still learn from every row, and how supplied Validation and Test inputs are prepared.
type: explanation
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-25
related:
  - data-bigger-than-memory.md
  - disk-caches.md
  - ../user-guide/data-studio/split-images-and-audio.md
---

A step that learns from the data (a scaler's mean, an encoder's categories,
a vocabulary) must not see the Validation and Test rows, or their scores
read better than new data would: the rows leak into the model. In a
training graph with a Data Split, the Engine therefore fits each such step
on the Train rows and then applies it to every row, like scikit-learn's
`fit(X_train).transform(X)`.

## How the rows are marked

The pipeline run marks each Data Input row Train, Validation or Test by the
Data Split's rule (a hidden `__partition__` column), before the first step.
Training then reads the same marks, so the rows a step learned from are the
rows the model trains on.

The marks are not data: steps that pick columns themselves skip them,
Unpivot keeps them with each row, and exports (Data Output, Save Dataset,
CSV / Parquet / JSON) leave them out.

Rows are not marked in:

- a graph without a Data Split;
- a time series;
- a Data Split with neither Shuffle nor Stratified and no supplied input
  (the rows keep their order);
- a table kept on disk (a warning says its steps learn from every row).

## Steps that learn from the Train rows

- the scalers and the Quantile / Power transforms;
- Fill Missing (mean, median, most frequent);
- Target Encoder: the Validation and Test labels never reach the encoding;
  a category with no Train row gets the Train mean;
- PCA, Truncated SVD, Whitening and Linear Solve: their directions or x
  come from the Train rows (see
  [Linear algebra](../usage/linear_algebra.md));
- TF-IDF and Count: a word seen only in Validation or Test rows is not a
  feature, and the IDF counts Train documents;
- the Text Tokenizer's vocabulary.

Label and Ordinal Encoder number the categories of every row: no label is
learned. Changing the Split's shares, seed, label or Stratified prepares the
data again.

## Steps that still learn from every row

The clustering steps (K-Means, DBSCAN, Hierarchical, GMM), the Outlier
Detector and the classic models (regression, trees, Naive Bayes, KNN,
Logistic Regression, SVM) cannot learn from the Train rows only. Placed
before a Data Split, they learn from the Validation and Test rows too, and
Compile warns (*CW-C-0206*):

> 'Groups (K-Means)' learns from every row before the Data Split, the
> Validation and Test rows too (it cannot learn from the Train rows only
> yet), so their scores read better than new data would

Training still runs; the Validation and Test scores are then optimistic.
Put such a step after the Data Split, or read its scores with that in mind.

## Supplied Validation and Test inputs

A Data Input on the Split's Validation or Test pin goes through the same
steps. Its rows join the training input's rows (Validation, then Test)
before the first step, so each step learns from the Train rows and
transforms the supplied rows with what it learned; training then reads them
from the prepared table. The log says so: "the Validation input's 48 rows
join 'ds_datainput_1'", "Training reads the supplied Validation rows from
'ds_operator_StandardScaler_5'".

- The supplied rows take the training input's columns by name; extra
  columns are left out, a supplied label column with another name stands
  for the training label, and numbers of different widths are widened
  alike.
- A supplied input without one of the training input's columns stops the
  run ("Validation input has no column 'a' (it needs the training input's
  columns)").
- A step between a supplied input and the Split is refused at compile.
- Its rows are prepared with the training rows, so the check for rows
  shared between Train and the supplied input does not run for it.

Limits: a supplied input kept on disk is read as it is, without the steps (a
warning says so), and so are the supplied inputs when the Training branch
has more than one Data Input (a Join).

## Related

- [Data bigger than memory](data-bigger-than-memory.md)
- [Disk caches](disk-caches.md)
- [Split images and audio](../user-guide/data-studio/split-images-and-audio.md)
