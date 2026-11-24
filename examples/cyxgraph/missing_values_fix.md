# Missing values: fix them in one click

`missing_values_fix.cyxgraph` is a small classifier on a table with gaps.
Training never uses an empty cell, or NaN in a number column, as 0 or as
class 0: it stops before the run and offers the fix.

```
Data Input (patients_missing.csv) -> Split 80/10/10 -> Loader (batch 32)
  -> Dense 16 -> ReLU -> Dense 3 -> Cross Entropy -> Adam -> Output
```

## Data

`examples/datasets/patients_missing.csv`: 300 made-up rows (no real
patients; `make_patients_missing.py` builds it) with five features (`age`,
`bmi`, `blood_pressure`, `cholesterol`, `smoker`) and a risk class
`diagnosis` 0, 1 or 2. The gaps: 18 empty `age` cells, 9 empty `cholesterol`
cells, 4 `NaN` in `bmi` (the CSV reader may read them as empty), and 5 rows
without a `diagnosis`.

## Run it

1. Open the graph (**File > Open**, or the Asset Browser), then **Train**
   (F5).
2. *Cannot start training* says "Training did not start: 4 columns have
   missing values", with a card for the feature columns and one for the
   label.
3. Either fix each group (**Fill numbers with** Mean / Median / 0 and **Fill
   these columns**, or **Drop these rows**; **Drop rows without a label**),
   then **Train**, or click **Fix and train**: it fills the features with the
   mean of the Train rows, drops the rows without a label and trains.
4. The canvas now has the steps between **Patients CSV** and the Split, in
   the card's order ("Fill age, bmi, cholesterol (mean)", then "Keep rows with
   a diagnosis"). **Undo** in the report removes a group's step; Ctrl+Z on
   the canvas removes the last one added, and the card follows.

More in [the compile report](../../docs/user-guide/graph-editor/fix-missing-values.md).
