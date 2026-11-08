# Testing a trained model

After training, test the model on the graph's test split:

- **Train > Test > Run Test** (F8) runs the whole test split.
- **Train > Test > Run Quick Test** runs the first 10 batches only. Its
  results say *quick test* and are never taken for a full run.

Image and audio graphs test their Data Split's test part (or the dataset
wired to its Test input; see image_split.md), through the same image steps
as training. Each image's file is kept with its result.

The test part is the one the model trained without: the Data Split it
trained with (its ratios, Stratified, seed and the images a Quality node
left out), kept with the model and in every checkpoint it saves. When the
graph's Data Split changed since, the log says so and the test still reads
the part the model never saw. A checkpoint saved before 2026-10-30 has no
split recorded and is tested with the graph's current one.

The test runs in the background (Task View) and, when it ends, Data Studio
opens **Results > Test** on it. **Train > Test > View Test Results** opens it
again later.

A graph with branches (Split, Add, Concatenate) is tested like any other: the
model it trained in this session is the one tested. It saves and exports as a
CyxWiz package with its graph (see Exporting a model). **Load Checkpoint for
Testing** builds the graph's own model and loads the checkpoint into it, and
training with *save best checkpoint* keeps and restores its best epoch, as
for a layer stack. Feature Importance measures it too (Permutation and
Drop-Column; see feature_importance.md).

**Train > Test > Load Checkpoint for Testing...** opens a folder picker in
the project's `checkpoints` folder, in front of the Engine, listing each
training run (`train-<id>`). Pick a run (its best checkpoint is used) or a
checkpoint folder. The graph on the canvas must compile with its Data Inputs
applied; otherwise the load stops and lists what to fix. Every file and
folder picker in the Engine opens where the Engine asks (the project's folder
for that kind of file, or the project's own folder), not in the folder
Windows last used.

## Results > Test

A test's results are Data Studio's **Results > Test** (the Test Results
window is gone; see [Data Studio > Results](data_studio.md#results)). The
header shows the headline figure (accuracy for classification; 1-NN accuracy
for metric learning; MAE for regression; token accuracy for sequence models),
how many samples were tested and how long it took, and the other figures as
chips (loss, F1, AUC, R², samples per second). If a test fails, it says *The
test failed* with the reason, above the last successful results (a failed run
is never compared against).

**Showing** picks the latest test, a kept test (Keep as node keeps one in the
project's `results/tests`), or any table read as a test. The sections:
Overview, Confusion matrix, Classes, **Errors**, Curves, **Threshold**,
**Calibration**, Predictions, **Plots** (regression), By column value and
Comparison (**Train > Test > Compare Test Results...** opens it). The
Overview of a regression also has its **Bias** (the mean of predicted - true:
positive predicts high).

**Errors** says how many rows are wrong and the commonest mistake ("20 of
the 1,007 test rows are wrong (98.01 % right). The commonest mistake is a
Class 6 read as Class 5 (3 rows)"), lists the mistakes by pair (true ->
predicted, with counts) and shows the most confident ones first, with their
images for an image model. **Explain the first in the Model window** and the
**Explain** link under each open them in Model > Explain (see
[The Model window](model_tools.md)); **Save the wrong rows as a table** keeps
them as a dataset.

**Threshold** picks the decision rule:

- One class against the rest (or a yes / no model): the class is predicted
  when its score reaches the typed **Threshold**. **Best F1**, **Recall at
  least** (the threshold that finds that share of the class) and **Cost of
  each error...** (a false alarm and a miss, in your units: **Use the lowest
  cost**) set it for you. The figures under it (precision, recall,
  specificity, balanced accuracy, accuracy) and the counts follow.
- A model with several classes also gets a **confidence floor**: below it a
  row is left for a person. The curve shows the share of rows answered and
  how many of those are right at every floor.

**Calibration** asks whether the model's confidence can be taken at its
word: bars of the share right in each probability band against the line of
perfect calibration, and the calibration error (ECE). The fit is made on
the even test rows and shown on the odd ones (**Before** / **After**).
Several classes: temperature scaling, which changes no prediction. Yes / no:
Platt (a logistic fit) or isotonic regression. The sweep, the recall target,
the costs, ECE, the temperature, Platt and isotonic are checked against
scikit-learn and scipy (ctest `threshold_calibration_sklearn_parity`).

**Keep 0.500 with the model** (the threshold shown) and **Keep the temperature
with the model** (or Platt's sigmoid, the isotonic fit) keep the decision with the
trained model: the next test applies it and says so ("This test used the
model's kept temperature 0.804"), the checkpoint's `metadata.json` and an
export carry it (`decision`), and the local inference server's
`/v1/predict` returns each row's calibrated `probabilities` and its
`predicted` class by the kept threshold (`/v1/model` shows the decision).
**Forget** drops it. A test that already used a kept calibration cannot fit
another on top: Forget it and test again first.

Class names come from the data: a Text Tokenizer's label keeps the class
names of its label column, so the sentiment demo shows *Anxiety*, *Normal*,
*Depression* and so on instead of class numbers.

- **Export CSV...** and **Export JSON...** save what is shown (the metrics,
  every class with its name, the confusion matrix, each prediction with its
  dataset row, a regression's rows, the tested table's name; JSON also has the
  curves and the quick-test flag). The save dialog starts in the project's
  `exports` folder; a note says where the file went. **Train > Test > Export
  Test Report...** is the same JSON export.
- **Clear** asks first, then forgets the latest test and the one before.
- **Run test again** tests the current graph's trained model again.

To check how far one test can be trusted, **Train > Evaluate >
Cross-validate** trains the graph once per fold and scores each fold once;
**Learning Curve** shows whether more rows would help (Data Studio > Results).

See [export_import.md](export_import.md) to save the trained model itself.
