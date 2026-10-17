# Feature Importance

**Tools > Evaluation > Feature Importance** measures how much each input
column matters to the active model: the model you trained in this session,
or a checkpoint loaded with **Load checkpoint...** (the same model Test and
Inference use). Any model works, a layer stack or a graph with branches
(Split, Add, Concatenate).

## Model

The card shows the active model and where it came from ("Trained this
session - 9 layers"). With no model it says *Nothing to measure yet*; train
the graph or load a checkpoint. A result belongs to the model it was computed
for: training again or loading another checkpoint clears it.

## Data

- **Dataset**: a loaded table (a Data Input you applied), for example
  *Metric blobs - Data Input - 480 rows - 9 cols*.
- **Features**: the columns the model reads, all selected at first ("8
  selected").
- **Target column**: the label, suggested from the training run ("class").

## Method

| Method | What it does | Models |
| --- | --- | --- |
| Permutation Importance | Shuffles one column at a time (**Repeats** times, 10 by default) and measures how much the score drops. | Any |
| Drop-Column Importance | Sets one column to zero at a time and measures the drop. | Any |
| Weight-Based | The mean absolute weight of each input in the first layer. | Layer stacks only |

**Scoring** (Permutation and Drop-Column): *Accuracy* for classification,
*MSE* for regression. For a graph with branches, Weight-Based stays off
with the reason on hover: its inputs feed several first layers, so there is
no single first layer to read.

**Compute importance** runs in the background. The result shows the method
and the baseline score, then a **Chart** (one bar per feature, largest on
top, scaled 0 to 1) and a **Ranking** table. **Export** saves it.

Example (the 17-node branching classifier `blobs_multibranch_classifier`,
trained 5 epochs on `metric_blobs.csv`): Permutation Importance, 10 repeats,
accuracy. Baseline 0.9771; ranking f3, f6, f1, f2, f5, f0, f4, f7.
