# Hyperparameter Search

**Train > Hyperparameter Search** trains the graph on the canvas several
times with different values for the properties you pick, and keeps the
values that scored best. Each trial trains the way **Train** does: the same
prepared data and Data Split, a fresh model, its own scratch checkpoints
(removed after). The model in use and the Training Dashboard's runs stay as
they are.

## Setup

**What to vary.** Click **Add from the graph**: the list comes from the
compiled graph (compile it first) and shows, per node, the properties a
trial can change:

| Node | Properties |
| --- | --- |
| Layers | units, filters, out channels, hidden size, number of layers, embedding size, heads, dropout rate |
| The optimizer | learning rate, weight decay, beta1 / beta2, momentum, warm-up and minimum learning-rate ratios |
| The scheduler | its steps, rates and periods |
| DataLoader | batch size, epochs |

The last layer (it sets the outputs the loss expects, such as the 10
classes of MNIST) is listed but not searched.

A picked property starts with a range around its value on the graph; change
it in the **Range** column:

| Values | Starting range | Example |
| --- | --- | --- |
| Log scale (rates) | a tenth to ten times the value | learning rate 0.001: 0.0001 - 0.01 |
| Whole number (counts) | a quarter to twice, in quarter steps | units 128: 32 - 256, step 32 |
| Choices (batch size) | half, the same, double | 64: 32, 64, 128 |
| Decimal (fractions) | around the value | dropout 0.2: 0.1 - 0.4 |

**How to search.**

- **Grid** tries every combination (a range is cut into 10 steps): best for
  one or two properties with few values.
- **Random** draws values within the ranges: a good start for many
  properties.
- **Bayesian** runs a few random trials, then puts each next trial where
  the results so far say the score should improve most (a Gaussian
  process). It finds good values in fewer trials.

**Trials** (Random and Bayesian), **Stop after** (hours; the best so far is
kept) and **Score by** (validation loss, validation accuracy, or validation
MAE for numbers).

These four start from your defaults in **Preferences > Search** (out of
the box: Random, 20 trials, 1 hour, validation loss). They
are saved with your Engine settings, apply to every project, and the panel
takes them again when you change them there (not while a search runs).
**Reset** puts the Engine defaults back.

Not listed above, the DataLoader's own settings apply to every trial: its
epochs, batch size and early stopping. The trials split the data as the
graph's Data Split does.

**Start search** goes through Train's checks and data preparation first
(the data is loaded and prepared once for all the trials). If a check
stops it, the reason shows under the button and the compile report opens.
Train is not available while a search runs, and a search does not start
while Train, a cross-validation or a learning curve runs.

## Trials

- The status line: the trial and epoch running, the time per trial and
  the time left, and **Stop**.
- **Best so far**: its scores and values. **Apply best to the graph** sets
  those values on the canvas (Ctrl+Z takes them back); train the graph
  afterwards to get the model. **Export CSV** saves a row per trial.
- The score per trial as bars (the best one light, a failed one short and
  red), and a row per trial: its values, validation loss, accuracy (or MAE),
  time and status. Hover a failed trial for its reason. A failed trial does
  not stop the search; three failures in a row do.

## Example

MNIST (`examples/cyxgraph/mnist_mlp.cyxgraph` with
`mnist_test.csv`, 10,000 rows): vary **Adam learning rate**
(0.0001 - 0.01), Random, 3 trials. Each trial took about 6 s; the best
(learning rate 0.0087) reached validation loss 0.309 and 94.4 % accuracy,
against 87.6 % at 0.0001.
