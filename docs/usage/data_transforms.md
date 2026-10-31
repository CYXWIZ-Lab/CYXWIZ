# Scaling and transforms

Scaling and shape changes are steps of the graph: nodes in Data Studio's
**Prepare** job, whose output the pipeline (and training) uses.
**Data > Transform** adds them:

| Data > Transform | Adds |
| --- | --- |
| Normalization (Min-Max) | a Min-Max Scaler step |
| Standardization (Z-Score) | a Standard Scaler (Z-score) step |
| Log Transform | a Log Transform step |
| Box-Cox Transform | a Power Transform step, family Box-Cox |
| Feature Scaling (All Methods) | a Z-score step, with Compare methods open on it |

The step goes after the data in use (the Data Input or node result picked in
Data Studio's rail; the graph's Data Input when it has only one) and is picked
in Prepare. Prepare's **Add a step** has the same steps under **Scale and
transform**. Ctrl+Z on the canvas undoes an added step.

All of them compute on the device selected in the Devices tab (CUDA, OpenCL, or
ArrayFire's CPU backend); there is no separate CPU code path. Results are
float64 where the device supports it, otherwise float32.

## The nodes

| Node | Result | Same as |
| --- | --- | --- |
| Standard Scaler (Z-score) | `(x - mean) / std`, std divides by n; a constant column gives 0 | scikit-learn `StandardScaler` |
| Min-Max Scaler | `(x - min) / (max - min)` into the typed range (default 0 .. 1; the range must rise) | scikit-learn `MinMaxScaler` |
| Robust Scaler | `(x - median) / (q_max - q_min)`, default quartiles 25 / 75; a zero spread scales by 1 | scikit-learn `RobustScaler` |
| Max-Abs Scaler | `x / max(|x|)`, into -1 .. 1; zeros stay zero (sparse data) | scikit-learn `MaxAbsScaler` |
| Quantile Transform | each value through min(n quantiles, rows) quantiles onto uniform 0 .. 1 or a standard normal | scikit-learn `QuantileTransformer` fitted on the whole column |
| Power Transform | Box-Cox `(x^λ - 1) / λ` (log x at λ = 0; values above 0) or Yeo-Johnson (any value); λ found per column by maximum likelihood, or typed | `scipy.stats.boxcox` / `yeojohnson` |
| Log Transform | ln, log10 or log2 of x, or of 1 + x (**log(1 + x)**, on by default) | |

Every node takes **Columns** (empty = every number column except the label
column and the excluded columns). A missing value stays missing in its row; the
fit uses the values present, as scikit-learn does.

**Fitting.** The scalers, Quantile and Power fit values on their input (a mean,
a range, quantiles, a λ). **Fit here** fits on the rows that reach the step
each run; **Save the fitted values** writes them to a `.cyxstate.json` file.
**Use saved values** reads such a file instead, so validation, test and
inference data are transformed with the training data's values; it is refused
when the file was fitted with other settings or columns. A fitting step is
refused when it reads a validation, test or inference Data Input without saved
values (it would learn from data it should not see). Fitted steps sit outside a
Preparation Recipe. Log Transform fits nothing.

## The step in Prepare

Picking one of these steps in Prepare shows:

- its settings as typed fields and switches: **Method** (Z-score, Min-Max,
  Robust, Max-Abs, Quantile: switching replaces the node with the other
  scaler, keeping its links and columns), **Range**, **Shape** and
  **Quantiles**, **Family** and **Lambda** (*Find the best* or *Typed*, with
  -1, -0.5, 0, 0.5, 1), **Base** and **log(1 + x)**, **Columns** (remove one
  with its x, **Pick columns** to choose), **Fit**;
- after **Run step**, the last run's **before / after** per column (range,
  mean, std after), with two histograms of the picked column; a Z-score step
  counts the values beyond |z| 3;
- for a Power Transform, per column λ, skewness before and after, and the
  normality test's p before and after (D'Agostino-Pearson,
  `scipy.stats.normaltest`, at least 20 values; p below 0.05 means not normal),
  the likelihood curve over λ -5 .. 5 with the best λ marked, and what it
  means: a column that mixes groups (more than one hump) is not made normal by
  any single power transform, and the view says so;
- **Compare methods**: one column through every scaling method (a preview,
  nothing is added), with histograms and a summary table; **Use this method**
  switches the step;
- **All settings**: every saved setting.

The figures read up to the first 200,000 rows of the step's input and output and
are worked out in the background.

## Accuracy

Every transform is checked against float64 PyTorch references of the library it
follows (and against scipy where scipy has the same function) to 1e-8; the step
view's histograms against `numpy.histogram`. On a GPU without float64 (for
example an Intel Iris Xe) the same checks pass at float32 precision: 5e-5
relative, and 2e-3 for a searched λ, whose float32 likelihood is flat near its
optimum.

## Refusals

A step says why it is refused: Box-Cox on a column with a value of 0 or less
(use Yeo-Johnson); plain log on a value of 0 or less (turn on log(1 + x)), or
log(1 + x) on -1 or less (use Power Transform, Yeo-Johnson); a Min-Max range
that does not rise; saved values from other settings or columns. A zero scale
(a constant column, or zero spread) divides by 1, as scikit-learn does.

The Outlier Detector node runs the same code: it flags a value outside
[q1 - t·IQR, q3 + t·IQR], or with |z| > t using the std that divides by n.
