# Data transformation tools

The Normalization, Standardization, Log Transform, Box-Cox and Feature Scaling
panels compute on the device selected in the Devices tab (CUDA, OpenCL, or
ArrayFire's CPU backend). There is no separate CPU code path; a build without
ArrayFire reports that the operation needs ArrayFire. Results are float64 where
the device supports it, otherwise float32.

Every transform is checked against float64 PyTorch references of the library it
follows (and against scipy where scipy has the same function) to 1e-8.
On a GPU without float64 (for example an Intel Iris Xe) the same checks pass at
float32 precision: 5e-5 relative, and 2e-3 for a searched Box-Cox / Yeo-Johnson
lambda, whose float32 likelihood is flat near its optimum.

## What each transform does

| Transform | Result | Same as |
| --- | --- | --- |
| Min-Max | `(x - min) / (max - min)` scaled to the chosen range; a constant column gives the range minimum | scikit-learn `MinMaxScaler` |
| Z-Score | `(x - mean) / std`, std divides by n; a constant column gives 0 | scikit-learn `StandardScaler` |
| Robust | `(x - median) / (q_max - q_min)`, default quartiles 25 / 75, interpolated linearly; a zero spread scales by 1 | scikit-learn `RobustScaler` |
| Max-Abs | `x / max(|x|)`; an all-zero column stays 0 | scikit-learn `MaxAbsScaler` |
| Quantile | Each value through min(n quantiles, rows) empirical quantiles onto uniform [0, 1] or a standard normal | scikit-learn `QuantileTransformer` fitted on the whole column (no 10,000-row subsample) |
| Log | ln, log10 or log2 of x, or of 1 + x with log1p | |
| Box-Cox | `(x^λ - 1) / λ`, `log x` at λ = 0; strictly positive data | `scipy.stats.boxcox` |
| Yeo-Johnson | Box-Cox extended to zero and negative values | `scipy.stats.yeojohnson` |

## Lambda search (Box-Cox panel)

"Auto-find optimal lambda" maximises the log-likelihood over λ in [-5, 5]: the
Lambda tab plots the profile at 101 points, and the optimum is refined well
past the grid spacing (it matches scipy's λ to about 1e-8). Yeo-Johnson uses its
own likelihood, so the Lambda tab works with "Use Yeo-Johnson" too.

## Statistics and the normality test

- Column statistics (min, max, mean, std, quartiles, IQR) use the population
  standard deviation, so a Z-Score column shows std 1.
- The normality line is the D'Agostino-Pearson K² test (`scipy.stats.normaltest`)
  and needs at least 20 values; with fewer, the panel says "not tested".
  "Normal" means p > 0.05.

## Refusals

A column is refused, with the reason shown, when Box-Cox data has a value ≤ 0,
or when log data has a value ≤ 0 (≤ -1 with log1p). A zero scale (a constant
column, or zero spread) divides by 1, as scikit-learn does.

The Standard Scaler, Min-Max Scaler, Robust Scaler, Log Transform and Outlier
Detector nodes run the same code on the device. The Outlier Detector flags a
value outside [q1 - t·IQR, q3 + t·IQR], or with |z| > t using the std that
divides by n.
