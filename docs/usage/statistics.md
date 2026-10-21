# Statistics: tests and regression

Hypothesis tests and regression live in Data Studio's **Explore** job, as the
**Tests** and **Regression** tabs, and as graph nodes: **Hypothesis Test**,
**Linear Regression** and **Polynomial Regression**. **Tools > Statistics >
Hypothesis Testing** and **Linear / Polynomial Regression** open the tabs on
the data in use; the old windows are gone.

Everything is computed on the device selected in the Devices tab (the
backend's `cyxwiz::StatisticalTests`: sums, variances, ranks and the
least-squares solve run on ArrayFire). Every test and fit is checked against
scipy and statsmodels in `test_computation_truth_statistical_tests`, on the
CUDA, OpenCL and CPU backends. p-values come from upper tails computed
directly, so a p of 1e-80 is exact rather than 0.

## Tests

Pick a **Question**, then its columns:

| Question | Columns | Test | Beside it |
| --- | --- | --- | --- |
| Two groups | **Values** split by a column with 2 values, or **two columns** | Welch's t (default) or Student's t | Mann-Whitney U, Levene |
| Paired columns | **Before** and **after** (rows kept together) | paired t | |
| Several groups | **Values** split by a column with 2 to 20 values | Welch's ANOVA (default) or the classic F, Tukey HSD pairs | Kruskal-Wallis, Levene |
| One column vs a value | **Values**, **compared with** a typed value | one-sample t | |
| Two categories | **Rows** and **columns** (two label columns) | chi-square of independence | expected counts |

**alpha** (typed, 0.001 to 0.5) sets what counts as significant; the CIs are
at 1 - alpha.

The answer shows:
- the groups: n, mean, SD, and whether each looks normal (D'Agostino-Pearson,
  20 values or more);
- box plots on one scale; for several groups, which pairs differ (Tukey HSD
  with CIs); for categories, the counts with the expected counts;
- the figures (difference or mean and its CI, t / F / chi-square, df, p,
  Cohen's d / eta² / Cramér's V with words: small, medium, large);
- a sentence that says what it means, and the **checks**: variances,
  normality, the rank test (and whether it agrees), expected counts below 5.

Rows without a value or a group are left out and counted. A value is
missing when it is empty, not a number, or one of the texts the project marks
as missing for its column. A refusal says why (a group column with too many
values, a column with the same value everywhere).

**Copy**, **Export CSV...**, and **Keep as a node**: a Hypothesis Test node
with these settings reading the data in use, so the test runs with the graph.

Penguins: bill length by sex, Welch t 6.672 on 329.3 df, p 1.1e-10, males
3.758 mm longer (95% CI 2.65 to 4.866); by species, Welch F 421.8 (2, 166.5),
every pair differs; species and island, chi-square 299.6 on 4 df, Cramér's V
0.66: Chinstrap only on Dream, Gentoo only on Biscoe.

## Regression

**Predict** a number column **from** one or more others (**Linear**), or from
one column as a curve (**Polynomial**, degree typed 1 to 6).

- **Coefficients**: estimate, 95% CI, SE, t, p (in colour when the CI
  excludes 0), and the equation with the column names.
- What the fit says: the predictor that carries the model, and the ones that
  add little once the others are known (their CI includes 0); then the
  simpler model on the predictors that matter, or for a polynomial the
  straight line, compared by R² and AIC.
- **Fit**: R², adjusted R², F with its df and p, residual SE, MAE, RMSE, AIC.
- Plots: **data and fit** (one predictor), **residuals vs fitted**, **Q-Q**
  of the residuals, **histogram** of the residuals, **actual vs predicted**;
  and whether the residuals look normal.
- **Add to the graph** (a Linear or Polynomial Regression node with these
  columns reading the data in use), **Export CSV...** (the coefficients, the
  fit, and every row with its prediction and residual), **Copy**.

The fit is ordinary least squares by QR on the device. A polynomial is fitted
on centred, scaled values and shown in the column's own units, so high
degrees stay accurate. A constant or dependent column, or too few rows, is
refused with the column named.

Penguins: body mass from flipper length, bill length and bill depth: R²
0.7615; flipper length carries it (50.27 g per mm); flipper length alone gives
R² 0.759 with a lower AIC. Degree 2 in flipper length: R² 0.775, AIC 5038.9
against 5060.9 for the line.

## Nodes

**Hypothesis Test** (Analytics): the Tests tab's questions on the node's
input. Parameters: `question` (two_groups, paired, several_groups, one_value,
two_categories), `values`, `groups`, `second`, `two_columns`, `variances`
(unequal = Welch, equal), `alpha`, `value`. Its output is one row: question,
test, statistic, df, df2, p, estimate, ci_low, ci_high, confidence, effect,
effect_name, n, left_out, significant, answer.

**Linear Regression**: `feature_cols` (comma-separated), `target_col`,
`fit_intercept` (off: R² is uncentred, as statsmodels does). **Polynomial
Regression**: `feature_col`, `target_col`, `degree` 1 to 6. Both append the
prediction and residual columns to their input and can save the fitted model.
