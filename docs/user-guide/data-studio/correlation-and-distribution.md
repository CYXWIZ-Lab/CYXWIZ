---
title: Correlation and distribution
description: See how the number columns move together in Explore > Correlation, and how one column is spread and which distribution fits it in Explore > Distribution.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - explore.md
  - profile.md
  - statistics.md
---

**Explore > Correlation** and **Explore > Distribution** in Data Studio
describe the number columns of the picked table. Neither changes the data.
**Data > Explore > Correlation Matrix** opens Correlation; **Tools >
Statistics > Summary Statistics** and **Distribution Fitter** open
Distribution.

## See how the columns move together

Correlation shows Pearson r for every pair of number columns, over the rows
where both values are present.

| Part | What it shows |
| --- | --- |
| The matrix | The profile's first 30 number columns (pick others in **Columns**, a searchable list with chips). Each cell shows r; hover for the pair, r, its strength in words and how many rows it used; click to pick it. |
| The picked pair (right) | r, a sentence (*very strong positive: when one is larger, so is the other*) and the rows it used |
| **STRONGEST PAIRS** | Every pair by \|r\|, strongest first; click one to pick it |
| Display | Cell size, values in the cells, the -1 .. +1 legend |

**Export CSV...** saves the matrix: a header of the columns, then one row
each.

Penguins: flipper length and body mass +0.87; bill depth and flipper length
-0.58; every pair uses 342 of 344 rows (2 rows have no measurements).

## See how a column is spread

1. In **Distribution**, pick the **Column**.
2. Optional: set **Bins** (5 to 100). The histogram follows when you leave
   the field.
3. Click a fit in **FITS, BEST AIC FIRST** to draw it over the histogram
   and the Q-Q plot.
4. Optional: **Export CSV...** saves the figures, percentiles and fits.

| Part | What it shows |
| --- | --- |
| Summary | Values, missing, skewness in words, and whether any distribution fits |
| Histogram | With the chosen fit drawn over it (expected count per bin) |
| Box plot | Box Q1 to Q3, line at the median, diamond at the mean, whiskers to the last value within 1.5 IQR, dots beyond |
| Q-Q plot | Against the chosen fit: points on the line mean the values follow it |
| Figures | Count, mean, median, mode (and how often), std, variance, SEM, min, max, range, sum, Q1, Q3, IQR, skewness, kurtosis (excess) and CV; hover a name for what it means |
| **PERCENTILES** | 5 to 95 |
| **FITS, BEST AIC FIRST** | Normal, Uniform, and for columns of positive values Exponential and Log-Normal, with BIC, the KS statistic and its p-value; a check mark when p is above 0.05. The picked fit's parameters and log-likelihood show below. |

The figures are SQL on the dataset (approximate quantiles on tables the
profile reads approximately). The fits use the column's values; a column of
more than 200,000 values is fitted on a 200,000-value sample and says so.

Penguins: `bill_length_mm` fits Normal (KS p 0.066); `body_mass_g` fits
nothing (best Log-Normal, p 0.018), because the three species mixed give it
more than one hump. Its mode is 3,800 g (12 birds), skewness 0.47, excess
kurtosis -0.72.

## Related

- [Profile a dataset](profile.md)
- [Statistics](statistics.md)
- [Find missing values and outliers](missing-values-and-outliers.md)
- [Explore a dataset](explore.md)
