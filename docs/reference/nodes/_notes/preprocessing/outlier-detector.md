## How it decides

The node learns each column's bounds from the Train rows (every row without
a Data Split), with the rule picked:

| Rule | A value is an outlier when |
| --- | --- |
| IQR | it is outside [q1 - t x IQR, q3 + t x IQR] |
| Z-score | \|z\| > t, with the std that divides by n |
| Modified Z | \|x - median\| > t x MAD / 0.6745 |

Then every row is tested, and the action decides what happens:

- **flag** adds an `is_outlier` column: 1 when any inspected column is
  outside its bounds. It is a number column, so training reads it as one
  more feature.
- **clip** pulls each value into its column's bounds (the column becomes
  float64; values inside the bounds and empty cells stay as they are).
- **remove** drops the rows outside.

A column with no spread (all values equal, Q1 = Q3, MAD = 0) flags nothing.
Data Studio's **Explore > Outliers** draws the same rules and adds this step
with its rule and factor
([Missing values and outliers](../../../user-guide/data-studio/missing-values-and-outliers.md)).
