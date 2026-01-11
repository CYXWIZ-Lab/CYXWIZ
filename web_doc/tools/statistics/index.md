# Statistics Tools

Statistical analysis tools for exploring and understanding data distributions, relationships, and significance.

## Overview

The Statistics tools provide:
- **Descriptive Statistics** - Summary statistics and distributions
- **Hypothesis Testing** - Statistical significance tests
- **Correlation Analysis** - Relationship detection
- **Regression Analysis** - Predictive modeling

## Tools Reference

### Descriptive Statistics

| Tool | Description | Shortcut |
|------|-------------|----------|
| **Summary Stats** | Mean, median, std, min, max, quartiles | `Ctrl+Shift+S` |
| **Distribution Plot** | Histogram with KDE overlay | |
| **Box Plot** | Quartile visualization with outliers | |
| **Violin Plot** | Distribution shape comparison | |

#### Summary Statistics Panel

```
+------------------------------------------------------------------+
|  Summary Statistics                                        [x]    |
+------------------------------------------------------------------+
|  Column: [price                    v]                             |
|                                                                   |
|  CENTRAL TENDENCY                                                 |
|  +----------------------------+-------------------------------+   |
|  | Mean                       | 45,234.56                     |   |
|  | Median                     | 42,100.00                     |   |
|  | Mode                       | 39,999.00                     |   |
|  | Trimmed Mean (5%)          | 44,123.45                     |   |
|  +----------------------------+-------------------------------+   |
|                                                                   |
|  DISPERSION                                                       |
|  +----------------------------+-------------------------------+   |
|  | Standard Deviation         | 12,456.78                     |   |
|  | Variance                   | 155,170,432.12                |   |
|  | Range                      | 89,500.00                     |   |
|  | IQR                        | 15,234.00                     |   |
|  | Coefficient of Variation   | 27.54%                        |   |
|  +----------------------------+-------------------------------+   |
|                                                                   |
|  SHAPE                                                            |
|  +----------------------------+-------------------------------+   |
|  | Skewness                   | 0.834 (Right-skewed)          |   |
|  | Kurtosis                   | 2.156 (Platykurtic)           |   |
|  +----------------------------+-------------------------------+   |
|                                                                   |
|  QUANTILES                                                        |
|  +----------------------------+-------------------------------+   |
|  | Min (0%)                   | 5,500.00                      |   |
|  | 25%                        | 35,000.00                     |   |
|  | 50%                        | 42,100.00                     |   |
|  | 75%                        | 50,234.00                     |   |
|  | Max (100%)                 | 95,000.00                     |   |
|  +----------------------------+-------------------------------+   |
|                                                                   |
|  [Export Report]  [Copy to Clipboard]  [Add to Console]           |
+------------------------------------------------------------------+
```

### Hypothesis Testing

| Tool | Description | Use Case |
|------|-------------|----------|
| **T-Test** | Compare two group means | A/B testing |
| **ANOVA** | Compare multiple group means | Multi-group comparison |
| **Chi-Square** | Test categorical independence | Feature relationships |
| **Kolmogorov-Smirnov** | Test distribution normality | Assumption checking |
| **Mann-Whitney U** | Non-parametric comparison | Non-normal data |
| **Wilcoxon** | Paired non-parametric test | Before/after analysis |

#### T-Test Panel

```
+------------------------------------------------------------------+
|  T-Test                                                    [x]    |
+------------------------------------------------------------------+
|  Test Type: ( ) One-Sample  (o) Two-Sample  ( ) Paired            |
|                                                                   |
|  GROUP A                         GROUP B                          |
|  Column: [control    v]          Column: [treatment  v]           |
|  N: 150                          N: 148                           |
|  Mean: 45.23                     Mean: 52.67                       |
|  Std: 12.34                      Std: 11.89                        |
|                                                                   |
|  HYPOTHESIS                                                       |
|  H0: mu_A = mu_B (No difference)                                  |
|  H1: mu_A != mu_B (Two-tailed)                                    |
|                                                                   |
|  Alternative: ( ) Less  (o) Two-Sided  ( ) Greater                |
|  Alpha: [0.05    ]                                                |
|                                                                   |
|  [ Run Test ]                                                     |
|                                                                   |
|  RESULTS                                                          |
|  +-----------------------------------------------------------+   |
|  | t-statistic         | -5.234                               |   |
|  | p-value             | 0.00001 ***                          |   |
|  | Degrees of Freedom  | 296                                  |   |
|  | Effect Size (d)     | 0.608 (Medium)                       |   |
|  | 95% CI for diff     | [-10.23, -4.65]                      |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  CONCLUSION                                                       |
|  Reject H0: Significant difference between groups (p < 0.05)      |
|                                                                   |
|  [Export Results]  [Generate Report]                              |
+------------------------------------------------------------------+
```

### Correlation Analysis

| Tool | Description | Output |
|------|-------------|--------|
| **Pearson Correlation** | Linear relationship strength | -1 to 1 coefficient |
| **Spearman Correlation** | Monotonic relationship | Rank correlation |
| **Kendall Tau** | Ordinal association | Concordance measure |
| **Correlation Matrix** | Multi-variable correlations | Heatmap visualization |
| **Partial Correlation** | Controlled correlations | Confounding removal |

#### Correlation Matrix Panel

```
+------------------------------------------------------------------+
|  Correlation Matrix                                        [x]    |
+------------------------------------------------------------------+
|  Variables: [x] price [x] area [x] rooms [x] age [x] rating       |
|                                                                   |
|  Method: (o) Pearson  ( ) Spearman  ( ) Kendall                   |
|                                                                   |
|  [ Compute ]                                                      |
|                                                                   |
|             price    area    rooms    age    rating               |
|  price     1.000    0.856   0.723   -0.234   0.567                |
|  area      0.856    1.000   0.812   -0.123   0.445                |
|  rooms     0.723    0.812   1.000   -0.089   0.334                |
|  age      -0.234   -0.123  -0.089    1.000  -0.456                |
|  rating    0.567    0.445   0.334   -0.456   1.000                |
|                                                                   |
|  [Heatmap colors: Strong negative (red) to Strong positive (blue)]|
|                                                                   |
|  SIGNIFICANT CORRELATIONS (p < 0.05)                              |
|  +-----------------------------------------------------------+   |
|  | price <-> area     | r = 0.856 | p < 0.001 | Strong +     |   |
|  | price <-> rooms    | r = 0.723 | p < 0.001 | Strong +     |   |
|  | area <-> rooms     | r = 0.812 | p < 0.001 | Strong +     |   |
|  | age <-> rating     | r = -0.456| p < 0.001 | Moderate -   |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Export Matrix]  [Save Heatmap]  [Show Scatter Plots]            |
+------------------------------------------------------------------+
```

### Regression Analysis

| Tool | Description | Use Case |
|------|-------------|----------|
| **Linear Regression** | Single predictor | Simple relationships |
| **Multiple Regression** | Multiple predictors | Complex modeling |
| **Logistic Regression** | Binary classification | Probability estimation |
| **Polynomial Regression** | Non-linear fitting | Curved relationships |

#### Linear Regression Panel

```
+------------------------------------------------------------------+
|  Linear Regression                                         [x]    |
+------------------------------------------------------------------+
|  Dependent (Y): [price          v]                                |
|  Independent (X): [x] area [x] rooms [x] age [ ] rating           |
|                                                                   |
|  [ Fit Model ]                                                    |
|                                                                   |
|  MODEL SUMMARY                                                    |
|  +-----------------------------------------------------------+   |
|  | R-squared           | 0.7845                               |   |
|  | Adjusted R-squared  | 0.7812                               |   |
|  | F-statistic         | 234.56                               |   |
|  | Prob (F-statistic)  | 0.0000                               |   |
|  | AIC                 | 1234.56                              |   |
|  | BIC                 | 1256.78                              |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  COEFFICIENTS                                                     |
|  +------------+--------+--------+--------+---------+              |
|  | Variable   | Coef   | Std Err| t      | p-value |              |
|  +------------+--------+--------+--------+---------+              |
|  | Intercept  | 5234.2 | 1234.5 | 4.24   | 0.000   |              |
|  | area       | 123.45 | 12.34  | 10.01  | 0.000   |              |
|  | rooms      | 4567.8 | 567.8  | 8.04   | 0.000   |              |
|  | age        | -234.5 | 45.6   | -5.14  | 0.000   |              |
|  +------------+--------+--------+--------+---------+              |
|                                                                   |
|  EQUATION                                                         |
|  price = 5234.2 + 123.45*area + 4567.8*rooms - 234.5*age          |
|                                                                   |
|  DIAGNOSTICS                                                      |
|  [x] Residual Plot  [x] Q-Q Plot  [x] Scale-Location              |
|                                                                   |
|  [Predict]  [Export Model]  [Generate Code]                       |
+------------------------------------------------------------------+
```

## Statistical Functions

### Descriptive Statistics Functions

```python
# In CyxWiz scripting console
import pycyxwiz.stats as stats

# Basic statistics
mean = stats.mean(data['column'])
median = stats.median(data['column'])
std = stats.std(data['column'])
variance = stats.variance(data['column'])

# Percentiles
q1, q2, q3 = stats.quartiles(data['column'])
percentile_90 = stats.percentile(data['column'], 90)

# Shape statistics
skewness = stats.skewness(data['column'])
kurtosis = stats.kurtosis(data['column'])

# Summary
summary = stats.describe(data['column'])
```

### Hypothesis Testing Functions

```python
# T-tests
result = stats.ttest_ind(group_a, group_b)
print(f"t={result.statistic}, p={result.pvalue}")

# ANOVA
result = stats.f_oneway(group1, group2, group3)

# Chi-square
contingency_table = stats.crosstab(data['cat1'], data['cat2'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Normality test
stat, p = stats.shapiro(data['column'])
```

### Correlation Functions

```python
# Pearson correlation
r, p = stats.pearsonr(x, y)

# Spearman correlation
rho, p = stats.spearmanr(x, y)

# Correlation matrix
corr_matrix = stats.correlation_matrix(data[['col1', 'col2', 'col3']])
```

### Regression Functions

```python
# Simple linear regression
model = stats.linear_regression(X, y)
print(f"R2: {model.r_squared}")
print(f"Coefficients: {model.coefficients}")

# Predictions
predictions = model.predict(X_new)

# Residuals
residuals = model.residuals()
```

## Integration with Node Editor

Statistics tools can be used in the node editor pipeline:

```
[Data Input] -> [Statistics Node] -> [Filter Node] -> [Model]
                     |
                     v
              [Summary Output]
```

### Statistics Nodes

| Node | Inputs | Outputs |
|------|--------|---------|
| **Describe** | Tensor | Statistics dict |
| **Correlation** | Tensor, Tensor | Correlation coefficient |
| **Normalize** | Tensor | Z-scored tensor |
| **Standardize** | Tensor | Min-max scaled tensor |

## Export Options

| Format | Content |
|--------|---------|
| **CSV** | Raw statistics data |
| **JSON** | Structured results |
| **HTML** | Formatted report |
| **LaTeX** | Publication-ready tables |
| **Python** | Reproducible script |

---

**Next**: [Clustering Tools](../clustering/index.md) | [Model Evaluation Tools](../model-evaluation/index.md)
