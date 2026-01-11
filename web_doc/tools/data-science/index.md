# Data Science Tools

Tools for data exploration, quality assessment, and understanding your datasets.

## Available Tools

| Tool | Description | Access |
|------|-------------|--------|
| [Data Profiler](#data-profiler) | Comprehensive data overview | Tools > Data Science > Data Profiler |
| [Correlation Matrix](#correlation-matrix) | Feature relationship analysis | Tools > Data Science > Correlation Matrix |
| [Missing Value Analysis](#missing-value-analysis) | Handle missing data | Tools > Data Science > Missing Values |
| [Outlier Detection](#outlier-detection) | Find anomalies | Tools > Data Science > Outlier Detection |

---

## Data Profiler

**Purpose:** Generate a comprehensive report about your dataset including data types, statistics, distributions, and quality metrics.

### Interface

```
+------------------------------------------------------------------+
|  Data Profiler                                            [x] [-] |
+------------------------------------------------------------------+
|  Dataset: [ iris.csv          v]   [ Load ]                       |
+------------------------------------------------------------------+
|                                                                   |
|  OVERVIEW                                                         |
|  +-----------------------+--------------------------------------+  |
|  | Rows                  | 150                                  |  |
|  | Columns               | 5                                    |  |
|  | Missing Cells         | 0 (0.0%)                             |  |
|  | Duplicate Rows        | 1 (0.7%)                             |  |
|  | Memory Usage          | 6.1 KB                               |  |
|  +-----------------------+--------------------------------------+  |
|                                                                   |
|  COLUMN TYPES                                                     |
|  [==========] Numeric: 4 (80%)                                    |
|  [==        ] Categorical: 1 (20%)                                |
|                                                                   |
|  WARNINGS                                                         |
|  ! Column 'species' has 3 unique values (categorical)             |
|  ! 1 duplicate row detected                                       |
|                                                                   |
|  COLUMN DETAILS  [Expand All] [Collapse All]                      |
|  +----------------------------------------------------------------|
|  | sepal_length  numeric  [Expand v]                              |
|  |   Mean: 5.84  Std: 0.83  Min: 4.3  Max: 7.9                   |
|  |   [========== Histogram ==========]                            |
|  +----------------------------------------------------------------|
|                                                                   |
|  [ Export Report ]  [ Export to Console ]                         |
+------------------------------------------------------------------+
```

### Features

- **Overview Statistics:** Row/column counts, memory usage
- **Data Types:** Automatic type detection
- **Missing Data:** Percentage and patterns
- **Distributions:** Histograms for numeric columns
- **Unique Values:** For categorical columns
- **Correlations:** Quick correlation overview
- **Warnings:** Data quality issues

### Usage

1. Select dataset from dropdown or load file
2. Click "Load" to analyze
3. Review generated report
4. Export as HTML/PDF if needed

---

## Correlation Matrix

**Purpose:** Visualize relationships between numerical features to identify redundant or highly related variables.

### Interface

```
+------------------------------------------------------------------+
|  Correlation Matrix                                       [x] [-] |
+------------------------------------------------------------------+
|  Dataset: [ iris.csv    v]                                        |
|  Method:  [ Pearson     v]  (Pearson / Spearman / Kendall)        |
|  [ Calculate ]                                                    |
+------------------------------------------------------------------+
|                                                                   |
|            sepal_l  sepal_w  petal_l  petal_w                     |
|  sepal_l     1.00    -0.12     0.87     0.82                     |
|  sepal_w    -0.12     1.00    -0.43    -0.37                     |
|  petal_l     0.87    -0.43     1.00     0.96                     |
|  petal_w     0.82    -0.37     0.96     1.00                     |
|                                                                   |
|  [============ Heatmap Visualization =============]               |
|  |        |+++++|     |     |                                     |
|  |     |++|-----|     |     |                                     |
|  |+++++|  |+++++|+++++|                                          |
|  |+++++|  |+++++|+++++|                                          |
|                                                                   |
|  High correlation (>0.8):                                         |
|  - petal_length & petal_width: 0.96                              |
|  - sepal_length & petal_length: 0.87                             |
|                                                                   |
|  [ Export Matrix ]  [ Export Heatmap ]                            |
+------------------------------------------------------------------+
```

### Correlation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Pearson** | Linear relationships | Normally distributed |
| **Spearman** | Monotonic relationships | Ordinal data, non-linear |
| **Kendall** | Rank-based | Small samples, ties |

### Features

- Heatmap visualization with color scale
- Highlight strong correlations
- Filter by threshold
- Export correlation matrix
- Statistical significance (p-values)

---

## Missing Value Analysis

**Purpose:** Identify, visualize, and handle missing data in your dataset.

### Interface

```
+------------------------------------------------------------------+
|  Missing Value Analysis                                   [x] [-] |
+------------------------------------------------------------------+
|  Dataset: [ customer_data.csv  v]   [ Analyze ]                   |
+------------------------------------------------------------------+
|                                                                   |
|  SUMMARY                                                          |
|  Total cells: 10,000    Missing: 423 (4.23%)                     |
|                                                                   |
|  BY COLUMN                                                        |
|  +-------------------+--------+------+-----------------------+    |
|  | Column            | Type   | Miss | Pattern               |    |
|  +-------------------+--------+------+-----------------------+    |
|  | age               | int    |  0%  | [==================] |    |
|  | income            | float  |  12% | [====          ====] |    |
|  | education         | string |  5%  | [=    =  =        =] |    |
|  | purchase_amount   | float  |  8%  | [=====    =====    ] |    |
|  +-------------------+--------+------+-----------------------+    |
|                                                                   |
|  MISSING PATTERN                                                  |
|  [=========== Missingness Heatmap ============]                   |
|  Shows which rows have missing values in which columns            |
|                                                                   |
|  IMPUTATION OPTIONS                                               |
|  ( ) Drop rows with missing values                                |
|  ( ) Fill with mean (numeric) / mode (categorical)                |
|  ( ) Fill with median                                             |
|  ( ) Forward fill                                                 |
|  ( ) Backward fill                                                |
|  ( ) Custom value: [_____]                                        |
|                                                                   |
|  [ Preview Imputation ]  [ Apply ]  [ Export ]                    |
+------------------------------------------------------------------+
```

### Features

- Missing value counts per column
- Missingness patterns (MCAR, MAR, MNAR detection)
- Visual heatmap of missing data
- Multiple imputation strategies
- Preview before applying changes

### Imputation Methods

| Method | Description | Use When |
|--------|-------------|----------|
| **Drop** | Remove rows | Few missing, MCAR |
| **Mean** | Replace with mean | Normal distribution |
| **Median** | Replace with median | Skewed distribution |
| **Mode** | Most frequent | Categorical data |
| **Forward Fill** | Previous value | Time series |
| **Interpolate** | Linear interpolation | Sequential data |

---

## Outlier Detection

**Purpose:** Identify unusual data points that may be errors or interesting anomalies.

### Interface

```
+------------------------------------------------------------------+
|  Outlier Detection                                        [x] [-] |
+------------------------------------------------------------------+
|  Dataset: [ sensor_data.csv  v]                                   |
|  Column:  [ temperature      v]                                   |
|  Method:  [ Z-Score          v]                                   |
|  Threshold: [ 3.0 ]                                               |
|  [ Detect Outliers ]                                              |
+------------------------------------------------------------------+
|                                                                   |
|  RESULTS                                                          |
|  Total points: 1,000    Outliers: 23 (2.3%)                      |
|                                                                   |
|  [============== Box Plot ==============]                         |
|          |-----|=======|---------|                                |
|    *  *         [  IQR  ]              *   *                     |
|  outliers       normal range       outliers                       |
|                                                                   |
|  OUTLIER INDICES                                                  |
|  +-------+-------+---------+                                      |
|  | Index | Value | Z-Score |                                      |
|  +-------+-------+---------+                                      |
|  |    15 |  98.5 |   4.21  |                                      |
|  |    42 | -12.3 |  -3.87  |                                      |
|  |   ... |  ...  |   ...   |                                      |
|  +-------+-------+---------+                                      |
|                                                                   |
|  ACTIONS                                                          |
|  ( ) Keep outliers                                                |
|  ( ) Remove outliers                                              |
|  ( ) Cap outliers (winsorize)                                     |
|  ( ) Replace with median                                          |
|                                                                   |
|  [ Apply ]  [ Export Outlier Indices ]                            |
+------------------------------------------------------------------+
```

### Detection Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| **Z-Score** | Standard deviations from mean | Threshold (default: 3) |
| **IQR** | Interquartile range | Multiplier (default: 1.5) |
| **Isolation Forest** | Ensemble anomaly detection | Contamination (0.01-0.5) |
| **LOF** | Local Outlier Factor | n_neighbors (default: 20) |
| **DBSCAN** | Density-based | eps, min_samples |

### Features

- Multiple detection algorithms
- Box plot visualization
- Scatter plot highlighting
- Configurable thresholds
- Action options (remove, cap, replace)
- Export outlier indices

### Z-Score Method

```
z = (x - mean) / std
outlier if |z| > threshold (typically 3)
```

### IQR Method

```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
outlier if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR
```

---

## Backend Integration

These tools utilize GPU-accelerated algorithms from `cyxwiz-backend`:

```cpp
// Correlation computation via ArrayFire
af::array corr = af::corrcoef(data);

// Outlier detection
af::array z_scores = (data - af::mean(data)) / af::stdev(data);
af::array outliers = af::abs(z_scores) > threshold;
```

## Common Workflows

### Exploratory Data Analysis (EDA)

1. **Data Profiler** - Get overview
2. **Missing Values** - Check and handle
3. **Outlier Detection** - Identify anomalies
4. **Correlation Matrix** - Find relationships

### Data Cleaning Pipeline

1. Handle missing values
2. Remove or cap outliers
3. Check distributions
4. Verify correlations

---

**Next**: [Statistics Tools](../statistics/index.md) | [Clustering Tools](../clustering/index.md)
