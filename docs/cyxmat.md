# CyxWiz MATLAB-Style Command Reference

Quick reference for MATLAB-style functions in the Command Window.

## Two Ways to Call Functions

```python
# Method 1: Flat namespace (MATLAB-style) - quick and easy
A = [[1, 2], [3, 4]]
U, S, V = svd(A)
spectrum = fft(signal)
labels = kmeans(data, k=3)['labels']

# Method 2: Grouped namespace (cyx.*) - organized by category
U, S, V = cyx.linalg.svd(A)
spectrum = cyx.signal.fft(signal)
labels = cyx.stats.kmeans(data, k=3)['labels']
forecast = cyx.timeseries.arima(data, horizon=5)
```

Both methods work identically - use whichever you prefer!

---

## Matrix Printing

Use `printmat` (or `pm` for short) to display matrices in a clean, aligned format:

```python
I = eye(3)
printmat(I)
#   1  0  0
#   0  1  0
#   0  0  1

A = [[1.5, 2.333], [3.14159, 4.0]]
pm(A)                    # Short alias
#   1.5     2.333
#   3.1416  4.0

pm(A, precision=2)       # Custom precision
#   1.5   2.33
#   3.14  4.0
```

| Function | Syntax | Description |
|----------|--------|-------------|
| `printmat` | `printmat(matrix, precision=4, suppress_small=True)` | Print matrix with alignment |
| `pm` | `pm(matrix)` | Short alias for printmat |

---

## Linear Algebra (`cyx.linalg`)

### Matrix Creation

| Function | Syntax | Description |
|----------|--------|-------------|
| `eye` | `eye(n)` | n×n identity matrix |
| `zeros` | `zeros(rows, cols)` | Zero matrix |
| `ones` | `ones(rows, cols)` | Ones matrix |

```python
# Flat
I = eye(3)           # [[1,0,0], [0,1,0], [0,0,1]]
Z = zeros(2, 3)      # [[0,0,0], [0,0,0]]
O = ones(2, 2)       # [[1,1], [1,1]]

# Grouped (cyx.linalg.*)
I = cyx.linalg.eye(3)
Z = cyx.linalg.zeros(2, 3)
O = cyx.linalg.ones(2, 2)
```

### Matrix Decompositions

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `svd` | `svd(A, full=False)` | `(U, S, Vt)` | Singular Value Decomposition |
| `eig` | `eig(A)` | `(real, imag, vectors)` | Eigenvalue decomposition |
| `qr` | `qr(A)` | `(Q, R)` | QR decomposition |
| `chol` | `chol(A)` | `L` | Cholesky decomposition |
| `lu` | `lu(A)` | `(L, U, P)` | LU decomposition |

```python
A = [[1, 2], [3, 4]]

# Flat namespace
U, S, Vt = svd(A)
print("Singular values:", S)  # [5.46, 0.37]

# Grouped namespace (cyx.linalg.*)
U, S, Vt = cyx.linalg.svd(A)
real, imag, vecs = cyx.linalg.eig(A)
Q, R = cyx.linalg.qr(A)
L = cyx.linalg.chol([[4, 2], [2, 5]])  # Must be positive definite
L, U, P = cyx.linalg.lu(A)
```

### Matrix Properties

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `det` | `det(A)` | scalar | Determinant |
| `rank` | `rank(A, tol=1e-10)` | int | Matrix rank |
| `trace` | `trace(A)` | scalar | Sum of diagonal |
| `norm` | `norm(A)` | scalar | Frobenius norm |
| `cond` | `cond(A)` | scalar | Condition number |

```python
A = [[1, 2], [3, 4]]

# Flat                          # Grouped (cyx.linalg.*)
d = det(A)           # -2.0     d = cyx.linalg.det(A)
r = rank(A)          # 2        r = cyx.linalg.rank(A)
t = trace(A)         # 5.0      t = cyx.linalg.trace(A)
n = norm(A)          # 5.477    n = cyx.linalg.norm(A)
c = cond(A)          # 14.93    c = cyx.linalg.cond(A)
```

### Matrix Operations

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `inv` | `inv(A)` | matrix | Matrix inverse |
| `transpose` | `transpose(A)` | matrix | Matrix transpose |
| `matmul` | `matmul(A, B)` | matrix | Matrix multiplication |
| `solve` | `solve(A, b)` | x | Solve Ax = b |
| `lstsq` | `lstsq(A, b)` | x | Least squares solution |

```python
A = [[1, 2], [3, 4]]
Ainv = inv(A)
At = transpose(A)
C = matmul(A, Ainv)  # Should be identity

# Solve linear system
b = [[5], [11]]
x = solve(A, b)      # [[1], [2]]

# Least squares (overdetermined system)
x = lstsq(A, b)
```

---

## Signal Processing (`cyx.signal`)

### Fourier Transform

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `fft` | `fft(x, fs=1.0)` | dict | Fast Fourier Transform |
| `ifft` | `ifft(X)` | array | Inverse FFT |

```python
x = sine(freq=10, fs=100, n=100)

# Flat namespace
result = fft(x, sample_rate=100)

# Grouped namespace (cyx.signal.*)
x = cyx.signal.sine(freq=10, fs=100, n=100)
result = cyx.signal.fft(x, sample_rate=100)

print(result['magnitude'])    # Amplitude spectrum
print(result['phase'])        # Phase spectrum
print(result['frequencies'])  # Frequency bins
```

### Convolution

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `conv` | `conv(x, h, mode='same')` | array | 1D convolution |
| `conv2` | `conv2(x, h, mode='same')` | matrix | 2D convolution |

```python
x = [1, 2, 3, 4, 5]
h = [0.25, 0.5, 0.25]  # Smoothing kernel
y = conv(x, h)

# 2D convolution (image filtering)
img = [[1,2,3], [4,5,6], [7,8,9]]
kernel = [[1,0,-1], [2,0,-2], [1,0,-1]]  # Sobel
edges = conv2(img, kernel)
```

### Spectrogram

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `spectrogram` | `spectrogram(x, win=256, hop=128, fs=1.0, window='hann')` | dict | Time-frequency representation |

```python
x = sine(freq=10, fs=1000, n=1000)
result = spectrogram(x, window_size=256, hop_size=128, sample_rate=1000)
S = result['S']              # Spectrogram matrix
freqs = result['frequencies']
times = result['times']
```

### Filter Design

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `lowpass` | `lowpass(cutoff, fs, order=4)` | dict | Lowpass filter |
| `highpass` | `highpass(cutoff, fs, order=4)` | dict | Highpass filter |
| `bandpass` | `bandpass(low, high, fs, order=4)` | dict | Bandpass filter |
| `filter` | `filter(x, b, a)` | array | Apply filter |

```python
# Design a 20 Hz lowpass filter at 100 Hz sample rate
filt = lowpass(cutoff=20, fs=100, order=4)
b, a = filt['b'], filt['a']

# Apply filter
y = filter(x, b, a)

# Bandpass 10-30 Hz
bp = bandpass(low=10, high=30, fs=100)
```

### Peak Detection

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `findpeaks` | `findpeaks(x, min_height=0, min_distance=1)` | dict | Find peaks in signal |

```python
x = [0, 1, 0, 2, 0, 1.5, 0]
peaks = findpeaks(x, min_height=0.5)
print(peaks['indices'])  # [1, 3, 5]
print(peaks['values'])   # [1, 2, 1.5]
```

### Signal Generation

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `sine` | `sine(freq, fs, n, amp=1, phase=0)` | array | Sine wave |
| `square` | `square(freq, fs, n, amp=1)` | array | Square wave |
| `noise` | `noise(n, amp=1)` | array | White noise |

```python
# 10 Hz sine wave, 100 Hz sample rate, 100 samples
x = sine(freq=10, fs=100, n=100)

# Square wave
sq = square(freq=5, fs=100, n=100)

# White noise
n = noise(n=100, amp=0.1)

# Signal + noise
noisy = [x[i] + n[i] for i in range(len(x))]
```

---

## Statistics & Clustering (`cyx.stats`)

### Clustering

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `kmeans` | `kmeans(data, k, max_iter=300, init='kmeans++')` | dict | K-Means clustering |
| `dbscan` | `dbscan(data, eps, min_samples=5)` | dict | DBSCAN clustering |
| `gmm` | `gmm(data, n_components, cov_type='full')` | dict | Gaussian Mixture Model |

```python
data = [[1,1], [1.1,1], [5,5], [5.1,5], [9,9], [9.1,9]]

# Flat namespace
result = kmeans(data, k=3)
result = dbscan(data, eps=0.5, min_samples=2)
result = gmm(data, n_components=3)

# Grouped namespace (cyx.stats.*)
result = cyx.stats.kmeans(data, k=3)
print(result['labels'])      # [0, 0, 1, 1, 2, 2]
print(result['centroids'])   # Cluster centers
print(result['inertia'])     # Sum of squared distances

result = cyx.stats.dbscan(data, eps=0.5, min_samples=2)
print(result['labels'])      # -1 = noise
print(result['n_clusters'])

result = cyx.stats.gmm(data, n_components=3)
print(result['labels'])
print(result['means'])
print(result['aic'], result['bic'])
```

### Dimensionality Reduction

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `pca` | `pca(data, n_components=2)` | dict | Principal Component Analysis |
| `tsne` | `tsne(data, n_dims=2, perplexity=30)` | array | t-SNE embedding |

```python
# High-dimensional data
data = [[1,2,3,4], [1.1,2.1,3.1,4.1], [5,6,7,8], [5.1,6.1,7.1,8.1]]

# Flat namespace
result = pca(data, n_components=2)
embeddings = tsne(data, n_dims=2, perplexity=2)

# Grouped namespace (cyx.stats.*)
result = cyx.stats.pca(data, n_components=2)
X_2d = result['transformed']
print(result['explained_variance'])  # Variance ratio per component
print(result['components'])          # Principal components

embeddings = cyx.stats.tsne(data, n_dims=2, perplexity=2)
```

### Evaluation Metrics

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `silhouette` | `silhouette(data, labels)` | float | Silhouette score |
| `confusion_matrix` | `confusion_matrix(y_true, y_pred)` | dict | Confusion matrix + metrics |
| `roc` | `roc(y_true, y_scores)` | dict | ROC curve + AUC |

```python
# Clustering quality
data = [[1,1], [1.1,1], [5,5], [5.1,5]]
labels = [0, 0, 1, 1]
score = silhouette(data, labels)  # -1 to 1, higher is better

# Classification metrics
y_true = [0, 0, 1, 1, 1]
y_pred = [0, 1, 1, 1, 0]
result = confusion_matrix(y_true, y_pred)
print(result['matrix'])
print(result['accuracy'])
print(result['precision'])
print(result['recall'])
print(result['f1'])

# ROC curve
y_scores = [0.1, 0.4, 0.35, 0.8, 0.7]
result = roc(y_true, y_scores)
print(result['auc'])         # Area under curve
print(result['fpr'])         # False positive rates
print(result['tpr'])         # True positive rates
```

---

## Time Series (`cyx.timeseries`)

### Autocorrelation

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `acf` | `acf(data, max_lag=-1)` | dict | Autocorrelation function |
| `pacf` | `pacf(data, max_lag=-1)` | array | Partial autocorrelation |

```python
data = [1, 2, 3, 2, 1, 2, 3, 2, 1]

# Flat namespace
result = acf(data, max_lag=5)
pacf_values = pacf(data, max_lag=5)

# Grouped namespace (cyx.timeseries.*)
result = cyx.timeseries.acf(data, max_lag=5)
print(result['acf'])                # ACF values
print(result['confidence_interval']) # 95% CI

pacf_values = cyx.timeseries.pacf(data, max_lag=5)
```

### Decomposition

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `decompose` | `decompose(data, period, method='additive')` | dict | Trend/Seasonal/Residual |

```python
# Monthly data with yearly seasonality
data = [10, 12, 15, 18, 22, 25, 24, 22, 18, 14, 11, 10,
        11, 13, 16, 19, 23, 26, 25, 23, 19, 15, 12, 11]

# Flat namespace
result = decompose(data, period=12, method='additive')

# Grouped namespace (cyx.timeseries.*)
result = cyx.timeseries.decompose(data, period=12, method='additive')
print(result['trend'])
print(result['seasonal'])
print(result['residual'])
```

### Stationarity Testing

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `stationarity` | `stationarity(data)` | dict | ADF + KPSS tests |

```python
data = [1, 2, 3, 4, 5, 6, 7, 8]  # Non-stationary (trend)

result = stationarity(data)
print(result['is_stationary'])    # False
print(result['adf_statistic'])
print(result['adf_pvalue'])
print(result['kpss_statistic'])
print(result['kpss_pvalue'])
```

### Forecasting

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `arima` | `arima(data, horizon, p=-1, d=-1, q=-1)` | dict | ARIMA forecasting |

```python
# Historical sales data
sales = [100, 110, 120, 115, 125, 130, 128, 135, 140, 138]

# Forecast next 5 periods (auto-select p,d,q if -1)
result = arima(sales, horizon=5)
print(result['forecast'])     # Predicted values
print(result['lower'])        # Lower confidence bound
print(result['upper'])        # Upper confidence bound
print(result['mse'])          # Mean squared error
print(result['aic'])          # Model selection criterion

# Or specify ARIMA(1,1,1) manually
result = arima(sales, horizon=5, p=1, d=1, q=1)
```

### Transformations

| Function | Syntax | Returns | Description |
|----------|--------|---------|-------------|
| `diff` | `diff(data, order=1)` | array | Difference series |
| `rolling_mean` | `rolling_mean(data, window)` | array | Rolling mean |
| `rolling_std` | `rolling_std(data, window)` | array | Rolling standard deviation |

```python
data = [1, 3, 6, 10, 15, 21]

# First difference (make stationary)
d1 = diff(data)           # [2, 3, 4, 5, 6]

# Second difference
d2 = diff(data, order=2)  # [1, 1, 1, 1]

# Rolling statistics
rm = rolling_mean(data, window=3)
rs = rolling_std(data, window=3)
```

---

## Quick Examples

### Complete Linear Algebra Workflow
```python
# Create matrix
A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]

# Check properties
print("Determinant:", det(A))
print("Rank:", rank(A))
print("Condition:", cond(A))

# Decompose
L = chol(A)  # Cholesky (A must be positive definite)
print("L @ L.T = A:", matmul(L, transpose(L)))

# Solve system
b = [[1], [2], [3]]
x = solve(A, b)
```

### Complete Signal Processing Workflow
```python
# Generate noisy signal
fs = 1000  # Sample rate
t = 1.0    # Duration
n = int(fs * t)

signal = sine(freq=50, fs=fs, n=n)
noisy = [signal[i] + noise(1, 0.5)[0] for i in range(n)]

# Filter noise
filt = lowpass(cutoff=100, fs=fs)
clean = filter(noisy, filt['b'], filt['a'])

# Analyze spectrum
spectrum = fft(clean, sample_rate=fs)
peaks = findpeaks(spectrum['magnitude'], min_height=10)
```

### Complete Clustering Workflow
```python
# Load/generate data
data = [[1,2], [1.5,1.8], [5,8], [8,8], [1,0.6], [9,11]]

# Try different algorithms
km = kmeans(data, k=2)
db = dbscan(data, eps=2, min_samples=2)

# Evaluate
score_km = silhouette(data, km['labels'])
score_db = silhouette(data, db['labels'])

print(f"K-Means silhouette: {score_km}")
print(f"DBSCAN silhouette: {score_db}")

# Reduce for visualization
pca_result = pca(data, n_components=2)
```

### Complete Time Series Workflow
```python
# Load time series
data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118]

# Test stationarity
stat = stationarity(data)
if not stat['is_stationary']:
    data_diff = diff(data)

# Analyze autocorrelation
acf_result = acf(data, max_lag=6)

# Decompose
decomp = decompose(data, period=4)

# Forecast
forecast = arima(data, horizon=4)
print("Next 4 predictions:", forecast['forecast'])
```

---

## Namespace Reference

| Flat | Grouped | Category |
|------|---------|----------|
| `svd`, `eig`, `qr`, `chol`, `lu` | `cyx.linalg.*` | Decompositions |
| `det`, `rank`, `trace`, `norm`, `cond` | `cyx.linalg.*` | Properties |
| `inv`, `transpose`, `solve`, `lstsq`, `matmul` | `cyx.linalg.*` | Operations |
| `eye`, `zeros`, `ones` | `cyx.linalg.*` | Creation |
| `fft`, `ifft`, `conv`, `conv2` | `cyx.signal.*` | Transforms |
| `spectrogram` | `cyx.signal.*` | Analysis |
| `lowpass`, `highpass`, `bandpass`, `filter` | `cyx.signal.*` | Filtering |
| `findpeaks` | `cyx.signal.*` | Detection |
| `sine`, `square`, `noise` | `cyx.signal.*` | Generation |
| `kmeans`, `dbscan`, `gmm` | `cyx.stats.*` | Clustering |
| `pca`, `tsne` | `cyx.stats.*` | Reduction |
| `silhouette`, `confusion_matrix`, `roc` | `cyx.stats.*` | Evaluation |
| `acf`, `pacf` | `cyx.timeseries.*` | Correlation |
| `decompose` | `cyx.timeseries.*` | Decomposition |
| `stationarity` | `cyx.timeseries.*` | Testing |
| `arima` | `cyx.timeseries.*` | Forecasting |
| `diff`, `rolling_mean`, `rolling_std` | `cyx.timeseries.*` | Transforms |

---

## DuckDB (SQL Analytics)

DuckDB provides fast SQL analytics directly on files.

| Function | Syntax | Description |
|----------|--------|-------------|
| `sql` | `sql(query)` | Execute SQL query |
| `read_csv` | `read_csv(path)` | Load CSV file |
| `read_parquet` | `read_parquet(path)` | Load Parquet file |
| `read_json` | `read_json(path)` | Load JSON file |
| `db` | `db` | DuckDB connection object |

```python
# Simple SQL
sql("SELECT 1 + 1 AS result")

# Query files directly (no import needed!)
sql("SELECT * FROM 'data.csv' LIMIT 10")
sql("SELECT category, SUM(amount) FROM 'sales.parquet' GROUP BY category")

# Load files
data = read_csv('data.csv')
data.show()

# Advanced: joins, CTEs, window functions
sql("""
    WITH top_sales AS (
        SELECT product, SUM(amount) as total
        FROM 'orders.csv'
        GROUP BY product
        ORDER BY total DESC
        LIMIT 10
    )
    SELECT * FROM top_sales
""")
```

---

## Polars (Fast DataFrames)

Polars provides lightning-fast DataFrame operations.

| Function | Syntax | Description |
|----------|--------|-------------|
| `pl` | `pl` | Polars module |
| `df` | `df(data)` | Create DataFrame |
| `lf` | `lf(data)` | Create LazyFrame |
| `col` | `col('name')` | Column expression |
| `lit` | `lit(value)` | Literal value |
| `when` | `when(condition)` | Conditional expression |
| `pl_csv` | `pl_csv(path)` | Read CSV |
| `pl_parquet` | `pl_parquet(path)` | Read Parquet |
| `pl_json` | `pl_json(path)` | Read JSON |
| `pl_excel` | `pl_excel(path)` | Read Excel |
| `scan_csv` | `scan_csv(path)` | Lazy CSV read |
| `scan_parquet` | `scan_parquet(path)` | Lazy Parquet read |

```python
# Create DataFrame
data = df({'name': ['Alice', 'Bob', 'Carol'], 'age': [25, 30, 28]})

# Filter and select
data.filter(col('age') > 25)
data.select(['name', 'age'])

# Transform
data.with_columns([
    (col('age') + 1).alias('next_year'),
    col('name').str.to_uppercase().alias('NAME')
])

# Aggregation
sales = pl_csv('sales.csv')
sales.group_by('region').agg([
    col('amount').sum().alias('total'),
    col('amount').mean().alias('avg')
])

# Lazy evaluation (for large files)
result = (scan_csv('huge_file.csv')
    .filter(col('status') == 'active')
    .group_by('category')
    .agg(col('value').sum())
    .collect())  # Execute here
```

### DuckDB + Polars Together

```python
# Use DuckDB for complex SQL
result = sql("SELECT * FROM 'data.parquet' WHERE amount > 100")

# Convert to Polars for further processing
df_result = result.pl()
df_result.with_columns(col('amount') * 1.1)
```
