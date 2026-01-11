# MATLAB Compatibility

CyxWiz provides MATLAB-style function aliases for users familiar with MATLAB syntax. These are available through the `pycyxwiz` module.

## Overview

The `pycyxwiz` module includes submodules with MATLAB-inspired function names:

```python
import pycyxwiz as cx

# MATLAB-style functions
A = cx.linalg.eye(3)      # Identity matrix
B = cx.linalg.zeros(3, 4)  # Zero matrix
U, S, Vt = cx.linalg.svd(A)  # SVD decomposition
```

## Linear Algebra (`cx.linalg`)

### Matrix Creation

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `eye(n)` | `eye(n)` | Identity matrix |
| `zeros(n)` | `zeros(n)` | Square zero matrix |
| `zeros(m, n)` | `zeros(m, n)` | Zero matrix |
| `ones(n)` | `ones(n)` | Square ones matrix |
| `ones(m, n)` | `ones(m, n)` | Ones matrix |
| `diag(v)` | `diag(v)` | Diagonal matrix from vector |

```python
>>> import pycyxwiz as cx

>>> cx.linalg.eye(3)
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]]

>>> cx.linalg.zeros(2, 3)
[[0, 0, 0],
 [0, 0, 0]]

>>> cx.linalg.ones(2, 2)
[[1, 1],
 [1, 1]]

>>> cx.linalg.diag([1, 2, 3])
[[1, 0, 0],
 [0, 2, 0],
 [0, 0, 3]]
```

### Decompositions

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `svd(A)` | `svd(A)` | Singular Value Decomposition |
| `eig(A)` | `eig(A)` | Eigenvalue decomposition |
| `qr(A)` | `qr(A)` | QR decomposition |
| `chol(A)` | `chol(A)` | Cholesky decomposition |
| `lu(A)` | `lu(A)` | LU decomposition |

```python
>>> A = [[1, 2], [3, 4]]

# SVD: A = U @ diag(S) @ Vt
>>> U, S, Vt = cx.linalg.svd(A)
>>> print("Singular values:", S)
[5.465, 0.366]

# Eigenvalues
>>> eigenvalues, eigenvectors = cx.linalg.eig(A)
>>> print("Eigenvalues:", eigenvalues)
[-0.372, 5.372]

# QR: A = Q @ R
>>> Q, R = cx.linalg.qr(A)

# Cholesky: A = L @ L.T (A must be positive definite)
>>> L = cx.linalg.chol([[4, 2], [2, 5]])

# LU: A = P @ L @ U
>>> L, U, P = cx.linalg.lu(A)
```

### Matrix Properties

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `det(A)` | `det(A)` | Determinant |
| `rank(A)` | `rank(A)` | Matrix rank |
| `trace(A)` | `trace(A)` | Sum of diagonal |
| `norm(A)` | `norm(A, 'fro')` | Frobenius norm |
| `cond(A)` | `cond(A)` | Condition number |

```python
>>> A = [[1, 2], [3, 4]]

>>> cx.linalg.det(A)
-2.0

>>> cx.linalg.rank(A)
2

>>> cx.linalg.trace(A)
5.0

>>> cx.linalg.norm(A)  # Frobenius norm
5.477

>>> cx.linalg.cond(A)
14.93
```

### Matrix Operations

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `inv(A)` | `inv(A)` | Matrix inverse |
| `transpose(A)` | `A'` or `transpose(A)` | Matrix transpose |
| `solve(A, b)` | `A \ b` | Solve Ax = b |
| `lstsq(A, b)` | `A \ b` (overdetermined) | Least squares |
| `matmul(A, B)` | `A * B` | Matrix multiplication |

```python
>>> A = [[1, 2], [3, 4]]
>>> b = [[5], [6]]

# Inverse
>>> A_inv = cx.linalg.inv(A)
>>> print(A_inv)
[[-2, 1], [1.5, -0.5]]

# Transpose
>>> A_T = cx.linalg.transpose(A)
>>> print(A_T)
[[1, 3], [2, 4]]

# Solve Ax = b
>>> x = cx.linalg.solve(A, b)

# Least squares (overdetermined system)
>>> A_over = [[1, 2], [3, 4], [5, 6]]
>>> b_over = [[1], [2], [3]]
>>> x = cx.linalg.lstsq(A_over, b_over)

# Matrix multiplication
>>> C = cx.linalg.matmul(A, A)
```

## Signal Processing (`cx.signal`)

### Spectral Analysis

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `fft(x)` | `fft(x)` | Fast Fourier Transform |
| `ifft(X)` | `ifft(X)` | Inverse FFT |
| `spectrogram(x)` | `spectrogram(x)` | Short-time Fourier Transform |

```python
>>> import numpy as np

# FFT
>>> x = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))
>>> result = cx.signal.fft(x, sample_rate=100)
>>> print(result.keys())
['magnitude', 'phase', 'frequencies', 'complex']

# Spectrogram
>>> spec = cx.signal.spectrogram(x, window_size=32, hop_size=16, sample_rate=100)
>>> print(spec.keys())
['S', 'frequencies', 'times']
```

### Filtering

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `lowpass(cutoff, fs)` | `lowpass` | Design lowpass filter |
| `highpass(cutoff, fs)` | `highpass` | Design highpass filter |
| `bandpass(low, high, fs)` | `bandpass` | Design bandpass filter |
| `filter(x, b, a)` | `filter(b, a, x)` | Apply filter |

```python
# Design filters
>>> b, a = cx.signal.lowpass(cutoff=10, fs=100, order=4)
>>> b, a = cx.signal.highpass(cutoff=5, fs=100, order=4)
>>> b, a = cx.signal.bandpass(low=5, high=15, fs=100, order=4)

# Apply filter
>>> filtered = cx.signal.filter(x, b, a)
```

### Convolution

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `conv(x, h)` | `conv(x, h)` | 1D convolution |
| `conv2(X, H)` | `conv2(X, H)` | 2D convolution |

```python
>>> x = [1, 2, 3, 4, 5]
>>> h = [1, 0, -1]
>>> y = cx.signal.conv(x, h, mode='same')

>>> X = np.random.randn(10, 10)
>>> H = np.ones((3, 3)) / 9  # Averaging filter
>>> Y = cx.signal.conv2(X, H, mode='same')
```

### Peak Detection

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `findpeaks(x)` | `findpeaks(x)` | Find local maxima |

```python
>>> x = [0, 1, 0, 2, 0, 3, 0]
>>> peaks = cx.signal.findpeaks(x, min_height=0.5)
>>> print(peaks['indices'])
[1, 3, 5]
>>> print(peaks['values'])
[1, 2, 3]
```

### Signal Generation

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `sine(freq, fs, n)` | `sin(2*pi*f*t)` | Generate sine wave |
| `square(freq, fs, n)` | `square(2*pi*f*t)` | Generate square wave |
| `noise(n)` | `randn(1, n)` | Generate white noise |

```python
>>> fs = 1000  # Sample rate
>>> t = 1.0    # Duration
>>> n = int(fs * t)

>>> sine_wave = cx.signal.sine(freq=10, fs=fs, n=n)
>>> square_wave = cx.signal.square(freq=5, fs=fs, n=n)
>>> white_noise = cx.signal.noise(n, amp=0.1)
```

## Statistics (`cx.stats`)

### Clustering

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `kmeans(data, k)` | `kmeans(data, k)` | K-Means clustering |
| `dbscan(data, eps)` | `dbscan(...)` | DBSCAN clustering |
| `gmm(data, k)` | `fitgmdist(...)` | Gaussian Mixture Model |

```python
>>> data = np.random.randn(100, 2)

>>> result = cx.stats.kmeans(data, k=3)
>>> print(result.keys())
['labels', 'centroids', 'inertia', 'n_iterations', 'converged']

>>> result = cx.stats.dbscan(data, eps=0.5, min_samples=5)
>>> print(result.keys())
['labels', 'n_clusters', 'n_noise']

>>> result = cx.stats.gmm(data, n_components=3)
>>> print(result.keys())
['labels', 'means', 'weights', 'aic', 'bic']
```

### Dimensionality Reduction

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `pca(data, n)` | `pca(data)` | Principal Component Analysis |
| `tsne(data, n)` | `tsne(data)` | t-SNE embedding |

```python
>>> result = cx.stats.pca(data, n_components=2)
>>> transformed = result['transformed']
>>> explained = result['explained_variance']

>>> embeddings = cx.stats.tsne(data, n_dims=2, perplexity=30)
```

### Evaluation Metrics

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `silhouette(data, labels)` | `silhouette(...)` | Silhouette score |
| `confusion_matrix(y_true, y_pred)` | `confusionmat(...)` | Confusion matrix |
| `roc(y_true, y_scores)` | `perfcurve(...)` | ROC curve and AUC |

```python
>>> y_true = [0, 0, 1, 1, 2, 2]
>>> y_pred = [0, 0, 1, 2, 2, 2]

>>> result = cx.stats.confusion_matrix(y_true, y_pred)
>>> print("Accuracy:", result['accuracy'])
>>> print("F1 Scores:", result['f1'])

>>> y_scores = [0.1, 0.4, 0.35, 0.8, 0.65, 0.9]
>>> result = cx.stats.roc(y_true, y_scores)
>>> print("AUC:", result['auc'])
```

## Time Series (`cx.timeseries`)

| Function | MATLAB Equivalent | Description |
|----------|-------------------|-------------|
| `acf(data)` | `autocorr(data)` | Autocorrelation function |
| `pacf(data)` | `parcorr(data)` | Partial autocorrelation |
| `decompose(data, period)` | `decompose(...)` | Seasonal decomposition |
| `stationarity(data)` | `adftest(...)` | Stationarity tests |
| `arima(data, horizon)` | `arima(...)` | ARIMA forecasting |
| `diff(data)` | `diff(data)` | Difference series |
| `rolling_mean(data, window)` | `movmean(data, window)` | Rolling mean |
| `rolling_std(data, window)` | `movstd(data, window)` | Rolling std |

```python
>>> data = np.cumsum(np.random.randn(100))

# Autocorrelation
>>> result = cx.timeseries.acf(data, max_lag=20)
>>> print(result.keys())
['acf', 'lags', 'confidence_upper', 'confidence_lower']

# Seasonal decomposition
>>> result = cx.timeseries.decompose(data, period=12, method='additive')
>>> trend = result['trend']
>>> seasonal = result['seasonal']
>>> residual = result['residual']

# Stationarity test
>>> result = cx.timeseries.stationarity(data)
>>> print("Is stationary:", result['is_stationary'])
>>> print("Suggested differencing:", result['suggested_differencing'])

# ARIMA forecasting
>>> forecast = cx.timeseries.arima(data, horizon=10)
>>> print(forecast.keys())
['forecast', 'lower', 'upper', 'mse', 'aic']

# Rolling statistics
>>> rm = cx.timeseries.rolling_mean(data, window=5)
>>> rs = cx.timeseries.rolling_std(data, window=5)
```

## Comparison Table

| MATLAB | CyxWiz (`pycyxwiz`) |
|--------|---------------------|
| `eye(3)` | `cx.linalg.eye(3)` |
| `zeros(3,4)` | `cx.linalg.zeros(3, 4)` |
| `A \ b` | `cx.linalg.solve(A, b)` |
| `[U,S,V] = svd(A)` | `U, S, Vt = cx.linalg.svd(A)` |
| `fft(x)` | `cx.signal.fft(x)` |
| `filter(b,a,x)` | `cx.signal.filter(x, b, a)` |
| `kmeans(X,3)` | `cx.stats.kmeans(X, 3)` |
| `autocorr(x)` | `cx.timeseries.acf(x)` |

## Usage Tips

### Import Convention

```python
import pycyxwiz as cx
import pycyxwiz.linalg as la  # Alternative for heavy linear algebra
```

### NumPy Integration

All functions accept and return NumPy-compatible data:

```python
>>> import numpy as np
>>> A = np.array([[1, 2], [3, 4]])
>>> U, S, Vt = cx.linalg.svd(A)
>>> type(U)
list  # Returns Python lists (convert with np.array if needed)
```

### Error Handling

Functions raise exceptions on failure:

```python
>>> try:
...     L = cx.linalg.chol([[1, 2], [2, 1]])  # Not positive definite
... except RuntimeError as e:
...     print("Error:", e)
Error: Matrix is not positive definite
```

---

**Back to**: [Scripting Index](index.md) | **See also**: [pycyxwiz API](../../backend/python/index.md)
