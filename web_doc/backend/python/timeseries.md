# Time Series (`cx.timeseries`)

The `timeseries` submodule provides functions for time series analysis, decomposition, stationarity testing, and forecasting.

## Overview

```python
import pycyxwiz as cx
import numpy as np

# Generate time series
data = np.cumsum(np.random.randn(100)).tolist()

# Autocorrelation
acf_result = cx.timeseries.acf(data, max_lag=20)

# Decomposition
decomp = cx.timeseries.decompose(data, period=12)

# Forecasting
forecast = cx.timeseries.arima(data, horizon=10)
```

## Autocorrelation Functions

### `acf(data, max_lag=-1)`

Compute autocorrelation function.

```python
result = cx.timeseries.acf(data, max_lag=20)

print("ACF values:", result['acf'][:5])
print("Lags:", result['lags'][:5])
print("Upper confidence:", result['confidence_upper'][:5])
print("Lower confidence:", result['confidence_lower'][:5])
```

**Parameters:**
- `data` (list): Time series data
- `max_lag` (int): Maximum lag to compute. Default: -1 (auto = len/4)

**Returns:** dict with keys:
- `acf`: List of autocorrelation values
- `lags`: List of lag values
- `confidence_upper`: Upper 95% confidence bound
- `confidence_lower`: Lower 95% confidence bound

**Example - ACF Plot:**
```python
import matplotlib.pyplot as plt

result = cx.timeseries.acf(data, max_lag=30)

plt.figure(figsize=(10, 4))
plt.bar(result['lags'], result['acf'], width=0.8, color='blue', alpha=0.7)
plt.axhline(y=result['confidence_upper'][0], color='r', linestyle='--')
plt.axhline(y=result['confidence_lower'][0], color='r', linestyle='--')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.show()
```

---

### `pacf(data, max_lag=-1)`

Compute partial autocorrelation function.

```python
pacf_values = cx.timeseries.pacf(data, max_lag=20)
# Returns list of PACF values
```

**Parameters:**
- `data` (list): Time series data
- `max_lag` (int): Maximum lag. Default: -1 (auto)

**Returns:** list of PACF values

**Use case:** PACF helps determine the order `p` for AR models. Significant spikes at lags 1-p suggest AR(p) process.

## Decomposition

### `decompose(data, period, method="additive")`

Seasonal decomposition into trend, seasonal, and residual components.

```python
# Monthly data with seasonal pattern
period = 12

result = cx.timeseries.decompose(data, period=period, method="additive")

trend = result['trend']       # Long-term trend
seasonal = result['seasonal']  # Seasonal component
residual = result['residual']  # Random residuals
```

**Parameters:**
- `data` (list): Time series data
- `period` (int): Seasonal period (e.g., 12 for monthly, 7 for daily)
- `method` (str): Decomposition method. Options:
  - `"additive"`: data = trend + seasonal + residual
  - `"multiplicative"`: data = trend * seasonal * residual

**Returns:** dict with keys:
- `trend`: List, trend component
- `seasonal`: List, seasonal component
- `residual`: List, residual component

**Example - Decomposition Plot:**
```python
import matplotlib.pyplot as plt

# Generate seasonal data
t = np.arange(100)
trend = 0.1 * t
seasonal = np.sin(2 * np.pi * t / 12)  # Period 12
noise = 0.3 * np.random.randn(100)
data = (trend + seasonal + noise).tolist()

# Decompose
result = cx.timeseries.decompose(data, period=12, method="additive")

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(data)
axes[0].set_title('Original')

axes[1].plot(result['trend'])
axes[1].set_title('Trend')

axes[2].plot(result['seasonal'])
axes[2].set_title('Seasonal')

axes[3].plot(result['residual'])
axes[3].set_title('Residual')

plt.tight_layout()
plt.show()
```

## Stationarity Testing

### `stationarity(data)`

Test for stationarity using ADF and KPSS tests.

```python
result = cx.timeseries.stationarity(data)

print("Is Stationary:", result['is_stationary'])
print("\nADF Test:")
print(f"  Statistic: {result['adf_statistic']:.4f}")
print(f"  P-value: {result['adf_pvalue']:.4f}")
print("\nKPSS Test:")
print(f"  Statistic: {result['kpss_statistic']:.4f}")
print(f"  P-value: {result['kpss_pvalue']:.4f}")
print(f"\nSuggested differencing: {result['suggested_differencing']}")
```

**Parameters:**
- `data` (list): Time series data

**Returns:** dict with keys:
- `is_stationary`: Boolean, overall stationarity assessment
- `adf_statistic`: ADF test statistic
- `adf_pvalue`: ADF test p-value
- `kpss_statistic`: KPSS test statistic
- `kpss_pvalue`: KPSS test p-value
- `suggested_differencing`: int, suggested order of differencing (0, 1, or 2)

**Interpretation:**
- **ADF Test**: Low p-value (< 0.05) suggests stationary
- **KPSS Test**: Low p-value (< 0.05) suggests non-stationary
- Conflicting results may indicate trend-stationarity

**Example - Making Data Stationary:**
```python
# Check stationarity
result = cx.timeseries.stationarity(data)

if not result['is_stationary']:
    d = result['suggested_differencing']
    print(f"Apply differencing of order {d}")

    # Apply differencing
    data_diff = cx.timeseries.diff(data, order=d)

    # Verify
    result_diff = cx.timeseries.stationarity(data_diff)
    print(f"After differencing: stationary = {result_diff['is_stationary']}")
```

## Forecasting

### `arima(data, horizon, p=-1, d=-1, q=-1)`

ARIMA forecasting with optional auto-selection of parameters.

```python
# Forecast next 10 periods
result = cx.timeseries.arima(data, horizon=10)

print("Forecast:", result['forecast'])
print("Lower bound:", result['lower'])
print("Upper bound:", result['upper'])
print("MSE:", result['mse'])
print("AIC:", result['aic'])
```

**Parameters:**
- `data` (list): Time series data
- `horizon` (int): Number of periods to forecast
- `p` (int): AR order. Default: -1 (auto-select)
- `d` (int): Differencing order. Default: -1 (auto-select)
- `q` (int): MA order. Default: -1 (auto-select)

**Returns:** dict with keys:
- `forecast`: List of forecasted values
- `lower`: List, lower confidence bound
- `upper`: List, upper confidence bound
- `mse`: Mean squared error on training data
- `aic`: Akaike Information Criterion

**Example - ARIMA Forecast Plot:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data with trend
t = np.arange(100)
data = (0.1 * t + 2 * np.sin(2 * np.pi * t / 20) + 0.5 * np.random.randn(100)).tolist()

# Forecast
horizon = 20
result = cx.timeseries.arima(data, horizon=horizon)

# Plot
plt.figure(figsize=(12, 6))

# Historical data
plt.plot(range(len(data)), data, 'b-', label='Historical')

# Forecast
future_idx = range(len(data), len(data) + horizon)
plt.plot(future_idx, result['forecast'], 'r-', linewidth=2, label='Forecast')

# Confidence interval
plt.fill_between(future_idx, result['lower'], result['upper'],
                color='red', alpha=0.2, label='95% CI')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('ARIMA Forecast')
plt.legend()
plt.grid(True)
plt.show()
```

**Specifying ARIMA Order:**
```python
# Use ACF/PACF to determine parameters
acf_result = cx.timeseries.acf(data, max_lag=20)
pacf_values = cx.timeseries.pacf(data, max_lag=20)

# Example: ARIMA(2, 1, 1)
result = cx.timeseries.arima(data, horizon=10, p=2, d=1, q=1)
```

## Preprocessing

### `diff(data, order=1)`

Compute differenced series.

```python
# First difference (removes linear trend)
data_diff1 = cx.timeseries.diff(data, order=1)
# len(data_diff1) = len(data) - 1

# Second difference (removes quadratic trend)
data_diff2 = cx.timeseries.diff(data, order=2)
```

**Parameters:**
- `data` (list): Time series data
- `order` (int): Differencing order. Default: 1

**Returns:** list, differenced series

---

### `rolling_mean(data, window)`

Compute rolling (moving) average.

```python
# 7-day moving average
ma = cx.timeseries.rolling_mean(data, window=7)
# First (window-1) values are NaN
```

**Parameters:**
- `data` (list): Time series data
- `window` (int): Window size

**Returns:** list, rolling mean values

---

### `rolling_std(data, window)`

Compute rolling standard deviation.

```python
# 7-day rolling volatility
std = cx.timeseries.rolling_std(data, window=7)
```

**Parameters:**
- `data` (list): Time series data
- `window` (int): Window size

**Returns:** list, rolling standard deviation values

**Example - Rolling Statistics:**
```python
import matplotlib.pyplot as plt

window = 10
ma = cx.timeseries.rolling_mean(data, window=window)
std = cx.timeseries.rolling_std(data, window=window)

plt.figure(figsize=(12, 6))
plt.plot(data, 'b-', alpha=0.5, label='Original')
plt.plot(ma, 'r-', linewidth=2, label=f'{window}-period MA')
plt.fill_between(range(len(data)),
                [m - 2*s if m and s else None for m, s in zip(ma, std)],
                [m + 2*s if m and s else None for m, s in zip(ma, std)],
                color='red', alpha=0.2, label='±2σ')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Rolling Mean and Standard Deviation')
plt.show()
```

## Complete Example

```python
import pycyxwiz as cx
import numpy as np
import matplotlib.pyplot as plt

# Generate realistic time series
np.random.seed(42)
n = 200

# Components
t = np.arange(n)
trend = 0.05 * t
seasonal = 3 * np.sin(2 * np.pi * t / 30)  # Period 30
noise = np.random.randn(n)
data = (trend + seasonal + noise).tolist()

print("=" * 60)
print("Time Series Analysis")
print("=" * 60)

# 1. Stationarity Test
print("\n1. Stationarity Test")
stat_result = cx.timeseries.stationarity(data)
print(f"   Stationary: {stat_result['is_stationary']}")
print(f"   ADF p-value: {stat_result['adf_pvalue']:.4f}")
print(f"   Suggested differencing: {stat_result['suggested_differencing']}")

# 2. Decomposition
print("\n2. Seasonal Decomposition")
decomp = cx.timeseries.decompose(data, period=30, method="additive")
print(f"   Trend range: {min(decomp['trend']):.2f} to {max(decomp['trend']):.2f}")
print(f"   Seasonal amplitude: {max(decomp['seasonal']) - min(decomp['seasonal']):.2f}")

# 3. ACF Analysis
print("\n3. Autocorrelation")
acf_result = cx.timeseries.acf(data, max_lag=40)
significant_lags = [i for i, v in enumerate(acf_result['acf'])
                   if abs(v) > acf_result['confidence_upper'][0]]
print(f"   Significant lags: {significant_lags[:10]}")

# 4. Forecasting
print("\n4. ARIMA Forecast")
horizon = 30
arima_result = cx.timeseries.arima(data, horizon=horizon)
print(f"   MSE: {arima_result['mse']:.4f}")
print(f"   AIC: {arima_result['aic']:.4f}")

# 5. Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Original data
axes[0, 0].plot(data)
axes[0, 0].set_title('Original Time Series')

# Decomposition - Trend
axes[0, 1].plot(decomp['trend'], 'r-')
axes[0, 1].set_title('Trend Component')

# Decomposition - Seasonal
axes[1, 0].plot(decomp['seasonal'], 'g-')
axes[1, 0].set_title('Seasonal Component')

# ACF
axes[1, 1].bar(acf_result['lags'], acf_result['acf'], width=0.8, alpha=0.7)
axes[1, 1].axhline(y=acf_result['confidence_upper'][0], color='r', linestyle='--')
axes[1, 1].axhline(y=acf_result['confidence_lower'][0], color='r', linestyle='--')
axes[1, 1].set_title('Autocorrelation Function')
axes[1, 1].set_xlabel('Lag')

# Forecast
future_idx = list(range(len(data), len(data) + horizon))
axes[2, 0].plot(range(len(data)), data, 'b-', label='Historical')
axes[2, 0].plot(future_idx, arima_result['forecast'], 'r-', linewidth=2, label='Forecast')
axes[2, 0].fill_between(future_idx, arima_result['lower'], arima_result['upper'],
                        color='red', alpha=0.2)
axes[2, 0].legend()
axes[2, 0].set_title('ARIMA Forecast')

# Residuals histogram
axes[2, 1].hist(decomp['residual'], bins=30, edgecolor='black', alpha=0.7)
axes[2, 1].set_title('Residual Distribution')
axes[2, 1].set_xlabel('Residual Value')
axes[2, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

---

**Next**: [Activations](activations.md) | [Back to Index](index.md)
