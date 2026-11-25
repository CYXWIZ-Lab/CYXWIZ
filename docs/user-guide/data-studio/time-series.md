---
title: Analyse a time series
description: Read a number column over time in Explore > Series - its trend and season, stationarity and correlogram - and add the matching nodes.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - forecast.md
  - signal-processing.md
  - explore.md
redirect_from:
  - time-series.md
---

Data Studio's **Explore > Series** tab reads one number column over time:
what it is made of, whether it repeats, whether it drifts, and what it
remembers. The **Forecast** tab says what comes next (see
[Forecast a time series](forecast.md)). The nodes listed at the end do the
same inside a graph, so the answer runs with the rest of the pipeline.
**Tools > Time Series** opens the tabs on a look (Time Series Decomposition,
ACF / PACF (Correlogram), Stationarity Testing, Seasonality Detection,
Forecasting).

The figures are the same as statsmodels and scipy give (`acf`, `pacf`,
`adfuller`, `kpss`, `seasonal_decompose`, `STL`, `periodogram`); the backend
is checked against them on six series. The device does the FFTs, the moving
averages and the least squares of the tests; the recursions that are
sequential by nature (STL's loess passes) run over the series on the host,
as statsmodels does.

## Before you start

- A loaded dataset with a number column, shown in Data Studio's rail. A
  time column helps: set it in the Data Input's dialog (**Use the rows as >
  Time series**).

## Read the series

1. In **Data Studio > Explore**, open the **Series** tab.
2. Pick the number column. **Time** and **Period** are found for you (see
   [The time column, the step and the period](#the-time-column-the-step-and-the-period));
   type over them when they are wrong.
3. Pick a look: **Parts**, **Seasonality**, **Stationarity** or
   **Correlogram**. Each says the series ("144 months, monthly, 1949-01 ..
   1960-12; period found: 12 (50% of the detrended power)") and what was
   left out.
4. Optional: put the look's node into the graph with **Add as a step** or
   **Keep as a node** (see each look).

## The time column, the step and the period

The tab reads **Time** from the Data Input's time column, else the one date
column, else the row order. A date column (DATE, TIMESTAMP) is used as it
is; a text column is read when its values are dates (`2020-02-01 00:00:00`,
`1949-01`, `2020/02/01`, `01/02/2020`, `2020`); a number column is read as
numbers. The rows are sorted by it; rows missing the time or the value are
left out and counted; a table that was not in time order says so.

The **step** between rows is read from the dates: monthly (28 to 31 days),
quarterly, yearly, weekly, daily, hourly, every N seconds / minutes / days.
The step says what "next month" or "in 12 months" means, continues the dates
of a forecast (calendar months for a monthly step, clamped to the month's
end), and names the positions of a season (Jan .. Dec for months, Mon .. Sun
for days).

The **period** is found from the periodogram of the detrended series: the
strongest peak that is not a harmonic of a stronger one, when a seasonal
pattern of that length explains at least 30% of what is left after the
trend and the peak carries at least 5% of the detrended power. A season has
to repeat: the found period goes up to a quarter of the rows (four full
cycles). When nothing stands out, the step's own period is used (12 for
months, 4 for quarters, 7 for days, 24 for hours), else there is none.
**Period** can be typed over in both tabs and in the nodes. On a long series
dominated by slow drift (a sensor over months at 10-second steps) the daily
cycle may carry too little of the power to be found: type it (8,640 rows of
10 seconds in a day).

## The looks

**Parts** splits the series into trend, season and what is left. **STL**
(the default: loess smoothing, a season that may change over the years,
every row filled; its seasonal window is odd, 7 by default, larger smooths
the season across the cycles; **Robust** weighs outliers down) or
**Classical** (moving averages, additive or multiplicative; half a period is
empty at each end, as statsmodels leaves it). It shows the trend and
seasonal strengths (0 to 1), the series with its trend, the season and the
residual on a dated axis, and a sentence: "Passengers is almost all trend
and season (strengths 1.00 and 0.98): the trend climbs from 123 to 496; the
season peaks in Jul (+21.9) and dips in Nov (−27.7); the season grows with
the level, so a multiplicative view may fit better; what is left is within
±16.7." **Add as a step** puts a Decomposition node after the data in use,
so the rest of the graph gets `trend`, `seasonal` and `residual` columns.

**Seasonality** ranks the periods of the detrended periodogram by their
share of the power, names the harmonics ("6, 15.3%, harmonic of 12"), gives
how much a season of the found length explains, and draws the season by
position (the mean of the detrended series in each month or weekday).
**Keep as a node** adds a Seasonality Detector node as a branch.

**Stationarity** runs ADF (a unit root; lags by AIC, MacKinnon p-values)
and KPSS (a steady level; the Newey-West lags, the p clipped to 0.01 ..
0.10) on the series and on its first and second differences, around a
**Level (c)** or a **Level and trend (ct)**. A series is stationary enough
when KPSS does not reject a steady level (p ≥ 0.05) and ADF rejects a unit
root at 10% (p < 0.10); the suggested differencing is the smallest d that is
enough. Rolling mean and spread over a typed window (the period by default)
show the drift. **Add a Differencing step** puts a Differencing node at
that order after the data in use; **Keep as a node** adds a Stationarity
Test node.

**Correlogram** shows the ACF with its Bartlett band (it widens with the
lag) and the PACF with its z / √n band, to a typed number of lags, on the
series or on its first difference, the seasonal lags marked, with the
Ljung-Box test ("Q(10) 857.1, p < 0.0001": not white noise) and the orders
the cut-offs suggest for an ARIMA. **Keep as a node** adds an ACF and a PACF
node. They read the column as it is: put a Differencing step before them for
the differenced correlogram.

## The nodes

Every node reads `signal_col` in table order and refuses a missing value
("fill or drop those rows first"): a node keeps every row, so it cannot
leave rows out the way the tabs do. **Add as a step** and **Keep as a node**
fill the parameters from what the tab showed.

| Node | Parameters | Output |
| --- | --- | --- |
| Decomposition | `period` (−1 finds it), `algorithm` stl / classical, `method` additive / multiplicative (classical), `seasonal_window`, `robust` | the input plus `trend`, `seasonal`, `residual` (classical: empty at the ends) |
| ACF | `max_lag` (−1: min(n/2, 40)), `confidence` | one row per lag: `lag`, `acf`, `confidence_lower`, `confidence_upper`, `significant`, `ljung_box_q`, `ljung_box_p` |
| PACF | `max_lag`, `confidence` | `lag`, `pacf`, `confidence_lower`, `confidence_upper`, `significant` |
| Stationarity Test | `regression` c / ct, `max_lags` | one row per differencing (0, 1, 2): `adf_statistic`, `adf_pvalue`, `adf_lags`, `kpss_statistic`, `kpss_pvalue`, `kpss_lags`, `stationary`, `suggested_differencing`, `analysis` |
| Seasonality Detector | `min_period`, `max_period` (−1: a quarter of the rows) | ranked: `period`, `share`, `harmonic_of` (0: none), `is_primary`, `has_seasonality`, `strength`, `analysis` |
| Differencing | `value_col`, `lag`, `order` | the differenced column |

The forecasting nodes (ARIMA, Exp. Smoothing) are in
[Forecast a time series](forecast.md#the-nodes).

## Related

- [Forecast a time series](forecast.md)
- [Analyse a signal](signal-processing.md)
- [Explore your data](explore.md)
