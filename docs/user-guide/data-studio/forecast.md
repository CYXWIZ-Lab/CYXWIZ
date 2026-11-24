---
title: Forecast a time series
description: Forecast a number column ahead in Explore > Forecast with exponential smoothing or ARIMA, checked on held-out rows, and keep it as a node.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - time-series.md
  - explore.md
---

Data Studio's **Explore > Forecast** tab forecasts a number column a typed
horizon ahead, with the model's prediction interval and the dates continued
from the series' step. It reads the series as the
[Series tab](time-series.md#the-time-column-the-step-and-the-period) does.
**Tools > Time Series > Forecasting** opens it.

The figures are the same as statsmodels gives (`ExponentialSmoothing`,
`ARIMA`). The smoothing filters and ARIMA's Kalman filter are sequential by
nature and run over the series on the host, as statsmodels does.

## Before you start

- A loaded dataset with a number column over time, shown in Data Studio's
  rail.

## Forecast the series

1. In **Data Studio > Explore**, open the **Forecast** tab.
2. Pick the number column; check **Time** and **Period**.
3. Pick the method:
   - **Simple** exponential smoothing;
   - **Holt** (additive or damped trend);
   - **Holt-Winters** (additive or multiplicative season);
   - **ARIMA** (orders p, d, q typed, or −1 to choose: d from the
     stationarity tests, p and q up to 3 by AIC);
   - **Seasonal naive** (each value repeats the one a period before: the
     baseline).
4. Type the horizon and the **Level** of the prediction interval (0.95:
   nineteen times in twenty). The smoothing weights are estimated
   (statsmodels' grid, then a local search) or typed.
5. Leave **Check on the last** on to score the methods on held-out rows
   (see the next section).
6. Read the page: the next step's value with its range, the value at the
   end of the horizon, the AIC, the series with the fit and the forecast
   band, the forecast table with its dates, and the held-out table.
7. Optional: **Add as a step** or **Keep as a node** adds the ARIMA or Exp.
   Smoothing node with the tab's settings (the baseline has no node).
   **Export CSV...** saves the series, the fit and the forecast rows;
   **Copy** copies the sentences.

The range is analytic for the smoothing models and comes from the state
space for ARIMA.

## Check on held-out rows

**Check on the last** H rows (on by default; twice the period found, else
12, at most a third of the rows) holds them out, fits every method on the
rest, and scores each against repeating the last cycle: "Checked on the last
24 months held out, it was 12.5% off on average: 1.2 times better than
repeating last year (15.5%); Holt-Winters did better (6.4%)." MAPE is how far
off on average as a share of the true value; RMSE is in the column's units.
Then the chosen method is fitted on everything for the forecast.

## Long series

On a long series the fits are kept within reach, with the same answers
where the references are checked:

- the smoothing grid is made coarser so it spends at most about 200 million
  point evaluations, and the local search after it at most about 400
  million (at least 200 evaluations);
- a season longer than 50 rows keeps its heuristic initial states and
  estimates the weights only (statsmodels estimates every state);
- STL takes the jumps Cleveland et al. advise above 50,000 rows;
- the ADF lag search solves every lag length from one Gram matrix on the
  device instead of one regression per lag;
- ARIMA's order search tries p and q up to 1 (not 3) above 200,000 rows.

On the public MetroPT-3 dataset (1.5 million rows at 10 seconds, a typed
period of 8,640) the parts take about 25 seconds and a Holt-Winters
forecast with its held-out check about two minutes. The tabs run in the
background with **Stop**, and the status bar says "Reading the series" or
"Forecasting". Stop ends the fit itself within a pass (a smoothing run, a
Kalman pass, an STL pass, an ADF lag), not after it.

## The nodes

The nodes read `signal_col` in table order and refuse a missing value
("fill or drop those rows first").

| Node | Parameters | Output |
| --- | --- | --- |
| ARIMA | `p`, `d`, `q` (−1 chooses), `horizon` (0: the fit only), `level`, `time_col` | the input rows plus `fitted` and `residual`, then `horizon` future rows with `forecast`, `low`, `high` |
| Exp. Smoothing | `method` simple / holt / holt_winters, `seasonal` additive / multiplicative, `period` (−1 finds it), `damped`, `alpha` / `beta` / `gamma` (−1 estimated), `horizon`, `level`, `time_col` | as ARIMA |

On the future rows every input column is empty except `time_col`, which
continues its step in its own type (a date32 by days, a timestamp by its
unit, calendar months for a monthly step, numbers by their step), so the
output can be plotted or joined as it is. The Seasonal Naive node (a
training baseline over Time Series Window rows) is a separate node.

## Related

- [Analyse a time series](time-series.md)
- [Explore your data](explore.md)
