# Time series

One number column over time: what it is made of, whether it repeats, whether
it drifts, what it remembers, and what comes next. Data Studio's Explore tabs
**Series** and **Forecast** answer those on any table ([Data
Studio](data_studio.md#series)); the nodes below do the same inside a graph,
so the answer runs with the rest of the pipeline. Tools > Time Series opens
the tabs.

The figures are the same as statsmodels and scipy give (`acf`, `pacf`,
`adfuller`, `kpss`, `seasonal_decompose`, `STL`, `ExponentialSmoothing`,
`ARIMA`, `periodogram`): the backend is checked against them on six series
(airline passengers, an AR(1), a random walk, white noise, a trend with a
season, a short series) in `test_computation_truth_time_series`. The device
does the FFTs (autocovariance, periodogram), the moving averages and the
least squares of the tests; the recursions that are sequential by nature
(the smoothing filters, STL's loess passes, ARIMA's Kalman filter) run over
the series on the host, as statsmodels does.

## The time column, the step and the period

The tabs read **Time** from the Data Input's time column (Use the rows as >
Time series), else the one date column, else the row order. A date column
(DATE, TIMESTAMP) is used as it is; a text column is read when its values
are dates (`2020-02-01 00:00:00`, `1949-01`, `2020/02/01`, `01/02/2020`,
`2020`); a number column is read as numbers. The rows are sorted by it; rows
missing the time or the value are left out and counted; a table that was not
in time order says so.

The **step** between rows is read from the dates: monthly (28 to 31 days),
quarterly, yearly, weekly, daily, hourly, every N seconds / minutes / days.
The step says what "next month" or "in 12 months" means, continues the dates
of a forecast (calendar months for a monthly step, clamped to the month's
end), and names the positions of a season (Jan .. Dec for months, Mon .. Sun
for days).

The **period** is found from the periodogram of the detrended series: the
strongest peak that is not a harmonic of a stronger one, when a seasonal
pattern of that length explains at least 30% of what is left after the trend
and the peak carries at least 5% of the detrended power. A season has to
repeat: the found period goes up to a quarter of the rows (four full cycles).
When nothing stands out, the step's own period is used (12 for months, 4 for
quarters, 7 for days, 24 for hours), else there is none; **Period** can be
typed over in both tabs and in the nodes. On a long series dominated by slow
drift (a sensor over months at 10-second steps) the daily cycle may carry too
little of the power to be found: type it (8,640 rows of 10 seconds in a day).

## Series

Four looks on the same series; every one says the series ("144 months,
monthly, 1949-01 .. 1960-12; period found: 12 (50% of the detrended power)")
and what was left out.

**Parts** splits it into trend, season and what is left. **STL** (the default:
loess smoothing, a season that may change over the years, every row filled;
its seasonal window is odd, 7 by default, larger smooths the season across
the cycles; **Robust** weighs outliers down) or **Classical** (moving averages,
additive or multiplicative; half a period is empty at each end, as
statsmodels leaves it). It shows the trend and seasonal strengths (0 to 1),
the series with its trend, the season and the residual on a dated axis, and
a sentence: "Passengers is almost all trend and season (strengths 1.00 and
0.98): the trend climbs from 123 to 496; the season peaks in Jul (+21.9) and
dips in Nov (−27.7); the season grows with the level (its swing goes from
49.5 to 237), so a multiplicative view may fit better; what is left is
within ±16.7." **Add as a step** puts a Decomposition node after the data in
use, so the rest of the graph gets `trend`, `seasonal` and `residual`
columns.

**Seasonality** ranks the periods of the detrended periodogram by their
share of the power, names the harmonics ("6, 15.3%, harmonic of 12"), gives
how much a season of the found length explains, and draws the season by
position (the mean of the detrended series in each month or weekday).
**Keep as a node** adds a Seasonality Detector node as a branch.

**Stationarity** runs ADF (a unit root; lags by AIC, MacKinnon p-values)
and KPSS (a steady level; the Newey-West lags, the p clipped to 0.01 .. 0.10)
on the series and on its first and second differences, around a **Level
(c)** or a **Level and trend (ct)**. A series is stationary enough when KPSS
does not reject a steady level (p ≥ 0.05) and ADF rejects a unit root at 10%
(p < 0.10); the suggested differencing is the smallest d that is enough.
Rolling mean and spread over a typed window (the period by default) show the
drift. **Add a Differencing step** puts a Differencing node at that order
after the data in use; **Keep as a node** adds a Stationarity Test node.

**Correlogram** shows the ACF with its Bartlett band (it widens with the
lag) and the PACF with its z / √n band, to a typed number of lags, on the
series or on its first difference, the seasonal lags marked, with the
Ljung-Box test ("Q(10) 857.1, p < 0.0001": not white noise) and the orders
the cut-offs suggest for an ARIMA. **Keep as a node** adds an ACF and a PACF
node (they read the column as it is: put a Differencing step before them
for the differenced correlogram).

## Forecast

**Simple** exponential smoothing, **Holt** (additive or damped trend),
**Holt-Winters** (additive or multiplicative season), **ARIMA** (orders p, d,
q typed, or −1 to choose: d from the stationarity tests, p and q up to 3 by
AIC) and **Seasonal naive** (each value repeats the one a period before: the
baseline). The horizon is typed; the range is the model's prediction interval
at the typed **Level** (0.95: nineteen times in twenty), analytic for the
smoothing models, from the state space for ARIMA. The smoothing weights are
estimated (statsmodels' grid, then a local search) or typed.

**Check on the last** H rows (on by default; twice the period found, else 12,
at most a third of the rows) holds them out, fits every method on the rest,
and scores each against repeating last cycle: "Checked on the last 24 months
held out, it was 12.5% off on average: 1.2 times better than repeating last
year (15.5%); Holt-Winters did better (6.4%)." MAPE is how far off on
average as a share of the true value; RMSE is in the column's units. Then
the chosen method is fitted on everything for the forecast.

The page shows the next step's value with its range, the value at the end
of the horizon, the AIC, the series with the fit and the forecast band, the
forecast table with its dates, and the held-out table. **Add as a step** /
**Keep as a node** add the ARIMA or Exp. Smoothing node with the tab's
settings (the baseline has no node); **Export CSV...** saves the series, the
fit and the forecast rows; **Copy** copies the sentences. A long search
(ARIMA with the orders chosen on a long series) shows **Stop**.

On a long series the fits are kept within reach, with the same answers
where the references are checked: the smoothing grid is made coarser so it
spends at most about 200 million point evaluations and the local search
after it at most about 400 million (at least 200 evaluations); a season
longer than 50 rows keeps its heuristic initial states and estimates the
weights only (statsmodels estimates every state, which is hopeless over a
day of seconds); STL takes the jumps Cleveland et al. advise above 50,000
rows; the ADF lag search solves every lag length from one Gram matrix on
the device instead of one regression per lag; ARIMA's order search tries p
and q up to 1 (not 3) above 200,000 rows. On MetroPT-3 (1.5 million rows at
10 seconds, a typed period of 8,640) the parts take about 25 seconds and a
Holt-Winters forecast with its held-out check about two minutes. The tabs
run in the background with **Stop**, and the status bar says "Reading the
series" or "Forecasting".

## The nodes

Every node reads `signal_col` in table order and refuses a missing value
("fill or drop those rows first"): a node keeps every row, so it cannot leave
rows out the way the tabs do. The tabs' "Add as a step" / "Keep as a node"
fill the parameters from what they showed.

| Node | Parameters | Output |
| --- | --- | --- |
| Decomposition | `period` (−1 finds it), `algorithm` stl / classical, `method` additive / multiplicative (classical), `seasonal_window`, `robust` | the input plus `trend`, `seasonal`, `residual` (classical: empty at the ends) |
| ACF | `max_lag` (−1: min(n/2, 40)), `confidence` | one row per lag: `lag`, `acf`, `confidence_lower`, `confidence_upper`, `significant`, `ljung_box_q`, `ljung_box_p` |
| PACF | `max_lag`, `confidence` | `lag`, `pacf`, `confidence_lower`, `confidence_upper`, `significant` |
| Stationarity Test | `regression` c / ct, `max_lags` | one row per differencing (0, 1, 2): `adf_statistic`, `adf_pvalue`, `adf_lags`, `kpss_statistic`, `kpss_pvalue`, `kpss_lags`, `stationary`, `suggested_differencing`, `analysis` |
| Seasonality Detector | `min_period`, `max_period` (−1: a quarter of the rows) | ranked: `period`, `share`, `harmonic_of` (0: none), `is_primary`, `has_seasonality`, `strength`, `analysis` |
| ARIMA | `p`, `d`, `q` (−1 chooses), `horizon` (0: the fit only), `level`, `time_col` | the input rows plus `fitted` and `residual`, then `horizon` future rows with `forecast`, `low`, `high` |
| Exp. Smoothing | `method` simple / holt / holt_winters, `seasonal` additive / multiplicative, `period` (−1 finds it), `damped`, `alpha` / `beta` / `gamma` (−1 estimated), `horizon`, `level`, `time_col` | as ARIMA |
| Differencing | `value_col`, `lag`, `order` | the differenced column |

On the future rows every input column is empty except `time_col`, which
continues its step in its own type (a date32 by days, a timestamp by its
unit, calendar months for a monthly step, numbers by their step), so the
output can be plotted or joined as it is. The Seasonal Naive node (a training
baseline over Time Series Window rows) is unchanged.
