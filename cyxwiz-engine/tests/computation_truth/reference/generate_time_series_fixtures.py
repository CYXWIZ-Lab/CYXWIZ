"""Truth for cyxwiz::TimeSeries (TOFIX129 D6): statsmodels and scipy on a few
series, for test_computation_truth_time_series.

Series: the airline passengers (144 months), and seeded synthetic ones: an
AR(1), a random walk, white noise, a trend with a yearly season, and a short
series of 30. For each: acf (Bartlett band), pacf (Yule-Walker), Ljung-Box,
adfuller (c / ct, lags by AIC and fixed), kpss (c / ct, auto and fixed lags),
seasonal_decompose (additive / multiplicative), STL, scipy's periodogram of
the detrended series, ExponentialSmoothing fits (simple, Holt, damped Holt,
Holt-Winters additive / multiplicative) with their fitted values, forecasts
and ETS prediction intervals, and ARIMA (state-space MLE) fits with forecasts
and intervals.

Run with the oracle venv (numpy, pandas, scipy, statsmodels 0.15):
  python generate_time_series_fixtures.py path/to/airline_passengers.csv
Writes ../fixtures/time_series_statsmodels.json.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.signal import detrend, periodogram
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

warnings.simplefilter("ignore")


def nan_none(a):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in np.asarray(a, dtype=float)]


def series_set(airline_csv):
    rng = np.random.default_rng(20261030)
    airline = pd.read_csv(airline_csv)["Passengers"].astype(float).values
    ar1 = np.empty(200)
    ar1[0] = 0.0
    e = rng.standard_normal(200)
    for t in range(1, 200):
        ar1[t] = 0.7 * ar1[t - 1] + e[t]
    ar1 += 5.0
    walk = np.cumsum(rng.standard_normal(200)) + 100.0
    noise = rng.standard_normal(150) * 2.0 + 1.0
    t = np.arange(120)
    trend_season = 50 + 0.4 * t + 8 * np.sin(2 * np.pi * t / 12) + 3 * np.cos(2 * np.pi * t / 6) + rng.standard_normal(120)
    short = 10 + np.cumsum(rng.standard_normal(30))
    return {"airline": airline, "ar1": ar1, "walk": walk, "noise": noise, "trend_season": trend_season, "short": short}


def correlogram(y):
    n = len(y)
    nlags = min(40, n // 2 - 1)
    a, aci = acf(y, nlags=nlags, alpha=0.05, fft=True)
    p, pci = pacf(y, nlags=nlags, alpha=0.05, method="ywm")
    lags = [k for k in (5, 10, 20) if k < n // 2]
    lb = acorr_ljungbox(y, lags=lags, return_df=True)
    return {"nlags": nlags, "acf": a.tolist(), "acf_ci": aci.tolist(), "pacf": p.tolist(), "pacf_ci": pci.tolist(),
            "ljung_box": {"lags": lags, "stat": lb["lb_stat"].tolist(), "p": lb["lb_pvalue"].tolist()}}


def stationarity(y):
    out = {}
    for reg in ("c", "ct"):
        r = adfuller(y, regression=reg, autolag="AIC", result_object=True)
        out[f"adf_{reg}"] = {"stat": r.statistic, "p": r.pvalue, "lags": int(r.lags), "nobs": int(r.nobs),
                             "critical": {k: float(v) for k, v in r.critical_values.items()}, "icbest": float(r.icbest)}
        k = kpss(y, regression=reg, nlags="auto", result_object=True)
        out[f"kpss_{reg}"] = {"stat": k.statistic, "p": k.pvalue, "lags": int(k.lags), "critical": {kk: float(v) for kk, v in k.critical_values.items()}}
    fixed = adfuller(y, maxlag=3, regression="c", autolag=None, result_object=True)
    out["adf_c_lag3"] = {"stat": fixed.statistic, "p": fixed.pvalue, "lags": int(fixed.lags), "nobs": int(fixed.nobs)}
    k4 = kpss(y, regression="c", nlags=4, result_object=True)
    out["kpss_c_lag4"] = {"stat": k4.statistic, "p": k4.pvalue}
    return out


def decomposition(y, period):
    out = {}
    for model in ("additive", "multiplicative"):
        d = seasonal_decompose(y, model=model, period=period)
        out[model] = {"trend": nan_none(d.trend), "seasonal": d.seasonal.tolist(), "resid": nan_none(d.resid)}
    for name, kw in (("stl", {}), ("stl_s13", {"seasonal": 13}), ("stl_robust", {"robust": True})):
        s = STL(y, period=period, **kw).fit()
        out[name] = {"trend": s.trend.tolist(), "seasonal": s.seasonal.tolist(), "resid": s.resid.tolist(),
                     "config": {k: (int(v) if not isinstance(v, bool) else bool(v)) for k, v in s.config.items()} if hasattr(s, "config") else None}
    f, pxx = periodogram(detrend(y), scaling="spectrum")
    out["periodogram"] = {"freq": f.tolist(), "power": pxx.tolist()}
    return out


def smoothing(y, period):
    out = {}
    cases = {"simple": dict(trend=None, seasonal=None), "holt": dict(trend="add", seasonal=None),
             "holt_damped": dict(trend="add", seasonal=None, damped_trend=True)}
    if period:
        cases["hw_add"] = dict(trend="add", seasonal="add", seasonal_periods=period)
        cases["hw_mul"] = dict(trend="add", seasonal="mul", seasonal_periods=period)
    h = 24 if period else 12
    for name, kw in cases.items():
        m = ExponentialSmoothing(y, initialization_method="estimated", **kw).fit()
        p = m.params
        out[name] = {"alpha": float(p["smoothing_level"]), "beta": None if p["smoothing_trend"] is None or np.isnan(p["smoothing_trend"]) else float(p["smoothing_trend"]),
                     "gamma": None if p["smoothing_seasonal"] is None or np.isnan(p["smoothing_seasonal"]) else float(p["smoothing_seasonal"]),
                     "phi": None if np.isnan(p["damping_trend"]) else float(p["damping_trend"]),
                     "initial_level": float(p["initial_level"]), "initial_trend": None if p["initial_trend"] is None or np.isnan(p["initial_trend"]) else float(p["initial_trend"]),
                     "initial_seasons": None if p["initial_seasons"] is None else [float(v) for v in np.atleast_1d(p["initial_seasons"])],
                     "sse": float(m.sse), "aic": float(m.aic), "bic": float(m.bic), "fitted": m.fittedvalues.tolist(), "forecast": m.forecast(h).tolist()}
    # The additive-error prediction interval from the fit itself (Hyndman et al. 2008, class 1):
    # var_h = sigma2 (1 + sum_{j<h} c_j^2), c_j = alpha + beta phi_j + gamma 1[j mod m = 0], sigma2 = SSE / n.
    from scipy.stats import norm
    z = norm.ppf(0.975)
    for name, kw in cases.items():
        v = out[name]
        if kw.get("seasonal") == "mul":
            continue
        n = len(y)
        sigma2 = v["sse"] / n
        alpha, beta, gamma, phi = v["alpha"], v["beta"] or 0.0, v["gamma"] or 0.0, v["phi"] if v["phi"] is not None else 1.0
        m = kw.get("seasonal_periods", 0)
        lower, upper, total, phi_sum, phi_pow = [], [], 0.0, 0.0, 1.0
        for hh in range(1, h + 1):
            half = z * np.sqrt(sigma2 * (1.0 + total))
            lower.append(v["forecast"][hh - 1] - half)
            upper.append(v["forecast"][hh - 1] + half)
            phi_pow *= phi
            phi_sum += phi_pow
            c = alpha + (beta * phi_sum if kw.get("trend") else 0.0) + (gamma if m and hh % m == 0 else 0.0)
            total += c * c
        v["interval"] = {"lower": lower, "upper": upper}
    # ETS prediction intervals (analytic for the linear models; simulated for the multiplicative season).
    ys = pd.Series(y, index=pd.RangeIndex(len(y)))
    ets_cases = {"simple": dict(error="add", trend=None, seasonal=None), "holt": dict(error="add", trend="add", seasonal=None),
                 "holt_damped": dict(error="add", trend="add", seasonal=None, damped_trend=True)}
    if period:
        ets_cases["hw_add"] = dict(error="add", trend="add", seasonal="add", seasonal_periods=period)
        ets_cases["hw_mul"] = dict(error="add", trend="add", seasonal="mul", seasonal_periods=period)
    for name, kw in ets_cases.items():
        e = ETSModel(ys, **kw).fit(disp=False)
        np.random.seed(1)  # the multiplicative season simulates its interval
        pred = e.get_prediction(start=len(y), end=len(y) + h - 1, simulate_repetitions=10000)
        sf = pred.summary_frame(alpha=0.05)
        out[name]["ets"] = {"alpha": float(e.alpha), "beta": float(e.beta) if kw.get("trend") else None,
                            "gamma": float(e.gamma) if kw.get("seasonal") else None, "phi": float(e.phi) if kw.get("damped_trend") else None,
                            "sigma2": float(e.mse), "aic": float(e.aic), "forecast": sf["mean"].tolist(),
                            "lower": sf["pi_lower"].tolist(), "upper": sf["pi_upper"].tolist(),
                            "analytic": kw.get("seasonal") != "mul"}
    return out


def arima(y, orders, h=24):
    out = {}
    for order in orders:
        m = ARIMA(y, order=order).fit()
        fc = m.get_forecast(h)
        ci = fc.conf_int(alpha=0.05)
        out["%d,%d,%d" % order] = {"params": {k: float(v) for k, v in zip(m.param_names, m.params)}, "llf": float(m.llf), "aic": float(m.aic),
                                   "bic": float(m.bic), "fitted": m.fittedvalues.tolist(), "forecast": fc.predicted_mean.tolist(),
                                   "lower": ci[:, 0].tolist(), "upper": ci[:, 1].tolist(), "nobs": int(m.nobs)}
    return out


def main():
    series = series_set(sys.argv[1])
    out = {"series": {k: v.tolist() for k, v in series.items()}, "cases": {}}
    for name, y in series.items():
        case = {"correlogram": correlogram(y)}
        if name != "short":
            case["stationarity"] = stationarity(y)
        if name in ("airline", "trend_season"):
            case["decomposition"] = decomposition(y, 12)
            case["smoothing"] = smoothing(y, 12)
        elif name in ("ar1", "walk"):
            case["smoothing"] = smoothing(y, 0)
        out["cases"][name] = case
        print(name, "done")
    out["cases"]["airline"]["arima"] = arima(series["airline"], [(2, 1, 1), (1, 1, 0), (0, 1, 1)])
    out["cases"]["ar1"]["arima"] = arima(series["ar1"], [(1, 0, 0), (2, 0, 0), (1, 0, 1), (0, 0, 1), (2, 0, 2)])
    out["cases"]["walk"]["arima"] = arima(series["walk"], [(0, 1, 0), (0, 1, 1), (1, 1, 0)])
    out["cases"]["trend_season"]["arima"] = arima(series["trend_season"], [(1, 1, 1)])
    out["versions"] = {"numpy": np.__version__, "statsmodels": __import__("statsmodels").__version__, "scipy": __import__("scipy").__version__}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "time_series_statsmodels.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", os.path.normpath(path), os.path.getsize(path) // 1024, "KB")
    a = out["cases"]["airline"]
    print("airline: adf_c %.3f p %.4f lags %d; kpss_c %.3f p %.3f; stl trend[0] %.2f; hw_mul alpha %.3f sse %.1f aic %.1f; arima211 aic %.1f" % (
        a["stationarity"]["adf_c"]["stat"], a["stationarity"]["adf_c"]["p"], a["stationarity"]["adf_c"]["lags"],
        a["stationarity"]["kpss_c"]["stat"], a["stationarity"]["kpss_c"]["p"], a["decomposition"]["stl"]["trend"][0],
        a["smoothing"]["hw_mul"]["alpha"], a["smoothing"]["hw_mul"]["sse"], a["smoothing"]["hw_mul"]["aic"], a["arima"]["2,1,1"]["aic"]))


if __name__ == "__main__":
    main()
