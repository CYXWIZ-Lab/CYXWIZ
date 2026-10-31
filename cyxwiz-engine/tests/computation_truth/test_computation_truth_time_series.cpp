// Truth for cyxwiz::TimeSeries (TOFIX129 D6) against statsmodels / scipy
// fixtures (reference/generate_time_series_fixtures.py): the correlogram
// (ACF with Bartlett's band, Yule-Walker PACF, Ljung-Box), ADF and KPSS, the
// classical and STL decompositions, the periodogram, exponential smoothing
// fits with their forecasts and ETS intervals, ARIMA fits with forecasts and
// intervals; and the rules of our own (seasonality detection, the suggested
// differencing, the seasonal naive baseline, refusals).

#include "test_device_selection.h"

#include <cyxwiz/time_series.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using nlohmann::json;
using cyxwiz::TimeSeries;

namespace {

int g_checks = 0;

void Check(bool condition, const std::string& message) {
    ++g_checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Near(double actual, double expected, double tolerance, const std::string& what) {
    const bool both_nan = std::isnan(actual) && std::isnan(expected);
    Check(both_nan || std::fabs(actual - expected) <= tolerance * (1.0 + std::fabs(expected)),
          what + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}

std::vector<double> Doubles(const json& j) {
    std::vector<double> out;
    for (const auto& v : j) out.push_back(v.is_null() ? NAN : v.get<double>());
    return out;
}

void NearAll(const std::vector<double>& actual, const json& expected, double tolerance, const std::string& what, size_t from = 0) {
    const std::vector<double> e = Doubles(expected);
    Check(actual.size() == e.size(), what + ": size " + std::to_string(actual.size()) + " vs " + std::to_string(e.size()));
    for (size_t i = from; i < e.size(); ++i) Near(actual[i], e[i], tolerance, what + "[" + std::to_string(i) + "]");
}

double Tol(double f64, double f32) { return cyxwiz::test::DeviceTolerance(f64, f32); }

void CheckCorrelogram(const std::string& name, const std::vector<double>& y, const json& c) {
    const int nlags = c["nlags"].get<int>();
    const auto r = TimeSeries::ComputeACFPACF(y, nlags, 0.95);
    Check(r.success, name + " correlogram: " + r.error_message);
    NearAll(r.acf, c["acf"], Tol(1e-9, 1e-4), name + " acf");
    NearAll(r.pacf, c["pacf"], Tol(1e-7, 1e-3), name + " pacf");
    const auto aci = c["acf_ci"], pci = c["pacf_ci"];
    for (size_t k = 0; k < r.acf.size(); ++k) {
        Near(r.confidence_lower[k], aci[k][0].get<double>(), Tol(1e-7, 1e-3), name + " acf band low " + std::to_string(k));
        Near(r.confidence_upper[k], aci[k][1].get<double>(), Tol(1e-7, 1e-3), name + " acf band high " + std::to_string(k));
        if (k > 0) {
            Near(r.pacf_confidence_lower[k], pci[k][0].get<double>(), Tol(1e-7, 1e-3), name + " pacf band low " + std::to_string(k));
            Near(r.pacf_confidence_upper[k], pci[k][1].get<double>(), Tol(1e-7, 1e-3), name + " pacf band high " + std::to_string(k));
        }
    }
    const auto lb = c["ljung_box"];
    for (size_t i = 0; i < lb["lags"].size(); ++i) {
        const int lag = lb["lags"][i].get<int>();
        Near(r.ljung_box_q[static_cast<size_t>(lag)], lb["stat"][i].get<double>(), Tol(1e-8, 1e-3), name + " Ljung-Box Q(" + std::to_string(lag) + ")");
        const double p = lb["p"][i].get<double>(), ours = r.ljung_box_p[static_cast<size_t>(lag)];
        Check(p < 1e-12 ? ours < 1e-10 : std::fabs(ours - p) <= Tol(1e-8, 1e-4) * (1.0 + p), name + " Ljung-Box p(" + std::to_string(lag) + ")");
    }
}

void CheckStationarity(const std::string& name, const std::vector<double>& y, const json& s) {
    for (const std::string reg : {"c", "ct"}) {
        const auto& e = s["adf_" + reg];
        const auto adf = TimeSeries::ADFTest(y, -1, reg, true);
        Check(adf.success, name + " ADF " + reg + ": " + adf.error_message);
        Check(adf.adf_lags == e["lags"].get<int>() && adf.adf_nobs == e["nobs"].get<int>(),
              name + " ADF " + reg + " lags " + std::to_string(adf.adf_lags) + " nobs " + std::to_string(adf.adf_nobs));
        Near(adf.adf_statistic, e["stat"].get<double>(), Tol(1e-7, 1e-3), name + " ADF " + reg + " statistic");
        Near(adf.adf_pvalue, e["p"].get<double>(), Tol(1e-7, 1e-3), name + " ADF " + reg + " p");
        Near(adf.adf_icbest, e["icbest"].get<double>(), Tol(1e-7, 1e-3), name + " ADF " + reg + " AIC");
        for (const auto& [level, value] : e["critical"].items()) Near(adf.adf_critical.at(level), value.get<double>(), 1e-10, name + " ADF critical " + level);
        const auto& k = s["kpss_" + reg];
        const auto kpss = TimeSeries::KPSSTest(y, reg, -1);
        Check(kpss.success, name + " KPSS " + reg + ": " + kpss.error_message);
        Check(kpss.kpss_lags == k["lags"].get<int>(), name + " KPSS " + reg + " lags " + std::to_string(kpss.kpss_lags));
        Near(kpss.kpss_statistic, k["stat"].get<double>(), Tol(1e-7, 1e-3), name + " KPSS " + reg + " statistic");
        Near(kpss.kpss_pvalue, k["p"].get<double>(), Tol(1e-7, 1e-3), name + " KPSS " + reg + " p");
    }
    const auto fixed = TimeSeries::ADFTest(y, 3, "c", false);
    Check(fixed.success && fixed.adf_lags == 3 && fixed.adf_nobs == s["adf_c_lag3"]["nobs"].get<int>(), name + " ADF with 3 lags");
    Near(fixed.adf_statistic, s["adf_c_lag3"]["stat"].get<double>(), Tol(1e-7, 1e-3), name + " ADF 3 lags statistic");
    const auto k4 = TimeSeries::KPSSTest(y, "c", 4);
    Near(k4.kpss_statistic, s["kpss_c_lag4"]["stat"].get<double>(), Tol(1e-7, 1e-3), name + " KPSS 4 lags statistic");
}

void CheckDecomposition(const std::string& name, const std::vector<double>& y, const json& d) {
    for (const std::string model : {"additive", "multiplicative"}) {
        const auto r = TimeSeries::Decompose(y, 12, model);
        Check(r.success, name + " " + model + ": " + r.error_message);
        NearAll(r.trend, d[model]["trend"], Tol(1e-9, 1e-4), name + " " + model + " trend");
        NearAll(r.seasonal, d[model]["seasonal"], Tol(1e-9, 1e-4), name + " " + model + " seasonal");
        NearAll(r.residual, d[model]["resid"], Tol(1e-8, 1e-3), name + " " + model + " resid");
    }
    const auto stl = TimeSeries::STLDecompose(y, 12);
    Check(stl.success && stl.seasonal_window == 7 && stl.low_pass_window == 13, name + " STL: " + stl.error_message);
    NearAll(stl.trend, d["stl"]["trend"], 1e-8, name + " STL trend");
    NearAll(stl.seasonal, d["stl"]["seasonal"], 1e-8, name + " STL seasonal");
    NearAll(stl.residual, d["stl"]["resid"], 1e-7, name + " STL resid");
    const auto s13 = TimeSeries::STLDecompose(y, 12, 13);
    NearAll(s13.trend, d["stl_s13"]["trend"], 1e-8, name + " STL(13) trend");
    NearAll(s13.seasonal, d["stl_s13"]["seasonal"], 1e-8, name + " STL(13) seasonal");
    const auto robust = TimeSeries::STLDecompose(y, 12, 7, -1, -1, true);
    NearAll(robust.trend, d["stl_robust"]["trend"], 1e-7, name + " robust STL trend");
    NearAll(robust.seasonal, d["stl_robust"]["seasonal"], 1e-7, name + " robust STL seasonal");
    const auto [power, freq] = TimeSeries::Periodogram(y);
    NearAll(freq, d["periodogram"]["freq"], 1e-12, name + " periodogram frequency");
    NearAll(power, d["periodogram"]["power"], Tol(1e-8, 1e-3), name + " periodogram power");
}

void CheckSmoothing(const std::string& name, const std::vector<double>& y, const json& s, int period) {
    struct Case {
        const char* key;
        std::function<cyxwiz::ForecastResult(int)> run;
    };
    const int h = period ? 24 : 12;
    std::vector<Case> cases = {
        {"simple", [&](int horizon) { return TimeSeries::SimpleES(y, horizon); }},
        {"holt", [&](int horizon) { return TimeSeries::HoltLinear(y, horizon); }},
        {"holt_damped", [&](int horizon) { return TimeSeries::HoltLinear(y, horizon, -1, -1, true); }},
    };
    if (period) {
        cases.push_back({"hw_add", [&](int horizon) { return TimeSeries::HoltWinters(y, horizon, period, "additive"); }});
        cases.push_back({"hw_mul", [&](int horizon) { return TimeSeries::HoltWinters(y, horizon, period, "multiplicative"); }});
    }
    for (const auto& c : cases) {
        const auto& e = s[c.key];
        const auto r = c.run(h);
        const std::string what = name + " " + c.key;
        Check(r.success, what + ": " + r.error_message);
        // The least-squares optimum: the same SSE, so the same fits and forecasts.
        Near(r.sse, e["sse"].get<double>(), 2e-3, what + " SSE");
        Near(r.aic, e["aic"].get<double>(), 2e-3, what + " AIC");
        NearAll(r.fitted_values, e["fitted"], 2e-2, what + " fitted");
        NearAll(r.forecast, e["forecast"], 2e-2, what + " forecast");
        Check(std::fabs(r.parameters.at("alpha") - e["alpha"].get<double>()) < 0.05, what + " alpha " + std::to_string(r.parameters.at("alpha")));
        // The interval: the additive-error formula from the fit (exact), and the ETS
        // model's own interval where its optimiser found the same fit.
        if (e.contains("interval")) {
            for (int i = 0; i < h; ++i) {
                Near(r.upper_bound[static_cast<size_t>(i)] - r.lower_bound[static_cast<size_t>(i)],
                     e["interval"]["upper"][static_cast<size_t>(i)].get<double>() - e["interval"]["lower"][static_cast<size_t>(i)].get<double>(), 2e-2,
                     what + " interval width at " + std::to_string(i + 1));
            }
        }
        const auto& ets = e["ets"];
        if (std::fabs(ets["alpha"].get<double>() - e["alpha"].get<double>()) < 0.05) {
            const double tolerance = ets["analytic"].get<bool>() ? 0.1 : 0.25;
            for (int i : {0, h / 2, h - 1}) {
                const double theirs = ets["upper"][static_cast<size_t>(i)].get<double>() - ets["lower"][static_cast<size_t>(i)].get<double>();
                const double ours = r.upper_bound[static_cast<size_t>(i)] - r.lower_bound[static_cast<size_t>(i)];
                Check(std::fabs(ours - theirs) <= tolerance * theirs, what + " ETS interval width at " + std::to_string(i + 1) + ": " + std::to_string(ours) + " vs " + std::to_string(theirs));
            }
        }
    }
}

void CheckArima(const std::string& name, const std::vector<double>& y, const json& a) {
    for (const auto& [order, e] : a.items()) {
        int p = 0, d = 0, q = 0;
        std::sscanf(order.c_str(), "%d,%d,%d", &p, &d, &q);
        const auto r = TimeSeries::ARIMA(y, 24, p, d, q);
        const std::string what = name + " ARIMA(" + order + ")";
        Check(r.success, what + ": " + r.error_message);
        Near(r.log_likelihood, e["llf"].get<double>(), 2e-4, what + " log-likelihood");
        Near(r.aic, e["aic"].get<double>(), 2e-4, what + " AIC");
        for (const auto& [key, value] : e["params"].items()) {
            Check(r.parameters.count(key) == 1, what + " has " + key);
            const double tolerance = key == "sigma2" ? 5e-3 : (key == "const" ? 2e-2 : 1e-2);
            Check(std::fabs(r.parameters.at(key) - value.get<double>()) <= tolerance * (1.0 + std::fabs(value.get<double>())),
                  what + " " + key + " " + std::to_string(r.parameters.at(key)) + " vs " + std::to_string(value.get<double>()));
        }
        const std::vector<double> fitted = Doubles(e["fitted"]);
        for (size_t t = static_cast<size_t>(d) + 10; t < fitted.size(); ++t)
            Near(r.fitted_values[t], fitted[t], 5e-3, what + " fitted " + std::to_string(t));
        NearAll(r.forecast, e["forecast"], 5e-3, what + " forecast");
        NearAll(r.lower_bound, e["lower"], 5e-3, what + " lower");
        NearAll(r.upper_bound, e["upper"], 5e-3, what + " upper");
    }
}

}  // namespace

int main() {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return 1;
    std::ifstream in(CYXWIZ_TIME_SERIES_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_TIME_SERIES_FIXTURE);
    const json f = json::parse(in);
    std::map<std::string, std::vector<double>> series;
    for (const auto& [name, values] : f["series"].items()) series[name] = Doubles(values);

    for (const auto& [name, c] : f["cases"].items()) {
        const auto& y = series.at(name);
        CheckCorrelogram(name, y, c["correlogram"]);
        if (c.contains("stationarity")) CheckStationarity(name, y, c["stationarity"]);
        if (c.contains("decomposition")) CheckDecomposition(name, y, c["decomposition"]);
        if (c.contains("smoothing")) CheckSmoothing(name, y, c["smoothing"], c.contains("decomposition") ? 12 : 0);
        if (c.contains("arima")) CheckArima(name, y, c["arima"]);
        std::cout << name << " ok\n";
    }

    // ---- rules of our own
    const auto& airline = series.at("airline");
    const auto season = TimeSeries::DetectSeasonality(airline);
    Check(season.success && season.detected_period == 12 && season.has_seasonality, "airline: a yearly season is found");
    Check(!season.candidate_periods.empty() && season.candidate_periods[0] == 12, "airline: 12 is the strongest period");
    bool six_is_harmonic = false;
    for (size_t i = 0; i < season.candidate_periods.size(); ++i)
        if (season.candidate_periods[i] == 6) six_is_harmonic = season.harmonic_of[i] == 12;
    Check(six_is_harmonic, "airline: 6 months is named a harmonic of 12");
    Check(season.strength > 0.7, "airline: the seasonal strength is high: " + std::to_string(season.strength));
    const auto noise_season = TimeSeries::DetectSeasonality(series.at("noise"));
    Check(noise_season.success && !noise_season.has_seasonality, "noise: no season");

    const auto airline_test = TimeSeries::TestStationarity(airline);
    Check(airline_test.success && !airline_test.is_stationary && airline_test.suggested_differencing == 1 && airline_test.differenced.size() == 1,
          "airline: not stationary, one difference suggested: " + airline_test.analysis);
    const auto walk_test = TimeSeries::TestStationarity(series.at("walk"));
    Check(walk_test.success && walk_test.suggested_differencing == 1, "walk: one difference");
    const auto ar1_test = TimeSeries::TestStationarity(series.at("ar1"));
    Check(ar1_test.success && ar1_test.is_stationary && ar1_test.suggested_differencing == 0, "ar1: stationary: " + ar1_test.analysis);

    const auto naive = TimeSeries::SeasonalNaive(airline, 24, 12);
    Check(naive.success && naive.forecast.size() == 24, "seasonal naive runs");
    for (int h = 0; h < 24; ++h) Near(naive.forecast[static_cast<size_t>(h)], airline[airline.size() - 12 + static_cast<size_t>(h % 12)], 1e-12, "seasonal naive repeats the last cycle");
    Near((naive.upper_bound[12] - naive.lower_bound[12]) / (naive.upper_bound[0] - naive.lower_bound[0]), std::sqrt(2.0), 1e-9, "seasonal naive interval grows by the cycle");

    // ---- refusals
    Check(!TimeSeries::ComputeACF({1.0, 2.0}).success, "two values are refused");
    Check(!TimeSeries::Decompose(airline, 100, "additive").success, "a period longer than half the series is refused");
    Check(!TimeSeries::HoltWinters(airline, 12, 12, "multiplicative", 1.5).success, "alpha above 1 is refused");
    Check(!TimeSeries::ADFTest(std::vector<double>(50, 3.0)).success, "a constant series is refused");
    Check(!TimeSeries::ARIMA(airline, 12, 6, 1, 0).success, "p above 5 is refused");
    const auto with_nan = [&] {
        std::vector<double> v = airline;
        v[10] = NAN;
        return v;
    }();
    Check(!TimeSeries::STLDecompose(with_nan, 12).success, "a missing value is refused");

    std::cout << "time series match the references: " << g_checks << " checks\n";
    return 0;
}
