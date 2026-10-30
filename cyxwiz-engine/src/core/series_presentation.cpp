#include "series_presentation.h"

#include "explore_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace cyxwiz::series_view {

using explore_view::Figure;
using explore_view::Number;
using series::Method;
using series::Series;
using series::TimeKind;

namespace {

std::string Percent(double share, int decimals = 0) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), decimals == 0 ? "%.0f%%" : "%.1f%%", 100.0 * share);
    return buf;
}

std::string Full(double v) {
    if (!std::isfinite(v)) return "";
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    return buf;
}

std::string Round(double v) {
    if (!std::isfinite(v)) return "";
    const double a = std::fabs(v);
    char buf[40];
    if (a >= 100.0) std::snprintf(buf, sizeof(buf), "%.0f", v);
    else if (a >= 10.0) std::snprintf(buf, sizeof(buf), "%.1f", v);
    else std::snprintf(buf, sizeof(buf), "%.2f", v);
    return buf;
}

// A date label without its day for monthly and coarser steps ("1949-01").
std::string ShortLabel(const std::string& label, const series::Step& step) {
    if (label.size() >= 10 && (step.name == "monthly" || step.name == "quarterly")) return label.substr(0, 7);
    if (label.size() >= 10 && step.name == "yearly") return label.substr(0, 4);
    return label;
}

std::string TimesBetter(double worse, double better) {
    if (!(better > 0.0) || !std::isfinite(worse)) return "";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", worse / better);
    return buf;
}

std::string Level(double level) {
    if (std::fabs(level - 0.95) < 1e-9) return "nineteen times in twenty";
    if (std::fabs(level - 0.9) < 1e-9) return "nine times in ten";
    if (std::fabs(level - 0.8) < 1e-9) return "four times in five";
    return Percent(level) + " of the time";
}

// "in Aug" for a monthly series, "on Mon" for a daily one, else "at position 8".
std::string PositionName(const Series& s, size_t row) {
    static const char* const kMonths[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    static const char* const kDays[7] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
    if (s.kind == TimeKind::Dates && row < s.t.size()) {
        int y, m, d, hh, mm, ss;
        series::SecondsToCivil(s.t[row], y, m, d, hh, mm, ss);
        if (s.step.name == "monthly" && s.period == 12) return std::string("in ") + kMonths[m - 1];
        if (s.step.name == "daily" && s.period == 7) {
            const long long day = static_cast<long long>(std::floor(s.t[row] / 86400.0));
            return std::string("on ") + kDays[((day + 3) % 7 + 7) % 7];  // 1970-01-01 was a Thursday
        }
    }
    return "at position " + std::to_string(row % static_cast<size_t>(std::max(s.period, 1)) + 1);
}

size_t FirstFinite(const std::vector<double>& v) {
    for (size_t i = 0; i < v.size(); ++i)
        if (std::isfinite(v[i])) return i;
    return v.size();
}

size_t LastFinite(const std::vector<double>& v) {
    for (size_t i = v.size(); i-- > 0;)
        if (std::isfinite(v[i])) return i;
    return v.size();
}

}  // namespace

const char* MethodLabel(Method m) { return series::MethodName(m); }

std::string StepUnit(const series::Step& step) {
    if (step.name == "monthly") return "month";
    if (step.name == "daily") return "day";
    if (step.name == "weekly") return "week";
    if (step.name == "quarterly") return "quarter";
    if (step.name == "yearly") return "year";
    if (step.name == "hourly") return "hour";
    return "step";
}

std::string Num(double v) { return Figure(v); }

std::string SeriesWords(const Series& s) {
    const std::string unit = StepUnit(s.step);
    std::string out = Number(static_cast<size_t>(s.used)) + " " + (unit == "step" ? "rows" : unit + "s");
    if (s.kind == TimeKind::Order) return out + " in table order";
    if (!s.step.name.empty()) out += ", " + s.step.name + (s.step.even ? "" : " (uneven)");
    if (s.kind == TimeKind::Dates && !s.labels.empty()) out += ", " + ShortLabel(s.labels.front(), s.step) + " .. " + ShortLabel(s.labels.back(), s.step);
    else if (s.kind == TimeKind::Numbers && !s.t.empty()) out += ", " + Figure(s.t.front()) + " .. " + Figure(s.t.back());
    return out;
}

std::string LeftOut(const Series& s) {
    std::string out;
    const int64_t left = s.rows - s.used;
    if (left > 0) out += explore_view::Count(static_cast<size_t>(left), "row has", "rows have") + " no value" + (s.kind == TimeKind::Order ? "" : " or time") + " and " +
                         (left == 1 ? "is" : "are") + " left out.";
    if (s.unordered > 0) out += std::string(out.empty() ? "" : " ") + "The table was not in time order; the rows were sorted.";
    if (!s.step.even && s.kind != TimeKind::Order && !s.step.name.empty()) out += std::string(out.empty() ? "" : " ") + "The rows are not evenly spaced; they are taken in order.";
    return out;
}

std::string PeriodWords(const Series& s) {
    if (s.period < 2) return "none found";
    if (s.period_found) return "found: " + std::to_string(s.period) + " (" + Percent(s.period_share) + " of the detrended power)";
    if (s.step.natural_period == s.period) return "from the step: " + std::to_string(s.period);
    return "typed: " + std::to_string(s.period);
}

std::string PartsSentence(const series::PartsAnswer& a) {
    if (!a.ok()) return a.error;
    const auto& p = a.parts;
    const std::string& name = a.request.base.value;
    const double ts = p.trend_strength, ss = p.seasonal_strength;
    std::string out = name;
    if (ts >= 0.9 && ss >= 0.9) out += " is almost all trend and season";
    else if (ts >= 0.6 && ss >= 0.6) out += " has a clear trend and a clear season";
    else if (ts >= 0.6) out += " is mostly trend" + std::string(ss >= 0.3 ? " with some season" : ", with little season");
    else if (ss >= 0.6) out += " is mostly season" + std::string(ts >= 0.3 ? " with some trend" : ", with little trend");
    else out += " is mostly irregular";
    out += " (strengths " + Round(ts) + " and " + Round(ss) + "): ";
    const size_t first = FirstFinite(p.trend), last = LastFinite(p.trend);
    if (first < p.trend.size()) {
        const double from = p.trend[first], to = p.trend[last];
        out += "the trend " + std::string(to > from * 1.02 ? "climbs" : to < from * 0.98 ? "falls" : "stays") + " from " + Round(from) + " to " + Round(to);
    }
    if (!p.seasonal.empty()) {
        const size_t period = static_cast<size_t>(std::max(p.period, 1));
        const auto hi = std::max_element(p.seasonal.begin(), p.seasonal.begin() + std::min(period, p.seasonal.size()));
        const auto lo = std::min_element(p.seasonal.begin(), p.seasonal.begin() + std::min(period, p.seasonal.size()));
        const bool multiplicative = p.method == "multiplicative";
        out += "; the season peaks " + PositionName(a.series, static_cast<size_t>(hi - p.seasonal.begin())) + " (" + (multiplicative ? Round(*hi) + "x" : (*hi >= 0 ? "+" : "") + Round(*hi)) +
               ") and dips " + PositionName(a.series, static_cast<size_t>(lo - p.seasonal.begin())) + " (" + (multiplicative ? Round(*lo) + "x" : Round(*lo)) + ")";
        if (p.method == "stl") {
            // STL lets the season change: compare the first and last cycle's swing.
            const size_t n = p.seasonal.size();
            if (n >= 2 * period) {
                const auto swing = [&](size_t start) {
                    const auto a2 = std::max_element(p.seasonal.begin() + start, p.seasonal.begin() + start + period);
                    const auto b2 = std::min_element(p.seasonal.begin() + start, p.seasonal.begin() + start + period);
                    return *a2 - *b2;
                };
                const double early = swing(0), late = swing(n - period);
                if (late > 1.5 * early) out += "; the season grows with the level (its swing goes from " + Round(early) + " to " + Round(late) + "), so a multiplicative view may fit better";
                else if (early > 1.5 * late) out += "; the season shrinks over time (its swing goes from " + Round(early) + " to " + Round(late) + ")";
            }
        }
    }
    double worst = 0.0;
    for (const double v : p.residual)
        if (std::isfinite(v)) worst = std::max(worst, std::fabs(p.method == "multiplicative" ? v - 1.0 : v));
    out += "; what is left is within " + std::string(p.method == "multiplicative" ? Percent(worst) : "+-" + Round(worst)) + ".";
    return out;
}

std::string SeasonalitySentence(const series::SeasonalityAnswer& a) {
    if (!a.ok()) return a.error;
    const auto& s = a.seasonality;
    if (s.detected_period < 2) return "No seasonal period stands out after removing the trend.";
    const std::string unit = StepUnit(a.series.step);
    const auto name = [&](int period) {
        if (unit == "month" && period == 12) return std::string("A yearly season (period 12)");
        if (unit == "month" && period == 6) return std::string("A half-year period");
        if (unit == "month" && period == 3) return std::string("A quarterly period");
        if (unit == "day" && period == 7) return std::string("A weekly season (period 7)");
        if (unit == "hour" && period == 24) return std::string("A daily season (period 24)");
        return "A period of " + std::to_string(period) + " " + (unit == "step" ? "rows" : unit + "s");
    };
    std::string out = name(s.detected_period) + " carries " + Percent(s.detected_share) + " of the detrended power";
    for (size_t i = 0; i < s.candidate_periods.size(); ++i)
        if (s.harmonic_of[i] == s.detected_period) {
            out += ", with its harmonic at " + std::to_string(s.candidate_periods[i]) + " (" + Percent(s.candidate_strengths[i]) + ")";
            break;
        }
    out += s.has_seasonality ? "; a seasonal pattern of that length explains " + Percent(s.strength) + " of what is left after the trend."
                             : "; but a seasonal pattern of that length explains only " + Percent(s.strength) + " of what is left after the trend, so the season is weak.";
    if (!a.season_by_position.empty()) {
        const auto hi = std::max_element(a.season_by_position.begin(), a.season_by_position.end());
        const auto lo = std::min_element(a.season_by_position.begin(), a.season_by_position.end());
        const size_t ih = static_cast<size_t>(hi - a.season_by_position.begin()), il = static_cast<size_t>(lo - a.season_by_position.begin());
        out += " " + a.position_names[ih] + " is the high (" + (*hi >= 0 ? "+" : "") + Round(*hi) + " on average), " + a.position_names[il] + " the low (" + Round(*lo) + ").";
    }
    return out;
}

std::string StationaritySentence(const series::StationarityAnswer& a) {
    if (!a.ok()) return a.error;
    std::string out = a.tests.analysis;
    const size_t first = FirstFinite(a.rolling_mean), last = LastFinite(a.rolling_mean);
    if (first < a.rolling_mean.size() && last > first) {
        const double m0 = a.rolling_mean[first], m1 = a.rolling_mean[last];
        const double s0 = a.rolling_sd[first], s1 = a.rolling_sd[last];
        const bool mean_moves = std::fabs(m1 - m0) > 0.5 * std::max(s0, s1);
        const bool spread_grows = s1 > 1.5 * s0, spread_shrinks = s0 > 1.5 * s1;
        if (mean_moves || spread_grows || spread_shrinks) {
            out += " The rolling mean " + std::string(mean_moves ? (m1 > m0 ? "climbs" : "falls") : "holds") + " (" + Round(m0) + " to " + Round(m1) + ")";
            if (spread_grows) out += " and the rolling spread grows with it (" + Round(s0) + " to " + Round(s1) + "), which also points to a multiplicative season or a log";
            else if (spread_shrinks) out += " and the rolling spread shrinks (" + Round(s0) + " to " + Round(s1) + ")";
            out += ".";
        }
    }
    return out;
}

std::string CorrelogramSentence(const series::CorrelogramAnswer& a) {
    if (!a.ok()) return a.error;
    const auto& c = a.correlogram;
    const std::string what = a.request.differenced ? "The differenced series" : "The series";
    std::string out;
    if (c.ljung_box_pvalue < 0.05) out = what + " is autocorrelated (Ljung-Box rejects white noise at " + std::to_string(c.ljung_box_lags) + " lags, p " + (c.ljung_box_pvalue < 1e-4 ? std::string("< 0.0001") : Round(c.ljung_box_pvalue)) + ").";
    else out = what + " looks like white noise (Ljung-Box p " + Round(c.ljung_box_pvalue) + " at " + std::to_string(c.ljung_box_lags) + " lags).";
    const int period = a.series.period;
    const bool slow = c.acf.size() > 10 && c.acf[1] > 0.8 && std::fabs(c.acf[10]) > c.confidence_upper[10] - c.acf[10];
    if (slow) out += " The ACF decays slowly, which is a trend";
    if (period >= 2 && static_cast<size_t>(period) < c.acf.size()) {
        bool seasonal = c.acf[static_cast<size_t>(period)] > c.confidence_upper[static_cast<size_t>(period)] - c.acf[static_cast<size_t>(period)] &&
                        c.acf[static_cast<size_t>(period)] > c.acf[static_cast<size_t>(period) - 1];
        if (seasonal) out += std::string(slow ? ", and rises again at " : " The ACF rises at ") + std::to_string(period) + (static_cast<size_t>(2 * period) < c.acf.size() ? ", " + std::to_string(2 * period) : "") + ", which is the season";
    }
    if (slow || (period >= 2)) out += ".";
    if (c.suggested_ar_order > 0 || c.suggested_ma_order > 0) {
        out += " For an ARIMA" + std::string(a.request.differenced ? "" : " on the differenced series") + ", start from p = " + std::to_string(c.suggested_ar_order) +
               ", q = " + std::to_string(c.suggested_ma_order) + " (the PACF cuts off after " + std::to_string(c.suggested_ar_order) + ", the ACF after " +
               std::to_string(c.suggested_ma_order) + ")";
        out += period >= 2 ? "; the seasonal lags call for a seasonal method (Holt-Winters, or a seasonal difference)." : ".";
    }
    return out;
}

std::string ModelWords(const series::ForecastAnswer& a) {
    const auto& r = a.request;
    const auto& p = a.fit.parameters;
    const auto param = [&](const char* key) { return p.count(key) ? Round(p.at(key)) : std::string(); };
    std::string out;
    switch (r.method) {
        case Method::Simple: out = "level only; alpha " + param("alpha"); break;
        case Method::Holt: out = std::string(r.damped ? "damped" : "additive") + " trend; alpha " + param("alpha") + ", beta " + param("beta") + (r.damped ? ", phi " + param("phi") : ""); break;
        case Method::HoltWinters:
            out = std::string("additive trend, ") + (r.multiplicative ? "multiplicative" : "additive") + " season, period " + std::to_string(a.series.period) + "; alpha " + param("alpha") +
                  ", beta " + param("beta") + ", gamma " + param("gamma");
            break;
        case Method::Arima:
            for (const auto& [key, value] : p)
                if (key.rfind("ar.", 0) == 0 || key.rfind("ma.", 0) == 0 || key == "const") out += (out.empty() ? "" : ", ") + key + " " + Round(value);
            break;
        case Method::SeasonalNaive: out = "each value repeats the one " + std::to_string(a.series.period) + " steps before"; break;
    }
    return out;
}

std::string ForecastSentence(const series::ForecastAnswer& a) {
    if (!a.ok()) return a.error;
    const auto& f = a.fit;
    if (f.forecast.empty()) return "";
    const std::string unit = StepUnit(a.series.step);
    // "Holt-Winters (additive trend, ...)" / "ARIMA(2,1,1) (ar.L1 1.09, ...)".
    const std::string label = a.request.method == Method::Arima ? f.method : std::string(MethodLabel(a.request.method));
    const std::string model = ModelWords(a);
    std::string out = label + (model.empty() ? "" : " (" + model + ")") + " forecasts " + Round(f.forecast[0]) + " " + a.request.base.value + " next " + unit;
    if (!f.lower_bound.empty()) out += ", between " + Round(f.lower_bound[0]) + " and " + Round(f.upper_bound[0]) + " " + Level(a.request.level);
    if (f.forecast.size() > 1) {
        const auto hi = std::max_element(f.forecast.begin(), f.forecast.end());
        const auto lo = std::min_element(f.forecast.begin(), f.forecast.end());
        const size_t last = f.forecast.size() - 1;
        if (*hi > f.forecast[0] * 1.02 && static_cast<size_t>(hi - f.forecast.begin()) != 0) {
            const size_t at = static_cast<size_t>(hi - f.forecast.begin());
            out += ", rising to " + Round(*hi) + (at < a.future_labels.size() ? " in " + ShortLabel(a.future_labels[at], a.series.step) : "");
        } else if (*lo < f.forecast[0] * 0.98 && static_cast<size_t>(lo - f.forecast.begin()) != 0) {
            const size_t at = static_cast<size_t>(lo - f.forecast.begin());
            out += ", falling to " + Round(*lo) + (at < a.future_labels.size() ? " in " + ShortLabel(a.future_labels[at], a.series.step) : "");
        } else {
            out += ", and " + Round(f.forecast[last]) + " in " + std::to_string(last + 1) + " " + unit + "s";
        }
    }
    return out + ".";
}

std::string HoldoutSentence(const series::ForecastAnswer& a) {
    if (!a.ok() || a.holdout == 0 || a.scores.empty()) return "";
    const auto& chosen = a.scores.front();
    if (!chosen.error.empty()) return "The held-out check could not run " + chosen.name + " on the shorter series: " + chosen.error;
    const std::string unit = StepUnit(a.series.step);
    std::string out = "Checked on the last " + std::to_string(a.holdout) + " " + unit + "s held out, it was " + Percent(chosen.mape / 100.0, 1) + " off on average";
    const series::HoldoutScore* naive = nullptr;
    const series::HoldoutScore* best = &chosen;
    for (const auto& s : a.scores) {
        if (s.method == Method::SeasonalNaive) naive = &s;
        if (s.error.empty() && s.mape < best->mape) best = &s;
    }
    if (naive && naive->error.empty() && naive != &chosen) {
        const std::string ratio = TimesBetter(naive->mape, chosen.mape);
        if (chosen.mape < naive->mape) out += ": " + ratio + " times better than repeating last " + (unit == "month" || unit == "quarter" ? "year" : "cycle") + " (" + Percent(naive->mape / 100.0, 1) + ")";
        else out += ": no better than repeating last " + std::string(unit == "month" || unit == "quarter" ? "year" : "cycle") + " (" + Percent(naive->mape / 100.0, 1) + ")";
    }
    if (best != &chosen && best->error.empty()) out += "; " + best->name + " did better (" + Percent(best->mape / 100.0, 1) + ")";
    return out + ".";
}

std::string PartsCsv(const series::PartsAnswer& a) {
    std::vector<std::vector<std::string>> rows = {{"time", a.request.base.value, "trend", "seasonal", "residual"}};
    for (size_t i = 0; i < a.series.values.size(); ++i)
        rows.push_back({i < a.series.labels.size() ? a.series.labels[i] : Full(a.series.t[i]), Full(a.series.values[i]), Full(a.parts.trend[i]), Full(a.parts.seasonal[i]), Full(a.parts.residual[i])});
    return explore_view::Csv(rows);
}

std::string SeasonalityCsv(const series::SeasonalityAnswer& a) {
    std::vector<std::vector<std::string>> rows = {{"period", "share of power", "harmonic of"}};
    for (size_t i = 0; i < a.seasonality.candidate_periods.size(); ++i)
        rows.push_back({std::to_string(a.seasonality.candidate_periods[i]), Full(a.seasonality.candidate_strengths[i]), a.seasonality.harmonic_of[i] ? std::to_string(a.seasonality.harmonic_of[i]) : ""});
    rows.push_back({});
    rows.push_back({"position", "season"});
    for (size_t i = 0; i < a.season_by_position.size(); ++i) rows.push_back({a.position_names[i], Full(a.season_by_position[i])});
    return explore_view::Csv(rows);
}

std::string StationarityCsv(const series::StationarityAnswer& a) {
    const auto& t = a.tests;
    std::vector<std::vector<std::string>> rows = {{"series", "ADF statistic", "ADF p", "ADF lags", "KPSS statistic", "KPSS p", "KPSS lags"},
                                                  {a.request.base.value, Full(t.adf_statistic), Full(t.adf_pvalue), std::to_string(t.adf_lags), Full(t.kpss_statistic), Full(t.kpss_pvalue), std::to_string(t.kpss_lags)}};
    for (const auto& d : t.differenced) rows.push_back({"differenced " + std::to_string(d.d), Full(d.adf_statistic), Full(d.adf_pvalue), "", Full(d.kpss_statistic), Full(d.kpss_pvalue), ""});
    rows.push_back({});
    rows.push_back({"time", "rolling mean", "rolling sd"});
    for (size_t i = 0; i < a.rolling_mean.size(); ++i) rows.push_back({i < a.series.labels.size() ? a.series.labels[i] : Full(a.series.t[i]), Full(a.rolling_mean[i]), Full(a.rolling_sd[i])});
    return explore_view::Csv(rows);
}

std::string CorrelogramCsv(const series::CorrelogramAnswer& a) {
    const auto& c = a.correlogram;
    std::vector<std::vector<std::string>> rows = {{"lag", "acf", "acf low", "acf high", "pacf", "pacf low", "pacf high", "Ljung-Box Q", "Ljung-Box p"}};
    for (size_t k = 0; k < c.acf.size(); ++k)
        rows.push_back({std::to_string(k), Full(c.acf[k]), Full(c.confidence_lower[k]), Full(c.confidence_upper[k]), Full(c.pacf[k]), Full(c.pacf_confidence_lower[k]),
                        Full(c.pacf_confidence_upper[k]), k > 0 ? Full(c.ljung_box_q[k]) : "", k > 0 ? Full(c.ljung_box_p[k]) : ""});
    return explore_view::Csv(rows);
}

std::string ForecastCsv(const series::ForecastAnswer& a) {
    std::vector<std::vector<std::string>> rows = {{"time", a.request.base.value, "fitted", "forecast", "low", "high"}};
    for (size_t i = 0; i < a.series.values.size(); ++i)
        rows.push_back({i < a.series.labels.size() ? a.series.labels[i] : Full(a.series.t[i]), Full(a.series.values[i]), i < a.fit.fitted_values.size() ? Full(a.fit.fitted_values[i]) : "", "", "", ""});
    for (size_t h = 0; h < a.fit.forecast.size(); ++h)
        rows.push_back({h < a.future_labels.size() ? a.future_labels[h] : "", "", "", Full(a.fit.forecast[h]), h < a.fit.lower_bound.size() ? Full(a.fit.lower_bound[h]) : "",
                        h < a.fit.upper_bound.size() ? Full(a.fit.upper_bound[h]) : ""});
    return explore_view::Csv(rows);
}

}  // namespace cyxwiz::series_view
