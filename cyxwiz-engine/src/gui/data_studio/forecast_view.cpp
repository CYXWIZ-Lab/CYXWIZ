#include "forecast_view.h"

#include "explore_plots.h"
#include "series_view.h"

#include "../../core/explore_presentation.h"
#include "../../core/file_dialogs.h"
#include "../../core/series_presentation.h"
#include "../../core/session_query_service.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace cyxwiz {

namespace {

namespace sv = series_view;
using series::Method;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::SameLine();
}

void Faint(const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_faint, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Heading(const char* text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::PopTextWrapPos();
}

void NumberCell(const std::string& text, const ImVec4* colour = nullptr) {
    ImGui::TableNextColumn();
    const float w = ImGui::CalcTextSize(text.c_str()).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
    if (colour) ImGui::TextColored(*colour, "%s", text.c_str());
    else ImGui::TextUnformatted(text.c_str());
}

float TableHeight(size_t rows) {
    return static_cast<float>(rows) * (ImGui::GetTextLineHeight() + 2.0f * ImGui::GetStyle().CellPadding.y) + 4.0f;
}

void Setting(const char* label, float width) {
    ImGui::SameLine(0.0f, 16.0f);
    if (ImGui::GetContentRegionAvail().x < ImGui::CalcTextSize(label).x + width) ImGui::NewLine();
    Label(label);
}

std::string Round(double v, const char* format = "%.3g") {
    if (!std::isfinite(v)) return "";
    char buf[32];
    std::snprintf(buf, sizeof(buf), format, v);
    return buf;
}

std::string Percent(double v) { return std::isfinite(v) ? Round(v, "%.1f") + "%" : ""; }

std::string Key(const std::string& dataset, const series::ForecastRequest& r) {
    std::string key = dataset + "|" + r.base.time + "|" + r.base.value + "|" + std::to_string(r.base.period) + "|" + series::MethodName(r.method) + "|" +
                      std::to_string(r.horizon) + "|" + Round(r.level, "%.4f") + "|" + std::to_string(r.holdout);
    switch (r.method) {
        case Method::Simple: key += "|" + Round(r.alpha); break;
        case Method::Holt: key += "|" + Round(r.alpha) + "|" + Round(r.beta) + (r.damped ? "|damped" : ""); break;
        case Method::HoltWinters: key += "|" + Round(r.alpha) + "|" + Round(r.beta) + "|" + Round(r.gamma) + (r.multiplicative ? "|mul" : "|add"); break;
        case Method::Arima: key += "|" + std::to_string(r.p) + "," + std::to_string(r.d) + "," + std::to_string(r.q); break;
        case Method::SeasonalNaive: break;
    }
    return key;
}

// A short label for the axis and the table ("1961-01" for a monthly step).
std::string Short(const std::string& label, const series::Step& step) {
    if (label.size() >= 10 && (step.name == "monthly" || step.name == "quarterly")) return label.substr(0, 7);
    if (label.size() >= 10 && step.name == "yearly") return label.substr(0, 4);
    return label;
}

std::string AxisLabel(const series::Series& s, size_t i) {
    if (s.kind == series::TimeKind::Dates && i < s.labels.size()) return Short(s.labels[i], s.step);
    if (s.kind == series::TimeKind::Numbers && i < s.t.size()) return explore_view::Figure(s.t[i]);
    return std::to_string(i + 1);
}

}  // namespace

void ForecastView::Message(const std::string& text, bool error) {
    message_ = text;
    message_error_ = error;
}

void ForecastView::StartFrom(const series::SeriesRequest& base) {
    if (!chosen_) request_.base = base;
}

void ForecastView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to forecast one of its columns.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        job_.Reset();
        message_.clear();
        stopped_key_.clear();
    }
    RenderControls(context, p);
    if (!message_.empty()) Wrapped(message_error_ ? t.error : t.text_dim, message_);
    ImGui::Spacing();
    if (request_.base.value.empty()) {
        Faint("Pick the number column to forecast.");
        return;
    }
    // The held-out check: the last H rows; -1 lets the model take twice the period found (else 12), at most a third of the rows.
    request_.holdout = holdout_ ? (holdout_rows_ > 0 ? holdout_rows_ : -1) : 0;
    if (estimate_) request_.alpha = request_.beta = request_.gamma = -1.0;
    const std::string key = Key(context.dataset, request_);
    if (key == stopped_key_) {
        Faint("Stopped. Change a setting to run again.");
        return;
    }
    const std::string dataset = context.dataset;
    const series::ForecastRequest req = request_;
    const explore::MissingTexts texts = context.missing_texts;
    job_.Ask(std::string("Forecast: ") + series::MethodName(req.method), key, [dataset, req, texts](LambdaTask&) { return series::RunForecast(dataset, req, RunNow, texts); });
    const series::ForecastAnswer* a = job_.Answer(key);
    if (!a) {
        if (!job_.Error().empty()) {
            Wrapped(t.error, job_.Error());
            return;
        }
        Faint(request_.method == Method::Arima && (request_.p < 0 || request_.q < 0) ? "Trying the ARIMA orders on the device..." : "Fitting on the device...");
        if (ui::LinkButton("Stop")) {
            job_.Cancel();
            stopped_key_ = key;
        }
        return;
    }
    last_holdout_ = a->holdout;
    RenderAnswer(*a);
}

void ForecastView::RenderControls(const ExploreContext& context, const DatasetProfile& p) {
    if (RenderSeriesPickers("##forecast", context, p, picker_, request_.base)) chosen_ = true;
    static const char* const kMethods[] = {"Simple", "Holt", "Holt-Winters", "ARIMA", "Seasonal naive"};
    int m = static_cast<int>(request_.method);
    ui::FlowRow row;
    row.Next(ImGui::CalcTextSize("MethodSimpleHoltHolt-WintersARIMASeasonal naive").x + 130.0f);
    Label("Method");
    if (ExploreChoices("##forecast_method", kMethods, 5, &m)) request_.method = static_cast<Method>(m);
    Setting("Horizon", 70.0f);
    ui::NumberField("##forecast_horizon", &request_.horizon, 1, 10000, 1, 56.0f, "Steps ahead");
    Setting("Level", 80.0f);
    {
        ui::NumberFieldOptions o;
        o.min = 0.5;
        o.max = 0.999;
        o.step = 0.05;
        o.format = "%.2f";
        o.width = 56.0f;
        o.hint = "The prediction interval's level (0.95: nineteen times in twenty)";
        ui::NumberField("##forecast_level", &request_.level, o);
    }
    RenderMethodSettings();
    ui::FlowRow check;
    check.Next(ImGui::CalcTextSize("Check on the last").x + 260.0f);
    ImGui::Checkbox("Check on the last", &holdout_);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Holds out the last rows, fits every method on the rest and scores each against repeating last cycle; then the chosen method is fitted on everything");
    if (holdout_) {
        ImGui::SameLine();
        int rows = holdout_rows_ > 0 ? holdout_rows_ : last_holdout_ > 0 ? last_holdout_ : 12;
        if (ui::NumberField("##forecast_holdout", &rows, 1, 100000, 1, 56.0f, "Rows held out; twice the period found unless typed")) holdout_rows_ = rows;
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(ui::CurrentTokens().text_dim, "rows");
    }
}

void ForecastView::RenderMethodSettings() {
    switch (request_.method) {
        case Method::Simple: break;
        case Method::Holt: {
            static const char* const kTrend[] = {"Additive trend", "Damped trend"};
            int trend = request_.damped ? 1 : 0;
            Setting("Trend", 220.0f);
            if (ExploreChoices("##forecast_trend", kTrend, 2, &trend)) request_.damped = trend == 1;
            break;
        }
        case Method::HoltWinters: {
            static const char* const kSeason[] = {"Additive season", "Multiplicative season"};
            int season = request_.multiplicative ? 1 : 0;
            Setting("Season", 260.0f);
            if (ExploreChoices("##forecast_season", kSeason, 2, &season)) request_.multiplicative = season == 1;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Multiplicative: the season grows with the level (airline passengers)");
            break;
        }
        case Method::Arima: {
            ui::FlowRow orders;
            orders.Next(ImGui::CalcTextSize("ARIMA orders p d q").x + 240.0f);
            Label("Orders p, d, q");
            ui::NumberField("##forecast_p", &request_.p, -1, 5, 1, 48.0f, "AR order; -1: chosen by AIC");
            ImGui::SameLine();
            ui::NumberField("##forecast_d", &request_.d, -1, 2, 1, 48.0f, "Differencing; -1: from the stationarity tests");
            ImGui::SameLine();
            ui::NumberField("##forecast_q", &request_.q, -1, 5, 1, 48.0f, "MA order; -1: chosen by AIC");
            break;
        }
        case Method::SeasonalNaive: break;
    }
    if (request_.method == Method::Simple || request_.method == Method::Holt || request_.method == Method::HoltWinters) {
        Setting("Weights", 120.0f);
        static const char* const kWeights[] = {"Estimated", "Typed"};
        int w = estimate_ ? 0 : 1;
        if (ExploreChoices("##forecast_weights", kWeights, 2, &w)) {
            estimate_ = w == 0;
            if (!estimate_) {
                if (request_.alpha < 0) request_.alpha = 0.3;
                if (request_.beta < 0) request_.beta = 0.1;
                if (request_.gamma < 0) request_.gamma = 0.1;
            }
        }
        if (!estimate_) {
            ui::NumberFieldOptions o;
            o.min = 0.0;
            o.max = 1.0;
            o.step = 0.05;
            o.format = "%.2f";
            o.width = 52.0f;
            Setting("alpha", 60.0f);
            ui::NumberField("##forecast_alpha", &request_.alpha, o);
            if (request_.method != Method::Simple) {
                Setting("beta", 60.0f);
                ui::NumberField("##forecast_beta", &request_.beta, o);
            }
            if (request_.method == Method::HoltWinters) {
                Setting("gamma", 60.0f);
                ui::NumberField("##forecast_gamma", &request_.gamma, o);
            }
        }
    }
}

void ForecastView::RenderAnswer(const series::ForecastAnswer& a) {
    const ui::Tokens& t = ui::CurrentTokens();
    const series::Series& s = a.series;
    const auto& f = a.fit;
    const std::string unit = sv::StepUnit(s.step);
    std::vector<std::pair<std::string, std::string>> figures;
    if (!f.forecast.empty()) {
        std::string next = Round(f.forecast[0]);
        if (!f.lower_bound.empty()) next += " (" + Round(f.lower_bound[0]) + " .. " + Round(f.upper_bound[0]) + ")";
        figures.emplace_back("next " + unit, next);
        if (f.forecast.size() > 1) figures.emplace_back("in " + std::to_string(f.forecast.size()) + " " + unit + "s", Round(f.forecast.back()));
    }
    if (a.request.method != Method::SeasonalNaive && std::isfinite(f.aic)) figures.emplace_back("AIC", Round(f.aic, "%.1f"));
    if (!a.scores.empty() && a.scores.front().error.empty()) {
        std::string held = Percent(a.scores.front().mape);
        const auto& naive = a.scores.back();
        if (naive.method == Method::SeasonalNaive && naive.error.empty() && &naive != &a.scores.front()) held += " vs " + Percent(naive.mape) + " repeating last cycle";
        figures.emplace_back("held-out MAPE", held);
    }
    ExploreFigures("##forecast_figures", figures);
    Wrapped(t.text, sv::ForecastSentence(a));
    Wrapped(t.text, sv::HoldoutSentence(a));
    Faint(sv::SeriesWords(s) + "; period " + sv::PeriodWords(s) + ".");
    Faint(sv::LeftOut(s));
    ImGui::Spacing();

    // The series, the fit, and the forecast with its band, over one axis.
    std::vector<double> xs = s.t;
    xs.insert(xs.end(), a.future_t.begin(), a.future_t.end());
    const size_t n = s.values.size(), h = a.future_t.size();
    explore_plot::Line observed{a.request.base.value, s.values};
    observed.ys.resize(n + h, NAN);
    explore_plot::Line fit{"fit and forecast", std::vector<double>(n + h, NAN)};
    std::vector<double> low(n + h, NAN), high(n + h, NAN);
    for (size_t i = 0; i < n && i < f.fitted_values.size(); ++i) fit.ys[i] = f.fitted_values[i];
    for (size_t k = 0; k < h && k < f.forecast.size(); ++k) {
        fit.ys[n + k] = f.forecast[k];
        if (k < f.lower_bound.size()) low[n + k] = f.lower_bound[k];
        if (k < f.upper_bound.size()) high[n + k] = f.upper_bound[k];
    }
    // The band starts at the last observed row so it joins the fit.
    if (n > 0 && h > 0 && n - 1 < f.fitted_values.size()) low[n - 1] = high[n - 1] = f.fitted_values[n - 1];
    const float w = ImGui::GetContentRegionAvail().x;
    const std::string lo = AxisLabel(s, 0);
    const std::string hi = h > 0 && !a.future_labels.empty() ? Short(a.future_labels.back(), s.step) : AxisLabel(s, n - 1);
    Heading(("THE SERIES, THE FIT AND THE FORECAST WITH ITS " + Round(100.0 * a.request.level, "%.0f") + "% RANGE").c_str());
    explore_plot::Band(xs, observed, fit, low, high, w, std::clamp(w * 0.34f, 160.0f, 280.0f), n > 0 ? s.t[n - 1] : NAN, "forecast", "", lo, hi);
    ImGui::Spacing();
    const bool split = w >= 820.0f && !a.scores.empty();
    if (split && ImGui::BeginTable("##forecast_split", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("table", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("holdout", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
    }
    RenderTable(a);
    if (!a.scores.empty()) {
        if (split) ImGui::TableNextColumn();
        else ImGui::Spacing();
        RenderHoldout(a);
    }
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    RenderActions(a);
}

void ForecastView::RenderTable(const series::ForecastAnswer& a) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& f = a.fit;
    const std::string unit = sv::StepUnit(a.series.step);
    Heading("THE FORECAST");
    const size_t rows = f.forecast.size();
    const bool band = !f.lower_bound.empty();
    const float height = TableHeight(std::min<size_t>(rows, 13) + 1);
    if (ImGui::BeginTable("##forecast_table", band ? 4 : 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_ScrollY,
                          ImVec2(0.0f, height))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn(unit.c_str(), ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableSetupColumn("forecast", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        if (band) {
            ImGui::TableSetupColumn("low", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("high", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        }
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        Heading(a.series.kind == series::TimeKind::Order ? "row" : unit == "step" ? "time" : unit.c_str());
        NumberCell("forecast", &t.text_dim);
        if (band) {
            NumberCell("low", &t.text_dim);
            NumberCell("high", &t.text_dim);
        }
        for (size_t k = 0; k < rows; ++k) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const std::string label = k < a.future_labels.size() ? Short(a.future_labels[k], a.series.step) : std::to_string(k + 1);
            ImGui::TextColored(t.text_dim, "%s", label.c_str());
            NumberCell(Round(f.forecast[k]));
            if (band) {
                NumberCell(k < f.lower_bound.size() ? Round(f.lower_bound[k]) : "", &t.text_dim);
                NumberCell(k < f.upper_bound.size() ? Round(f.upper_bound[k]) : "", &t.text_dim);
            }
        }
        ImGui::EndTable();
    }
}

void ForecastView::RenderHoldout(const series::ForecastAnswer& a) {
    const ui::Tokens& t = ui::CurrentTokens();
    Heading(("CHECKED ON THE LAST " + std::to_string(a.holdout) + " ROWS HELD OUT").c_str());
    if (ImGui::BeginTable("##forecast_holdout", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings,
                          ImVec2(0.0f, TableHeight(a.scores.size() + 1)))) {
        ImGui::TableSetupColumn("method", ImGuiTableColumnFlags_WidthStretch, 1.8f);
        ImGui::TableSetupColumn("MAPE", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("RMSE", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        Heading("method");
        NumberCell("MAPE", &t.text_dim);
        NumberCell("RMSE", &t.text_dim);
        double best = INFINITY;
        for (const auto& sc : a.scores)
            if (sc.error.empty() && std::isfinite(sc.mape)) best = std::min(best, sc.mape);
        for (size_t i = 0; i < a.scores.size(); ++i) {
            const auto& sc = a.scores[i];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool chosen = i == 0;
            ImGui::TextColored(chosen ? t.text : t.text_dim, "%s%s", sc.name.c_str(), chosen ? " (chosen)" : sc.method == Method::SeasonalNaive ? " (baseline)" : "");
            if (!sc.error.empty()) {
                ImGui::TableNextColumn();
                ImGui::TextColored(t.text_faint, "%s", sc.error.c_str());
                ImGui::TableNextColumn();
                continue;
            }
            const bool is_best = sc.error.empty() && sc.mape == best;
            NumberCell(Percent(sc.mape), is_best ? &t.success : nullptr);
            NumberCell(Round(sc.rmse));
        }
        ImGui::EndTable();
    }
    Faint("MAPE: how far off on average, as a share of the true value. RMSE: in the column's units. Each method is fitted on the rows before the held-out ones.");
}

void ForecastView::RenderActions(const series::ForecastAnswer& a) {
    const char* type = series::ForecastNodeType(a.request.method);
    const std::string name = std::string(a.request.method == Method::Arima ? "ARIMA" : "Exp. Smoothing") + " (" + a.request.base.value + ")";
    if (add_to_graph) {
        const bool has_node = type[0] != '\0';
        if (ui::PrimaryButton("Add as a step", has_node, has_node ? nullptr : "The seasonal naive baseline has no node; pick a method", ui::ButtonSize::Small))
            Message(add_to_graph(type, name, series::NodeParameters(a.request), true), false);
        if (has_node && ImGui::IsItemHovered())
            ImGui::SetTooltip("Puts the %s node after the data in use: the rest of the graph gets fitted / residual columns and %d future rows with forecast / low / high",
                              a.request.method == Method::Arima ? "ARIMA" : "Exp. Smoothing", a.request.horizon);
        ExploreNextButton("Keep as a node");
        if (ui::SecondaryButton("Keep as a node", has_node, has_node ? nullptr : "The seasonal naive baseline has no node"))
            Message(add_to_graph(type, name, series::NodeParameters(a.request), false), false);
        ExploreNextButton("Export CSV...");
    }
    if (ui::SecondaryButton("Export CSV...")) {
        if (const auto path = FileDialogs::SaveFile("Export forecast", {{"CSV", "csv"}}, nullptr, "forecast.csv")) {
            std::ofstream out(*path, std::ios::binary);
            out << sv::ForecastCsv(a);
            Message(out ? "Saved " + *path : "Could not write " + *path, !out);
        }
    }
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        std::string text = sv::ForecastSentence(a) + "\n";
        const std::string held = sv::HoldoutSentence(a);
        if (!held.empty()) text += held + "\n";
        ImGui::SetClipboardText(text.c_str());
        Message("Copied the forecast.", false);
    }
}

}  // namespace cyxwiz
