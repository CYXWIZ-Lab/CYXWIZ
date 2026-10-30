#include "series_view.h"

#include "explore_plots.h"

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

// A setting after the one before it, on a new line when it would not fit.
void Setting(const char* label, float width) {
    ImGui::SameLine(0.0f, 16.0f);
    if (ImGui::GetContentRegionAvail().x < ImGui::CalcTextSize(label).x + width) ImGui::NewLine();
    Label(label);
}

std::string BaseKey(const std::string& dataset, const series::SeriesRequest& b) {
    return dataset + "|" + b.time + "|" + b.value + "|" + std::to_string(b.period);
}

// The axis ends of a series: "1949-01" .. "1960-12", numbers, or row numbers.
std::string AxisLabel(const series::Series& s, size_t i) {
    if (s.kind == series::TimeKind::Dates && i < s.labels.size()) {
        const std::string& l = s.labels[i];
        if (l.size() >= 10 && (s.step.name == "monthly" || s.step.name == "quarterly")) return l.substr(0, 7);
        if (l.size() >= 10 && s.step.name == "yearly") return l.substr(0, 4);
        return l;
    }
    if (s.kind == series::TimeKind::Numbers && i < s.t.size()) return explore_view::Figure(s.t[i]);
    return std::to_string(i + 1);
}

std::string Round(double v, const char* format = "%.3g") {
    if (!std::isfinite(v)) return "";
    char buf[32];
    std::snprintf(buf, sizeof(buf), format, v);
    return buf;
}

std::string P(double p) {
    if (!std::isfinite(p)) return "";
    if (p < 0.0001) return "< 0.0001";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", p);
    return buf;
}

float PlotHeight(float w) { return std::clamp(w * 0.3f, 140.0f, 240.0f); }

}  // namespace

void SeriesView::Message(const std::string& text, bool error) {
    message_ = text;
    message_error_ = error;
}

void SeriesView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to read one of its columns as a series over time.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        parts_job_.Reset();
        season_job_.Reset();
        stationarity_job_.Reset();
        correlogram_job_.Reset();
        message_.clear();
    }
    dataset_ = context.dataset;
    RenderControls(context, p);
    if (!message_.empty()) Wrapped(message_error_ ? t.error : t.text_dim, message_);
    ImGui::Spacing();
    if (base_.value.empty()) {
        Faint("Pick the number column to read over time.");
        return;
    }
    switch (look_) {
        case Look::Parts: RenderParts(context); break;
        case Look::Seasonality: RenderSeasonality(context); break;
        case Look::Stationarity: RenderStationarity(context); break;
        case Look::Correlogram: RenderCorrelogram(context); break;
    }
}

bool RenderSeriesPickers(const char* id, const ExploreContext& context, const DatasetProfile& p, plot::ColumnPicker& picker, series::SeriesRequest& base) {
    // Time candidates: date columns, texts that read as dates, numbers; the
    // Data Input's time column first; "(row order)" when none.
    std::vector<plot::ColumnSummary> times, numbers;
    std::string dated;
    int dated_count = 0;
    for (const auto& c : p.columns) {
        const bool temporal = c.facts.type == ColumnFacts::Type::Temporal;
        const bool date_text = c.facts.type == ColumnFacts::Type::Text && c.facts.date_share >= 0.9;
        plot::ColumnSummary s;
        s.name = c.facts.name;
        s.numeric = c.numeric;
        s.distinct = c.facts.distinct;
        if (temporal || date_text || c.numeric) times.push_back(s);
        if (temporal || date_text) {
            dated = c.facts.name;
            ++dated_count;
        }
        if (c.numeric && std::isfinite(c.min) && std::isfinite(c.max)) numbers.push_back(s);
    }
    const auto has = [&](const std::vector<plot::ColumnSummary>& list, const std::string& name) {
        return std::any_of(list.begin(), list.end(), [&](const plot::ColumnSummary& s) { return s.name == name; });
    };
    if (!base.time.empty() && !has(times, base.time)) base.time.clear();
    if (!base.value.empty() && !has(numbers, base.value)) base.value.clear();
    bool changed = false;
    if (base.time.empty() && base.value.empty()) {
        if (!context.time_column.empty() && has(times, context.time_column)) base.time = context.time_column;
        else if (dated_count == 1) base.time = dated;
        for (const auto& n : numbers)
            if (n.name != base.time) {
                base.value = n.name;
                break;
            }
        changed = true;
    }
    const std::string sid = id;
    ui::FlowRow row;
    row.Next(ImGui::CalcTextSize("Time").x + 240.0f);
    Label("Time");
    changed |= picker.Pick((sid + "_time").c_str(), base.time, times, false, "(row order)", 200.0f);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A date column, a text that reads as dates, or numbers; the rows are sorted by it. (row order): the table as it is.");
    row.Next(ImGui::CalcTextSize("Value").x + 240.0f);
    Label("Value");
    changed |= picker.Pick((sid + "_value").c_str(), base.value, numbers, true, nullptr, 200.0f);
    row.Next(ImGui::CalcTextSize("Period").x + 150.0f);
    Label("Period");
    changed |= ui::NumberField((sid + "_period").c_str(), &base.period, 0, 100000, 1, 56.0f,
                               "The seasonal period in rows (12 for months in a year). 0: found from the detrended periodogram");
    return changed;
}

void SeriesView::RenderControls(const ExploreContext& context, const DatasetProfile& p) {
    RenderSeriesPickers("##series", context, p, picker_, base_);
    static const char* const kLooks[] = {"Parts", "Seasonality", "Stationarity", "Correlogram"};
    int look = static_cast<int>(look_);
    ui::FlowRow looks;
    looks.Next(ImGui::CalcTextSize("LookPartsSeasonalityStationarityCorrelogram").x + 110.0f);
    Label("Look");
    if (ExploreChoices("##series_look", kLooks, 4, &look)) look_ = static_cast<Look>(look);
    RenderLookSettings();
}

void SeriesView::RenderLookSettings() {
    switch (look_) {
        case Look::Parts: {
            static const char* const kAlgorithm[] = {"STL", "Classical"};
            int algorithm = parts_.stl ? 0 : 1;
            Setting("Method", 160.0f);
            if (ExploreChoices("##series_algorithm", kAlgorithm, 2, &algorithm)) parts_.stl = algorithm == 0;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("STL: loess smoothing, a season that may change, every row filled. Classical: moving averages, half a period empty at each end");
            if (parts_.stl) {
                Setting("Seasonal window", 70.0f);
                if (ui::NumberField("##series_swin", &parts_.seasonal_window, 3, 999, 2, 48.0f, "Odd; larger smooths the season across the years")) {
                    if (parts_.seasonal_window % 2 == 0) ++parts_.seasonal_window;
                }
                Setting("Robust", 60.0f);
                ImGui::Checkbox("##series_robust", &parts_.robust);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Robustness iterations: outliers weigh less");
            } else {
                static const char* const kMethod[] = {"Additive", "Multiplicative"};
                int method = parts_.multiplicative ? 1 : 0;
                Setting("Season", 180.0f);
                if (ExploreChoices("##series_method", kMethod, 2, &method)) parts_.multiplicative = method == 1;
            }
            break;
        }
        case Look::Seasonality: break;
        case Look::Stationarity: {
            static const char* const kRegression[] = {"Level (c)", "Level and trend (ct)"};
            int regression = stationarity_.trend ? 1 : 0;
            Setting("Around", 220.0f);
            if (ExploreChoices("##series_regression", kRegression, 2, &regression)) stationarity_.trend = regression == 1;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("ct: stationary around a trend line counts as stationary");
            Setting("Rolling window", 70.0f);
            ui::NumberField("##series_window", &stationarity_.window, 0, 100000, 1, 56.0f, "Rows in the rolling mean and spread. 0: the period");
            break;
        }
        case Look::Correlogram: {
            Setting("Lags", 70.0f);
            ui::NumberField("##series_lags", &correlogram_.lags, 1, 1000, 1, 56.0f, "How many lags to show");
            static const char* const kOf[] = {"The series", "Its first difference"};
            int of = correlogram_.differenced ? 1 : 0;
            Setting("Of", 220.0f);
            if (ExploreChoices("##series_of", kOf, 2, &of)) correlogram_.differenced = of == 1;
            break;
        }
    }
}

void SeriesView::RenderSeriesLine(const series::Series& s) {
    Faint(sv::SeriesWords(s) + "; period " + sv::PeriodWords(s) + ".");
    Faint(sv::LeftOut(s));
}

// ---- Parts

void SeriesView::RenderParts(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    parts_.base = base_;
    const std::string key = BaseKey(context.dataset, base_) + "|parts|" + (parts_.stl ? "stl" + std::to_string(parts_.seasonal_window) + (parts_.robust ? "r" : "") : parts_.multiplicative ? "mul" : "add");
    const std::string dataset = context.dataset;
    const series::PartsRequest req = parts_;
    const explore::MissingTexts texts = context.missing_texts;
    parts_job_.Ask("Series: parts", key, [dataset, req, texts](LambdaTask&) { return series::RunParts(dataset, req, RunNow, texts); });
    const series::PartsAnswer* a = parts_job_.Answer(key);
    if (!a) {
        if (!parts_job_.Error().empty()) Wrapped(t.error, parts_job_.Error());
        else Faint("Splitting the series on the device...");
        return;
    }
    const series::Series& s = a->series;
    ExploreFigures("##series_parts_figures", {{"trend strength", Round(a->parts.trend_strength)},
                                              {"seasonal strength", Round(a->parts.seasonal_strength)},
                                              {"period", std::to_string(a->parts.period)},
                                              {"rows", std::to_string(s.used)}});
    Wrapped(t.text, sv::PartsSentence(*a));
    RenderSeriesLine(s);
    ImGui::Spacing();
    const float w = ImGui::GetContentRegionAvail().x;
    const std::string lo = AxisLabel(s, 0), hi = AxisLabel(s, s.t.size() - 1);
    Heading(("THE SERIES AND ITS TREND" + std::string(a->parts.method == "stl" ? " (STL)" : " (CLASSICAL)")).c_str());
    explore_plot::LinesOver(s.t, {{base_.value, s.values}, {"trend", a->parts.trend}}, w, PlotHeight(w), false, NAN, "", "", lo, hi);
    Heading(a->parts.method == "multiplicative" ? "THE SEASON (A FACTOR)" : "THE SEASON");
    explore_plot::LinesOver(s.t, {{"seasonal", a->parts.seasonal}}, w, PlotHeight(w) * 0.6f, false, NAN, "", "", lo, hi, 2);
    Heading(a->parts.method == "multiplicative" ? "WHAT IS LEFT (A FACTOR)" : "WHAT IS LEFT");
    explore_plot::LinesOver(s.t, {{"residual", a->parts.residual}}, w, PlotHeight(w) * 0.6f, false, NAN, "", "", lo, hi, 3);
    ImGui::Spacing();
    if (add_to_graph) {
        if (ui::PrimaryButton("Add as a step", true, nullptr, ui::ButtonSize::Small))
            Message(add_to_graph("TimeSeriesDecomposition", "Decomposition (" + base_.value + ")", series::NodeParameters(a->request), true), false);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Puts the Decomposition node after the data in use: the rest of the graph gets trend, seasonal and residual columns");
        ExploreNextButton("Export CSV...");
    }
    if (ui::SecondaryButton("Export CSV...")) Export("series_parts.csv", sv::PartsCsv(*a));
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((sv::PartsSentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

// ---- Seasonality

void SeriesView::RenderSeasonality(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    const std::string key = BaseKey(context.dataset, base_) + "|season";
    const std::string dataset = context.dataset;
    const series::SeriesRequest req = base_;
    const explore::MissingTexts texts = context.missing_texts;
    season_job_.Ask("Series: seasonality", key, [dataset, req, texts](LambdaTask&) { return series::RunSeasonality(dataset, req, RunNow, texts); });
    const series::SeasonalityAnswer* a = season_job_.Answer(key);
    if (!a) {
        if (!season_job_.Error().empty()) Wrapped(t.error, season_job_.Error());
        else Faint("Reading the periodogram on the device...");
        return;
    }
    const series::Series& s = a->series;
    const auto& r = a->seasonality;
    ExploreFigures("##series_season_figures", {{"period found", r.detected_period >= 2 ? std::to_string(r.detected_period) : "none"},
                                               {"share of the detrended power", r.detected_period >= 2 ? Round(100.0 * r.detected_share, "%.0f") + "%" : ""},
                                               {"season explains", r.detected_period >= 2 ? Round(100.0 * r.strength, "%.0f") + "%" : ""},
                                               {"rows", std::to_string(s.used)}});
    Wrapped(t.text, sv::SeasonalitySentence(*a));
    RenderSeriesLine(s);
    ImGui::Spacing();
    const float avail = ImGui::GetContentRegionAvail().x;
    const bool split = avail >= 820.0f && !a->season_by_position.empty();
    if (split && ImGui::BeginTable("##series_season_split", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("candidates", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("season", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
    }
    Heading("PERIODS BY THEIR SHARE OF THE DETRENDED POWER");
    const size_t shown = std::min<size_t>(r.candidate_periods.size(), 8);
    if (shown == 0) Faint("No period stands out.");
    else if (ImGui::BeginTable("##series_candidates", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings,
                               ImVec2(0.0f, TableHeight(shown + 1)))) {
        ImGui::TableSetupColumn("period", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("share", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("note", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        Heading("period");
        NumberCell("share of power", &t.text_dim);
        ImGui::TableNextColumn();
        Heading("");
        for (size_t i = 0; i < shown; ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool primary = r.candidate_periods[i] == r.detected_period;
            ImGui::TextColored(primary ? t.text : t.text_dim, "%d", r.candidate_periods[i]);
            NumberCell(Round(100.0 * r.candidate_strengths[i], "%.1f") + "%", primary ? nullptr : &t.text_dim);
            ImGui::TableNextColumn();
            if (primary) ImGui::TextColored(t.text_dim, "%s", "the season");
            else if (i < r.harmonic_of.size() && r.harmonic_of[i] > 0) ImGui::TextColored(t.text_faint, "harmonic of %d", r.harmonic_of[i]);
        }
        ImGui::EndTable();
    }
    if (!a->season_by_position.empty()) {
        if (split) ImGui::TableNextColumn();
        else ImGui::Spacing();
        const float w = ImGui::GetContentRegionAvail().x;
        Heading("THE SEASON BY POSITION (MEAN OF THE DETRENDED SERIES)");
        explore_plot::PositionBars(a->season_by_position, a->position_names, w, PlotHeight(w));
    }
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    if (add_to_graph) {
        if (ui::SecondaryButton("Keep as a node"))
            Message(add_to_graph("SeasonalityDetector", "Seasonality (" + base_.value + ")", series::NodeParameters(a->request), false), false);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds the Seasonality Detector node as a branch reading the data in use");
        ExploreNextButton("Export CSV...");
    }
    if (ui::SecondaryButton("Export CSV...")) Export("series_seasonality.csv", sv::SeasonalityCsv(*a));
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((sv::SeasonalitySentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

// ---- Stationarity

void SeriesView::RenderStationarity(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    stationarity_.base = base_;
    const std::string key = BaseKey(context.dataset, base_) + "|stationarity|" + (stationarity_.trend ? "ct" : "c") + "|" + std::to_string(stationarity_.window);
    const std::string dataset = context.dataset;
    const series::StationarityRequest req = stationarity_;
    const explore::MissingTexts texts = context.missing_texts;
    stationarity_job_.Ask("Series: stationarity", key, [dataset, req, texts](LambdaTask&) { return series::RunStationarity(dataset, req, RunNow, texts); });
    const series::StationarityAnswer* a = stationarity_job_.Answer(key);
    if (!a) {
        if (!stationarity_job_.Error().empty()) Wrapped(t.error, stationarity_job_.Error());
        else Faint("Running ADF and KPSS on the device...");
        return;
    }
    const series::Series& s = a->series;
    const auto& r = a->tests;
    ExploreFigures("##series_stationarity_figures",
                   {{"verdict", r.is_stationary ? "stationary" : "not stationary"},
                    {"suggested differencing", r.suggested_differencing < 0 ? "more than 2" : std::to_string(r.suggested_differencing)},
                    {"ADF lags", std::to_string(r.adf_lags)},
                    {"rolling window", std::to_string(a->window)}});
    Wrapped(t.text, sv::StationaritySentence(*a));
    RenderSeriesLine(s);
    ImGui::Spacing();
    Heading("ADF (A UNIT ROOT) AND KPSS (A STEADY LEVEL) ON THE SERIES AND ITS DIFFERENCES");
    const size_t rows = 1 + r.differenced.size();
    if (ImGui::BeginTable("##series_tests", 6, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings, ImVec2(0.0f, TableHeight(rows + 1)))) {
        ImGui::TableSetupColumn("series", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        for (const char* c : {"ADF", "p", "KPSS", "p"}) ImGui::TableSetupColumn(c, ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("verdict", ImGuiTableColumnFlags_WidthStretch, 1.6f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        Heading("series");
        for (const char* c : {"ADF", "p", "KPSS", "p"}) NumberCell(c, &t.text_dim);
        ImGui::TableNextColumn();
        Heading("verdict");
        const auto line = [&](const std::string& name, double adf, double adf_p, double kpss, double kpss_p, bool stationary, bool chosen) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(chosen ? t.text : t.text_dim, "%s", name.c_str());
            NumberCell(Round(adf));
            NumberCell(P(adf_p));
            NumberCell(Round(kpss));
            NumberCell(P(kpss_p));
            ImGui::TableNextColumn();
            ImGui::TextColored(stationary ? t.success : t.caution, "%s", stationary ? "stationary enough" : "not stationary");
        };
        line(base_.value, r.adf_statistic, r.adf_pvalue, r.kpss_statistic, r.kpss_pvalue, r.is_stationary, r.suggested_differencing == 0);
        for (const auto& d : r.differenced)
            line(d.d == 1 ? "first difference" : "second difference", d.adf_statistic, d.adf_pvalue, d.kpss_statistic, d.kpss_pvalue, d.stationary,
                 r.suggested_differencing == d.d);
        ImGui::EndTable();
    }
    Faint("Stationary enough: KPSS does not reject a steady level (p >= 0.05) and ADF rejects a unit root at 10% (p < 0.10). The ADF p-values are MacKinnon's; the KPSS p is clipped to 0.01 .. 0.10.");
    ImGui::Spacing();
    const float w = ImGui::GetContentRegionAvail().x;
    const std::string lo = AxisLabel(s, 0), hi = AxisLabel(s, s.t.size() - 1);
    Heading(("ROLLING MEAN AND SPREAD OVER " + std::to_string(a->window) + " ROWS").c_str());
    explore_plot::LinesOver(s.t, {{base_.value, s.values}, {"rolling mean", a->rolling_mean}}, w, PlotHeight(w), false, NAN, "", "", lo, hi);
    explore_plot::LinesOver(s.t, {{"rolling sd", a->rolling_sd}}, w, PlotHeight(w) * 0.6f, false, NAN, "", "", lo, hi, 2);
    ImGui::Spacing();
    if (add_to_graph) {
        const int d = r.suggested_differencing;
        if (d >= 1) {
            if (ui::PrimaryButton("Add a Differencing step", true, nullptr, ui::ButtonSize::Small))
                Message(add_to_graph("Differencing", "Difference " + base_.value, {{"value_col", base_.value}, {"lag", "1"}, {"order", std::to_string(d)}}, true), false);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Puts a Differencing node (order %d) after the data in use, so the rest of the graph works on a stationary series", d);
            ExploreNextButton("Keep as a node");
        }
        if (ui::SecondaryButton("Keep as a node"))
            Message(add_to_graph("StationarityTest", "Stationarity (" + base_.value + ")", series::NodeParameters(a->request), false), false);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds the Stationarity Test node as a branch reading the data in use");
        ExploreNextButton("Export CSV...");
    }
    if (ui::SecondaryButton("Export CSV...")) Export("series_stationarity.csv", sv::StationarityCsv(*a));
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((sv::StationaritySentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

// ---- Correlogram

void SeriesView::RenderCorrelogram(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    correlogram_.base = base_;
    const std::string key = BaseKey(context.dataset, base_) + "|correlogram|" + std::to_string(correlogram_.lags) + (correlogram_.differenced ? "|d" : "");
    const std::string dataset = context.dataset;
    const series::CorrelogramRequest req = correlogram_;
    const explore::MissingTexts texts = context.missing_texts;
    correlogram_job_.Ask("Series: correlogram", key, [dataset, req, texts](LambdaTask&) { return series::RunCorrelogram(dataset, req, RunNow, texts); });
    const series::CorrelogramAnswer* a = correlogram_job_.Answer(key);
    if (!a) {
        if (!correlogram_job_.Error().empty()) Wrapped(t.error, correlogram_job_.Error());
        else Faint("Computing the autocorrelations on the device...");
        return;
    }
    const series::Series& s = a->series;
    const auto& c = a->correlogram;
    ExploreFigures("##series_correlogram_figures",
                   {{"Ljung-Box Q(" + std::to_string(c.ljung_box_lags) + ")", Round(c.ljung_box_statistic, "%.1f") + ", p " + P(c.ljung_box_pvalue)},
                    {"significant ACF lags", std::to_string(c.significant_acf_lags.size()) + " of " + std::to_string(c.max_lag)},
                    {"PACF cuts off after", std::to_string(c.suggested_ar_order)},
                    {"ACF cuts off after", std::to_string(c.suggested_ma_order)}});
    Wrapped(t.text, sv::CorrelogramSentence(*a));
    RenderSeriesLine(s);
    ImGui::Spacing();
    const float avail = ImGui::GetContentRegionAvail().x;
    const bool split = avail >= 820.0f;
    if (split && ImGui::BeginTable("##series_correlogram_split", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
    }
    {
        const float w = ImGui::GetContentRegionAvail().x;
        Heading(("ACF" + std::string(s.period >= 2 ? ", THE SEASONAL LAGS MARKED" : "") + " (BARTLETT BAND AT 95%)").c_str());
        explore_plot::Stems(c.acf, c.confidence_lower, c.confidence_upper, w, PlotHeight(w), s.period >= 2 ? s.period : 0, "acf");
    }
    if (split) ImGui::TableNextColumn();
    else ImGui::Spacing();
    {
        const float w = ImGui::GetContentRegionAvail().x;
        Heading("PACF (BAND AT 95%)");
        explore_plot::Stems(c.pacf, c.pacf_confidence_lower, c.pacf_confidence_upper, w, PlotHeight(w), s.period >= 2 ? s.period : 0, "pacf");
    }
    if (split) ImGui::EndTable();
    Faint("A stem beyond the band is a lag the series remembers. The ACF band widens with the lag (Bartlett); the PACF band is z / sqrt(n).");
    ImGui::Spacing();
    if (add_to_graph) {
        if (ui::SecondaryButton("Keep as a node")) {
            const std::string acf = add_to_graph("ACFNode", "ACF (" + base_.value + ")", series::NodeParameters(a->request), false);
            add_to_graph("PACFNode", "PACF (" + base_.value + ")", series::NodeParameters(a->request), false);
            Message(acf + (correlogram_.differenced ? " The nodes read the column as it is; put a Differencing step before them for the differenced correlogram." : ""), false);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds the ACF and PACF nodes as branches reading the data in use");
        ExploreNextButton("Export CSV...");
    }
    if (ui::SecondaryButton("Export CSV...")) Export("series_correlogram.csv", sv::CorrelogramCsv(*a));
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((sv::CorrelogramSentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

void SeriesView::Export(const std::string& name, const std::string& csv) {
    if (const auto path = FileDialogs::SaveFile("Export series", {{"CSV", "csv"}}, nullptr, name.c_str())) {
        std::ofstream out(*path, std::ios::binary);
        out << csv;
        Message(out ? "Saved " + *path : "Could not write " + *path, !out);
    }
}

}  // namespace cyxwiz
