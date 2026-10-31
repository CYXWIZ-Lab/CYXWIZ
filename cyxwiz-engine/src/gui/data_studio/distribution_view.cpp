#include "distribution_view.h"

#include "../../core/explore_presentation.h"
#include "../../core/file_dialogs.h"
#include "../../core/session_query_service.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace cyxwiz {

namespace {

namespace ev = explore_view;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

bool Measurable(const ProfiledColumn& c) {
    return c.numeric && std::isfinite(c.min) && std::isfinite(c.max) && c.max > c.min;
}

std::string Fixed(double v, int decimals) {
    if (!std::isfinite(v)) return "";
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

// A p-value: 4 decimals, or scientific when tiny.
std::string PValue(double p) {
    char buf[32];
    if (p < 1e-4) std::snprintf(buf, sizeof(buf), "%.1e", p);
    else std::snprintf(buf, sizeof(buf), "%.4f", p);
    return buf;
}

std::string Params(const DistributionFitResult& fit) {
    std::string out;
    for (const auto& [name, value] : fit.parameters) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), std::fabs(value) < 1 ? "%s = %.6f" : "%s = %.4g", name.c_str(), value);
        out += (out.empty() ? "" : ", ") + std::string(buf);
    }
    return out;
}

}  // namespace

void DistributionView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to see how a number column is spread.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        figures_.Reset();
        fit_.Reset();
        bins_.Reset();
        fit_index_ = 0;
        message_.clear();
    }
    const ProfiledColumn* c = p.Find(column_);
    if (!c || !Measurable(*c)) {
        c = nullptr;
        for (const auto& col : p.columns)
            if (Measurable(col)) {
                c = &col;
                break;
            }
        column_ = c ? c->facts.name : std::string();
        fit_index_ = 0;
    }
    if (!c) {
        ui::EmptyState(ICON_FA_CHART_SIMPLE, "No number column with a range", "A distribution is drawn for number columns that hold more than one value.");
        return;
    }

    const std::string dataset = context.dataset, column = column_;
    const std::string key = dataset + "\x1f" + column;
    const bool exact = p.exact;
    const size_t non_null = c->facts.non_null;
    figures_.Ask("Figures of " + column, key, [dataset, column, exact](LambdaTask&) { return explore::Figures(dataset, column, exact, RunNow); });
    fit_.Ask("Distribution fits of " + column, key, [dataset, column, non_null](LambdaTask&) {
        return explore::FitColumn(dataset, column, non_null, RunNow);
    });
    const explore::ColumnFigures* figures = figures_.Answer(key);
    const explore::ColumnFit* fit = fit_.Answer(key);
    const DistributionFitResult* chosen = fit && !fit->fits.empty() ? &fit->fits[std::min(fit_index_, fit->fits.size() - 1)] : nullptr;

    RenderControls(context, p);
    c = p.Find(column_);  // the picker may have changed it
    if (!c || column_ != column) return;
    {
        ui::FlowRow row;
        const auto part = [&](const ImVec4& colour, const std::string& text) {
            row.Next(ImGui::CalcTextSize(text.c_str()).x);
            ImGui::TextColored(colour, "%s", text.c_str());
        };
        part(t.text, column_);
        part(t.text_dim, ev::Number(c->facts.non_null) + " values \xC2\xB7 " + ev::Number(c->missing) + " missing");
        if (figures) part(t.text_dim, "skewness " + Fixed(figures->skewness, 2) + " (" + ev::SkewWords(figures->skewness) + ")");
        if (fit) {
            const DistributionFitResult* good = nullptr;
            for (const auto& f : fit->fits)
                if (f.good_fit) {
                    good = &f;
                    break;
                }
            if (good) part(t.success, "fits " + good->name);
            else if (!fit->fits.empty()) part(t.warning, "no distribution fits well");
            if (fit->sampled) part(t.text_faint, "(fitted on a " + ev::Number(explore::kFitValues) + "-value sample)");
        } else if (!fit_.Error().empty()) {
            part(t.error, fit_.Error());
        } else {
            part(t.text_faint, "fitting...");
        }
    }
    if (!message_.empty()) ImGui::TextColored(message_error_ ? t.error : t.success, "%s", message_.c_str());
    if (!figures_.Error().empty()) ImGui::TextColored(t.error, "%s", figures_.Error().c_str());
    ImGui::Spacing();

    const float avail = ImGui::GetContentRegionAvail().x;
    const bool stacked = avail < 700.0f;
    const float pane_w = stacked ? avail : std::min(420.0f, avail * 0.42f);
    if (!stacked) ImGui::BeginChild("##dist_left", ImVec2(avail - pane_w - ImGui::GetStyle().ItemSpacing.x, 0), ImGuiChildFlags_None);
    RenderHistogram(context, *c, figures, chosen);
    if (figures) RenderBoxPlot(*figures, fit);
    if (fit && chosen) RenderQq(*fit, *chosen);
    if (!stacked) {
        ImGui::EndChild();
        ImGui::SameLine();
    } else {
        ImGui::Spacing();
    }
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    ImGui::BeginChild("##dist_side", ImVec2(stacked ? 0.0f : pane_w, 0.0f),
                      ImGuiChildFlags_AlwaysUseWindowPadding | (stacked ? ImGuiChildFlags_AutoResizeY : ImGuiChildFlags_None));
    ImGui::PopStyleColor();
    RenderSide(figures, fit);
    ImGui::EndChild();
}

void DistributionView::RenderControls(const ExploreContext& context, const DatasetProfile& p) {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<plot::ColumnSummary> numbers;
    for (const auto& c : p.columns) {
        if (!Measurable(c)) continue;
        plot::ColumnSummary s;
        s.name = c.facts.name;
        s.numeric = true;
        numbers.push_back(std::move(s));
    }
    ui::FlowRow row;
    row.Next(ImGui::CalcTextSize("Column").x + 250.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Column");
    ImGui::SameLine();
    if (picker_.Pick("##dist_column", column_, numbers, true, nullptr, 240.0f)) {
        fit_index_ = 0;
        message_.clear();
    }
    row.Next(ImGui::CalcTextSize("Bins").x + 80.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Bins");
    ImGui::SameLine();
    // Typed; the histogram follows when the field is left.
    ui::NumberField("##dist_bins", &bins_wanted_, 5, 100, 0, 60.0f, "Bins of the histogram (the profile keeps 20)");
    row.Next(ui::ButtonWidth("Export CSV...", ui::ButtonSize::Small));
    const auto* figures = figures_.Answer(context.dataset + "\x1f" + column_);
    if (ui::SecondaryButton("Export CSV...", figures != nullptr, "The figures are still being computed") && figures)
        Export(context, *figures, fit_.Answer(context.dataset + "\x1f" + column_));
}

void DistributionView::RenderHistogram(const ExploreContext& context, const ProfiledColumn& c, const explore::ColumnFigures* figures,
                                       const DistributionFitResult* fit) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "HISTOGRAM%s%s", fit ? " \xC2\xB7 " : "", fit ? (fit->name + " fit").c_str() : "");
    std::vector<double> edges = c.hist_edges;
    std::vector<size_t> counts = c.hist_counts;
    if (bins_wanted_ != 20 || counts.empty()) {
        const std::string key = context.dataset + "\x1f" + c.facts.name + "\x1f" + std::to_string(bins_wanted_);
        const std::string dataset = context.dataset, column = c.facts.name;
        const double lo = c.min, hi = c.max;
        const int bins = bins_wanted_;
        bins_.Ask("Histogram of " + column, key, [dataset, column, lo, hi, bins](LambdaTask&) {
            return explore::CountBins(dataset, column, lo, hi, bins, RunNow);
        });
        const auto* b = bins_.Answer(key);
        if (!b) {
            ImGui::TextColored(bins_.Error().empty() ? t.text_faint : t.error, "%s", bins_.Error().empty() ? "Counting the values..." : bins_.Error().c_str());
            return;
        }
        edges = b->edges;
        counts = b->counts;
    }
    if (counts.empty() || edges.size() != counts.size() + 1) return;
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = 150.0f;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const double span = edges.back() - edges.front();
    const double bin_w = span / static_cast<double>(counts.size());
    const auto X = [&](double v) { return at.x + static_cast<float>((v - edges.front()) / span) * width; };
    double top = static_cast<double>(*std::max_element(counts.begin(), counts.end()));
    // The fit as expected counts per bin: values x bin width x density.
    std::vector<ImVec2> curve;
    const double total = figures ? static_cast<double>(figures->count) : static_cast<double>(c.facts.non_null);
    std::vector<double> expected;
    if (fit) {
        for (int k = 0; k <= 160; ++k) {
            const double x = edges.front() + span * k / 160.0;
            const double y = total * bin_w * explore::FitDensity(*fit, x);
            expected.push_back(std::isfinite(y) ? y : 0.0);
            top = std::max(top, expected.back());
        }
    }
    top = std::max(top, 1.0);
    const float base = at.y + height;
    const auto Y = [&](double v) { return base - static_cast<float>(v / top) * height; };
    int hovered = -1;
    for (size_t i = 0; i < counts.size(); ++i) {
        const float x0 = X(edges[i]) + 1.0f, x1 = std::max(X(edges[i + 1]) - 1.0f, x0 + 1.0f);
        dl->AddRectFilled(ImVec2(x0, std::min(Y(static_cast<double>(counts[i])), base - 1.0f)), ImVec2(x1, base),
                          ui::ToU32(ui::WithAlpha(t.accent_text, 0.5f)), 2.0f);
        if (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(ImVec2(x0, at.y), ImVec2(x1, base))) hovered = static_cast<int>(i);
    }
    for (size_t k = 0; k < expected.size(); ++k)
        curve.emplace_back(X(edges.front() + span * static_cast<double>(k) / 160.0), Y(expected[k]));
    if (curve.size() > 1) dl->AddPolyline(curve.data(), static_cast<int>(curve.size()), ui::ToU32(t.warning), ImDrawFlags_None, 2.0f);
    ImGui::Dummy(ImVec2(width, height + 2.0f));
    const std::string lo = ev::Figure(edges.front()), mid = ev::Figure((edges.front() + edges.back()) / 2), hi = ev::Figure(edges.back());
    ImGui::TextColored(t.text_faint, "%s", lo.c_str());
    ImGui::SameLine(std::max(0.0f, width / 2 - ImGui::CalcTextSize(mid.c_str()).x / 2));
    ImGui::TextColored(t.text_faint, "%s", mid.c_str());
    ImGui::SameLine(std::max(0.0f, width - ImGui::CalcTextSize(hi.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", hi.c_str());
    if (hovered >= 0) {
        const size_t i = static_cast<size_t>(hovered);
        ImGui::SetTooltip("%s to %s: %s", ev::Figure(edges[i]).c_str(), ev::Figure(edges[i + 1]).c_str(), ev::Number(counts[i]).c_str());
    }
}

void DistributionView::RenderBoxPlot(const explore::ColumnFigures& f, const explore::ColumnFit* values) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "BOX PLOT");
    const float width = ImGui::GetContentRegionAvail().x;
    const float h = 26.0f;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const double span = f.max - f.min;
    if (!(span > 0)) return;
    const auto X = [&](double v) { return at.x + static_cast<float>((v - f.min) / span) * width; };
    const float mid = at.y + h / 2;
    const ImU32 line = ui::ToU32(t.text_dim);
    dl->AddLine(ImVec2(X(f.whisker_lo), mid), ImVec2(X(f.q1), mid), line);
    dl->AddLine(ImVec2(X(f.q3), mid), ImVec2(X(f.whisker_hi), mid), line);
    dl->AddLine(ImVec2(X(f.whisker_lo), mid - 7), ImVec2(X(f.whisker_lo), mid + 7), line);
    dl->AddLine(ImVec2(X(f.whisker_hi), mid - 7), ImVec2(X(f.whisker_hi), mid + 7), line);
    dl->AddRectFilled(ImVec2(X(f.q1), at.y), ImVec2(std::max(X(f.q3), X(f.q1) + 1.0f), at.y + h), ui::ToU32(ui::WithAlpha(t.accent_text, 0.35f)), 3.0f);
    dl->AddLine(ImVec2(X(f.median), at.y), ImVec2(X(f.median), at.y + h), ui::ToU32(t.text), 2.0f);
    const float mx = X(f.mean);
    dl->AddQuadFilled(ImVec2(mx, mid - 5), ImVec2(mx + 5, mid), ImVec2(mx, mid + 5), ImVec2(mx - 5, mid), ui::ToU32(t.warning));
    // Values past the whiskers as dots (from the fitted values; at most 300).
    size_t dots = 0;
    if (values) {
        for (double v : values->values) {
            if (v >= f.whisker_lo && v <= f.whisker_hi) continue;
            if (++dots > 300) break;
            dl->AddCircleFilled(ImVec2(X(v), mid), 2.5f, ui::ToU32(t.error));
        }
    }
    ImGui::Dummy(ImVec2(width, h));
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Box Q1 %s to Q3 %s, line the median, diamond the mean, whiskers to the last value within 1.5 IQR%s",
                       ev::Figure(f.q1).c_str(), ev::Figure(f.q3).c_str(), dots ? ", dots beyond." : " (none beyond).");
    ImGui::PopTextWrapPos();
}

void DistributionView::RenderQq(const explore::ColumnFit& values, const DistributionFitResult& fit) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "Q-Q AGAINST %s", fit.name.c_str());
    const auto points = explore::QqPoints(values, fit, 200);
    if (points.empty()) return;
    double lo = INFINITY, hi = -INFINITY;
    for (const auto& [a, b] : points) {
        if (!std::isfinite(a) || !std::isfinite(b)) continue;
        lo = std::min({lo, a, b});
        hi = std::max({hi, a, b});
    }
    if (!(hi > lo)) return;
    const float size = std::min(190.0f, ImGui::GetContentRegionAvail().x * 0.45f);
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const auto P = [&](double a, double b) {
        return ImVec2(at.x + static_cast<float>((a - lo) / (hi - lo)) * size, at.y + size - static_cast<float>((b - lo) / (hi - lo)) * size);
    };
    dl->AddRectFilled(at, ImVec2(at.x + size, at.y + size), ui::ToU32(t.plot_bg), 3.0f);
    dl->AddLine(P(lo, lo), P(hi, hi), ui::ToU32(t.warning), 1.5f);
    for (const auto& [a, b] : points)
        if (std::isfinite(a) && std::isfinite(b)) dl->AddCircleFilled(P(a, b), 2.5f, ui::ToU32(t.accent_text));
    ImGui::Dummy(ImVec2(size, size));
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Across: %s quantiles. Up: the values.", fit.name.c_str());
    ImGui::TextColored(t.text_faint, "Points on the line: the values follow %s. A curve or an S shape means they do not.", fit.name.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
}

void DistributionView::RenderSide(const explore::ColumnFigures* f, const explore::ColumnFit* fit) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "FIGURES");
    if (!f) {
        ImGui::TextColored(t.text_faint, "Computing...");
        return;
    }
    const std::string mode = ev::Figure(f->mode) + " (" + ev::Count(f->mode_count, "time", "times") + ")";
    ExploreFigures("##dist_figures", {{"Count", ev::Number(f->count)}, {"Mean", ev::Figure(f->mean)}, {"Median", ev::Figure(f->median)},
                                      {"Mode", mode}, {"Std", ev::Figure(f->std)}, {"Variance", ev::Figure(f->variance)},
                                      {"SEM", ev::Figure(f->sem)}, {"Min", ev::Figure(f->min)}, {"Max", ev::Figure(f->max)},
                                      {"Range", ev::Figure(f->range)}, {"Sum", ev::Figure(f->sum)}, {"Q1", ev::Figure(f->q1)},
                                      {"Q3", ev::Figure(f->q3)}, {"IQR", ev::Figure(f->iqr)}, {"Skewness", Fixed(f->skewness, 3)},
                                      {"Kurtosis", Fixed(f->kurtosis, 3)}, {"CV", Fixed(f->cv, 1) + "%"}});
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "PERCENTILES");
    static const char* kLabels[] = {"5%", "10%", "25%", "50%", "75%", "90%", "95%"};
    if (ImGui::BeginTable("##dist_percentiles", 7, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoSavedSettings)) {
        for (const char* label : kLabels) ImGui::TableSetupColumn(label);
        ImGui::TableHeadersRow();
        ImGui::TableNextRow();
        for (double v : f->percentiles) {
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(ev::Figure(v).c_str());
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "FITS, BEST AIC FIRST");
    if (!fit) {
        ImGui::TextColored(fit_.Error().empty() ? t.text_faint : t.error, "%s", fit_.Error().empty() ? "Fitting..." : fit_.Error().c_str());
        return;
    }
    if (ImGui::BeginTable("##dist_fits", 6, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("##good");
        ImGui::TableSetupColumn("distribution", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("Exponential").x);
        for (const char* h : {"AIC", "BIC", "KS", "p"}) ImGui::TableSetupColumn(h);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < fit->fits.size(); ++i) {
            const auto& d = fit->fits[i];
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ImGui::Selectable(d.good_fit ? ICON_FA_CHECK : "##none", i == fit_index_, ImGuiSelectableFlags_SpanAllColumns)) fit_index_ = i;
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(d.name.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(Fixed(d.aic, 1).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(Fixed(d.bic, 1).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(Fixed(d.ks_statistic, 4).c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(d.good_fit ? t.success : t.error, "%s", PValue(d.ks_p_value).c_str());
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "%s = good fit (KS p above 0.05). Lower AIC and BIC fit better. Click a fit to draw it.", ICON_FA_CHECK);
    if (!fit->fits.empty()) {
        const auto& d = fit->fits[std::min(fit_index_, fit->fits.size() - 1)];
        ImGui::Text("%s: %s; log-likelihood %s", d.name.c_str(), Params(d).c_str(), Fixed(d.log_likelihood, 2).c_str());
        ImGui::TextColored(d.good_fit ? t.success : t.error, "%s", d.good_fit ? "Good fit: KS p-value above 0.05." : "Poor fit: KS p-value 0.05 or below.");
    }
    if (fit->sampled) ImGui::TextColored(t.text_faint, "Fitted on a %s-value sample of the column.", ev::Number(explore::kFitValues).c_str());
    ImGui::PopTextWrapPos();
}

void DistributionView::Export(const ExploreContext& context, const explore::ColumnFigures& f, const explore::ColumnFit* fit) {
    std::vector<std::vector<std::string>> rows = {{"figure", "value"},
                                                  {"Count", std::to_string(f.count)},
                                                  {"Mean", Fixed(f.mean, 6)},
                                                  {"Median", Fixed(f.median, 6)},
                                                  {"Mode", Fixed(f.mode, 6)},
                                                  {"Mode count", std::to_string(f.mode_count)},
                                                  {"Std", Fixed(f.std, 6)},
                                                  {"Variance", Fixed(f.variance, 6)},
                                                  {"SEM", Fixed(f.sem, 6)},
                                                  {"Min", Fixed(f.min, 6)},
                                                  {"Max", Fixed(f.max, 6)},
                                                  {"Range", Fixed(f.range, 6)},
                                                  {"Sum", Fixed(f.sum, 6)},
                                                  {"Q1", Fixed(f.q1, 6)},
                                                  {"Q3", Fixed(f.q3, 6)},
                                                  {"IQR", Fixed(f.iqr, 6)},
                                                  {"Skewness", Fixed(f.skewness, 6)},
                                                  {"Kurtosis (excess)", Fixed(f.kurtosis, 6)},
                                                  {"CV %", Fixed(f.cv, 6)}};
    static const char* kLabels[] = {"P5", "P10", "P25", "P50", "P75", "P90", "P95"};
    for (size_t i = 0; i < f.percentiles.size() && i < 7; ++i) rows.push_back({kLabels[i], Fixed(f.percentiles[i], 6)});
    if (fit) {
        rows.push_back({});
        rows.push_back({"distribution", "aic", "bic", "ks", "p", "log_likelihood", "parameters"});
        for (const auto& d : fit->fits)
            rows.push_back({d.name, Fixed(d.aic, 4), Fixed(d.bic, 4), Fixed(d.ks_statistic, 6), PValue(d.ks_p_value), Fixed(d.log_likelihood, 4), Params(d)});
    }
    const std::string name = context.shown + "_" + column_ + "_distribution.csv";
    if (const auto path = FileDialogs::SaveFile("Export distribution", {{"CSV", "csv"}}, nullptr, name.c_str())) {
        std::ofstream out(*path, std::ios::binary);
        out << ev::Csv(rows);
        message_error_ = !out.good();
        message_ = message_error_ ? "Could not write " + *path : "Saved " + *path;
    }
}

}  // namespace cyxwiz
