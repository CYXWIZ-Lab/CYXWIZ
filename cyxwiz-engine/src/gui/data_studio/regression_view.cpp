#include "regression_view.h"

#include "../../core/data_explore.h"
#include "../../core/file_dialogs.h"
#include "../../core/session_query_service.h"
#include "../../core/stats_presentation.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <cyxwiz/data_transform.h>

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace cyxwiz {

namespace {

namespace sv = stats_view;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

bool Measurable(const ProfiledColumn& c) { return c.numeric && std::isfinite(c.min) && std::isfinite(c.max) && c.max > c.min; }

std::string Key(const std::string& dataset, const stats::RegressionRequest& r) {
    std::string key = dataset + "|" + r.response + "|" + (r.polynomial ? "poly" + std::to_string(r.degree) : std::string("linear"));
    for (const auto& p : r.predictors) key += "|" + p;
    return key;
}

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

void Heading(const char* text) { ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text); }

void NumberCell(const std::string& text, const ImVec4* colour = nullptr) {
    ImGui::TableNextColumn();
    const float w = ImGui::CalcTextSize(text.c_str()).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
    if (colour) ImGui::TextColored(*colour, "%s", text.c_str());
    else ImGui::TextUnformatted(text.c_str());
}

// A scatter of (x, y) on its own scale, with an optional curve, and a guide:
// 1 the line y = 0, 2 the diagonal y = x. At most kPoints points are drawn.
constexpr size_t kPoints = 4000;
void Scatter(const std::vector<double>& xs, const std::vector<double>& ys, float width, float height,
             const std::vector<std::pair<double, double>>* curve, int guide, const char* x_label, const char* y_label) {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t n = std::min(xs.size(), ys.size());
    if (n == 0) return;
    double x0 = INFINITY, x1 = -INFINITY, y0 = INFINITY, y1 = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        x0 = std::min(x0, xs[i]);
        x1 = std::max(x1, xs[i]);
        y0 = std::min(y0, ys[i]);
        y1 = std::max(y1, ys[i]);
    }
    if (curve)
        for (const auto& [cx, cy] : *curve) {
            y0 = std::min(y0, cy);
            y1 = std::max(y1, cy);
        }
    if (guide == 2) {
        x0 = y0 = std::min(x0, y0);
        x1 = y1 = std::max(x1, y1);
    }
    if (guide == 1) {
        y0 = std::min(y0, 0.0);
        y1 = std::max(y1, 0.0);
    }
    if (!(x1 > x0)) x1 = x0 + 1.0;
    if (!(y1 > y0)) y1 = y0 + 1.0;
    const double px = (x1 - x0) * 0.04, py = (y1 - y0) * 0.06;
    x0 -= px;
    x1 += px;
    y0 -= py;
    y1 += py;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.plot_bg), 4.0f);
    const auto P = [&](double x, double y) {
        return ImVec2(at.x + static_cast<float>((x - x0) / (x1 - x0)) * width, at.y + height - static_cast<float>((y - y0) / (y1 - y0)) * height);
    };
    if (guide == 1) {
        const ImVec2 a = P(x0, 0.0), b = P(x1, 0.0);
        for (float x = a.x; x < b.x; x += 8.0f) dl->AddLine(ImVec2(x, a.y), ImVec2(std::min(x + 4.0f, b.x), a.y), ui::ToU32(t.text_faint), 1.0f);
    } else if (guide == 2) {
        dl->AddLine(P(x0, x0), P(x1, x1), ui::ToU32(t.text_faint), 1.0f);
    }
    ImVec4 dot = t.accent_text;
    dot.w = 0.7f;
    const size_t stride = std::max<size_t>(1, n / kPoints);
    for (size_t i = 0; i < n; i += stride) dl->AddCircleFilled(P(xs[i], ys[i]), 2.4f, ui::ToU32(dot));
    if (curve && curve->size() > 1) {
        std::vector<ImVec2> line;
        for (const auto& [cx, cy] : *curve) line.push_back(P(cx, cy));
        dl->AddPolyline(line.data(), static_cast<int>(line.size()), ui::ToU32(t.caution), ImDrawFlags_None, 2.0f);
    }
    const std::string lx = sv::Num(x0 + px), hx = sv::Num(x1 - px), ly = sv::Num(y0 + py), hy = sv::Num(y1 - py);
    dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(t.text_faint), (std::string(y_label) + "  " + hy).c_str());
    dl->AddText(ImVec2(at.x + 4.0f, at.y + height - ImGui::GetTextLineHeight() - 2.0f), ui::ToU32(t.text_faint), ly.c_str());
    ImGui::TextColored(t.text_faint, "%s", lx.c_str());
    ImGui::SameLine();
    const std::string mid = x_label;
    const float x_start = ImGui::GetCursorPosX();
    ImGui::SetCursorPosX(std::max(x_start, x_start + (width - ImGui::CalcTextSize(mid.c_str()).x) * 0.5f - ImGui::CalcTextSize(lx.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", mid.c_str());
    ImGui::SameLine();
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), at.x - ImGui::GetWindowPos().x + ImGui::GetScrollX() + width - ImGui::CalcTextSize(hx.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", hx.c_str());
}

void Bars(const std::vector<int64_t>& bins, double lo, double hi, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height));
    if (bins.empty()) return;
    const int64_t top = *std::max_element(bins.begin(), bins.end());
    if (top <= 0) return;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.plot_bg), 4.0f);
    const float bw = width / static_cast<float>(bins.size());
    ImVec4 fill = t.accent_text;
    fill.w = 0.75f;
    for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i] == 0) continue;
        const float h = (height - 8.0f) * static_cast<float>(bins[i]) / static_cast<float>(top);
        const float x = at.x + static_cast<float>(i) * bw;
        dl->AddRectFilled(ImVec2(x + 1.0f, at.y + height - h), ImVec2(x + std::max(2.0f, bw - 1.0f), at.y + height), ui::ToU32(fill), 2.0f,
                          ImDrawFlags_RoundCornersTop);
    }
    const std::string l = sv::Num(lo), h = sv::Num(hi);
    ImGui::TextColored(t.text_faint, "%s", l.c_str());
    ImGui::SameLine();
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), at.x - ImGui::GetWindowPos().x + ImGui::GetScrollX() + width - ImGui::CalcTextSize(h.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", h.c_str());
}

}  // namespace

void RegressionView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to predict one of its number columns.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        job_.Reset();
        message_.clear();
    }
    RenderControls(p);
    if (!message_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.error : t.text_dim, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();
    if (request_.response.empty() || request_.predictors.empty()) {
        Faint("Pick the column to predict and at least one column to predict it from.");
        return;
    }
    if (request_.polynomial && request_.predictors.size() != 1) {
        Faint("A polynomial uses one column: keep one in \"from\", or choose Linear.");
        return;
    }
    const std::string key = Key(context.dataset, request_);
    key_ = key;
    const std::string dataset = context.dataset;
    const stats::RegressionRequest req = request_;
    const explore::MissingTexts texts = context.missing_texts;
    job_.Ask("Regression: " + req.response, key, [dataset, req, texts](LambdaTask&) { return stats::RunRegression(dataset, req, RunNow, texts); });
    const stats::RegressionAnswer* a = job_.Answer(key);
    if (!a) {
        if (!job_.Error().empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.error, "%s", job_.Error().c_str());
            ImGui::PopTextWrapPos();
        } else {
            Faint("Fitting on the device...");
        }
        return;
    }
    if (a->rows > a->used)
        Faint(sv::Num(static_cast<double>(a->used)) + " rows with every value (" + sv::Num(static_cast<double>(a->rows - a->used)) +
              " left out).");
    const float avail = ImGui::GetContentRegionAvail().x;
    if (avail >= 780.0f && ImGui::BeginTable("##regression_split", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("coefficients", ImGuiTableColumnFlags_WidthStretch, 1.25f);
        ImGui::TableSetupColumn("fit", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        RenderCoefficients(*a);
        ImGui::TableNextColumn();
        RenderFit(*a);
        ImGui::EndTable();
    } else {
        RenderCoefficients(*a);
        ImGui::Spacing();
        RenderFit(*a);
    }
}

void RegressionView::RenderControls(const DatasetProfile& p) {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<plot::ColumnSummary> numbers;
    for (const auto& c : p.columns) {
        if (!Measurable(c)) continue;
        plot::ColumnSummary s;
        s.name = c.facts.name;
        s.numeric = true;
        numbers.push_back(std::move(s));
    }
    const auto valid = [&](const std::string& name) {
        const ProfiledColumn* c = p.Find(name);
        return c && Measurable(*c);
    };
    // Defaults: predict the last number column from the first other one.
    if (!valid(request_.response)) request_.response = numbers.empty() ? std::string() : numbers.back().name;
    std::erase_if(request_.predictors, [&](const std::string& n) { return !valid(n) || n == request_.response; });
    if (request_.predictors.empty())
        for (const auto& c : numbers)
            if (c.name != request_.response) {
                request_.predictors.push_back(c.name);
                break;
            }

    ui::FlowRow row;
    row.Next(ImGui::CalcTextSize("Predict").x + 230.0f);
    Label("Predict");
    if (picker_.Pick("##regression_y", request_.response, numbers, true, nullptr, 220.0f))
        std::erase(request_.predictors, request_.response);
    row.Next(ImGui::CalcTextSize("from").x + 330.0f);
    Label("from");
    std::vector<plot::ColumnSummary> others;
    for (const auto& c : numbers)
        if (c.name != request_.response) others.push_back(c);
    picker_.PickMany("##regression_x", request_.predictors, others, true, 320.0f);

    ui::FlowRow model;
    static const char* const kModels[] = {"Linear", "Polynomial (one column)"};
    int m = request_.polynomial ? 1 : 0;
    model.Next(ImGui::CalcTextSize("ModelLinearPolynomial (one column)").x + 50.0f);
    Label("Model");
    if (ui::SegmentedControl("##regression_model", kModels, 2, &m)) {
        request_.polynomial = m == 1;
        if (request_.polynomial && request_.predictors.size() > 1) request_.predictors.resize(1);
    }
    if (request_.polynomial) {
        model.Next(ImGui::CalcTextSize("degree").x + 200.0f);
        Label("degree");
        ui::NumberField("##regression_degree", &request_.degree, 1, 6, 1, 40.0f, "Highest power of the column");
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_faint, "typed, 1 .. 6");
    }
}

void RegressionView::RenderCoefficients(const stats::RegressionAnswer& a) {
    const ui::Tokens& t = ui::CurrentTokens();
    const LinearFit& f = a.fit;
    Heading("COEFFICIENTS");
    if (ImGui::BeginTable("##regression_coef", 6, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("term", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableSetupColumn("estimate", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("ci", ImGuiTableColumnFlags_WidthStretch, 1.6f);
        for (int i = 0; i < 3; ++i) ImGui::TableSetupColumn("figure", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        Heading("term");
        for (const char* h : {"estimate", "95% CI", "SE", "t", "p"}) NumberCell(h, &t.text_dim);
        for (size_t j = 0; j < f.coef.size(); ++j) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(f.names[j].c_str());
            NumberCell(sv::Num(f.coef[j]));
            NumberCell(sv::Range(f.ci_low[j], f.ci_high[j]));
            NumberCell(sv::Num(f.se[j]));
            NumberCell(sv::Num(f.t[j]));
            const bool clear = f.ci_low[j] > 0.0 || f.ci_high[j] < 0.0;
            NumberCell(sv::P(f.p[j]), j > 0 ? (clear ? &t.success : &t.text_dim) : nullptr);
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_bright, "%s", sv::Equation(a).c_str());
    ImGui::TextColored(t.text, "%s", sv::RegressionSentence(a).c_str());
    const std::string comparison = sv::ComparisonSentence(a);
    if (!comparison.empty()) ImGui::TextColored(t.text_dim, "%s", comparison.c_str());
    if (a.request.polynomial) ImGui::TextColored(t.text_faint, "Fitted on centred, scaled %s for accuracy, shown in its own units.", a.request.predictors[0].c_str());
    ImGui::PopTextWrapPos();
}

void RegressionView::RenderFit(const stats::RegressionAnswer& a) {
    const ui::Tokens& t = ui::CurrentTokens();
    const LinearFit& f = a.fit;
    Heading("FIT");
    ExploreFigures("##regression_fit", {{"R\xC2\xB2", sv::Num(f.r2)},
                                        {"adjusted R\xC2\xB2", sv::Num(f.adj_r2)},
                                        {"F", sv::Num(f.f) + " (" + std::to_string(f.df_model) + ", " + std::to_string(f.df_resid) + ")"},
                                        {"p", sv::P(f.f_p)},
                                        {"residual SE", sv::Num(f.rse)},
                                        {"MAE", sv::Num(f.mae)},
                                        {"RMSE", sv::Num(f.rmse)},
                                        {"AIC", sv::Aic(f.aic)}});
    ImGui::Spacing();
    const bool one = a.x.size() == a.y.size() && !a.x.empty();
    std::vector<const char*> views;
    if (one) views.push_back("data and fit");
    for (const char* v : {"residuals vs fitted", "Q-Q", "histogram", "actual vs predicted"}) views.push_back(v);
    plot_ = std::clamp(plot_, 0, static_cast<int>(views.size()) - 1);
    // The choices side by side when they fit, else a list.
    float needed = 0.0f;
    for (const char* v : views) needed += ImGui::CalcTextSize(v).x + 22.0f;
    if (needed <= ImGui::GetContentRegionAvail().x) {
        ui::SegmentedControl("##regression_plot", views.data(), static_cast<int>(views.size()), &plot_);
    } else {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Plot");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::BeginCombo("##regression_plot_list", views[static_cast<size_t>(plot_)])) {
            for (int i = 0; i < static_cast<int>(views.size()); ++i)
                if (ImGui::Selectable(views[static_cast<size_t>(i)], i == plot_)) plot_ = i;
            ImGui::EndCombo();
        }
    }
    RenderPlot(a, ImGui::GetContentRegionAvail().x);
    Faint(sv::ResidualSentence(a));
    ImGui::Spacing();
    if (add_to_graph) {
        if (ui::PrimaryButton("Add to the graph", true, nullptr, ui::ButtonSize::Small)) add_to_graph(a.request);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Adds a %s Regression node with these columns after the data in use: its fitted values and model stay in the graph",
                              a.request.polynomial ? "Polynomial" : "Linear");
        ImGui::SameLine();
    }
    if (ui::SecondaryButton("Export CSV...")) Export(a);
    ImGui::SameLine();
    if (ui::SecondaryButton("Copy")) {
        std::string text = sv::Equation(a) + "\n" + sv::RegressionSentence(a) + "\n";
        for (size_t j = 0; j < f.coef.size(); ++j)
            text += f.names[j] + ": " + sv::Num(f.coef[j]) + " (95% CI " + sv::Range(f.ci_low[j], f.ci_high[j]) + ", p " + sv::P(f.p[j]) + ")\n";
        text += "R2 " + sv::Num(f.r2) + ", AIC " + sv::Aic(f.aic) + "\n";
        ImGui::SetClipboardText(text.c_str());
        message_ = "Copied the fit.";
        message_error_ = false;
    }
}

void RegressionView::RenderPlot(const stats::RegressionAnswer& a, float width) {
    const LinearFit& f = a.fit;
    const bool one = a.x.size() == a.y.size() && !a.x.empty();
    const int view = one ? plot_ : plot_ + 1;  // 0 data and fit, 1 vs fitted, 2 Q-Q, 3 histogram, 4 actual vs predicted
    const float height = std::clamp(width * 0.55f, 140.0f, 260.0f);
    if (plotted_key_ != key_) {
        plotted_key_ = key_;
        // Sorted residuals in SE units against the normal quantiles; the residual histogram (device).
        std::vector<double> r = f.residuals;
        std::sort(r.begin(), r.end());
        const size_t n = r.size(), step = std::max<size_t>(1, n / 2000);
        qq_x_.clear();
        qq_y_.clear();
        for (size_t i = 0; i < n; i += step) {
            qq_x_.push_back(explore::NormalQuantile((static_cast<double>(i) + 0.5) / static_cast<double>(n)));
            qq_y_.push_back(f.rse > 0.0 ? r[i] / f.rse : r[i]);
        }
        bins_lo_ = r.empty() ? 0.0 : r.front();
        bins_hi_ = r.empty() ? 0.0 : r.back();
        bins_ = DataTransform::Histogram(f.residuals, 24, bins_lo_, bins_hi_);
    }
    const std::string& x_name = one ? a.request.predictors[0] : a.request.response;
    switch (view) {
        case 0: {
            // The fitted line or curve across the column's range.
            const auto [lo, hi] = std::minmax_element(a.x.begin(), a.x.end());
            std::vector<std::pair<double, double>> curve;
            for (int i = 0; i <= 120; ++i) {
                const double x = *lo + (*hi - *lo) * i / 120.0;
                double y = 0.0, power = 1.0;
                for (const double c : f.coef) {
                    y += c * power;
                    power *= x;
                }
                curve.emplace_back(x, y);
            }
            Scatter(a.x, a.y, width, height, &curve, 0, x_name.c_str(), a.request.response.c_str());
            break;
        }
        case 1: Scatter(f.fitted, f.residuals, width, height, nullptr, 1, "fitted", "residual"); break;
        case 2: Scatter(qq_x_, qq_y_, width, height, nullptr, 2, "normal quantile", "residual / SE"); break;
        case 3: Bars(bins_, bins_lo_, bins_hi_, width, height * 0.8f); break;
        default: Scatter(f.fitted, a.y, width, height, nullptr, 2, "predicted", "actual"); break;
    }
}

void RegressionView::Export(const stats::RegressionAnswer& a) {
    const std::string name = a.request.response + "_regression.csv";
    if (const auto path = FileDialogs::SaveFile("Export regression", {{"CSV", "csv"}}, nullptr, name.c_str())) {
        std::ofstream out(*path, std::ios::binary);
        out << sv::RegressionCsv(a);
        message_ = out ? "Saved " + *path : "Could not write " + *path;
        message_error_ = !out;
    }
}

}  // namespace cyxwiz
