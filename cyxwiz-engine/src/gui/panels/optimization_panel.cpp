#include "optimization_panel.h"

#include "optimization_plots.h"

#include "../../core/file_dialogs.h"
#include "../../core/linear_algebra_presentation.h"
#include "../../core/session_query_service.h"
#include "../data_studio/explore_plots.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>

namespace cyxwiz {

namespace {

using optcalc::Op;
using optcalc::Tone;

constexpr double kInf = std::numeric_limits<double>::infinity();

std::string Plain(double v) {
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    char buffer[40];
    std::snprintf(buffer, sizeof buffer, "%.17g", v);
    return buffer;
}

// A typed bound: a number, a constant formula, inf or -inf.
bool BoundFrom(const std::string& text, double& value, std::string& error) {
    return optcalc::ReadValue(text, value, error, true);
}

ImVec4 ToneColour(Tone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    return tone == Tone::Good ? t.success : (tone == Tone::Warning ? t.warning : t.text);
}

// Points written as "(1, 1), (-2.805, 3.131)" in an example's minima.
std::vector<std::pair<double, double>> PointsIn(std::string text) {
    std::vector<std::pair<double, double>> out;
    for (size_t i; (i = text.find("\xE2\x88\x92")) != std::string::npos;) text.replace(i, 3, "-");
    size_t at = 0;
    while ((at = text.find('(', at)) != std::string::npos) {
        const size_t comma = text.find(',', at), close = text.find(')', at);
        if (comma == std::string::npos || close == std::string::npos || comma > close) break;
        double x = 0, y = 0;
        std::string error;
        if (optcalc::ReadValue(text.substr(at + 1, comma - at - 1), x, error) &&
            optcalc::ReadValue(text.substr(comma + 1, close - comma - 1), y, error))
            out.push_back({x, y});
        at = close + 1;
    }
    return out;
}

// The objective of a two-variable QP as a formula of x and y, for its map.
std::string QuadraticFormula(const optcalc::ProgramRequest& p) {
    return "0.5*(" + Plain(p.Q[0][0]) + "*x^2 + 2*" + Plain(p.Q[0][1]) + "*x*y + " + Plain(p.Q[1][1]) + "*y^2) + " +
           Plain(p.objective[0]) + "*x + " + Plain(p.objective[1]) + "*y";
}

void Paragraph(const std::string& text, const ImVec4* colour = nullptr) {
    ImGui::PushTextWrapPos(0.0f);
    if (colour) ImGui::TextColored(*colour, "%s", text.c_str());
    else ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
}

// A table of text rows, the first row the header.
void TextTable(const char* id, const std::vector<std::vector<std::string>>& rows, const std::vector<ImVec4>* tones = nullptr,
               size_t tone_column = 0) {
    if (rows.empty()) return;
    const ui::Tokens& t = ui::CurrentTokens();
    const int columns = static_cast<int>(rows.front().size());
    const float height = (std::min)(ImGui::GetTextLineHeightWithSpacing() * (static_cast<float>(rows.size()) + 1.0f), 300.0f);
    if (!ImGui::BeginTable(id, columns, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                           ImVec2(0, height)))
        return;
    ImGui::TableSetupScrollFreeze(1, 1);
    for (const auto& name : rows.front()) ImGui::TableSetupColumn(name.c_str());
    ImGui::TableHeadersRow();
    for (size_t r = 1; r < rows.size(); ++r) {
        ImGui::TableNextRow();
        for (size_t c = 0; c < rows[r].size(); ++c) {
            ImGui::TableNextColumn();
            if (tones && c == tone_column && r - 1 < tones->size()) ImGui::TextColored((*tones)[r - 1], "%s", rows[r][c].c_str());
            else if (c == 0) ImGui::TextColored(t.text_dim, "%s", rows[r][c].c_str());
            else ImGui::TextUnformatted(rows[r][c].c_str());
        }
    }
    ImGui::EndTable();
}

}  // namespace

OptimizationPanel::OptimizationPanel() {
    optimizers_ = optcalc::OfferedOptimizers();
    linear_ = optcalc::LinearExamples().front().request;
    quadratic_ = optcalc::QuadraticExamples().front().request;
}

void OptimizationPanel::Open(const std::string& id) {
    visible_ = true;
    if (const optcalc::OpInfo* info = optcalc::FindOp(id)) SetOp(info->op);
    ImGui::SetWindowFocus("###OptimizationCalculus");
}

void OptimizationPanel::SetOp(Op op) {
    if (op == op_) return;
    op_ = op;
    lower_text_.clear();  // the shown program's bounds are read again
    Changed();
}

void OptimizationPanel::Changed() {
    dirty_ = true;
    message_.clear();
    graph_message_.clear();
}

optcalc::ProgramRequest& OptimizationPanel::Program() { return op_ == Op::Quadratic ? quadratic_ : linear_; }

void OptimizationPanel::ApplyExample(const optcalc::FormulaExample& e) {
    surface_ = e.formula;
    x0_ = e.x0;
    y0_ = e.y0;
    x_min_ = e.x_min;
    x_max_ = e.x_max;
    y_min_ = e.y_min;
    y_max_ = e.y_max;
    minima_ = e.minima;
    shown_steps_ = 0;
    Changed();
}

void OptimizationPanel::ApplyExample(const optcalc::LineExample& e) {
    line_ = e.formula;
    a_ = e.a;
    b_ = e.b;
    at_ = Plain(e.x);
    Changed();
}

std::string OptimizationPanel::Key() const {
    std::string k = std::to_string(static_cast<int>(op_)) + "|";
    switch (op_) {
        case Op::Optimizers:
            k += surface_ + "|" + Plain(x0_) + "," + Plain(y0_) + "|" + std::to_string(steps_);
            for (const auto& o : optimizers_)
                k += "|" + std::to_string(static_cast<int>(o.type)) + ":" + Plain(o.learning_rate) + ":" + Plain(o.momentum) + (o.on ? "1" : "0");
            k += "|" + Plain(x_min_) + Plain(x_max_) + Plain(y_min_) + Plain(y_max_);
            break;
        case Op::Linear:
        case Op::Quadratic: {
            if (op_ == Op::Quadratic && qp_source_ == 1) {
                k += "table|" + portfolio_.table + "|" + portfolio_.target + (portfolio_.long_only ? "|long" : "|short") + "|" +
                     std::to_string(refresh_);
                for (const auto& a : portfolio_.assets) k += "|" + a;
                break;
            }
            const auto& p = op_ == Op::Quadratic ? quadratic_ : linear_;
            k += p.maximize ? "max" : "min";
            for (const auto& v : p.variables) k += "|" + v;
            for (const double c : p.objective) k += "," + Plain(c);
            for (const auto& row : p.Q)
                for (const double q : row) k += ";" + Plain(q);
            for (const auto& r : p.rows) {
                k += "|" + r.name + std::to_string(static_cast<int>(r.sense)) + Plain(r.rhs);
                for (const double c : r.coefficients) k += "," + Plain(c);
            }
            for (const auto& b : p.bounds) k += "|" + Plain(b.lower) + ":" + Plain(b.upper);
            break;
        }
        case Op::Integral: k += line_ + "|" + a_ + "|" + b_ + "|" + std::to_string(rule_) + "|" + std::to_string(n_); break;
        case Op::Derivative: k += line_ + "|" + at_ + "|" + std::to_string(difference_) + "|" + step_text_; break;
        case Op::Curvature:
            k += surface_ + "|" + point_x_ + "|" + point_y_ + "|" + Plain(x_min_) + Plain(x_max_) + Plain(y_min_) + Plain(y_max_);
            break;
    }
    return k;
}

void OptimizationPanel::Render() {
    if (!visible_) return;
    ImGui::SetNextWindowSize(ImVec2(1180, 760), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(ICON_FA_SQUARE_ROOT_VARIABLE " Optimization and Calculus###OptimizationCalculus", &visible_)) {
        ImGui::End();
        return;
    }
    if (dirty_) {
        dirty_ = false;
        key_ = Key();
    }
    if (key_ != stopped_key_) {
        Job request;
        request.op = op_;
        if (op_ == Op::Optimizers) {
            request.minimize_request.formula = surface_;
            request.minimize_request.x0 = x0_;
            request.minimize_request.y0 = y0_;
            request.minimize_request.steps = steps_;
            request.minimize_request.optimizers = optimizers_;
        }
        if (op_ == Op::Linear || op_ == Op::Quadratic) request.program_request = Program();
        if (op_ == Op::Quadratic && qp_source_ == 1) {
            request.from_table = true;
            request.portfolio = portfolio_;
        }
        optcalc::IntegralRequest integral{line_, a_, b_, static_cast<QuadratureRule>(rule_), n_};
        optcalc::DerivativeRequest derivative{line_, at_, static_cast<DifferenceRule>(difference_), 0.0};
        if (!step_text_.empty()) {
            std::string error;
            optcalc::ReadValue(step_text_, derivative.h, error);
        }
        optcalc::CurvatureRequest curvature{surface_, point_x_, point_y_, x_min_, x_max_, y_min_, y_max_, 40};
        const double x_min = x_min_, x_max = x_max_, y_min = y_min_, y_max = y_max_;
        job_.Ask(std::string("Optimization and Calculus: ") + optcalc::InfoOf(op_).label, key_,
                 [request, integral, derivative, curvature, x_min, x_max, y_min, y_max](LambdaTask& task) {
            Job job = request;
            const auto start = std::chrono::steady_clock::now();
            const auto stop = [&task] { return task.ShouldStop(); };
            switch (job.op) {
                case Op::Optimizers: {
                    const Expression f = Expression::Parse(job.minimize_request.formula, {"x", "y"});
                    if (f.Valid()) job.map = optcalc::MapOf(f, x_min, x_max, y_min, y_max);
                    job.minimize = optcalc::RunMinimize(job.minimize_request, stop);
                    break;
                }
                case Op::Linear:
                case Op::Quadratic:
                    if (job.from_table) {
                        const QueryRunner run = [](const QueryRequest& q) { return SessionQueryService::Instance().RunNow(q); };
                        const optcalc::PortfolioRead read = optcalc::PortfolioFromTable(job.portfolio, run);
                        if (!read.ok()) {
                            job.program.error = read.error;
                            break;
                        }
                        job.program_request = read.request;
                        job.portfolio_note = read.note;
                    }
                    job.program = optcalc::RunProgram(job.program_request);
                    if (job.program_request.quadratic && job.program_request.variables.size() == 2 && job.program.plot.drawn) {
                        const auto& plot = job.program.plot;
                        job.map = optcalc::MapOf(Expression::Parse(QuadraticFormula(job.program_request), {"x", "y"}), plot.x_min,
                                                 plot.x_max, plot.y_min, plot.y_max, 90, 10);
                    }
                    break;
                case Op::Integral: job.integral = optcalc::RunIntegral(integral); break;
                case Op::Derivative: job.derivative = optcalc::RunDerivative(derivative); break;
                case Op::Curvature: {
                    job.curvature = optcalc::RunCurvature(curvature);
                    const Expression f = Expression::Parse(curvature.formula, {"x", "y"});
                    if (f.Valid()) job.map = optcalc::MapOf(f, x_min, x_max, y_min, y_max);
                    break;
                }
            }
            if (task.ShouldStop()) job.error = "Cancelled.";
            job.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            return job;
        });
    }
    if (const Job* answer = job_.Answer(key_); answer && shown_key_ != key_) {
        shown_ = std::make_shared<const Job>(*answer);
        shown_key_ = key_;
    }

    RenderOperations();
    ImGui::Spacing();
    ui::BeginSettingsColumn("optimization_calculus", 400.0f);
    RenderInputs();
    ui::BeginResultsArea();
    RenderAnswer();
    ui::EndResultsArea();
    ImGui::End();
}

void OptimizationPanel::RenderOperations() {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImGuiStyle& style = ImGui::GetStyle();
    bool first = true;
    for (const std::string& group : optcalc::Groups()) {
        float width = ImGui::CalcTextSize(group.c_str()).x;
        for (const auto& info : optcalc::Ops())
            if (group == info.group) width += style.ItemSpacing.x + ImGui::CalcTextSize(info.label).x + 2.0f * style.FramePadding.x;
        if (!first) {
            ImGui::SameLine(0, 18.0f);
            if (ImGui::GetContentRegionAvail().x < width) ImGui::NewLine();
        }
        first = false;
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "%s", group.c_str());
        for (const auto& info : optcalc::Ops()) {
            if (group != info.group) continue;
            ImGui::SameLine();
            ImGui::PushID(info.id);
            if (ui::GhostButton(info.label, true, nullptr, info.op == op_)) SetOp(info.op);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", info.hint);
            ImGui::PopID();
        }
    }
}

bool OptimizationPanel::RenderFormula(const char* id, std::string& text, const char* label) {
    ui::FieldLabel(label);
    ImGui::SetNextItemWidth(-1);
    const bool changed = ImGui::InputText(id, &text);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Numbers, x%s, + - * / ^, pi, e, and sin cos tan exp log sqrt abs ...", id[2] == 's' ? " and y" : "");
    return changed;
}

bool OptimizationPanel::RenderRange() {
    bool changed = false;
    ui::FieldLabel("Range shown");
    ui::NumberFieldOptions o;
    o.format = "%.4g";
    o.width = 64.0f;
    const auto pair = [&](const char* name, double& lo, double& hi) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(name);
        ImGui::SameLine();
        ImGui::PushID(name);
        if (ui::NumberField("##lo", &lo, o)) changed = true;
        ImGui::SameLine();
        ImGui::TextUnformatted("to");
        ImGui::SameLine();
        if (ui::NumberField("##hi", &hi, o)) changed = true;
        ImGui::PopID();
    };
    pair("x", x_min_, x_max_);
    pair("y", y_min_, y_max_);
    if (changed) Changed();
    return changed;
}

void OptimizationPanel::RenderInputs() {
    switch (op_) {
        case Op::Optimizers: RenderMinimizeInputs(); break;
        case Op::Linear:
        case Op::Quadratic: RenderProgramInputs(); break;
        case Op::Integral:
        case Op::Derivative: RenderLineInputs(); break;
        case Op::Curvature: RenderCurvatureInputs(); break;
    }
}

void OptimizationPanel::RenderMinimizeInputs() {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::SettingsSection("Formula");
    if (RenderFormula("##surface", surface_, "f(x, y)")) {
        minima_.clear();
        shown_steps_ = 0;
        Changed();
    }
    ImGui::TextColored(t.text_dim, "Examples");
    for (const auto& e : optcalc::SurfaceExamples()) {
        ExploreNextButton(e.name);
        ImGui::PushID(e.name);
        if (ui::LinkButton(e.name)) ApplyExample(e);
        ImGui::PopID();
    }
    ImGui::NewLine();
    ImGui::Spacing();
    ui::SettingsSection("Start");
    ui::NumberFieldOptions o;
    o.format = "%.4g";
    o.width = 70.0f;
    o.step = 0.5;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("x");
    ImGui::SameLine();
    if (ui::NumberField("##x0", &x0_, o)) Changed();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("y");
    ImGui::SameLine();
    if (ui::NumberField("##y0", &y0_, o)) Changed();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Steps");
    ImGui::SameLine();
    if (ui::NumberField("##steps", &steps_, 1, 100000, 100, 80.0f)) {
        shown_steps_ = 0;
        Changed();
    }
    ImGui::Spacing();
    ui::SettingsSection("Optimizers (the ones training uses)");
    for (size_t i = 0; i < optimizers_.size(); ++i) {
        auto& o2 = optimizers_[i];
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Checkbox("##on", &o2.on)) Changed();
        ImGui::SameLine();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float r = ImGui::GetTextLineHeight() * 0.3f;
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + r, p.y + ImGui::GetFrameHeight() * 0.5f), r, ui::ToU32(ui::SeriesColour(t, i)));
        ImGui::Dummy(ImVec2(2 * r + 4.0f, ImGui::GetFrameHeight()));
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(optcalc::OptimizerName(o2).c_str());
        ui::NumberFieldOptions lr;
        lr.format = "%.6g";
        lr.width = 62.0f;
        lr.min = 1e-12;
        lr.hint = "Learning rate";
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "lr");
        ImGui::SameLine();
        if (ui::NumberField("##lr", &o2.learning_rate, lr)) Changed();
        if (o2.type == OptimizerType::SGD || o2.type == OptimizerType::RMSprop) {
            ui::NumberFieldOptions m;
            m.format = "%.3g";
            m.width = 44.0f;
            m.min = 0.0;
            m.max = 0.999;
            m.hint = "Momentum";
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "momentum");
            ImGui::SameLine();
            if (ui::NumberField("##m", &o2.momentum, m)) Changed();
        }
        ImGui::PopID();
    }
    ImGui::Spacing();
    RenderRange();
}

void OptimizationPanel::ResizeProgram(int variables) {
    auto& p = Program();
    const size_t n = static_cast<size_t>((std::max)(1, variables));
    const VariableBound fresh = p.quadratic ? VariableBound{-kInf, kInf} : VariableBound{};
    while (p.variables.size() < n) p.variables.push_back("x" + std::to_string(p.variables.size() + 1));
    p.variables.resize(n);
    p.objective.resize(n, 0.0);
    p.bounds.resize(n, fresh);
    for (auto& r : p.rows) r.coefficients.resize(n, 0.0);
    if (p.quadratic) {
        // A new variable gets 1 on Q's diagonal, so Q stays positive definite.
        const size_t old = p.Q.size();
        p.Q.resize(n);
        for (auto& row : p.Q) row.resize(n, 0.0);
        for (size_t i = old; i < n; ++i) p.Q[i][i] = 1.0;
    }
    lower_text_.clear();
    Changed();
}

void OptimizationPanel::RenderPortfolioInputs() {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::FieldLabel("Table of returns (one column per asset, one row per period)");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##table", portfolio_.table.empty() ? "Pick a table" : portfolio_.table.c_str())) {
        tables_ = SessionQueryService::Instance().QueryableNames();
        if (tables_.empty()) ImGui::TextColored(t.text_dim, "No table yet: load a Data Input, or run a node's result.");
        for (const auto& name : tables_)
            if (ImGui::Selectable(name.c_str(), name == portfolio_.table) && name != portfolio_.table) {
                portfolio_.table = name;
                portfolio_.assets.clear();
                Changed();
            }
        ImGui::EndCombo();
    }
    if (!portfolio_.table.empty() && columns_for_ != portfolio_.table && columns_task_ == 0) {
        // The number columns, from the table's schema (a background query).
        columns_for_ = portfolio_.table;
        table_columns_.clear();
        picked_.clear();
        columns_error_.clear();
        QueryRequest q;
        q.sql = "SELECT * FROM " + SessionQueryEngine::QuoteIdentifier(portfolio_.table) + " LIMIT 0";
        q.label = "Optimization and Calculus: " + portfolio_.table;
        const std::string table = portfolio_.table;
        columns_task_ = SessionQueryService::Instance().Submit(
            q,
            [this, table](const QueryResult& r) {
                columns_task_ = 0;
                if (table != portfolio_.table) return;
                if (!r.ok || !r.table) {
                    columns_error_ = r.error.empty() ? "Cannot read " + table + "." : r.error;
                    return;
                }
                std::string error;
                const QueryRunner schema = [&r](const QueryRequest&) { return r; };
                table_columns_ = linalg::NumberColumns(table, schema, error);
                picked_.assign(table_columns_.size(), 1);
                if (table_columns_.size() < 2) columns_error_ = table + " needs at least two number columns (assets).";
                portfolio_.assets = table_columns_;
                Changed();
            },
            alive_);
    }
    if (columns_task_ != 0) ImGui::TextColored(t.text_dim, "Reading the columns...");
    if (!columns_error_.empty()) Paragraph(columns_error_, &t.warning);
    if (!table_columns_.empty()) {
        ui::FieldLabel("Assets");
        ui::BeginInsetList("##assets", (std::min)(170.0f, ImGui::GetFrameHeightWithSpacing() * (static_cast<float>(table_columns_.size()) + 0.4f)));
        for (size_t j = 0; j < table_columns_.size(); ++j) {
            bool on = picked_[j] != 0;
            ImGui::PushID(static_cast<int>(j));
            if (ImGui::Checkbox(table_columns_[j].c_str(), &on)) {
                picked_[j] = on ? 1 : 0;
                portfolio_.assets.clear();
                for (size_t c = 0; c < table_columns_.size(); ++c)
                    if (picked_[c]) portfolio_.assets.push_back(table_columns_[c]);
                Changed();
            }
            ImGui::PopID();
        }
        ui::EndInsetList();
    }
    ui::FieldLabel("Target return a period");
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::InputTextWithHint("##target", "none", &portfolio_.target)) Changed();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("The least mean return the mix must earn a period; empty: the least risky mix");
    if (ImGui::Checkbox("Long only (no weight below 0)", &portfolio_.long_only)) Changed();
    if (!portfolio_.table.empty()) {
        if (ui::LinkButton("Read again")) {
            ++refresh_;
            if (!columns_error_.empty() || table_columns_.empty()) columns_for_.clear();
            Changed();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Read the table again (its data changed)");
    }
    Paragraph("minimize \xC2\xBD w'C w (C: the covariance of the returns) with the weights summing to 1 and earning the target.",
              &t.text_dim);
}

void OptimizationPanel::RenderProgramInputs() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (op_ == Op::Quadratic) {
        static const char* const kSources[] = {"Typed", "From a returns table"};
        if (ui::SegmentedControl("##qp_source", kSources, 2, &qp_source_)) Changed();
        ImGui::Spacing();
        if (qp_source_ == 1) {
            RenderPortfolioInputs();
            return;
        }
    }
    auto& p = Program();
    ImGui::TextColored(t.text_dim, "Examples");
    for (const auto& e : op_ == Op::Quadratic ? optcalc::QuadraticExamples() : optcalc::LinearExamples()) {
        ExploreNextButton(e.name);
        ImGui::PushID(e.name);
        if (ui::LinkButton(e.name)) {
            p = e.request;
            lower_text_.clear();
            Changed();
        }
        ImGui::PopID();
    }
    ImGui::NewLine();
    ImGui::Spacing();
    const size_t n = p.variables.size();
    if (lower_text_.size() != n) {
        lower_text_.clear();
        upper_text_.clear();
        for (const auto& b : p.bounds) {
            lower_text_.push_back(Plain(b.lower));
            upper_text_.push_back(Plain(b.upper));
        }
        bounds_error_.clear();
    }
    if (op_ == Op::Linear) {
        ui::SettingsSection("Goal");
        static const char* const kGoal[] = {"Maximize", "Minimize"};
        int goal = p.maximize ? 0 : 1;
        if (ui::SegmentedControl("##goal", kGoal, 2, &goal)) {
            p.maximize = goal == 0;
            Changed();
        }
    } else {
        ui::SettingsSection("Objective");
        ImGui::TextColored(t.text_dim, "minimize \xC2\xBD x'Q x + c'x");
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Variables");
    ImGui::SameLine();
    int count = static_cast<int>(n);
    if (ui::NumberField("##variables", &count, 1, 40, 1, 44.0f)) ResizeProgram(count);

    // The objective, the rows and the bounds, one column per variable.
    const int columns = static_cast<int>(n) + 4;  // remove, name, the variables, sense, b
    const float height = (std::min)(ImGui::GetFrameHeightWithSpacing() * (static_cast<float>(p.rows.size()) + 5.0f), 340.0f);
    int remove = -1;
    if (ImGui::BeginTable("##program", columns, ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                          ImVec2(0, height))) {
        ImGui::TableSetupScrollFreeze(2, 1);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFrameHeight());
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 84.0f);
        for (size_t j = 0; j < n; ++j) ImGui::TableSetupColumn(p.variables[j].c_str(), ImGuiTableColumnFlags_WidthFixed, 62.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("b", ImGuiTableColumnFlags_WidthFixed, 62.0f);
        ImGui::TableHeadersRow();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, op_ == Op::Quadratic ? "c" : "Objective");
        for (size_t j = 0; j < n; ++j) {
            ImGui::TableNextColumn();
            ImGui::PushID(static_cast<int>(j));
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputDouble("##c", &p.objective[j], 0, 0, "%.6g")) Changed();
            ImGui::PopID();
        }
        for (size_t i = 0; i < p.rows.size(); ++i) {
            auto& row = p.rows[i];
            ImGui::PushID(static_cast<int>(i) + 1000);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ui::GhostButton(ICON_FA_TRASH "##remove")) remove = static_cast<int>(i);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove %s", row.name.c_str());
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputText("##name", &row.name)) Changed();
            for (size_t j = 0; j < n; ++j) {
                ImGui::TableNextColumn();
                ImGui::PushID(static_cast<int>(j));
                ImGui::SetNextItemWidth(-1);
                if (ImGui::InputDouble("##a", &row.coefficients[j], 0, 0, "%.6g")) Changed();
                ImGui::PopID();
            }
            ImGui::TableNextColumn();
            static const char* const kSense[] = {"\xE2\x89\xA4", "\xE2\x89\xA5", "="};
            ImGui::SetNextItemWidth(-1);
            if (ImGui::BeginCombo("##sense", kSense[static_cast<int>(row.sense)], ImGuiComboFlags_NoArrowButton)) {
                for (int s = 0; s < 3; ++s)
                    if (ImGui::Selectable(kSense[s], static_cast<int>(row.sense) == s)) {
                        row.sense = static_cast<ConstraintSense>(s);
                        Changed();
                    }
                ImGui::EndCombo();
            }
            ImGui::TableNextColumn();
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputDouble("##b", &row.rhs, 0, 0, "%.6g")) Changed();
            ImGui::PopID();
        }
        const auto bound_row = [&](const char* label, std::vector<std::string>& texts) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(t.text_dim, "%s", label);
            for (size_t j = 0; j < n; ++j) {
                ImGui::TableNextColumn();
                ImGui::PushID(label);
                ImGui::PushID(static_cast<int>(j));
                ImGui::SetNextItemWidth(-1);
                if (ImGui::InputText("##bound", &texts[j])) {
                    bounds_error_.clear();
                    for (size_t k = 0; k < n; ++k) {
                        double lo = 0, hi = 0;
                        std::string error;
                        if (!BoundFrom(lower_text_[k], lo, error) || !BoundFrom(upper_text_[k], hi, error)) {
                            bounds_error_ = p.variables[k] + ": " + error;
                            continue;
                        }
                        p.bounds[k] = {lo, hi};
                    }
                    Changed();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("A number, or inf / -inf for no limit");
                ImGui::PopID();
                ImGui::PopID();
            }
        };
        bound_row("Lower bound", lower_text_);
        bound_row("Upper bound", upper_text_);
        ImGui::EndTable();
    }
    if (remove >= 0) {
        p.rows.erase(p.rows.begin() + remove);
        Changed();
    }
    if (!bounds_error_.empty()) Paragraph(bounds_error_, &t.warning);
    if (ui::SecondaryButton("+ Constraint")) {
        p.rows.push_back({"C" + std::to_string(p.rows.size() + 1), std::vector<double>(n, 0.0), ConstraintSense::LessEqual, 0.0});
        Changed();
    }
    ImGui::SameLine();
    if (ui::SecondaryButton(paste_open_ ? "Close paste" : "Paste rows")) paste_open_ = !paste_open_;
    if (paste_open_) {
        ImGui::TextColored(t.text_dim, "One row a line: coefficients, <= >= or =, then b (\"Labour: 1 2 1 1 <= 40\").");
        ImGui::InputTextMultiline("##paste", &paste_, ImVec2(-1, 110));
        if (ui::SecondaryButton("Use these rows")) {
            std::vector<optcalc::ProgramRow> rows;
            if (optcalc::ParseRows(paste_, n, rows, paste_error_)) {
                p.rows = rows;
                paste_error_.clear();
                paste_open_ = false;
                Changed();
            }
        }
        if (!paste_error_.empty()) Paragraph(paste_error_, &t.warning);
    }
    if (op_ == Op::Quadratic) {
        ImGui::Spacing();
        ui::SettingsSection("Q (symmetric: an entry sets its mirror)");
        if (ImGui::BeginTable("##Q", static_cast<int>(n) + 1, ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit,
                              ImVec2(0, (std::min)(ImGui::GetFrameHeightWithSpacing() * (static_cast<float>(n) + 1.4f), 280.0f)))) {
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 40.0f);
            for (size_t j = 0; j < n; ++j) ImGui::TableSetupColumn(p.variables[j].c_str(), ImGuiTableColumnFlags_WidthFixed, 66.0f);
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < n; ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(t.text_dim, "%s", p.variables[i].c_str());
                for (size_t j = 0; j < n; ++j) {
                    ImGui::TableNextColumn();
                    ImGui::PushID(static_cast<int>(i * n + j));
                    ImGui::SetNextItemWidth(-1);
                    if (ImGui::InputDouble("##q", &p.Q[i][j], 0, 0, "%.6g")) {
                        p.Q[j][i] = p.Q[i][j];
                        Changed();
                    }
                    ImGui::PopID();
                }
            }
            ImGui::EndTable();
        }
    }
}

void OptimizationPanel::RenderLineInputs() {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::SettingsSection("Formula");
    if (RenderFormula("##line", line_, "f(x)")) Changed();
    ImGui::TextColored(t.text_dim, "Examples");
    for (const auto& e : optcalc::LineExamples()) {
        ExploreNextButton(e.name);
        ImGui::PushID(e.name);
        if (ui::LinkButton(e.name)) ApplyExample(e);
        ImGui::PopID();
    }
    ImGui::NewLine();
    ImGui::Spacing();
    if (op_ == Op::Integral) {
        ui::SettingsSection("From a to b");
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("a");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        if (ImGui::InputText("##a", &a_)) Changed();
        ImGui::SameLine();
        ImGui::TextUnformatted("b");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        if (ImGui::InputText("##b", &b_)) Changed();
        ImGui::TextColored(t.text_dim, "Numbers, pi, e, or -inf / inf.");
        ImGui::Spacing();
        ui::SettingsSection("Rule");
        static const char* const kRules[] = {"Adaptive (Gauss\xE2\x80\x93Kronrod)", "Trapezoid", "Midpoint", "Simpson", "Gauss\xE2\x80\x93Legendre"};
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##rule", &rule_, kRules, 5)) Changed();
        if (rule_ != 0) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(rule_ == 4 ? "Points" : "Intervals");
            ImGui::SameLine();
            if (ui::NumberField("##n", &n_, 1, 100000, rule_ == 3 ? 2 : 10, 80.0f)) Changed();
        }
        Paragraph(rule_ == 0 ? "Splits where f is hard and stops when its error estimate is below 1.5e-8 of the answer, as "
                               "scipy.integrate.quad. The fixed rules are compared with it under the answer."
                             : "The adaptive answer is worked out too, to measure this rule against.",
                  &t.text_dim);
    } else {
        ui::SettingsSection("At");
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("x");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110.0f);
        if (ImGui::InputText("##at", &at_)) Changed();
        ImGui::Spacing();
        ui::SettingsSection("Differences");
        static const char* const kRules[] = {"Forward", "Backward", "Central"};
        if (ui::SegmentedControl("##difference", kRules, 3, &difference_)) Changed();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Step h");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110.0f);
        if (ImGui::InputTextWithHint("##h", "automatic", &step_text_)) Changed();
        Paragraph("Empty: the step that suits float64 for the rule. f\xE2\x80\xB2 and f\xE2\x80\xB3 also come exactly from the formula.",
                  &t.text_dim);
    }
}

void OptimizationPanel::RenderCurvatureInputs() {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::SettingsSection("Formula");
    if (RenderFormula("##surface", surface_, "f(x, y)")) {
        minima_.clear();
        Changed();
    }
    ImGui::TextColored(t.text_dim, "Examples");
    for (const auto& e : optcalc::SurfaceExamples()) {
        ExploreNextButton(e.name);
        ImGui::PushID(e.name);
        if (ui::LinkButton(e.name)) ApplyExample(e);
        ImGui::PopID();
    }
    ImGui::NewLine();
    ImGui::Spacing();
    ui::SettingsSection("Point");
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("x");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::InputText("##px", &point_x_)) Changed();
    ImGui::SameLine();
    ImGui::TextUnformatted("y");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::InputText("##py", &point_y_)) Changed();
    ImGui::Spacing();
    RenderRange();
}

void OptimizationPanel::RenderAnswer() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (job_.Running()) {
        ImGui::TextColored(t.running, "Working out %s...", optcalc::InfoOf(op_).label);
        ImGui::SameLine();
        if (ui::DangerButton("Stop")) {
            job_.Cancel();
            stopped_key_ = key_;
        }
        if (shown_ && shown_key_ != key_) ImGui::TextColored(t.text_dim, "Below: the answer before your edit.");
        ImGui::Spacing();
    } else if (key_ == stopped_key_ && shown_key_ != key_) {
        ImGui::TextColored(t.text_dim, "Stopped. Change an input to work it out again.");
        ImGui::Spacing();
    }
    if (!shown_) {
        if (!job_.Error().empty()) ImGui::TextColored(t.error, "%s", job_.Error().c_str());
        return;
    }
    const Job& job = *shown_;
    if (!job.ok()) {
        ImGui::TextColored(t.text_dim, "Stopped before an answer.");
        return;
    }
    if (job.op != op_) return;  // the answer for another operation, until this one's comes
    switch (job.op) {
        case Op::Optimizers: RenderMinimize(job); break;
        case Op::Linear:
        case Op::Quadratic: RenderProgram(job); break;
        case Op::Integral: RenderIntegral(job); break;
        case Op::Derivative: RenderDerivative(job); break;
        case Op::Curvature: RenderCurvature(job); break;
    }
}

namespace {

void Refusal(const std::string& error) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.warning, ICON_FA_TRIANGLE_EXCLAMATION);
    ImGui::SameLine();
    Paragraph(error);
}

float PlotWidth() { return (std::max)(240.0f, ImGui::GetContentRegionAvail().x); }

}  // namespace

void OptimizationPanel::RenderMinimize(const Job& job) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& a = job.minimize;
    if (!a.ok()) {
        Refusal(a.error);
        return;
    }
    Paragraph(a.sentence);
    ImGui::Spacing();
    size_t longest = 0;
    for (const auto& p : a.paths) longest = (std::max)(longest, p.points.size() - 1);
    if (ui::SecondaryButton("Step 10")) shown_steps_ = (std::min)(longest, (shown_steps_ == 0 ? 0 : shown_steps_) + 10);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show the paths 10 steps further");
    ImGui::SameLine();
    if (ui::LinkButton("Whole paths", shown_steps_ != 0)) shown_steps_ = 0;
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "%s", shown_steps_ == 0 ? ("Steps 0 to " + optcalc::Count(static_cast<long long>(longest))).c_str()
                                                          : ("Step " + optcalc::Count(static_cast<long long>(shown_steps_)) + " of " +
                                                             optcalc::Count(static_cast<long long>(longest))).c_str());
    if (job.map.ok()) {
        std::vector<optplot::Path> paths;
        for (size_t k = 0; k < a.paths.size() && k < a.rows.size(); ++k)
            paths.push_back({a.paths[k].points, shown_steps_, ui::SeriesColour(t, a.rows[k].choice)});
        std::vector<optplot::Mark> marks = {{job.minimize_request.x0, job.minimize_request.y0, "start", false}};
        if (job.minimize_request.formula == surface_)
            for (const auto& [mx, my] : PointsIn(minima_)) marks.push_back({mx, my, "", true});
        const float width = (std::min)(PlotWidth(), 760.0f);
        optplot::Map(job.map, paths, marks, width, width * 0.66f);
        ImGui::TextColored(t.text_dim, "%s", job.map.log_scale ? "Contour lines of log10(f \xE2\x88\x92 min + 1); a diamond marks a known minimum."
                                                               : "Contour lines of f; a diamond marks a known minimum.");
    } else {
        Paragraph(job.map.error, &t.text_dim);
    }
    ImGui::Spacing();
    std::vector<std::vector<std::string>> rows = {{"Optimizer", "Learning rate", "Ended at", "f", "|grad f|", "Result"}};
    std::vector<ImVec4> tones;
    for (const auto& r : a.rows) {
        rows.push_back({r.name, r.learning_rate, r.ended_at, r.value, r.gradient, r.word});
        tones.push_back(ToneColour(r.tone));
    }
    TextTable("##paths", rows, &tones, 5);
    Paragraph("The same update rules as training (and PyTorch's torch.optim); the gradient comes exactly from the formula.", &t.text_dim);
    RenderActions(job);
}

void OptimizationPanel::RenderProgram(const Job& job) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& a = job.program;
    if (!a.ok()) {
        Refusal(a.error);
        return;
    }
    const ImVec4 tone = ToneColour(a.tone);
    ImGui::TextColored(tone, "%s", a.status.c_str());
    Paragraph(a.sentence);
    if (job.from_table) {
        Paragraph(job.portfolio_note, &t.text_dim);
        if (a.result.success) {
            const optcalc::GraphHandOff hand = optcalc::PortfolioHandOff(job.portfolio);
            if (ui::PrimaryButton(hand.label.c_str(), static_cast<bool>(add_to_graph_), "No graph to add to")) {
                graph_message_ = add_to_graph_(job.portfolio.table, hand);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", hand.hint.c_str());
            if (!graph_message_.empty()) Paragraph(graph_message_, &t.text_dim);
        }
    }
    ImGui::Spacing();
    if (!a.variables.empty()) {
        std::vector<std::vector<std::string>> vars = {{"Variable", "Value"}};
        for (const auto& v : a.variables) vars.push_back({v.name, v.value});
        TextTable("##variables", vars);
        std::vector<std::vector<std::string>> rows = {{"Constraint", "Room", "Shadow price", "Binding"}};
        for (const auto& c : a.constraints) rows.push_back({c.name, c.room, c.price, c.binding});
        if (rows.size() > 1) {
            TextTable("##constraints", rows);
            ImGui::TextColored(t.text_dim, "Shadow price: how much the best value changes when that row's b grows by 1.");
        }
    }
    if (a.plot.drawn) {
        std::vector<std::string> names;
        for (const auto& r : job.program_request.rows) names.push_back(r.name);
        const float width = (std::min)(PlotWidth(), 640.0f);
        optplot::Program(a.plot, names, job.program_request.variables, job.map.ok() ? &job.map : nullptr, width, width * 0.6f);
        ImGui::TextColored(t.text_dim, "%s", a.plot.region.empty() ? "No point of this view meets every row." : "Shaded: every point that meets every row.");
    } else if (job.program_request.variables.size() > 2) {
        ImGui::TextColored(t.text_dim, "With more than two variables there is no region to draw; the tables carry the answer.");
    }
    RenderActions(job);
}

void OptimizationPanel::RenderIntegral(const Job& job) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& a = job.integral;
    if (!a.ok()) {
        Refusal(a.error);
        return;
    }
    ImGui::TextColored(t.text_bright, "%s", a.title.c_str());
    Paragraph(a.sentence);
    ImGui::Spacing();
    const float width = (std::min)(PlotWidth(), 640.0f);
    optplot::ShadedCurve(a.curve_x, a.curve_y, a.shade_from, a.shade_to, width, width * 0.45f);
    ImGui::Spacing();
    std::vector<std::vector<std::string>> rows = {{"Rule", "Value", "Evaluations", "Difference from adaptive"}};
    for (const auto& r : a.rules) rows.push_back({r.name, r.value, r.evaluations, r.difference});
    TextTable("##rules", rows);
    RenderActions(job);
}

void OptimizationPanel::RenderDerivative(const Job& job) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& a = job.derivative;
    if (!a.ok()) {
        Refusal(a.error);
        return;
    }
    ImGui::TextColored(t.text_bright, "%s", a.title.c_str());
    Paragraph(a.sentence);
    ImGui::Spacing();
    const float width = (std::min)(PlotWidth(), 640.0f);
    if (a.curve.success) {
        optplot::LogLines(a.curve.steps,
                          {{"forward", a.curve.forward_error, ui::SeriesColour(t, 1)}, {"central", a.curve.central_error, ui::SeriesColour(t, 0)}},
                          true, "step h", "error", width, width * 0.45f);
        Paragraph("The error of the numerical f\xE2\x80\xB2 against the exact one. Large h: the rule's own error (forward's falls "
                  "with h, central's with h\xC2\xB2). Small h: rounding in f(x + h) \xE2\x88\x92 f(x) grows as h shrinks.",
                  &t.text_dim);
    }
    std::vector<explore_plot::Line> lines = {{"f", a.curve_f}};
    if (!a.curve_df.empty()) lines.push_back({"f\xE2\x80\xB2", a.curve_df});
    explore_plot::Lines(a.curve_x, lines, width, width * 0.4f, false, a.curve_x.empty() ? NAN : a.curve_x[a.curve_x.size() / 2],
                        "x = " + at_, "x");
    RenderActions(job);
}

void OptimizationPanel::RenderCurvature(const Job& job) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& a = job.curvature;
    if (!a.ok()) {
        Refusal(a.error);
        return;
    }
    ImGui::TextColored(t.text_bright, "%s", a.title.c_str());
    Paragraph(a.sentence);
    ImGui::Spacing();
    const size_t n = a.result.eigenvalues.size();
    std::vector<std::vector<std::string>> hessian = {{"Hessian"}};
    for (size_t j = 0; j < n; ++j) hessian.front().push_back(j == 0 ? "x" : (j == 1 ? "y" : "v" + std::to_string(j + 1)));
    for (size_t i = 0; i < n; ++i) {
        std::vector<std::string> row = {hessian.front()[i + 1]};
        for (size_t j = 0; j < n; ++j) row.push_back(a.hessian[i * n + j]);
        hessian.push_back(row);
    }
    TextTable("##hessian", hessian);
    std::vector<std::vector<std::string>> eigen = {{"Eigenvalue", "Value"}};
    for (size_t i = 0; i < a.eigenvalues.size(); ++i) eigen.push_back({"\xCE\xBB" + std::to_string(i + 1), a.eigenvalues[i]});
    TextTable("##eigen", eigen);
    ImGui::Spacing();
    if (!a.range_sentence.empty()) Paragraph(a.range_sentence);
    if (job.map.ok()) {
        double px = 0, py = 0;
        std::string error;
        std::vector<optplot::Mark> marks;
        if (optcalc::ReadValue(point_x_, px, error) && optcalc::ReadValue(point_y_, py, error)) marks.push_back({px, py, "here", false});
        const float width = (std::min)(PlotWidth(), 640.0f);
        optplot::Map(job.map, {}, marks, width, width * 0.66f);
    }
    RenderActions(job);
}

void OptimizationPanel::RenderActions(const Job& job) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto text = [&job](char separator) {
        switch (job.op) {
            case Op::Optimizers: return optcalc::MinimizeText(job.minimize_request, job.minimize, separator);
            case Op::Linear:
            case Op::Quadratic: return optcalc::ProgramText(job.program_request, job.program, separator);
            case Op::Integral: return optcalc::IntegralText(job.integral, separator);
            case Op::Derivative: return optcalc::DerivativeText(job.derivative, separator);
            case Op::Curvature: return optcalc::CurvatureText(job.curvature, separator);
        }
        return std::string();
    };
    ImGui::Spacing();
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText(text('\t').c_str());
        message_ = "Copied as tab-separated rows.";
        message_error_ = false;
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Export CSV...")) {
        const std::string name = std::string("optimization_") + optcalc::InfoOf(job.op).id + ".csv";
        if (const auto path = FileDialogs::SaveFile("Export the answer", {{"CSV", "csv"}}, nullptr, name.c_str())) {
            std::ofstream out(*path, std::ios::binary);
            out << text(',');
            message_ = out ? "Saved " + *path : "Could not write " + *path;
            message_error_ = !out;
        }
    }
    if (!message_.empty()) Paragraph(message_, message_error_ ? &t.warning : &t.text_dim);
    ImGui::TextColored(t.text_faint, "Worked out in %.0f ms.", job.milliseconds);
}

}  // namespace cyxwiz
