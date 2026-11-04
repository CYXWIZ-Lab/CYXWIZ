#include "linear_algebra_panel.h"

#include "../../core/file_dialogs.h"
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
#include <random>

namespace cyxwiz {

namespace {

using linalg::Matrix;
using linalg::Op;

// The examples the five windows opened on, and two for the operations they had none for.
Matrix Example(Op op) {
    switch (op) {
        case Op::Eigen: return {{4, -2, 1}, {-2, 4, -2}, {1, -2, 4}};
        case Op::QR: return {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}, {-1, 1, 0}};
        case Op::Cholesky: return {{4, 12, -16}, {12, 37, -43}, {-16, -43, 98}};
        case Op::LU: return {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
        case Op::SVD:
        case Op::Rank:
        case Op::Transpose:
        case Op::Norm:
        case Op::LeastSquares: return {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        default: return {{1, 2}, {3, 4}};
    }
}

void Resize(Matrix& m, int rows, int cols) {
    m.resize(static_cast<size_t>(rows));
    for (auto& r : m) r.resize(static_cast<size_t>(cols), 0.0);
}

std::string Key(const linalg::Request& r) {
    std::string key = std::to_string(static_cast<int>(r.op)) + "|" + std::to_string(r.k) + "|" + std::to_string(r.low_rank);
    char buffer[32];
    for (const Matrix* m : {&r.a, &r.b}) {
        key += "|" + std::to_string(m->size()) + "x" + std::to_string(m->empty() ? 0 : (*m)[0].size());
        for (const auto& row : *m)
            for (const double v : row) {
                std::snprintf(buffer, sizeof(buffer), ",%.17g", v);
                key += buffer;
            }
    }
    return std::to_string(std::hash<std::string>{}(key)) + ":" + std::to_string(key.size());
}

double Scale(const Matrix& m) {
    double s = 0;
    for (const auto& r : m)
        for (const double v : r) s = std::max(s, std::fabs(v));
    return s;
}

}  // namespace

LinearAlgebraPanel::LinearAlgebraPanel() { LoadExample(); }

void LinearAlgebraPanel::Open(const std::string& id) {
    visible_ = true;
    if (const linalg::OpInfo* info = linalg::FindOp(id)) SetOp(info->op);
    ImGui::SetWindowFocus("###LinearAlgebra");
}

void LinearAlgebraPanel::SetOp(Op op) {
    if (op == op_) return;
    op_ = op;
    tab_ = 0;
    if (!edited_) LoadExample();
    // The second input follows A's rows where it must.
    const auto second = linalg::InfoOf(op_).second;
    if (second == linalg::Second::VectorB && (b_.size() != a_.size() || b_.empty())) {
        b_.assign(a_.size(), std::vector<double>(1, 0.0));
        for (size_t i = 0; i < b_.size(); ++i) b_[i][0] = static_cast<double>(i + 1) + (i % 2 ? 1.0 : 4.0);
    } else if (second == linalg::Second::MatrixB && (b_.empty() || b_[0].size() == 1)) {
        b_ = {{5, 6}, {7, 8}};
    }
    Changed();
}

void LinearAlgebraPanel::LoadExample() {
    a_ = Example(op_);
    const auto second = linalg::InfoOf(op_).second;
    if (second == linalg::Second::MatrixB) b_ = {{5, 6}, {7, 8}};
    else if (second == linalg::Second::VectorB) b_ = op_ == Op::Solve ? Matrix{{5}, {6}} : Matrix{{1}, {2}, {2}, {4}};
    else b_.clear();
    Changed();
}

void LinearAlgebraPanel::Changed() {
    dirty_ = true;
    message_.clear();
}

void LinearAlgebraPanel::Render() {
    if (!visible_) return;
    ImGui::SetNextWindowSize(ImVec2(1120, 720), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(ICON_FA_TABLE_CELLS " Linear Algebra###LinearAlgebra", &visible_)) {
        ImGui::End();
        return;
    }
    if (dirty_) {
        dirty_ = false;
        request_.op = op_;
        request_.a = a_;
        request_.b = linalg::InfoOf(op_).second == linalg::Second::MatrixB || linalg::InfoOf(op_).second == linalg::Second::VectorB ? b_ : Matrix{};
        request_.k = k_;
        request_.low_rank = low_rank_;
        key_ = Key(request_);
    }
    if (key_ != stopped_key_) {
        const linalg::Request request = request_;
        job_.Ask(std::string("Linear Algebra: ") + linalg::InfoOf(op_).label, key_, [request](LambdaTask& task) {
            Job job;
            const auto start = std::chrono::steady_clock::now();
            job.answer = linalg::Run(request, [&task] { return task.ShouldStop(); });
            job.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (job.answer.error == "Cancelled.") job.error = "Cancelled.";
            return job;
        });
    }
    if (const Job* answer = job_.Answer(key_); answer && shown_key_ != key_) {
        shown_ = std::make_shared<const Job>(*answer);
        shown_key_ = key_;
        if (tab_ >= static_cast<int>(shown_->answer.tabs.size())) tab_ = 0;
    }

    RenderOperations();
    ImGui::Spacing();
    ui::BeginSettingsColumn("linear_algebra", 400.0f);
    RenderInputs();
    ui::BeginResultsArea();
    RenderAnswer();
    ui::EndResultsArea();
    ImGui::End();
}

void LinearAlgebraPanel::RenderOperations() {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImGuiStyle& style = ImGui::GetStyle();
    bool first = true;
    for (const std::string& group : linalg::Groups()) {
        // A group stays on one line: it starts a new line when it would not fit.
        float width = ImGui::CalcTextSize(group.c_str()).x;
        for (const auto& info : linalg::Ops())
            if (group == info.group) width += style.ItemSpacing.x + ImGui::CalcTextSize(info.label).x + 2.0f * style.FramePadding.x;
        if (!first) {
            ImGui::SameLine(0, 18.0f);
            if (ImGui::GetContentRegionAvail().x < width) ImGui::NewLine();
        }
        first = false;
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "%s", group.c_str());
        for (const auto& info : linalg::Ops()) {
            if (group != info.group) continue;
            ImGui::SameLine();
            ImGui::PushID(info.id);
            if (ui::GhostButton(info.label, true, nullptr, info.op == op_)) SetOp(info.op);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", info.hint);
            ImGui::PopID();
        }
    }
}

bool LinearAlgebraPanel::RenderGrid(const char* id, Matrix& m, bool rows_fixed) {
    const ui::Tokens& t = ui::CurrentTokens();
    bool changed = false;
    int rows = static_cast<int>(m.size()), cols = m.empty() ? 0 : static_cast<int>(m[0].size());
    ImGui::PushID(id);
    if (!rows_fixed) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "Rows");
        ImGui::SameLine();
        if (ui::NumberField("##rows", &rows, 1, linalg::kMaxTyped, 1, 44.0f)) changed = true;
        ImGui::SameLine();
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Columns");
    ImGui::SameLine();
    if (ui::NumberField("##cols", &cols, 1, linalg::kMaxTyped, 1, 44.0f)) changed = true;
    if (changed) Resize(m, rows, cols);
    const float cell = 68.0f;
    const float height = std::min(ImGui::GetFrameHeightWithSpacing() * (static_cast<float>(m.size()) + 1.2f), 300.0f);
    if (ImGui::BeginChild("##grid", ImVec2(0, height), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar)) {
        for (size_t i = 0; i < m.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            for (size_t j = 0; j < m[i].size(); ++j) {
                if (j > 0) ImGui::SameLine(0, 4.0f);
                ImGui::PushID(static_cast<int>(j));
                ImGui::SetNextItemWidth(cell);
                if (ImGui::InputDouble("##v", &m[i][j], 0.0, 0.0, "%.6g")) changed = true;
                ImGui::PopID();
            }
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
    ImGui::PopID();
    return changed;
}

void LinearAlgebraPanel::RenderInputs() {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::SettingsSection("Matrix A");
    static const char* const kSources[] = {"Typed", "Paste"};
    if (ui::SegmentedControl("##source", kSources, 2, &source_)) paste_note_.clear();
    ImGui::Spacing();
    if (source_ == 0) {
        if (RenderGrid("a", a_, false)) {
            edited_ = true;
            Changed();
        }
        const auto preset = [&](const char* label, const std::function<void()>& apply, bool enabled = true) {
            ExploreNextButton(label);
            if (ui::LinkButton(label, enabled)) {
                apply();
                edited_ = true;
                Changed();
            }
        };
        const size_t rows = a_.size(), cols = a_.empty() ? 0 : a_[0].size();
        preset("Identity", [&] {
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j) a_[i][j] = i == j ? 1.0 : 0.0;
        });
        preset("Zero", [&] {
            for (auto& r : a_) std::fill(r.begin(), r.end(), 0.0);
        });
        preset("Symmetrize", [&] {
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = i + 1; j < cols; ++j) a_[i][j] = a_[j][i] = (a_[i][j] + a_[j][i]) / 2.0;
        }, rows == cols);
        if (rows != cols && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("A square A only");
        preset("Random", [&] {
            std::mt19937 rng(static_cast<unsigned>(random_seed_++));
            std::uniform_int_distribution<int> value(-9, 9);
            for (auto& r : a_)
                for (double& v : r) v = value(rng);
        });
        preset("Transpose", [&] {
            Matrix m(cols, std::vector<double>(rows));
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j) m[j][i] = a_[i][j];
            a_ = m;
        });
        preset("Example", [&] { LoadExample(); });
        ImGui::NewLine();
    } else {
        ImGui::TextColored(t.text_dim, "Rows from a spreadsheet or CSV (tabs, commas or spaces).");
        if (ImGui::InputTextMultiline("##paste", &paste_, ImVec2(-1, 150))) {
            Matrix m;
            std::string error;
            if (linalg::ParseMatrix(paste_, m, error)) {
                a_ = m;
                edited_ = true;
                paste_note_ = "Read A: " + std::to_string(m.size()) + " × " + std::to_string(m[0].size()) + ".";
                paste_error_ = false;
                Changed();
            } else {
                paste_note_ = error;
                paste_error_ = true;
            }
        }
        if (!paste_note_.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(paste_error_ ? t.warning : t.text_dim, "%s", paste_note_.c_str());
            ImGui::PopTextWrapPos();
        }
    }
    RenderSecond();
}

void LinearAlgebraPanel::RenderSecond() {
    const auto second = linalg::InfoOf(op_).second;
    if (second == linalg::Second::None) return;
    ImGui::Spacing();
    if (second == linalg::Second::Scalar) {
        ui::SettingsSection("k");
        ui::NumberFieldOptions o;
        o.format = "%.6g";
        o.width = 100.0f;
        if (ui::NumberField("##k", &k_, o)) Changed();
        return;
    }
    if (second == linalg::Second::VectorB) {
        ui::SettingsSection("Right-hand side b");
        if (b_.size() != a_.size()) Resize(b_, static_cast<int>(a_.size()), b_.empty() ? 1 : static_cast<int>(b_[0].size()));
        if (RenderGrid("b", b_, true)) Changed();
        return;
    }
    ui::SettingsSection("Matrix B");
    if (RenderGrid("B", b_, false)) Changed();
}

void LinearAlgebraPanel::RenderTable(const linalg::Table& table) {
    const ui::Tokens& t = ui::CurrentTokens();
    const double scale = Scale(table.values);
    const int columns = static_cast<int>(table.columns.size()) + 1;
    if (ImGui::BeginTable("##result", columns, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                          ImVec2(0, std::min(ImGui::GetTextLineHeightWithSpacing() * (static_cast<float>(table.values.size()) + 2.0f), 320.0f)))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("##row", ImGuiTableColumnFlags_WidthFixed, 44.0f);
        for (const auto& c : table.columns) ImGui::TableSetupColumn(c.c_str(), ImGuiTableColumnFlags_WidthFixed, 96.0f);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < table.values.size(); ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", i < table.row_names.size() ? table.row_names[i].c_str() : "");
            for (const double v : table.values[i]) {
                ImGui::TableNextColumn();
                const std::string text = table.matrix ? linalg::Number(v, scale) : linalg::Number(v);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetColumnWidth() - ImGui::CalcTextSize(text.c_str()).x - 8.0f));
                ImGui::TextUnformatted(text.c_str());
            }
        }
        ImGui::EndTable();
    }
    if (!table.note.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", table.note.c_str());
        ImGui::PopTextWrapPos();
    }
}

void LinearAlgebraPanel::RenderAnswer() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (job_.Running()) {
        ImGui::TextColored(t.running, "Working out %s...", linalg::InfoOf(op_).label);
        ImGui::SameLine();
        if (ui::DangerButton("Stop")) {
            job_.Cancel();
            stopped_key_ = key_;
        }
        if (shown_ && shown_key_ != key_) ImGui::TextColored(t.text_dim, "Below: the answer for the matrix before your edit.");
        ImGui::Spacing();
    } else if (key_ == stopped_key_ && shown_key_ != key_) {
        ImGui::TextColored(t.text_dim, "Stopped. Change A to work it out again.");
        ImGui::Spacing();
    }
    if (!shown_) {
        if (!job_.Error().empty()) ImGui::TextColored(t.error, "%s", job_.Error().c_str());
        return;
    }
    const linalg::Answer& a = shown_->answer;
    if (!a.ok()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.warning, ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::SameLine();
        ImGui::TextUnformatted(a.error.c_str());
        ImGui::PopTextWrapPos();
        for (const std::string& id : a.try_instead) {
            if (const linalg::OpInfo* info = linalg::FindOp(id)) {
                ExploreNextButton(info->label);
                ImGui::PushID(info->id);
                if (ui::SecondaryButton(info->label)) SetOp(info->op);
                ImGui::PopID();
            }
        }
        return;
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(a.sentence.c_str());
    ImGui::PopTextWrapPos();
    if (!a.figures.empty()) {
        std::vector<std::pair<std::string, std::string>> rows;
        for (const auto& f : a.figures) rows.emplace_back(f.name, f.value);
        ExploreFigures("##figures", rows);
    }
    if (!a.bars.empty()) {
        ImGui::Spacing();
        double top = 0;
        for (const auto& b : a.bars) top = std::max(top, std::fabs(b.value));
        ImDrawList* draw = ImGui::GetWindowDrawList();
        const float label_w = 34.0f, value_w = 170.0f;
        for (const auto& b : a.bars) {
            ImGui::TextColored(t.text_dim, "%s", b.name.c_str());
            ImGui::SameLine(label_w);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float width = std::max(40.0f, ImGui::GetContentRegionAvail().x - value_w);
            const float h = ImGui::GetTextLineHeight();
            draw->AddRectFilled(p, ImVec2(p.x + width, p.y + h), ImGui::GetColorU32(t.bg_input), 3.0f);
            const float fill = top > 0 ? static_cast<float>(std::fabs(b.value) / top) * width : 0.0f;
            if (fill > 0.5f) draw->AddRectFilled(p, ImVec2(p.x + fill, p.y + h), ImGui::GetColorU32(t.accent), 3.0f);
            ImGui::Dummy(ImVec2(width, h));
            ImGui::SameLine();
            std::string value = linalg::Number(b.value, top);
            if (std::isfinite(b.share)) {
                char share[32];
                std::snprintf(share, sizeof(share), " (%.2f %%)", 100.0 * b.share);
                value += share;
            }
            ImGui::TextUnformatted(value.c_str());
        }
    }
    ImGui::Spacing();
    if (!a.tabs.empty() && ImGui::BeginTabBar("##tabs")) {
        for (size_t i = 0; i < a.tabs.size(); ++i) {
            if (ImGui::BeginTabItem((a.tabs[i].title + "###tab" + std::to_string(i)).c_str())) {
                tab_ = static_cast<int>(i);
                if (op_ == Op::SVD && a.tabs[i].title.rfind("Low rank", 0) == 0) {
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Keep");
                    ImGui::SameLine();
                    const int most = std::max(1, static_cast<int>(a.bars.size()));
                    if (ui::NumberField("##keep", &low_rank_, 1, most, 1, 44.0f, "The directions kept")) Changed();
                }
                RenderTable(a.tabs[i]);
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
    if (!a.verify.empty()) {
        ImGui::TextColored(t.success, ICON_FA_CIRCLE_CHECK);
        ImGui::SameLine();
        ImGui::TextUnformatted(a.verify.c_str());
    }
    ImGui::Spacing();
    const bool has_tab = tab_ < static_cast<int>(a.tabs.size());
    if (ui::SecondaryButton("Copy")) ImGui::SetClipboardText(linalg::CopyText(a, static_cast<size_t>(tab_)).c_str());
    ExploreNextButton("Export CSV...");
    if (ui::SecondaryButton("Export CSV...", has_tab, "Nothing to save: this answer has no table")) Export(a.tabs[static_cast<size_t>(tab_)]);
    ExploreNextButton("Use as A");
    const bool usable = has_tab && a.tabs[static_cast<size_t>(tab_)].matrix;
    if (ui::SecondaryButton("Use as A", usable, "Only a matrix result can become A")) {
        a_ = a.tabs[static_cast<size_t>(tab_)].values;
        source_ = 0;
        edited_ = true;
        Changed();
    }
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "%.0f ms on the device", shown_->milliseconds);
    if (!message_.empty()) ImGui::TextColored(message_error_ ? t.error : t.text_dim, "%s", message_.c_str());
}

void LinearAlgebraPanel::Export(const linalg::Table& table) {
    std::string name = std::string("linear_algebra_") + linalg::InfoOf(op_).id + ".csv";
    if (const auto path = FileDialogs::SaveFile("Export the result", {{"CSV", "csv"}}, nullptr, name.c_str())) {
        std::ofstream out(*path, std::ios::binary);
        out << linalg::Csv(table);
        message_ = out ? "Saved " + *path : "Could not write " + *path;
        message_error_ = !out;
    }
}

}  // namespace cyxwiz
