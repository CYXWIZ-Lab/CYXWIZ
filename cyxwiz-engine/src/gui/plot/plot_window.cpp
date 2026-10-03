#include "plot_window.h"

#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../../core/plot/plot_arrow_source.h"
#include "../../core/plot/plot_presets.h"
#include "../../core/plot/plot_table_source.h"
#include "../../core/plot_script.h"
#include "../../data/data_table.h"

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>

namespace cyxwiz::plot {

namespace {

bool NumericX(Kind k) {
    return k != Kind::Bar && k != Kind::Pie && k != Kind::ErrorBars && k != Kind::Heatmap;
}

std::string DefaultTitle(const PlotSpec& s) {
    const std::string kind = Info(s.kind).label;
    if (s.kind == Kind::Histogram && !s.x_column.empty()) return "Histogram of " + s.x_column;
    if (!s.y_columns.empty() && !s.x_column.empty()) return s.y_columns.front() + " by " + s.x_column;
    if (!s.y_columns.empty()) return kind + " of " + s.y_columns.front();
    if (!s.x_column.empty()) return kind + " of " + s.x_column;
    return kind;
}

}  // namespace

PlotWindow::PlotWindow(std::string id) : id_(std::move(id)), view_(id_ + "_view") {}

PlotWindow::~PlotWindow() {
    if (job_.valid()) job_.wait();
}

void PlotWindow::Open(const std::string& source_name, std::shared_ptr<DataTable> table, PlotSpec spec, size_t row_limit,
                      size_t total_rows) {
    source_name_ = source_name;
    spec_ = std::move(spec);
    if (spec_.title.empty()) spec_.title = DefaultTitle(spec_);
    std::snprintf(title_buf_, sizeof(title_buf_), "%s", spec_.title.c_str());
    visible = true;
    view_.RequestFit();
    SetTable(std::move(table), row_limit, total_rows);
    ImGui::SetWindowFocus(("Plot###" + id_).c_str());
}

void PlotWindow::SetArrowTable(const std::string& source_name, std::shared_ptr<arrow::Table> table, size_t row_limit,
                               size_t total_rows) {
    source_name_ = source_name;
    table_.reset();
    arrow_table_ = std::move(table);
    row_limit_ = row_limit;
    total_rows_ = total_rows;
    headers_.clear();
    numeric_.clear();
    if (arrow_table_) {
        for (const auto& c : ArrowColumns(*arrow_table_)) {
            headers_.push_back(c.name);
            numeric_.push_back(c.numeric);
        }
    }
    ReadColumns();
    // First plot of an evaluation table (Confusion Matrix, ROC, PR): its
    // preset (board 5).
    if (spec_.x_column.empty() && spec_.y_columns.empty() && arrow_table_) {
        const auto first_value = [table = arrow_table_](const std::string& column) {
            const Source one = SourceFromArrow(*table, {column}, 1);
            return !one.columns.empty() && !one.columns.front().numbers.empty() ? one.columns.front().numbers.front() : NAN;
        };
        if (auto preset = EvaluationPreset(headers_, first_value)) {
            spec_ = *preset;
            std::snprintf(title_buf_, sizeof(title_buf_), "%s", spec_.title.c_str());
            view_.RequestFit();
        }
    }
    // First plot: a label column's counts, else a text column's counts, else
    // the first numeric column's histogram.
    if (spec_.x_column.empty() && spec_.y_columns.empty() && !headers_.empty()) {
        int label = -1, text = -1, number = -1;
        for (size_t i = 0; i < headers_.size(); ++i) {
            std::string lower = headers_[i];
            for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (label < 0 && (lower == "class" || lower == "label" || lower == "target" || lower == "y")) label = static_cast<int>(i);
            if (text < 0 && !numeric_[i]) text = static_cast<int>(i);
            if (number < 0 && numeric_[i]) number = static_cast<int>(i);
        }
        const int category = label >= 0 ? label : text;
        if (category >= 0) {
            spec_.kind = Kind::Bar;
            spec_.x_column = headers_[static_cast<size_t>(category)];
        } else if (number >= 0) {
            spec_.kind = Kind::Histogram;
            spec_.x_column = headers_[static_cast<size_t>(number)];
        }
        spec_.title = DefaultTitle(spec_);
        std::snprintf(title_buf_, sizeof(title_buf_), "%s", spec_.title.c_str());
        view_.RequestFit();
    }
    Rebuild();
}

void PlotWindow::ClearData(const std::string& source_name, const std::string& message) {
    source_name_ = source_name;
    table_.reset();
    arrow_table_.reset();
    headers_.clear();
    numeric_.clear();
    columns_.clear();
    empty_message_ = message;
    view_.Clear();
}

void PlotWindow::SetSpec(PlotSpec spec) {
    spec_ = std::move(spec);
    condition_values_.clear();  // re-read from the spec
    std::snprintf(title_buf_, sizeof(title_buf_), "%s", spec_.title.c_str());
    view_.RequestFit();
    Rebuild();
}

void PlotWindow::SetTable(std::shared_ptr<DataTable> table, size_t row_limit, size_t total_rows) {
    arrow_table_.reset();
    table_ = std::move(table);
    row_limit_ = row_limit;
    total_rows_ = total_rows;
    headers_ = table_ ? table_->GetHeaders() : std::vector<std::string>{};
    numeric_ = table_ ? NumericColumns(*table_) : std::vector<bool>{};
    ReadColumns();
    Rebuild();
}

size_t PlotWindow::TableRows() const {
    if (arrow_table_) return static_cast<size_t>(arrow_table_->num_rows());
    return table_ ? table_->GetRowCount() : 0;
}

void PlotWindow::ReadColumns() {
    // Names and types now; the summaries follow.
    columns_.clear();
    for (size_t i = 0; i < headers_.size(); ++i) {
        ColumnSummary c;
        c.name = headers_[i];
        c.numeric = numeric_[i];
        columns_.push_back(std::move(c));
    }
    if (arrow_table_) {
        // An Arrow table does not change: one column at a time off the UI
        // thread (a wide table is never copied whole).
        columns_job_ = std::async(std::launch::async, [table = arrow_table_]() {
            std::vector<ColumnSummary> out;
            for (const auto& c : ArrowColumns(*table)) {
                const Source one = SourceFromArrow(*table, {c.name});
                if (one.columns.empty()) {
                    ColumnSummary plain;
                    plain.name = c.name;
                    plain.numeric = c.numeric;
                    out.push_back(std::move(plain));
                } else {
                    out.push_back(SummarizeColumn(one.columns.front()));
                }
            }
            return out;
        });
    } else if (table_ && headers_.size() * table_->GetRowCount() <= 2000000) {
        // A DataTable is read on the UI thread, so only small ones.
        const Source all = SourceFromTable(*table_, headers_, numeric_);
        for (auto& c : columns_)
            if (const SourceColumn* sc = all.Find(c.name)) c = SummarizeColumn(*sc);
    }
}

int PlotWindow::ColumnIndex(const std::string& name) const {
    for (size_t i = 0; i < headers_.size(); ++i)
        if (headers_[i] == name) return static_cast<int>(i);
    return -1;
}

void PlotWindow::Rebuild() {
    if (on_spec_changed) on_spec_changed(spec_);
    if (!table_ && !arrow_table_) {
        view_.Clear();
        return;
    }
    if (busy_) {  // prepare again when the running job is done
        dirty_ = true;
        return;
    }
    const std::vector<std::string> needed = ColumnsNeeded(spec_);
    if (arrow_table_) {
        // An Arrow table does not change: read and prepare off the UI thread.
        busy_ = true;
        dirty_ = false;
        job_ = std::async(std::launch::async, [spec = spec_, table = arrow_table_, needed, limit = row_limit_,
                                               total = total_rows_]() {
            Source src = SourceFromArrow(*table, needed);
            if (limit > 0) {
                src.row_limit = limit;
                src.total_rows = total;
            }
            return Prepare(spec, src);
        });
        return;
    }
    // Copy the columns on the UI thread (the table is not thread-safe), then
    // prepare off it.
    Source src = SourceFromTable(*table_, needed, numeric_);
    src.row_limit = row_limit_;
    src.total_rows = total_rows_;
    busy_ = true;
    dirty_ = false;
    job_ = std::async(std::launch::async, [spec = spec_, src = std::move(src)]() { return Prepare(spec, src); });
}

void PlotWindow::Poll() {
    if (columns_job_.valid() && columns_job_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        std::vector<ColumnSummary> read = columns_job_.get();
        if (read.size() == columns_.size()) columns_ = std::move(read);  // still the same table
    }
    if (!busy_ || !job_.valid()) return;
    if (job_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    view_.SetData(job_.get());
    busy_ = false;
    if (dirty_) Rebuild();
}

void PlotWindow::Render() {
    Poll();
    PlotView::Options vo;
    vo.export_name = spec_.title.empty() ? std::string("plot") : spec_.title;
    vo.python_script = [this]() { return PythonScript(); };
    view_.DrawOwnWindow(vo);
    if (!visible) return;
    ImGui::SetNextWindowSize(ImVec2(1200, 720), ImGuiCond_FirstUseEver);
    const std::string title = "Plot" + (source_name_.empty() ? std::string() : " \xC2\xB7 " + source_name_) + "###" + id_;
    if (!ImGui::Begin(title.c_str(), &visible)) {
        ImGui::End();
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    // Source line (a Plot node draws its own status instead).
    if (draw_header) {
        draw_header();
    } else if (table_) {
        ImGui::PushStyleColor(ImGuiCol_Text, t.success);
        ImGui::Bullet();
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextUnformatted(source_name_.c_str());
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s rows \xC3\x97 %zu columns", Thousands(static_cast<long long>(table_->GetRowCount())).c_str(),
                           headers_.size());
        if (busy_) {
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "preparing...");
        }
    } else {
        ImGui::TextDisabled("%s", empty_message_.c_str());
    }

    const float kinds_w = 190.0f, settings_w = 290.0f;
    const float h = ImGui::GetContentRegionAvail().y;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    ImGui::BeginChild("##kinds", ImVec2(kinds_w, h), ImGuiChildFlags_AlwaysUseWindowPadding);
    DrawKinds();
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##plot", ImVec2(ImGui::GetContentRegionAvail().x - settings_w - ImGui::GetStyle().ItemSpacing.x, h),
                      ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    {
        ui::FontScope heading(ui::Font::Medium);
        ImGui::TextUnformatted(spec_.title.c_str());
    }
    if (!table_ && !arrow_table_) {
        // No data: the reason where the plot would be.
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + 12.0f, ImGui::GetCursorPosY() + avail.y * 0.4f));
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + avail.x - 24.0f);
        ImGui::TextDisabled("%s", empty_message_.c_str());
        ImGui::PopTextWrapPos();
    } else {
        view_.Draw(ImVec2(0, 0), vo);
    }
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.plot_bg);
    ImGui::BeginChild("##settings", ImVec2(0, h), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    DrawSettings();
    ImGui::EndChild();
    ImGui::End();
}

void PlotWindow::DrawKinds() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "PLOT TYPE");
    Group group = Group::Basic;
    bool first = true;
    for (const auto& k : Kinds()) {
        if (first || k.group != group) {
            ImGui::Spacing();
            ImGui::TextColored(t.text_dim, "%s", GroupLabel(k.group));
            group = k.group;
            first = false;
        }
        const bool selected = spec_.kind == k.kind;
        if (ImGui::Selectable(k.label, selected)) {
            if (!selected) {
                const bool was_numeric = NumericX(spec_.kind);
                const bool auto_title = spec_.title == DefaultTitle(spec_);
                spec_.kind = k.kind;
                // Keep what still fits; Box/Violin read values from Y.
                if ((k.kind == Kind::Box || k.kind == Kind::Violin || k.kind == Kind::Kde || k.kind == Kind::Matrix) &&
                    spec_.y_columns.empty() && !spec_.x_column.empty())
                    spec_.y_columns = {spec_.x_column};
                if ((k.required & kEncX) && spec_.x_column.empty() && !spec_.y_columns.empty())
                    spec_.x_column = spec_.y_columns.front();
                if (k.kind == Kind::Histogram && spec_.x_column.empty() && !spec_.y_columns.empty())
                    spec_.x_column = spec_.y_columns.front();
                if (NumericX(k.kind) && !was_numeric) {
                    const int c = ColumnIndex(spec_.x_column);
                    if (c >= 0 && !numeric_[static_cast<size_t>(c)]) spec_.x_column.clear();
                }
                if (!k.multi_y && spec_.y_columns.size() > 1) spec_.y_columns.resize(1);
                if (auto_title) {
                    spec_.title = DefaultTitle(spec_);
                    std::snprintf(title_buf_, sizeof(title_buf_), "%s", spec_.title.c_str());
                }
                view_.RequestFit();
                Rebuild();
            }
        }
    }
    ImGui::Spacing();
    ImGui::TextColored(t.text_faint, "More kinds: P2b");
    ImGui::TextColored(t.text_faint, "3D: P4");
}

bool PlotWindow::DrawRows(float w) {
    const ui::Tokens& t = ui::CurrentTokens();
    bool changed = false;
    ImGui::TextColored(t.text_dim, "ROWS");
    static const char* const kModes[] = {"All", "First", "Range", "Filter"};
    int mode = static_cast<int>(spec_.rows);
    if (ui::SegmentedControl("##rows", kModes, 4, &mode)) {
        spec_.rows = static_cast<RowMode>(mode);
        if (spec_.rows == RowMode::Filter && spec_.conditions.empty()) spec_.conditions.push_back({});
        changed = true;
    }
    const std::string all = Thousands(static_cast<long long>(TableRows()));
    switch (spec_.rows) {
        case RowMode::All: ImGui::TextColored(t.text_dim, "All %s rows.", all.c_str()); break;
        case RowMode::First: {
            int n = static_cast<int>(std::min<size_t>(spec_.first_rows, 2000000000));
            ImGui::SetNextItemWidth(w * 0.45f);
            if (ImGui::InputInt("##first", &n, 0, 0)) {
                spec_.first_rows = static_cast<size_t>(std::max(1, n));
                changed = true;
            }
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "of %s rows", all.c_str());
            break;
        }
        case RowMode::Range: {
            int from = static_cast<int>(std::min<size_t>(spec_.row_from, 2000000000));
            int to = static_cast<int>(std::min<size_t>(spec_.row_to, 2000000000));
            const float field = (w - ImGui::CalcTextSize("to").x - ImGui::GetStyle().ItemSpacing.x * 2) * 0.5f;
            ImGui::SetNextItemWidth(field);
            if (ImGui::InputInt("##from", &from, 0, 0)) {
                spec_.row_from = static_cast<size_t>(std::max(1, from));
                spec_.row_to = std::max(spec_.row_to, spec_.row_from);
                changed = true;
            }
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "to");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(field);
            if (ImGui::InputInt("##to", &to, 0, 0)) {
                spec_.row_to = std::max(spec_.row_from, static_cast<size_t>(std::max(1, to)));
                changed = true;
            }
            ImGui::TextColored(t.text_dim, "Rows are numbered from 1, of %s.", all.c_str());
            break;
        }
        case RowMode::Filter: {
            if (condition_values_.size() != spec_.conditions.size()) {
                condition_values_.assign(spec_.conditions.size(), {});
                for (size_t i = 0; i < spec_.conditions.size(); ++i)
                    std::snprintf(condition_values_[i].data(), condition_values_[i].size(), "%s",
                                  spec_.conditions[i].value.c_str());
            }
            const auto& ops = ConditionOps();
            const float remove_w = ImGui::CalcTextSize(ICON_FA_XMARK).x + 8.0f;
            const float gap = ImGui::GetStyle().ItemSpacing.x;
            for (size_t i = 0; i < spec_.conditions.size(); ++i) {
                RowCondition& c = spec_.conditions[i];
                ImGui::PushID(static_cast<int>(i));
                const float col_w = (w - remove_w - gap * 3) * 0.46f, op_w = (w - remove_w - gap * 3) * 0.24f;
                const std::string col_id = "##cond_col" + std::to_string(i);
                changed |= picker_.Pick(col_id.c_str(), c.column, columns_, false, nullptr, col_w);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(op_w);
                if (ImGui::BeginCombo("##op", c.op.c_str())) {
                    for (const auto& op : ops)
                        if (ImGui::Selectable(op.c_str(), op == c.op)) {
                            c.op = op;
                            changed = true;
                        }
                    ImGui::EndCombo();
                }
                ImGui::SameLine();
                ImGui::SetNextItemWidth(std::max(30.0f, w - col_w - op_w - remove_w - gap * 3));
                if (ImGui::InputTextWithHint("##value", "value", condition_values_[i].data(), condition_values_[i].size())) {
                    c.value = condition_values_[i].data();
                    changed = true;
                }
                ImGui::SameLine();
                bool removed = false;
                if (ui::LinkButton(ICON_FA_XMARK "##remove")) {
                    spec_.conditions.erase(spec_.conditions.begin() + static_cast<std::ptrdiff_t>(i));
                    condition_values_.erase(condition_values_.begin() + static_cast<std::ptrdiff_t>(i));
                    changed = removed = true;
                }
                if (!removed && ImGui::IsItemHovered()) ImGui::SetTooltip("Remove this condition");
                ImGui::PopID();
                if (removed) break;
            }
            if (ui::LinkButton("Add condition")) {
                spec_.conditions.push_back({});
                condition_values_.push_back({});
                changed = true;
            }
            const Prepared& p = view_.Data();
            if (view_.HasData() && p.problem.empty() && !p.label.selection.empty()) {
                ImGui::SameLine();
                ImGui::TextColored(t.text_dim, "%s of %s rows match.", Thousands(static_cast<long long>(p.rows_selected)).c_str(),
                                   Thousands(static_cast<long long>(p.rows_total)).c_str());
            }
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_faint, "All conditions must match. To filter for the whole graph, use a Filter Rows node.");
            ImGui::PopTextWrapPos();
            break;
        }
    }
    return changed;
}

void PlotWindow::DrawSettings() {
    const ui::Tokens& t = ui::CurrentTokens();
    const KindInfo& k = Info(spec_.kind);
    bool changed = false;
    // An automatic title follows the columns until the user edits it.
    const bool auto_title = spec_.title == DefaultTitle(spec_);
    const float w = ImGui::GetContentRegionAvail().x;

    changed |= DrawRows(w);
    // Axis names set by a preset follow their column: a new column drops them.
    const std::string x_before = spec_.x_column;
    const std::vector<std::string> y_before = spec_.y_columns;

    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "DATA");
    // X / categories / values.
    if ((k.required | k.optional) & kEncX) {
        ImGui::TextColored(t.text_dim, "%s", k.x_hint);
        changed |= picker_.Pick("##x", spec_.x_column, columns_, NumericX(spec_.kind),
                                (k.required & kEncX) ? nullptr : "(row number)", w);
    }
    // Y: several (one series each) or one.
    if ((k.required | k.optional) & kEncY) {
        ImGui::TextColored(t.text_dim, "%s", k.y_hint);
        const bool numeric_only = spec_.kind != Kind::Heatmap;
        if (k.multi_y) {
            changed |= picker_.PickMany("##y_many", spec_.y_columns, columns_, numeric_only, w);
            if (spec_.y_columns.size() > kMaxLegendSeries && spec_.kind != Kind::Box && spec_.kind != Kind::Violin) {
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(t.text_dim, "%zu series: the legend lists %zu, hover shows all.", spec_.y_columns.size(),
                                   kMaxLegendSeries);
                ImGui::PopTextWrapPos();
            }
        } else {
            std::string y = spec_.y_columns.empty() ? std::string() : spec_.y_columns.front();
            if (picker_.Pick("##y", y, columns_, numeric_only, (k.required & kEncY) ? nullptr : "(none)", w)) {
                spec_.y_columns.clear();
                if (!y.empty()) spec_.y_columns.push_back(y);
                changed = true;
            }
        }
    }
    if (k.optional & kEncColor) {
        ImGui::TextColored(t.text_dim, "Colour by");
        changed |= picker_.Pick("##colour", spec_.color_column, columns_, false, "(none)", w);
        // A number column on a scatter: groups or a colour scale.
        const int c = ColumnIndex(spec_.color_column);
        if (spec_.kind == Kind::Scatter && c >= 0 && numeric_[static_cast<size_t>(c)]) {
            static const char* const kModes[] = {"Groups", "Scale"};
            int mode = view_.HasData() && view_.Data().colour_scale ? 1 : 0;
            if (ui::SegmentedControl("##colour_mode", kModes, 2, &mode)) {
                spec_.color_mode = mode == 1 ? ColourMode::Scale : ColourMode::Groups;
                changed = true;
            }
        }
    }
    if (k.optional & kEncValue) {
        ImGui::TextColored(t.text_dim, "%s", k.value_hint);
        changed |= picker_.Pick("##value", spec_.value_column, columns_, true, "(count rows)", w);
    }
    if (spec_.x_column != x_before) spec_.x_label.clear();
    if (spec_.y_columns != y_before) spec_.y_label.clear();
    if (spec_.kind == Kind::Line || spec_.kind == Kind::Scatter) {
        changed |= ImGui::Checkbox("Show the y = x line", &spec_.show_diagonal);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("A reference line where y equals x (chance on a ROC curve)");
    }
    if (spec_.kind == Kind::Histogram || spec_.kind == Kind::Histogram2D || spec_.kind == Kind::Hexbin ||
        spec_.kind == Kind::Contour || spec_.kind == Kind::FilledContour) {
        ImGui::TextColored(t.text_dim, "%s", spec_.kind == Kind::Hexbin ? "Hexagons across"
                                             : spec_.kind == Kind::Contour || spec_.kind == Kind::FilledContour ? "Grid cells across"
                                                                                                               : "Bins");
        ImGui::SetNextItemWidth(w);
        if (ImGui::InputInt("##bins", &spec_.bins, 0, 0)) {
            spec_.bins = std::clamp(spec_.bins, spec_.kind == Kind::Histogram ? 1 : 2,
                                    spec_.kind == Kind::Histogram ? 1000 : spec_.kind == Kind::Histogram2D ? 400 : 200);
            changed = true;
        }
    }
    // P2b group 1 options (board 8).
    if (spec_.kind == Kind::Bar && !spec_.color_column.empty()) {
        ImGui::TextColored(t.text_dim, "Layout");
        static const char* const kLayouts[] = {"Grouped", "Stacked", "100%"};
        int layout = static_cast<int>(spec_.bar_layout);
        if (ui::SegmentedControl("##bar_layout", kLayouts, 3, &layout)) {
            spec_.bar_layout = static_cast<PlotSpec::BarLayout>(layout);
            changed = true;
        }
    }
    if (spec_.kind == Kind::Pie) changed |= ImGui::Checkbox("Donut (the total in the middle)", &spec_.donut);
    if (spec_.kind == Kind::Kde) {
        ImGui::TextColored(t.text_dim, "Bandwidth (times Silverman's)");
        float bw = static_cast<float>(spec_.kde_bandwidth);
        ImGui::SetNextItemWidth(w);
        if (ImGui::SliderFloat("##kde_bw", &bw, 0.2f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic)) {
            spec_.kde_bandwidth = bw;
            changed = true;
        }
    }
    if (spec_.kind == Kind::Matrix) {
        ImGui::TextColored(t.text_dim, "Values");
        static const char* const kMatrix[] = {"Correlation (Pearson)", "Correlation (Spearman, ranks)", "The values (rows x columns)"};
        int mode = static_cast<int>(spec_.matrix_values);
        ImGui::SetNextItemWidth(w);
        if (ImGui::Combo("##matrix_values", &mode, kMatrix, 3)) {
            spec_.matrix_values = static_cast<PlotSpec::MatrixValues>(mode);
            changed = true;
        }
    }
    if (spec_.kind == Kind::Hexbin && spec_.value_column.empty())
        changed |= ImGui::Checkbox("Colour by the log of the count", &spec_.log_colour);
    // P2b group 2 options (board 9).
    if (spec_.kind == Kind::Polar) {
        ImGui::TextColored(t.text_dim, "Angle in");
        static const char* const kUnits[] = {"Auto (categories for text, degrees for numbers)", "Degrees", "Radians",
                                             "Categories (share the turn)"};
        int unit = static_cast<int>(spec_.angle_unit);
        ImGui::SetNextItemWidth(w);
        if (ImGui::Combo("##angle_unit", &unit, kUnits, 4)) {
            spec_.angle_unit = static_cast<PlotSpec::AngleUnit>(unit);
            changed = true;
        }
        changed |= ImGui::Checkbox("Points instead of lines", &spec_.polar_points);
    }
    if (k.required & kEncVector) {
        ImGui::TextColored(t.text_dim, "Arrows from");
        static const char* const kFrom[] = {"u, v", "Direction + length"};
        int from = static_cast<int>(spec_.vector_from);
        if (ui::SegmentedControl("##vector_from", kFrom, 2, &from)) {
            spec_.vector_from = static_cast<PlotSpec::VectorFrom>(from);
            changed = true;
        }
        const bool uv = spec_.vector_from == PlotSpec::VectorFrom::UV;
        ImGui::TextColored(t.text_dim, "%s", uv ? "Arrow X (u)" : "Direction (degrees clockwise from north)");
        changed |= picker_.Pick("##u", spec_.u_column, columns_, true, nullptr, w);
        ImGui::TextColored(t.text_dim, "%s", uv ? "Arrow Y (v)" : "Length");
        changed |= picker_.Pick("##v", spec_.v_column, columns_, true, nullptr, w);
        if (!uv) {
            changed |= ImGui::Checkbox("Wind: the direction is where it comes from", &spec_.wind_from);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Weather data gives where the wind comes from; the arrows then point the other way");
        }
        if (spec_.kind == Kind::Quiver) {
            ImGui::TextColored(t.text_dim, "Show every Nth arrow (0: as many as fit)");
            ImGui::SetNextItemWidth(w);
            if (ImGui::InputInt("##arrow_every", &spec_.arrow_every, 0, 0)) {
                spec_.arrow_every = std::clamp(spec_.arrow_every, 0, 1000);
                changed = true;
            }
        } else {
            ImGui::TextColored(t.text_dim, "Density");
            float density = static_cast<float>(spec_.stream_density);
            ImGui::SetNextItemWidth(w);
            if (ImGui::SliderFloat("##stream_density", &density, 0.2f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic)) {
                spec_.stream_density = density;
                changed = true;
            }
        }
    }
    if (spec_.kind == Kind::Contour || spec_.kind == Kind::FilledContour) {
        ImGui::TextColored(t.text_dim, "Levels");
        ImGui::SetNextItemWidth(w);
        if (ImGui::InputInt("##levels", &spec_.levels, 0, 0)) {
            spec_.levels = std::clamp(spec_.levels, 1, 50);
            changed = true;
        }
    }
    if (spec_.kind == Kind::Histogram) {
        changed |= ImGui::Checkbox("Show median", &spec_.show_median);
        changed |= ImGui::Checkbox("Show mean", &spec_.show_mean);
        changed |= ImGui::Checkbox("Density instead of count", &spec_.density);
    }
    if (spec_.kind == Kind::Line || spec_.kind == Kind::Area || spec_.kind == Kind::Step) {
        ImGui::TextColored(t.text_dim, "Smoothing (moving average, 0 = off)");
        ImGui::SetNextItemWidth(w);
        if (ImGui::InputInt("##smooth", &spec_.smooth, 0, 0)) {
            spec_.smooth = std::clamp(spec_.smooth, 0, 100000);
            changed = true;
        }
    }

    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "LABELS");
    ImGui::TextColored(t.text_dim, "Title");
    ImGui::SetNextItemWidth(w);
    if (ImGui::InputText("##title", title_buf_, sizeof(title_buf_))) {
        spec_.title = title_buf_;
        changed = true;
    }

    // Values of the plotted column.
    if (view_.HasData() && view_.Data().problem.empty() && view_.Data().stats.count > 0) {
        const ColumnStats& st = view_.Data().stats;
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "VALUES");
        if (ImGui::BeginTable("##values", 2, ImGuiTableFlags_SizingStretchSame)) {
            const auto row = [&](const char* name, const std::string& value) {
                ImGui::TableNextColumn();
                ImGui::TextColored(t.text_dim, "%s", name);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(value.c_str());
            };
            char buf[48];
            const auto num = [&](double v) {
                std::snprintf(buf, sizeof(buf), "%.4g", v);
                return std::string(buf);
            };
            row("Count", Thousands(static_cast<long long>(st.count)));
            row("Missing", Thousands(static_cast<long long>(st.missing)));
            row("Min", num(st.min));
            row("Max", num(st.max));
            row("Mean", num(st.mean));
            row("Median", num(st.median));
            ImGui::EndTable();
        }
    }

    ImGui::Spacing();
    // The script is built on click (a histogram reads the whole column).
    const bool new_kind = spec_.kind == Kind::Kde || spec_.kind == Kind::Matrix || spec_.kind == Kind::Hexbin ||
                          spec_.kind == Kind::Polar || spec_.kind == Kind::Quiver || spec_.kind == Kind::Stream ||
                          spec_.kind == Kind::Contour || spec_.kind == Kind::FilledContour ||
                          (spec_.kind == Kind::Bar && !spec_.color_column.empty()) || (spec_.kind == Kind::Pie && spec_.donut);
    const bool scriptable = spec_.kind != Kind::Violin && spec_.kind != Kind::ErrorBars && spec_.kind != Kind::Heatmap &&
                            spec_.kind != Kind::Histogram2D && !new_kind && view_.HasData() && view_.Data().problem.empty();
    if (ui::SecondaryButton("Plot with Python (copy script)", scriptable, "Not for this plot type",
                            ui::ButtonSize::Small, w)) {
        const std::string script = PythonScript();
        if (!script.empty()) ImGui::SetClipboardText(script.c_str());
    }
    if (on_open_visualizer) {
        const int x = ColumnIndex(spec_.x_column.empty() && !spec_.y_columns.empty() ? spec_.y_columns.front() : spec_.x_column);
        if (ui::SecondaryButton("Open in Visualizer", x >= 0, "Choose a column first", ui::ButtonSize::Small, w))
            on_open_visualizer(x);
    }
    if (changed && auto_title && spec_.title == title_buf_) {
        spec_.title = DefaultTitle(spec_);
        std::snprintf(title_buf_, sizeof(title_buf_), "%s", spec_.title.c_str());
    }
    if (changed) Rebuild();
}

std::string PlotWindow::PythonScript() const {
    if (!view_.HasData() || !view_.Data().problem.empty() || view_.Data().series.empty()) return "";
    const Prepared& p = view_.Data();
    const Series& s = p.series.front();
    const auto& xs = s.all_x.empty() ? s.x : s.all_x;
    const auto& ys = s.all_y.empty() ? s.y : s.all_y;
    using K = plotscript::Kind;
    switch (p.spec.kind) {
        case Kind::Line: return plotscript::MatplotlibScript(K::Line, spec_.title, ys);
        case Kind::Area: return plotscript::MatplotlibScript(K::Area, spec_.title, ys);
        case Kind::Step: return plotscript::MatplotlibScript(K::Stairs, spec_.title, ys);
        case Kind::Stem: return plotscript::MatplotlibScript(K::Stem, spec_.title, ys);
        case Kind::Scatter: return plotscript::MatplotlibScript(K::Scatter, spec_.title, xs, ys);
        case Kind::Bar: return plotscript::MatplotlibScript(K::Bar, spec_.title, s.y);
        case Kind::Pie: return plotscript::MatplotlibScript(K::Pie, spec_.title, s.y);
        case Kind::Histogram:
        case Kind::Box: {
            // All finite values of the column.
            const std::string col = p.spec.kind == Kind::Histogram ? spec_.x_column
                                                                   : (spec_.y_columns.empty() ? "" : spec_.y_columns.front());
            const int c = ColumnIndex(col);
            if (c < 0 || (!table_ && !arrow_table_)) return "";
            std::vector<double> values;
            std::vector<std::string> read = ColumnsNeeded(spec_);
            if (std::find(read.begin(), read.end(), col) == read.end()) read.push_back(col);
            const Source src = arrow_table_ ? SourceFromArrow(*arrow_table_, read) : SourceFromTable(*table_, read, numeric_);
            const RowSelection rows = SelectRows(spec_, src);  // the same rows as the plot
            if (!rows.problem.empty()) return "";
            const SourceColumn* values_column = (rows.all ? src : rows.source).Find(col);
            if (!values_column) return "";
            for (double v : values_column->numbers)
                if (std::isfinite(v)) values.push_back(v);
            return plotscript::MatplotlibScript(p.spec.kind == Kind::Histogram ? K::Histogram : K::Box, spec_.title, values);
        }
        default: return "";
    }
}

}  // namespace cyxwiz::plot
