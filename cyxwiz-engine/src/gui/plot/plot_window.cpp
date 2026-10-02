#include "plot_window.h"

#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../../core/plot/plot_table_source.h"
#include "../../core/plot_script.h"
#include "../../data/data_table.h"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

void PlotWindow::SetTable(std::shared_ptr<DataTable> table, size_t row_limit, size_t total_rows) {
    table_ = std::move(table);
    row_limit_ = row_limit;
    total_rows_ = total_rows;
    headers_ = table_ ? table_->GetHeaders() : std::vector<std::string>{};
    numeric_ = table_ ? NumericColumns(*table_) : std::vector<bool>{};
    Rebuild();
}

int PlotWindow::ColumnIndex(const std::string& name) const {
    for (size_t i = 0; i < headers_.size(); ++i)
        if (headers_[i] == name) return static_cast<int>(i);
    return -1;
}

void PlotWindow::Rebuild() {
    if (!table_) {
        view_.Clear();
        return;
    }
    if (busy_) {  // prepare again when the running job is done
        dirty_ = true;
        return;
    }
    // Copy the columns on the UI thread (the table is not thread-safe), then
    // prepare off it.
    std::vector<std::string> needed = spec_.y_columns;
    needed.push_back(spec_.x_column);
    needed.push_back(spec_.color_column);
    Source src = SourceFromTable(*table_, needed, numeric_);
    src.row_limit = row_limit_;
    src.total_rows = total_rows_;
    busy_ = true;
    dirty_ = false;
    job_ = std::async(std::launch::async, [spec = spec_, src = std::move(src)]() { return Prepare(spec, src); });
}

void PlotWindow::Poll() {
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
    // Source line.
    if (table_) {
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
        ImGui::TextDisabled("No table. Open a table in the Table Viewer and choose Plot on a column.");
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
    view_.Draw(ImVec2(0, 0), vo);
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
                if ((k.kind == Kind::Box || k.kind == Kind::Violin) && spec_.y_columns.empty() && !spec_.x_column.empty())
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
    ImGui::TextColored(t.text_faint, "Model results: P2");
    ImGui::TextColored(t.text_faint, "3D: P4");
}

void PlotWindow::DrawSettings() {
    const ui::Tokens& t = ui::CurrentTokens();
    const KindInfo& k = Info(spec_.kind);
    bool changed = false;
    // An automatic title follows the columns until the user edits it.
    const bool auto_title = spec_.title == DefaultTitle(spec_);
    const float w = ImGui::GetContentRegionAvail().x;

    ImGui::TextColored(t.text_dim, "DATA");
    // X / categories / values.
    if ((k.required | k.optional) & kEncX) {
        ImGui::TextColored(t.text_dim, "%s", k.x_hint);
        ImGui::SetNextItemWidth(w);
        const bool numeric_only = NumericX(spec_.kind);
        const std::string preview = spec_.x_column.empty() ? std::string("(row number)") : spec_.x_column;
        if (ImGui::BeginCombo("##x", preview.c_str())) {
            if (!(k.required & kEncX) && ImGui::Selectable("(row number)", spec_.x_column.empty())) {
                spec_.x_column.clear();
                changed = true;
            }
            for (size_t i = 0; i < headers_.size(); ++i) {
                if (numeric_only && !numeric_[i]) continue;
                if (ImGui::Selectable(headers_[i].c_str(), headers_[i] == spec_.x_column)) {
                    spec_.x_column = headers_[i];
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
    }
    // Y: several (one series each) or one.
    if ((k.required | k.optional) & kEncY) {
        ImGui::TextColored(t.text_dim, "%s", k.y_hint);
        const bool text_ok = spec_.kind == Kind::Heatmap;
        if (k.multi_y) {
            for (size_t i = 0; i < headers_.size(); ++i) {
                if (!numeric_[i]) continue;
                bool on = std::find(spec_.y_columns.begin(), spec_.y_columns.end(), headers_[i]) != spec_.y_columns.end();
                if (ImGui::Checkbox(headers_[i].c_str(), &on)) {
                    if (on) spec_.y_columns.push_back(headers_[i]);
                    else spec_.y_columns.erase(std::remove(spec_.y_columns.begin(), spec_.y_columns.end(), headers_[i]),
                                               spec_.y_columns.end());
                    changed = true;
                }
            }
        } else {
            ImGui::SetNextItemWidth(w);
            const std::string preview = spec_.y_columns.empty() ? std::string(k.required & kEncY ? "(choose)" : "(none)")
                                                                : spec_.y_columns.front();
            if (ImGui::BeginCombo("##y", preview.c_str())) {
                if (!(k.required & kEncY) && ImGui::Selectable("(none)", spec_.y_columns.empty())) {
                    spec_.y_columns.clear();
                    changed = true;
                }
                for (size_t i = 0; i < headers_.size(); ++i) {
                    if (!text_ok && !numeric_[i]) continue;
                    if (ImGui::Selectable(headers_[i].c_str(), !spec_.y_columns.empty() && spec_.y_columns.front() == headers_[i])) {
                        spec_.y_columns = {headers_[i]};
                        changed = true;
                    }
                }
                ImGui::EndCombo();
            }
        }
    }
    if (k.optional & kEncColor) {
        ImGui::TextColored(t.text_dim, "Colour by");
        ImGui::SetNextItemWidth(w);
        if (ImGui::BeginCombo("##colour", spec_.color_column.empty() ? "(none)" : spec_.color_column.c_str())) {
            if (ImGui::Selectable("(none)", spec_.color_column.empty())) {
                spec_.color_column.clear();
                changed = true;
            }
            for (const auto& hname : headers_)
                if (ImGui::Selectable(hname.c_str(), hname == spec_.color_column)) {
                    spec_.color_column = hname;
                    changed = true;
                }
            ImGui::EndCombo();
        }
    }
    if (spec_.kind == Kind::Histogram || spec_.kind == Kind::Histogram2D) {
        ImGui::TextColored(t.text_dim, "Bins");
        ImGui::SetNextItemWidth(w);
        if (ImGui::InputInt("##bins", &spec_.bins, 0, 0)) {
            spec_.bins = std::clamp(spec_.bins, 1, spec_.kind == Kind::Histogram2D ? 400 : 1000);
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
    const bool scriptable = spec_.kind != Kind::Violin && spec_.kind != Kind::ErrorBars && spec_.kind != Kind::Heatmap &&
                            spec_.kind != Kind::Histogram2D && view_.HasData() && view_.Data().problem.empty();
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
            if (c < 0 || !table_) return "";
            std::vector<double> values;
            const Source src = SourceFromTable(*table_, {col}, numeric_);
            if (src.columns.empty()) return "";
            for (double v : src.columns.front().numbers)
                if (std::isfinite(v)) values.push_back(v);
            return plotscript::MatplotlibScript(p.spec.kind == Kind::Histogram ? K::Histogram : K::Box, spec_.title, values);
        }
        default: return "";
    }
}

}  // namespace cyxwiz::plot
