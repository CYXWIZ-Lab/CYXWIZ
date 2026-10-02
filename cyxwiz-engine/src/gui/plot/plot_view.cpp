#include "plot_view.h"

#include "plot_style.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"

#include <implot.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace cyxwiz::plot {

namespace {

ViewHooks& HookStore() {
    static ViewHooks hooks;
    return hooks;
}

ImVec4 SeriesColour(size_t i) {
    return ui::CurrentTokens().series[i % ui::Tokens::kSeriesCount];
}

std::string Value(double v) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.4g", v);
    return buf;
}

// One tooltip row: a colour dot, a name, a value.
void TooltipRow(const ImVec4& colour, const std::string& name, const std::string& value) {
    const float h = ImGui::GetTextLineHeight();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + h * 0.3f, p.y + h * 0.5f), h * 0.28f, ui::ToU32(colour));
    ImGui::Dummy(ImVec2(h * 0.75f, h));
    ImGui::SameLine();
    ImGui::TextUnformatted(name.c_str());
    ImGui::SameLine();
    ImGui::TextColored(ui::CurrentTokens().text_bright, "%s", value.c_str());
}

bool IsLineLike(Kind k) {
    return k == Kind::Line || k == Kind::Area || k == Kind::Step || k == Kind::Stem;
}

std::string XLabel(const Prepared& p) {
    if (!p.spec.x_label.empty()) return p.spec.x_label;
    if (!p.spec.x_column.empty()) return p.spec.x_column;
    return IsLineLike(p.spec.kind) ? "row" : "";
}

std::string YLabel(const Prepared& p) {
    if (!p.spec.y_label.empty()) return p.spec.y_label;
    if (p.spec.kind == Kind::Histogram) return p.spec.density ? "density" : "count";
    if ((p.spec.kind == Kind::Bar || p.spec.kind == Kind::Pie) && p.spec.y_columns.empty()) return "rows";
    if (p.spec.y_columns.size() == 1) return p.spec.y_columns.front();
    return "";
}

ImVec4 LabelColour(DataLabel::State s) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (s) {
        case DataLabel::State::Exact: return t.success;
        case DataLabel::State::Reduced:
        case DataLabel::State::Sampled: return t.info;
        case DataLabel::State::Truncated: return t.warning;
    }
    return t.text_dim;
}

// Nearest index of `x` in an ascending array.
size_t NearestSorted(const std::vector<double>& xs, double x) {
    auto it = std::lower_bound(xs.begin(), xs.end(), x);
    if (it == xs.end()) return xs.size() - 1;
    if (it == xs.begin()) return 0;
    const size_t hi = static_cast<size_t>(it - xs.begin());
    return std::fabs(xs[hi] - x) < std::fabs(xs[hi - 1] - x) ? hi : hi - 1;
}

size_t NearestScan(const std::vector<double>& xs, double x) {
    size_t best = 0;
    for (size_t i = 1; i < xs.size(); ++i)
        if (std::fabs(xs[i] - x) < std::fabs(xs[best] - x)) best = i;
    return best;
}

}  // namespace

void SetViewHooks(ViewHooks hooks) { HookStore() = std::move(hooks); }
const ViewHooks& Hooks() { return HookStore(); }

PlotView::PlotView(std::string id) : id_(std::move(id)) {}

void PlotView::SetData(Prepared data) {
    const bool kind_changed = !has_data_ || data.spec.kind != data_.spec.kind ||
                              data.spec.x_column != data_.spec.x_column || data.spec.y_columns != data_.spec.y_columns;
    data_ = std::move(data);
    has_data_ = true;
    log_y_ = data_.spec.log_y;
    legend_ = data_.spec.legend;
    if (kind_changed) fit_ = true;
}

void PlotView::Clear() {
    data_ = Prepared{};
    has_data_ = false;
}

std::string PlotView::Title() const {
    if (!data_.spec.title.empty()) return data_.spec.title;
    return std::string(Info(data_.spec.kind).label);
}

void PlotView::FinishCaptures() {
    if (!pending_->ready) return;
    const auto action = pending_->action;
    std::vector<unsigned char> png = std::move(pending_->png);
    *pending_ = Pending{};
    const ViewHooks& h = Hooks();
    if (png.empty()) {
        note_ = "Could not read the plot image.";
    } else if (action == Pending::Action::Copy && h.copy_png) {
        note_ = h.copy_png(png) ? "Image copied." : "Could not copy the image.";
    } else if (action == Pending::Action::Save && h.save_path) {
        if (auto path = h.save_path("Save plot as PNG", "png", export_name_ + ".png")) {
            std::ofstream(*path, std::ios::binary).write(reinterpret_cast<const char*>(png.data()),
                                                          static_cast<std::streamsize>(png.size()));
            note_ = "Saved " + *path;
        }
    }
    note_until_ = ImGui::GetTime() + 4.0;
}

void PlotView::ExportMenu(const Options& o) {
    const ViewHooks& h = Hooks();
    const bool can_capture = static_cast<bool>(h.capture_png);
    const bool can_save = static_cast<bool>(h.save_path);
    if (ImGui::MenuItem("Save image as PNG...", nullptr, false, can_capture && can_save)) {
        pending_->action = Pending::Action::Save;
        capture_frame_ = ImGui::GetFrameCount() + 1;  // after this menu is gone
    }
    if (ImGui::MenuItem("Save image as SVG...", nullptr, false, can_save)) {
        if (auto path = h.save_path("Save plot as SVG", "svg", o.export_name + ".svg")) {
            const ui::Tokens& t = ui::CurrentTokens();
            SvgStyle st;
            st.background = HexColour(t.plot_bg.x, t.plot_bg.y, t.plot_bg.z);
            const ImVec4 grid = ui::Mix(t.plot_bg, t.text_dim, 0.25f);
            st.grid = HexColour(grid.x, grid.y, grid.z);
            st.text = HexColour(t.text.x, t.text.y, t.text.z);
            st.text_dim = HexColour(t.text_dim.x, t.text_dim.y, t.text_dim.z);
            for (int i = 0; i < 6; ++i) st.series[i] = HexColour(t.series[i].x, t.series[i].y, t.series[i].z);
            st.scale_low = HexColour(t.scale_sequential[0].x, t.scale_sequential[0].y, t.scale_sequential[0].z);
            st.scale_high = HexColour(t.scale_sequential[3].x, t.scale_sequential[3].y, t.scale_sequential[3].z);
            std::ofstream(*path, std::ios::binary) << ToSvg(data_, range_, st);
            note_ = "Saved " + *path;
            note_until_ = ImGui::GetTime() + 4.0;
        }
    }
    if (ImGui::MenuItem("Copy image", nullptr, false, can_capture && static_cast<bool>(h.copy_png))) {
        pending_->action = Pending::Action::Copy;
        capture_frame_ = ImGui::GetFrameCount() + 1;
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Save data as CSV...", nullptr, false, can_save)) {
        if (auto path = h.save_path("Save plot data as CSV", "csv", o.export_name + ".csv")) {
            std::ofstream(*path, std::ios::binary) << ToCsv(data_);
            note_ = "Saved " + *path;
            note_until_ = ImGui::GetTime() + 4.0;
        }
    }
    if (ImGui::MenuItem("Copy data")) {
        ImGui::SetClipboardText(ToCsv(data_).c_str());
        note_ = "Data copied as CSV.";
        note_until_ = ImGui::GetTime() + 4.0;
    }
    if (o.python_script) {
        ImGui::Separator();
        if (ImGui::MenuItem("Plot with Python (copy script)")) {
            ImGui::SetClipboardText(o.python_script().c_str());
            note_ = "Python script copied: paste it in the Script Editor and run it.";
            note_until_ = ImGui::GetTime() + 5.0;
        }
    }
}

void PlotView::DrawToolbar(const Options& o) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (o.toolbar_left) {
        o.toolbar_left();
        ImGui::SameLine();
    }
    if (has_data_ && data_.problem.empty()) {
        const std::string label = data_.label.Text();
        ui::StatusPill(("##label" + id_).c_str(), label.c_str(), LabelColour(data_.label.state));
        if (ImGui::IsItemHovered()) {
            switch (data_.label.state) {
                case DataLabel::State::Exact: ImGui::SetTooltip("All values are drawn."); break;
                case DataLabel::State::Reduced:
                    ImGui::SetTooltip("Long series are drawn with the lowest and highest value of each step,\nso spikes stay visible. Hover values and exports use all points.");
                    break;
                case DataLabel::State::Sampled:
                    ImGui::SetTooltip("An even sample of the rows (the same each time). Exports use all rows.");
                    break;
                case DataLabel::State::Truncated: ImGui::SetTooltip("The table was read with a row limit."); break;
            }
        }
        ImGui::SameLine();
    }
    if (ImGui::GetTime() < note_until_) {
        ImGui::TextColored(t.text_dim, "%s", note_.c_str());
        ImGui::SameLine();
    }
    // Right-aligned actions.
    const char* own = ICON_FA_WINDOW_RESTORE;
    const float w = ui::ButtonWidth("Fit", ui::ButtonSize::Small) + ui::ButtonWidth("Log Y", ui::ButtonSize::Small) +
                    ui::ButtonWidth("Legend", ui::ButtonSize::Small) + ui::ButtonWidth("Export", ui::ButtonSize::Small) +
                    ui::ButtonWidth(own, ui::ButtonSize::Small) + ImGui::GetStyle().ItemSpacing.x * 4;
    const float right = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - w;
    if (right > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(right);
    const bool usable = has_data_ && data_.problem.empty();
    if (ui::GhostButton(("Fit##" + id_).c_str(), usable)) fit_ = true;
    ImGui::SameLine();
    const bool can_log = usable && data_.spec.kind != Kind::Pie && data_.spec.kind != Kind::Heatmap &&
                         data_.spec.kind != Kind::Histogram2D;
    if (ui::GhostButton(("Log Y##" + id_).c_str(), can_log, "Not for this plot type", log_y_)) {
        log_y_ = !log_y_;
        fit_ = true;
    }
    ImGui::SameLine();
    if (ui::GhostButton(("Legend##" + id_).c_str(), usable, nullptr, legend_)) legend_ = !legend_;
    ImGui::SameLine();
    if (ui::GhostButton(("Export##" + id_).c_str(), usable)) ImGui::OpenPopup(("##export" + id_).c_str());
    if (ImGui::BeginPopup(("##export" + id_).c_str())) {
        ExportMenu(o);
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ui::GhostButton((std::string(own) + "##own" + id_).c_str(), true, nullptr, own_window)) own_window = !own_window;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(own_window ? "Back into this panel" : "Open in its own window");
}

void PlotView::Draw(ImVec2 size, const Options& o) {
    FinishCaptures();
    export_name_ = o.export_name;
    ImGui::PushID(id_.c_str());
    if (o.toolbar) DrawToolbar(o);
    if (o.show_title && has_data_) {
        ui::FontScope bold(ui::Font::Medium);
        ImGui::TextUnformatted(Title().c_str());
    }
    if (size.y <= 0) size.y = std::max(120.0f, ImGui::GetContentRegionAvail().y + size.y);
    if (size.x <= 0) size.x = ImGui::GetContentRegionAvail().x + size.x;
    if (own_window && !drawing_own_window_) {
        ImGui::TextDisabled("Shown in its own window.");
        ImGui::SameLine();
        if (ui::LinkButton("Bring it back")) own_window = false;
        ImGui::PopID();
        return;
    }
    if (!has_data_) {
        ImGui::Dummy(ImVec2(size.x, std::min(size.y, 40.0f)));
    } else if (!data_.problem.empty()) {
        const ImVec2 at = ImGui::GetCursorPos();
        ImGui::Dummy(size);
        const ImVec2 ts = ImGui::CalcTextSize(data_.problem.c_str());
        ImGui::SetCursorPos(ImVec2(at.x + std::max(0.0f, (size.x - ts.x) * 0.5f), at.y + size.y * 0.4f));
        ImGui::TextDisabled("%s", data_.problem.c_str());
    } else {
        DrawPlot(size);
        if (pending_->action != Pending::Action::None && capture_frame_ > 0 && ImGui::GetFrameCount() >= capture_frame_) {
            capture_frame_ = 0;
            auto sink = pending_;
            if (Hooks().capture_png)
                Hooks().capture_png(frame_min_, frame_max_, [sink](std::vector<unsigned char> png) {
                    sink->png = std::move(png);
                    sink->ready = true;
                });
        }
    }
    ImGui::PopID();
}

void PlotView::DrawOwnWindow(const Options& o) {
    if (!own_window) return;
    ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_FirstUseEver);
    const std::string title = Title() + "###plotview_own_" + id_;
    if (ImGui::Begin(title.c_str(), &own_window)) {
        drawing_own_window_ = true;
        Draw(ImVec2(0, 0), o);
        drawing_own_window_ = false;
    }
    ImGui::End();
}

void PlotView::DrawPlot(ImVec2 size) {
    const Prepared& p = data_;
    const Kind kind = p.spec.kind;
    const ui::Tokens& t = ui::CurrentTokens();
    const bool scale_bar = kind == Kind::Heatmap || kind == Kind::Histogram2D;
    ImVec2 plot_size = size;
    if (scale_bar) plot_size.x = std::max(100.0f, size.x - 76.0f);

    ImPlotFlags flags = ImPlotFlags_NoTitle | ImPlotFlags_NoMenus;
    if (!legend_) flags |= ImPlotFlags_NoLegend;
    if (kind == Kind::Pie) flags |= ImPlotFlags_Equal | ImPlotFlags_NoMouseText;

    // Fixed-layout kinds set their limits; the rest fit to the data.
    const ImPlotCond cond = fit_ ? ImPlotCond_Always : ImPlotCond_Once;
    if (fit_ && kind != Kind::Pie && kind != Kind::Box && kind != Kind::Violin && kind != Kind::Heatmap &&
        kind != Kind::Histogram2D)
        ImPlot::SetNextAxesToFit();

    const std::string plot_id = "##plot";
    if (!ImPlot::BeginPlot(plot_id.c_str(), plot_size, flags)) return;

    const std::string xl = XLabel(p), yl = YLabel(p);
    if (kind == Kind::Pie) {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxesLimits(0, 1, 0, 1, ImPlotCond_Always);
    } else {
        ImPlotAxisFlags xf = ImPlotAxisFlags_None, yf = ImPlotAxisFlags_None;
        if (kind == Kind::Heatmap) xf = yf = ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
        ImPlot::SetupAxes(xl.empty() ? nullptr : xl.c_str(), yl.empty() ? nullptr : yl.c_str(), xf, yf);
        if (log_y_ && kind != Kind::Heatmap && kind != Kind::Histogram2D) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
    }
    ImPlot::SetupLegend(ImPlotLocation_NorthEast);

    // Category ticks (bar, error bars, box, violin, heatmap).
    std::vector<const char*> names;
    std::vector<double> positions;
    const auto set_ticks = [&](ImAxis axis, const std::vector<std::string>& labels, double offset) {
        names.clear();
        positions.clear();
        if (labels.empty() || labels.size() > 40) return;
        for (size_t i = 0; i < labels.size(); ++i) {
            names.push_back(labels[i].c_str());
            positions.push_back(static_cast<double>(i) + offset);
        }
        ImPlot::SetupAxisTicks(axis, positions.data(), static_cast<int>(positions.size()), names.data());
    };
    std::vector<std::string> series_names;
    for (const auto& s : p.series) series_names.push_back(s.label);
    std::vector<const char*> xnames, ynames;
    std::vector<double> xpos, ypos;

    switch (kind) {
        case Kind::Bar:
        case Kind::ErrorBars: set_ticks(ImAxis_X1, p.categories, 0.0); break;
        case Kind::Box:
        case Kind::Violin: {
            set_ticks(ImAxis_X1, series_names, 0.0);
            double lo = 0, hi = 1;
            bool first = true;
            for (size_t i = 0; i < p.boxes.size(); ++i) {
                const auto& b = p.boxes[i];
                double a = b.low, z = b.high;
                if (kind == Kind::Violin && i < p.series.size() && !p.series[i].y.empty()) {
                    a = std::min(a, p.series[i].y.front());
                    z = std::max(z, p.series[i].y.back());
                }
                lo = first ? a : std::min(lo, a);
                hi = first ? z : std::max(hi, z);
                first = false;
            }
            const double pad = (hi - lo) * 0.08 + 1e-9;
            ImPlot::SetupAxesLimits(-0.6, static_cast<double>(p.boxes.size()) - 0.4, lo - pad, hi + pad, cond);
            break;
        }
        case Kind::Heatmap: {
            for (size_t i = 0; i < p.col_names.size() && p.col_names.size() <= 40; ++i) {
                xnames.push_back(p.col_names[i].c_str());
                xpos.push_back(static_cast<double>(i) + 0.5);
            }
            // Row 0 is drawn at the top.
            for (size_t i = 0; i < p.row_names.size() && p.row_names.size() <= 40; ++i) {
                ynames.push_back(p.row_names[i].c_str());
                ypos.push_back(static_cast<double>(p.grid_rows) - static_cast<double>(i) - 0.5);
            }
            if (!xpos.empty()) ImPlot::SetupAxisTicks(ImAxis_X1, xpos.data(), static_cast<int>(xpos.size()), xnames.data());
            if (!ypos.empty()) ImPlot::SetupAxisTicks(ImAxis_Y1, ypos.data(), static_cast<int>(ypos.size()), ynames.data());
            ImPlot::SetupAxesLimits(0, p.grid_cols, 0, p.grid_rows, cond);
            break;
        }
        case Kind::Histogram2D: ImPlot::SetupAxesLimits(p.x_min, p.x_max, p.y_min, p.y_max, cond); break;
        default: break;
    }
    fit_ = false;

    const auto n = [](const std::vector<double>& v) { return static_cast<int>(v.size()); };
    switch (kind) {
        case Kind::Line:
        case Kind::Area:
        case Kind::Step:
        case Kind::Stem:
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                const ImVec4 c = SeriesColour(i);
                const bool smoothed = !s.smooth_y.empty();
                const char* label = s.label.c_str();
                if (kind == Kind::Area) {
                    ImPlot::SetNextFillStyle(c, 0.25f);
                    ImPlot::PlotShaded(label, s.x.data(), s.y.data(), n(s.x), 0.0);
                }
                ImPlot::SetNextLineStyle(smoothed ? ui::WithAlpha(c, 0.55f) : c, smoothed ? 1.2f : 1.8f);
                if (kind == Kind::Step) ImPlot::PlotStairs(label, s.x.data(), s.y.data(), n(s.x));
                else if (kind == Kind::Stem) {
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f, c, 0.0f);
                    ImPlot::PlotStems(label, s.x.data(), s.y.data(), n(s.x));
                } else ImPlot::PlotLine(label, s.x.data(), s.y.data(), n(s.x));
                if (smoothed) {
                    const std::string sl = s.label + ", smoothed (" + std::to_string(p.spec.smooth) + ")";
                    ImPlot::SetNextLineStyle(ui::Mix(c, t.text_bright, 0.35f), 2.6f);
                    ImPlot::PlotLine(sl.c_str(), s.smooth_x.data(), s.smooth_y.data(), n(s.smooth_x));
                }
            }
            break;
        case Kind::Scatter:
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f, ui::WithAlpha(SeriesColour(i), 0.7f), 0.0f);
                ImPlot::PlotScatter(s.label.c_str(), s.x.data(), s.y.data(), n(s.x));
            }
            break;
        case Kind::Histogram: {
            const double width = p.edges.size() > 1 ? p.edges[1] - p.edges[0] : 1.0;
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                ImPlot::SetNextFillStyle(SeriesColour(i), p.series.size() > 1 ? 0.6f : 0.9f);
                ImPlot::PlotBars(s.label.c_str(), s.x.data(), s.y.data(), n(s.x), width * 0.94);
            }
            if (p.spec.show_median) {
                const std::string l = "median " + Value(p.stats.median);
                ImPlot::SetNextLineStyle(SeriesColour(2), 2.0f);
                ImPlot::PlotInfLines(l.c_str(), &p.stats.median, 1);
            }
            if (p.spec.show_mean) {
                const std::string l = "mean " + Value(p.stats.mean);
                ImPlot::SetNextLineStyle(SeriesColour(1), 2.0f);
                ImPlot::PlotInfLines(l.c_str(), &p.stats.mean, 1);
            }
            break;
        }
        case Kind::Bar:
            if (!p.series.empty()) {
                ImPlot::SetNextFillStyle(SeriesColour(0), 0.9f);
                ImPlot::PlotBars(p.series[0].label.c_str(), p.series[0].x.data(), p.series[0].y.data(), n(p.series[0].x), 0.67);
            }
            break;
        case Kind::ErrorBars:
            if (!p.series.empty()) {
                const auto& s = p.series[0];
                ImPlot::SetNextErrorBarStyle(SeriesColour(0), 1.5f, 6.0f);
                ImPlot::PlotErrorBars(s.label.c_str(), s.x.data(), s.y.data(), s.low.data(), s.high.data(), n(s.x));
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f, SeriesColour(0), 0.0f);
                ImPlot::PlotScatter(s.label.c_str(), s.x.data(), s.y.data(), n(s.x));
            }
            break;
        case Kind::Pie:
            if (!p.series.empty() && !p.categories.empty()) {
                std::vector<const char*> labels;
                for (const auto& c : p.categories) labels.push_back(c.c_str());
                ImPlot::PlotPieChart(labels.data(), p.series[0].y.data(), static_cast<int>(labels.size()), 0.5, 0.5, 0.4,
                                     "%.0f", 90);
            }
            break;
        case Kind::Box:
        case Kind::Violin: {
            ImDrawList* dl = ImPlot::GetPlotDrawList();
            ImPlot::PushPlotClipRect();
            for (size_t i = 0; i < p.boxes.size(); ++i) {
                const auto& b = p.boxes[i];
                const double x = static_cast<double>(i);
                const ImVec4 c = SeriesColour(i);
                if (kind == Kind::Violin && i < p.series.size()) {
                    const auto& s = p.series[i];
                    for (size_t k = 0; k + 1 < s.y.size(); ++k) {
                        dl->AddQuadFilled(ImPlot::PlotToPixels(s.low[k], s.y[k]), ImPlot::PlotToPixels(s.high[k], s.y[k]),
                                          ImPlot::PlotToPixels(s.high[k + 1], s.y[k + 1]),
                                          ImPlot::PlotToPixels(s.low[k + 1], s.y[k + 1]), ui::ToU32(ui::WithAlpha(c, 0.45f)));
                    }
                }
                const double w = kind == Kind::Violin ? 0.08 : 0.3;
                const ImVec2 a = ImPlot::PlotToPixels(x - w, b.q3), z = ImPlot::PlotToPixels(x + w, b.q1);
                dl->AddRectFilled(a, z, ui::ToU32(ui::WithAlpha(c, 0.35f)), 2.0f);
                dl->AddRect(a, z, ui::ToU32(c), 2.0f);
                dl->AddLine(ImPlot::PlotToPixels(x - w, b.median), ImPlot::PlotToPixels(x + w, b.median), ui::ToU32(t.text_bright), 2.0f);
                dl->AddLine(ImPlot::PlotToPixels(x, b.q3), ImPlot::PlotToPixels(x, b.high), ui::ToU32(c), 1.5f);
                dl->AddLine(ImPlot::PlotToPixels(x, b.q1), ImPlot::PlotToPixels(x, b.low), ui::ToU32(c), 1.5f);
                dl->AddLine(ImPlot::PlotToPixels(x - w * 0.5, b.high), ImPlot::PlotToPixels(x + w * 0.5, b.high), ui::ToU32(c), 1.5f);
                dl->AddLine(ImPlot::PlotToPixels(x - w * 0.5, b.low), ImPlot::PlotToPixels(x + w * 0.5, b.low), ui::ToU32(c), 1.5f);
            }
            ImPlot::PopPlotClipRect();
            break;
        }
        case Kind::Heatmap:
        case Kind::Histogram2D: {
            double peak = 0;
            for (double v : p.grid) peak = std::max(peak, v);
            ImPlot::PushColormap(SequentialColormap());
            const bool numbers = kind == Kind::Heatmap && p.grid.size() <= 400;
            const ImPlotPoint lo = kind == Kind::Heatmap ? ImPlotPoint(0, 0) : ImPlotPoint(p.x_min, p.y_min);
            const ImPlotPoint hi = kind == Kind::Heatmap ? ImPlotPoint(p.grid_cols, p.grid_rows) : ImPlotPoint(p.x_max, p.y_max);
            ImPlot::PlotHeatmap("##grid", p.grid.data(), p.grid_rows, p.grid_cols, 0.0, peak > 0 ? peak : 1.0,
                                numbers ? "%g" : nullptr, lo, hi);
            ImPlot::PopColormap();
            break;
        }
    }

    if (ImPlot::IsPlotHovered() && pending_->action == Pending::Action::None) DrawHover();
    const ImPlotRect lim = ImPlot::GetPlotLimits();
    range_ = AxisRange{lim.X.Min, lim.X.Max, lim.Y.Min, lim.Y.Max, log_y_};
    ImPlot::EndPlot();
    frame_min_ = ImGui::GetItemRectMin();
    frame_max_ = ImGui::GetItemRectMax();

    if (scale_bar) {
        double peak = 0;
        for (double v : p.grid) peak = std::max(peak, v);
        ImGui::SameLine();
        ImPlot::ColormapScale("##scale", 0.0, peak > 0 ? peak : 1.0, ImVec2(60, plot_size.y), "%g", 0, SequentialColormap());
    }
}

void PlotView::DrawHover() {
    const Prepared& p = data_;
    const ImPlotPoint m = ImPlot::GetPlotMousePos();
    const Kind kind = p.spec.kind;
    const ui::Tokens& t = ui::CurrentTokens();
    bool shown = false;
    const auto begin = [&](const std::string& header) {
        if (!shown) {
            ImGui::BeginTooltip();
            ImGui::TextColored(t.text_bright, "%s", header.c_str());
            shown = true;
        }
    };
    const auto category = [&](const std::vector<std::string>& names, double x) -> int {
        const int i = static_cast<int>(std::lround(x));
        return i >= 0 && i < static_cast<int>(names.size()) && std::fabs(x - i) <= 0.45 ? i : -1;
    };
    switch (kind) {
        case Kind::Line:
        case Kind::Area:
        case Kind::Step:
        case Kind::Stem: {
            // The value of each series at the x nearest the mouse (all points).
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                const auto& xs = s.all_x.empty() ? s.x : s.all_x;
                const auto& ys = s.all_y.empty() ? s.y : s.all_y;
                if (xs.empty()) continue;
                const size_t k = s.x_sorted ? NearestSorted(xs, m.x) : NearestScan(xs, m.x);
                begin(XLabel(p) + " " + Value(xs[k]));
                TooltipRow(SeriesColour(i), s.label, Value(ys[k]));
                if (!s.smooth_y.empty()) {
                    const size_t j = NearestScan(s.smooth_x, xs[k]);
                    TooltipRow(ui::Mix(SeriesColour(i), t.text_bright, 0.35f),
                               s.label + ", smoothed (" + std::to_string(p.spec.smooth) + ")", Value(s.smooth_y[j]));
                }
            }
            break;
        }
        case Kind::Scatter: {
            // The nearest drawn point within 12 px.
            const ImVec2 mouse = ImGui::GetMousePos();
            float best = 144.0f;
            size_t bs = 0, bk = 0;
            bool found = false;
            for (size_t i = 0; i < p.series.size(); ++i)
                for (size_t k = 0; k < p.series[i].x.size(); ++k) {
                    const ImVec2 q = ImPlot::PlotToPixels(p.series[i].x[k], p.series[i].y[k]);
                    const float d = (q.x - mouse.x) * (q.x - mouse.x) + (q.y - mouse.y) * (q.y - mouse.y);
                    if (d < best) {
                        best = d;
                        bs = i;
                        bk = k;
                        found = true;
                    }
                }
            if (found) {
                const auto& s = p.series[bs];
                begin(s.label);
                TooltipRow(SeriesColour(bs), XLabel(p), Value(s.x[bk]));
                TooltipRow(SeriesColour(bs), p.spec.y_columns.empty() ? "y" : p.spec.y_columns.front(), Value(s.y[bk]));
            }
            break;
        }
        case Kind::Histogram: {
            if (p.edges.size() < 2 || m.x < p.edges.front() || m.x > p.edges.back()) break;
            const double width = p.edges[1] - p.edges[0];
            const size_t b = std::min(p.edges.size() - 2, static_cast<size_t>((m.x - p.edges.front()) / width));
            begin(Value(p.edges[b]) + " to " + Value(p.edges[b + 1]));
            for (size_t i = 0; i < p.series.size(); ++i) {
                double total = 0;
                for (double c : p.series[i].y) total += c;
                char share[32];
                std::snprintf(share, sizeof(share), "  (%.1f%%)", total > 0 ? 100.0 * p.series[i].y[b] / total : 0.0);
                TooltipRow(SeriesColour(i), p.series[i].label,
                           Value(p.series[i].y[b]) + (p.spec.density ? std::string() : std::string(share)));
            }
            break;
        }
        case Kind::Bar:
        case Kind::ErrorBars: {
            const int i = category(p.categories, m.x);
            if (i < 0 || p.series.empty()) break;
            begin(p.categories[static_cast<size_t>(i)]);
            const auto& s = p.series[0];
            TooltipRow(SeriesColour(0), s.label, Value(s.y[static_cast<size_t>(i)]) +
                                                     (kind == Kind::ErrorBars ? " \xC2\xB1 " + Value(s.low[static_cast<size_t>(i)]) : ""));
            break;
        }
        case Kind::Pie: {
            if (p.series.empty()) break;
            const double dx = m.x - 0.5, dy = m.y - 0.5;
            if (dx * dx + dy * dy > 0.16) break;
            // ImPlot starts at 90 degrees and goes counter-clockwise.
            double angle = std::atan2(dy, dx) * 180.0 / 3.141592653589793 - 90.0;
            while (angle < 0) angle += 360.0;
            double total = 0;
            for (double v : p.series[0].y) total += std::max(0.0, v);
            double acc = 0;
            for (size_t k = 0; k < p.series[0].y.size() && total > 0; ++k) {
                acc += std::max(0.0, p.series[0].y[k]) / total * 360.0;
                if (angle <= acc) {
                    char share[32];
                    std::snprintf(share, sizeof(share), "  (%.1f%%)", 100.0 * p.series[0].y[k] / total);
                    begin(p.categories[k]);
                    TooltipRow(SeriesColour(k), p.series[0].label, Value(p.series[0].y[k]) + share);
                    break;
                }
            }
            break;
        }
        case Kind::Box:
        case Kind::Violin: {
            const int i = category(std::vector<std::string>(p.boxes.size()), m.x);
            if (i < 0 || i >= static_cast<int>(p.series.size())) break;
            const auto& b = p.boxes[static_cast<size_t>(i)];
            const ImVec4 c = SeriesColour(static_cast<size_t>(i));
            begin(p.series[static_cast<size_t>(i)].label);
            TooltipRow(c, "high whisker", Value(b.high));
            TooltipRow(c, "q3", Value(b.q3));
            TooltipRow(c, "median", Value(b.median));
            TooltipRow(c, "q1", Value(b.q1));
            TooltipRow(c, "low whisker", Value(b.low));
            TooltipRow(c, "mean", Value(b.mean));
            break;
        }
        case Kind::Heatmap: {
            const int c = static_cast<int>(std::floor(m.x));
            const int r = p.grid_rows - 1 - static_cast<int>(std::floor(m.y));
            if (c < 0 || r < 0 || c >= p.grid_cols || r >= p.grid_rows) break;
            begin(p.row_names[static_cast<size_t>(r)] + " \xC2\xB7 " + p.col_names[static_cast<size_t>(c)]);
            TooltipRow(SeriesColour(0), "rows", Value(p.grid[static_cast<size_t>(r * p.grid_cols + c)]));
            break;
        }
        case Kind::Histogram2D: {
            if (m.x < p.x_min || m.x > p.x_max || m.y < p.y_min || m.y > p.y_max || p.grid_cols == 0) break;
            const double dx = (p.x_max - p.x_min) / p.grid_cols, dy = (p.y_max - p.y_min) / p.grid_rows;
            const int c = std::min(p.grid_cols - 1, static_cast<int>((m.x - p.x_min) / dx));
            const int rb = std::min(p.grid_rows - 1, static_cast<int>((m.y - p.y_min) / dy));
            const int r = p.grid_rows - 1 - rb;
            begin(XLabel(p) + " " + Value(p.x_min + dx * c) + " to " + Value(p.x_min + dx * (c + 1)));
            TooltipRow(SeriesColour(0), YLabel(p) + " " + Value(p.y_min + dy * rb) + " to " + Value(p.y_min + dy * (rb + 1)),
                       Value(p.grid[static_cast<size_t>(r * p.grid_cols + c)]) + " rows");
            break;
        }
    }
    if (shown) ImGui::EndTooltip();
}

}  // namespace cyxwiz::plot
