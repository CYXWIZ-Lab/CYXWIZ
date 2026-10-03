#include "plot_view.h"

#include "plot_style.h"
#include "../../core/plot/plot_prepare.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"

#include <implot.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <fstream>

#include <spdlog/spdlog.h>

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

// Grid kinds lay out their own axes and draw a colour bar.
bool IsGridKind(Kind k) {
    return k == Kind::Heatmap || k == Kind::Histogram2D || k == Kind::Matrix || k == Kind::Hexbin || k == Kind::Contour ||
           k == Kind::FilledContour || k == Kind::Quiver || k == Kind::Stream;
}

constexpr double kTurn = 6.283185307179586;

// A compass direction (degrees clockwise from north) of an arrow.
double CompassOf(double u, double v) {
    double d = std::atan2(u, v) * 360.0 / kTurn;
    return d < 0 ? d + 360.0 : d;
}

// Polar angle text: the category, or degrees.
std::string PolarAngleText(const Prepared& p, double a) {
    if (!p.polar_names.empty()) {
        const size_t k = static_cast<size_t>(std::llround(a / kTurn * static_cast<double>(p.polar_names.size())));
        return k < p.polar_names.size() ? p.polar_names[k] : std::string();
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4g\xC2\xB0", a * 360.0 / kTurn);
    return buf;
}

std::string XLabel(const Prepared& p) {
    if (!p.spec.x_label.empty()) return p.spec.x_label;
    if (p.spec.kind == Kind::Kde) return p.spec.y_columns.size() == 1 ? p.spec.y_columns.front() : std::string("value");
    if (!p.spec.x_column.empty()) return p.spec.x_column;
    return IsLineLike(p.spec.kind) ? "row" : "";
}

std::string YLabel(const Prepared& p) {
    if (!p.spec.y_label.empty()) return p.spec.y_label;
    if (p.spec.kind == Kind::Histogram) return p.spec.density ? "density" : "count";
    if (p.spec.kind == Kind::Kde) return "density";
    if (p.spec.kind == Kind::Bar && p.spec.bar_layout == PlotSpec::BarLayout::Percent && p.series.size() > 1) return "percent";
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

// The legend lists the first kMaxLegendSeries series; the rest are drawn
// and shown on hover ("##" hides an ImPlot item from the legend).
std::string LegendLabel(const Prepared& p, size_t i, const std::string& label) {
    return p.series.size() > kMaxLegendSeries && i >= kMaxLegendSeries ? "##" + label : label;
}

// A grid value's colour on the plot's grid range (hexbin: log option).
ImVec4 GridColourOf(const Prepared& p, double v) {
    double lo = p.grid_lo, hi = p.grid_hi;
    if (p.spec.kind == Kind::Hexbin && p.spec.log_colour) {
        v = std::log1p(std::max(0.0, v));
        lo = std::log1p(std::max(0.0, lo));
        hi = std::log1p(std::max(0.0, hi));
    }
    float t = static_cast<float>(hi > lo ? std::clamp((v - lo) / (hi - lo), 0.0, 1.0) : 0.0);
    // Arrows and flow lines start a quarter up the scale, so the slow ones show on the plot.
    if (p.spec.kind == Kind::Quiver || p.spec.kind == Kind::Stream) t = 0.25f + 0.75f * t;
    return ImPlot::SampleColormap(t, p.grid_diverging ? DivergingColormap() : SequentialColormap());
}

// Colour of a value on the plot's colour scale (a missing value: faint).
ImVec4 ScaleColourOf(const Prepared& p, double v) {
    if (!std::isfinite(v)) return ui::CurrentTokens().text_faint;
    const double span = p.colour_max - p.colour_min;
    const float t = static_cast<float>(span > 0 ? std::clamp((v - p.colour_min) / span, 0.0, 1.0) : 0.0);
    return ImPlot::SampleColormap(t, p.colour_diverging ? DivergingColormap() : SequentialColormap());
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
    const bool log_changed = has_data_ && data.spec.log_y != data_.spec.log_y;
    // Other rows (Rows in the Data panel) or another colour: fit to them.
    const bool rows_changed = has_data_ && (data.label.selection != data_.label.selection ||
                                            data.rows_selected != data_.rows_selected ||
                                            data.spec.color_column != data_.spec.color_column);
    data_ = std::move(data);
    has_data_ = true;
    if (rows_changed) fit_ = true;
    // New values keep the user's Log Y and Legend; a new plot or a changed
    // log setting takes the spec's.
    if (kind_changed || log_changed) log_y_ = data_.spec.log_y;
    if (kind_changed) legend_ = data_.spec.legend;
    if (kind_changed) fit_ = true;
}

ImVec4 PlotView::ColourOf(size_t i) const {
    if (i < data_.series.size() && data_.series[i].colour >= 0) return SeriesColour(static_cast<size_t>(data_.series[i].colour));
    return SeriesColour(colour_offset_ + i);
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
    spdlog::info("Plot '{}': {} ({} bytes of PNG)", id_, note_, png.size());
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
            st.diverging_low = HexColour(t.scale_diverging[0].x, t.scale_diverging[0].y, t.scale_diverging[0].z);
            st.diverging_mid = HexColour(t.scale_diverging[1].x, t.scale_diverging[1].y, t.scale_diverging[1].z);
            st.diverging_high = HexColour(t.scale_diverging[2].x, t.scale_diverging[2].y, t.scale_diverging[2].z);
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
                case DataLabel::State::Exact:
                    ImGui::SetTooltip(data_.label.selection.empty() ? "All values are drawn."
                                                                    : "All values of the chosen rows are drawn (Rows in the Data panel).");
                    break;
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
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float w = (o.tool_fit ? ui::ButtonWidth("Fit", ui::ButtonSize::Small) + gap : 0.0f) +
                    (o.tool_log ? ui::ButtonWidth("Log Y", ui::ButtonSize::Small) + gap : 0.0f) +
                    (o.tool_legend ? ui::ButtonWidth("Legend", ui::ButtonSize::Small) + gap : 0.0f) +
                    ui::ButtonWidth("Export", ui::ButtonSize::Small) +
                    (o.own_window_button ? ui::ButtonWidth(own, ui::ButtonSize::Small) + gap : 0.0f);
    const float right = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - w;
    if (right > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(right);
    const bool usable = has_data_ && data_.problem.empty();
    if (o.tool_fit) {
        if (ui::GhostButton(("Fit##" + id_).c_str(), usable)) fit_ = true;
        ImGui::SameLine();
    }
    const bool can_log = usable && data_.spec.kind != Kind::Pie && data_.spec.kind != Kind::Polar && !IsGridKind(data_.spec.kind);
    if (o.tool_log) {
        if (ui::GhostButton(("Log Y##" + id_).c_str(), can_log, "Not for this plot type", log_y_)) {
            log_y_ = !log_y_;
            fit_ = true;
        }
        ImGui::SameLine();
    }
    if (o.tool_legend) {
        if (ui::GhostButton(("Legend##" + id_).c_str(), usable, nullptr, legend_)) legend_ = !legend_;
        ImGui::SameLine();
    }
    if (ui::GhostButton(("Export##" + id_).c_str(), usable)) ImGui::OpenPopup(("##export" + id_).c_str());
    if (ImGui::BeginPopup(("##export" + id_).c_str())) {
        ExportMenu(o);
        ImGui::EndPopup();
    }
    if (o.own_window_button) {
        ImGui::SameLine();
        if (ui::GhostButton((std::string(own) + "##own" + id_).c_str(), true, nullptr, own_window)) own_window = !own_window;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(own_window ? "Back into this panel" : "Open in its own window");
    }
}

void PlotView::Draw(ImVec2 size, const Options& o) {
    FinishCaptures();
    export_name_ = o.export_name;
    colour_offset_ = o.colour_offset;
    x_range_ = o.x_range;
    y_range_ = o.y_range;
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
            spdlog::info("Plot '{}': reading the image ({:.0f} x {:.0f})", id_, frame_max_.x - frame_min_.x,
                         frame_max_.y - frame_min_.y);
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
    const bool scale_bar = IsGridKind(kind) || p.colour_scale;
    ImVec2 plot_size = size;
    // A colour bar with a column name needs room for its ticks and the name.
    const float bar_w = p.colour_scale ? 60.0f + ImGui::GetTextLineHeight() + 8.0f : 60.0f;
    if (scale_bar) plot_size.x = std::max(100.0f, size.x - bar_w - 16.0f);

    // No mouse-position text: the hover card shows the values.
    ImPlotFlags flags = ImPlotFlags_NoTitle | ImPlotFlags_NoMenus | ImPlotFlags_NoMouseText;
    if (!legend_) flags |= ImPlotFlags_NoLegend;
    if (kind == Kind::Pie) flags |= ImPlotFlags_Equal | ImPlotFlags_NoMouseText;

    // Fixed-layout kinds set their limits; the rest fit to the data.
    const ImPlotCond cond = fit_ ? ImPlotCond_Always : ImPlotCond_Once;
    if (fit_ && !x_range_.on && !y_range_.on && kind != Kind::Pie && kind != Kind::Polar && kind != Kind::Box && kind != Kind::Violin &&
        !IsGridKind(kind))
        ImPlot::SetNextAxesToFit();

    const std::string plot_id = "##plot";
    if (!ImPlot::BeginPlot(plot_id.c_str(), plot_size, flags)) return;

    const std::string xl = XLabel(p), yl = YLabel(p);
    if (kind == Kind::Pie) {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxesLimits(0, 1, 0, 1, ImPlotCond_Always);
    } else if (kind == Kind::Polar) {
        // Its own rings and spokes; the radius 1 is the largest value. The
        // limits follow the plot's shape so -1.25..1.25 fits both ways
        // (the names sit at 1.12).
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        const double aspect = plot_size.y > 0 ? static_cast<double>(plot_size.x) / plot_size.y : 1.0;
        const double hx = aspect >= 1.0 ? 1.25 * aspect : 1.25, hy = aspect >= 1.0 ? 1.25 : 1.25 / aspect;
        ImPlot::SetupAxesLimits(-hx, hx, -hy, hy, ImPlotCond_Always);
    } else {
        ImPlotAxisFlags xf = ImPlotAxisFlags_None, yf = ImPlotAxisFlags_None;
        if (kind == Kind::Heatmap || kind == Kind::Matrix) xf = yf = ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
        ImPlot::SetupAxes(xl.empty() ? nullptr : xl.c_str(), yl.empty() ? nullptr : yl.c_str(), xf, yf);
        if (log_y_ && !IsGridKind(kind)) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
    }
    ImPlot::SetupLegend(ImPlotLocation_NorthEast);
    if (x_range_.on) ImPlot::SetupAxisLimits(ImAxis_X1, x_range_.lo, x_range_.hi, x_range_.once ? ImPlotCond_Once : ImPlotCond_Always);
    if (y_range_.on) ImPlot::SetupAxisLimits(ImAxis_Y1, y_range_.lo, y_range_.hi, y_range_.once ? ImPlotCond_Once : ImPlotCond_Always);
    // Log Y over counts: fit to the positive counts (the bars' base of 0
    // would pull a log axis down to 1e-300 and fill the plot).
    if (log_y_ && fit_ && !y_range_.on && (kind == Kind::Histogram || kind == Kind::Bar)) {
        double lo = 0, hi = 0;
        for (const auto& s : p.series)
            for (double v : s.y)
                if (v > 0) {
                    lo = lo == 0 ? v : std::min(lo, v);
                    hi = std::max(hi, v);
                }
        if (hi > 0) ImPlot::SetupAxisLimits(ImAxis_Y1, lo * 0.5, hi * 2.0, ImPlotCond_Always);
    }

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
    std::vector<std::string> short_names;

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
        case Kind::Heatmap:
        case Kind::Matrix: {
            // Column names longer than their cell are shortened ("artist_po..");
            // the rows and the hover keep the full names.
            float row_w = 0.0f;
            for (const auto& r : p.row_names) row_w = std::max(row_w, ImGui::CalcTextSize(r.c_str()).x);
            const float cell_w = p.grid_cols > 0 ? (plot_size.x - row_w - 40.0f) / static_cast<float>(p.grid_cols) - 6.0f : 0.0f;
            for (size_t i = 0; i < p.col_names.size() && p.col_names.size() <= 40; ++i) {
                std::string name = p.col_names[i];
                if (ImGui::CalcTextSize(name.c_str()).x > cell_w) {
                    while (name.size() > 1 && ImGui::CalcTextSize((name + "..").c_str()).x > cell_w) name.pop_back();
                    name += "..";
                }
                short_names.push_back(std::move(name));
                xpos.push_back(static_cast<double>(i) + 0.5);
            }
            for (const auto& name : short_names) xnames.push_back(name.c_str());
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
        case Kind::Histogram2D:
        case Kind::Hexbin:
        case Kind::Contour:
        case Kind::FilledContour: ImPlot::SetupAxesLimits(p.x_min, p.x_max, p.y_min, p.y_max, cond); break;
        case Kind::Quiver:
        case Kind::Stream: {
            // A margin so the arrows at the edge show.
            const double mx = (p.x_max - p.x_min) * 0.04, my = (p.y_max - p.y_min) * 0.04;
            ImPlot::SetupAxesLimits(p.x_min - mx, p.x_max + mx, p.y_min - my, p.y_max + my, cond);
            break;
        }
        default: break;
    }
    fit_ = false;

    const auto n = [](const std::vector<double>& v) { return static_cast<int>(v.size()); };
    switch (kind) {
        case Kind::Line:
        case Kind::Area:
        case Kind::Step:
        case Kind::Stem:
        case Kind::Kde:
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                const ImVec4 c = ColourOf((i));
                const bool smoothed = !s.smooth_y.empty();
                const std::string legend_label = LegendLabel(p, i, s.label);
                const char* label = legend_label.c_str();
                if (kind == Kind::Area || kind == Kind::Kde) {
                    ImPlot::SetNextFillStyle(c, kind == Kind::Kde ? 0.18f : 0.25f);
                    ImPlot::PlotShaded(label, s.x.data(), s.y.data(), n(s.x), 0.0);
                }
                ImPlot::SetNextLineStyle(smoothed ? ui::WithAlpha(c, 0.55f) : c, smoothed ? 1.2f : 1.8f);
                if (kind == Kind::Step) ImPlot::PlotStairs(label, s.x.data(), s.y.data(), n(s.x));
                else if (kind == Kind::Stem) {
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f, c, 0.0f);
                    ImPlot::PlotStems(label, s.x.data(), s.y.data(), n(s.x));
                } else {
                    if (s.markers) ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f, c, 0.0f);
                    ImPlot::PlotLine(label, s.x.data(), s.y.data(), n(s.x));
                }
                if (smoothed) {
                    const std::string sl = LegendLabel(p, i, s.label + ", smoothed (" + std::to_string(p.spec.smooth) + ")");
                    ImPlot::SetNextLineStyle(ui::Mix(c, t.text_bright, 0.35f), 2.6f);
                    ImPlot::PlotLine(sl.c_str(), s.smooth_x.data(), s.smooth_y.data(), n(s.smooth_x));
                }
            }
            break;
        case Kind::Scatter:
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                if (p.colour_scale) {
                    // Invisible markers keep the axes fitting the points; each
                    // point is then drawn in its colour on the scale.
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f, ui::WithAlpha(t.text, 0.0f), 0.0f);
                    ImPlot::PlotScatter(("##" + s.label).c_str(), s.x.data(), s.y.data(), n(s.x));
                    ImDrawList* dl = ImPlot::GetPlotDrawList();
                    ImPlot::PushPlotClipRect();
                    for (size_t k = 0; k < s.x.size(); ++k) {
                        const ImVec4 c = ScaleColourOf(p, k < s.c.size() ? s.c[k] : NAN);
                        dl->AddCircleFilled(ImPlot::PlotToPixels(s.x[k], s.y[k]), 2.5f, ui::ToU32(ui::WithAlpha(c, 0.8f)), 8);
                    }
                    ImPlot::PopPlotClipRect();
                    continue;
                }
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f, ui::WithAlpha(ColourOf((i)), 0.7f), 0.0f);
                ImPlot::PlotScatter(LegendLabel(p, i, s.label).c_str(), s.x.data(), s.y.data(), n(s.x));
            }
            break;
        case Kind::Histogram: {
            const double width = p.edges.size() > 1 ? p.edges[1] - p.edges[0] : 1.0;
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                ImPlot::SetNextFillStyle(ColourOf((i)), p.series.size() > 1 ? 0.6f : 0.9f);
                ImPlot::PlotBars(s.label.c_str(), s.x.data(), s.y.data(), n(s.x), width * 0.94);
            }
            if (p.spec.show_median) {
                const std::string l = "median " + Value(p.stats.median);
                ImPlot::SetNextLineStyle(ColourOf((2)), 2.0f);
                ImPlot::PlotInfLines(l.c_str(), &p.stats.median, 1);
            }
            if (p.spec.show_mean) {
                const std::string l = "mean " + Value(p.stats.mean);
                ImPlot::SetNextLineStyle(ColourOf((1)), 2.0f);
                ImPlot::PlotInfLines(l.c_str(), &p.stats.mean, 1);
            }
            break;
        }
        case Kind::Bar:
            if (p.series.size() > 1) {
                std::vector<const char*> labels;
                std::vector<double> values;
                for (const auto& s : p.series) {
                    labels.push_back(s.label.c_str());
                    values.insert(values.end(), s.y.begin(), s.y.end());
                }
                ImPlot::PushColormap(SeriesColormap());
                ImPlot::PlotBarGroups(labels.data(), values.data(), static_cast<int>(p.series.size()),
                                      static_cast<int>(p.categories.size()), 0.67, 0.0,
                                      p.spec.bar_layout == PlotSpec::BarLayout::Grouped ? 0 : ImPlotBarGroupsFlags_Stacked);
                ImPlot::PopColormap();
            } else if (!p.series.empty()) {
                ImPlot::SetNextFillStyle(ColourOf((0)), 0.9f);
                ImPlot::PlotBars(p.series[0].label.c_str(), p.series[0].x.data(), p.series[0].y.data(), n(p.series[0].x), 0.67);
            }
            break;
        case Kind::ErrorBars:
            if (!p.series.empty()) {
                const auto& s = p.series[0];
                ImPlot::SetNextErrorBarStyle(ColourOf((0)), 1.5f, 6.0f);
                ImPlot::PlotErrorBars(s.label.c_str(), s.x.data(), s.y.data(), s.low.data(), s.high.data(), n(s.x));
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f, ColourOf((0)), 0.0f);
                ImPlot::PlotScatter(s.label.c_str(), s.x.data(), s.y.data(), n(s.x));
            }
            break;
        case Kind::Pie:
            if (!p.series.empty() && !p.categories.empty()) {
                std::vector<const char*> labels;
                for (const auto& c : p.categories) labels.push_back(c.c_str());
                ImPlot::PlotPieChart(labels.data(), p.series[0].y.data(), static_cast<int>(labels.size()), 0.5, 0.5, 0.4,
                                     p.spec.donut ? nullptr : "%.0f", 90);
                if (p.spec.donut) {
                    double total = 0;
                    for (double v : p.series[0].y) total += std::max(0.0, v);
                    const ImVec2 c = ImPlot::PlotToPixels(0.5, 0.5), e = ImPlot::PlotToPixels(0.5 + 0.4 * 0.58, 0.5);
                    ImDrawList* dl = ImPlot::GetPlotDrawList();
                    ImPlot::PushPlotClipRect();
                    dl->AddCircleFilled(c, e.x - c.x, ui::ToU32(t.plot_bg), 64);
                    const std::string text = Thousands(static_cast<long long>(std::llround(total)));
                    ImFont* font = ImGui::GetFont();
                    const float big = ImGui::GetFontSize() * 1.6f;
                    const ImVec2 ts = font->CalcTextSizeA(big, FLT_MAX, 0.0f, text.c_str());
                    dl->AddText(font, big, ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.6f), ui::ToU32(t.text_bright), text.c_str());
                    const ImVec2 ls = ImGui::CalcTextSize("total");
                    dl->AddText(ImVec2(c.x - ls.x * 0.5f, c.y + ts.y * 0.45f), ui::ToU32(t.text_dim), "total");
                    ImPlot::PopPlotClipRect();
                }
            }
            break;
        case Kind::Box:
        case Kind::Violin: {
            ImDrawList* dl = ImPlot::GetPlotDrawList();
            ImPlot::PushPlotClipRect();
            for (size_t i = 0; i < p.boxes.size(); ++i) {
                const auto& b = p.boxes[i];
                const double x = static_cast<double>(i);
                const ImVec4 c = ColourOf((i));
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
        case Kind::Matrix:
        case Kind::Histogram2D: {
            const bool named = kind != Kind::Histogram2D;
            ImPlot::PushColormap(p.grid_diverging ? DivergingColormap() : SequentialColormap());
            const bool numbers = named && p.grid.size() <= 400;
            const char* format = kind == Kind::Matrix && p.spec.matrix_values != PlotSpec::MatrixValues::Values ? "%.2f" : "%g";
            const ImPlotPoint lo = named ? ImPlotPoint(0, 0) : ImPlotPoint(p.x_min, p.y_min);
            const ImPlotPoint hi = named ? ImPlotPoint(p.grid_cols, p.grid_rows) : ImPlotPoint(p.x_max, p.y_max);
            ImPlot::PlotHeatmap("##grid", p.grid.data(), p.grid_rows, p.grid_cols, p.grid_lo,
                                p.grid_hi > p.grid_lo ? p.grid_hi : p.grid_lo + 1.0, numbers ? format : nullptr, lo, hi);
            ImPlot::PopColormap();
            break;
        }
        case Kind::Polar: {
            // Rings at round radii, spokes, then each series (0 at the top, clockwise).
            ImDrawList* dl = ImPlot::GetPlotDrawList();
            const double rmax = p.polar_rmax > 0 ? p.polar_rmax : 1.0;
            const auto at = [&](double a, double r) { return ImPlot::PlotToPixels(r / rmax * std::sin(a), r / rmax * std::cos(a)); };
            const ImVec2 c0 = ImPlot::PlotToPixels(0.0, 0.0);
            const float unit = ImPlot::PlotToPixels(1.0, 0.0).x - c0.x;
            ImPlot::PushPlotClipRect();
            const double step = [&] {
                const double raw = rmax / 3.0, mag = std::pow(10.0, std::floor(std::log10(raw)));
                const double nrm = raw / mag;
                return (nrm < 1.5 ? 1.0 : nrm < 3.0 ? 2.0 : nrm < 7.0 ? 5.0 : 10.0) * mag;
            }();
            const size_t spokes = !p.polar_names.empty() ? std::min<size_t>(p.polar_names.size(), 36) : 12;
            // Ring values sit between the first two spokes (clear of the names).
            const double label_angle = 0.5 / static_cast<double>(spokes) * kTurn;
            for (double r = step; r <= rmax * 1.0001; r += step) {
                dl->AddCircle(c0, static_cast<float>(r / rmax) * unit, ui::ToU32(t.plot_grid), 96, 1.0f);
                const std::string lbl = Value(r);
                const ImVec2 q = at(label_angle, r);
                dl->AddText(ImVec2(q.x + 2.0f, q.y - ImGui::GetTextLineHeight()), ui::ToU32(t.text_dim), lbl.c_str());
            }
            for (size_t k = 0; k < spokes; ++k) {
                const double a = static_cast<double>(k) / static_cast<double>(spokes) * kTurn;
                dl->AddLine(c0, at(a, rmax), ui::ToU32(t.plot_grid), 1.0f);
                const std::string name = !p.polar_names.empty() ? p.polar_names[k] : Value(static_cast<double>(k) * 30.0);
                const ImVec2 q = at(a, rmax * 1.12), ts = ImGui::CalcTextSize(name.c_str());
                dl->AddText(ImVec2(q.x - ts.x * 0.5f, q.y - ts.y * 0.5f), ui::ToU32(t.text_dim), name.c_str());
            }
            ImPlot::PopPlotClipRect();
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                std::vector<double> xs, ys;
                for (size_t k = 0; k < s.x.size(); ++k) {
                    xs.push_back(s.y[k] / rmax * std::sin(s.x[k]));
                    ys.push_back(s.y[k] / rmax * std::cos(s.x[k]));
                }
                if (p.polar_closed && !p.spec.polar_points && !xs.empty()) {
                    xs.push_back(xs.front());
                    ys.push_back(ys.front());
                }
                const ImVec4 c = ColourOf(i);
                if (p.spec.polar_points) {
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.8f, ui::WithAlpha(c, 0.85f), 0.0f);
                    ImPlot::PlotScatter(LegendLabel(p, i, s.label).c_str(), xs.data(), ys.data(), n(xs));
                } else {
                    ImPlot::SetNextLineStyle(c, 1.8f);
                    ImPlot::PlotLine(LegendLabel(p, i, s.label).c_str(), xs.data(), ys.data(), n(xs));
                }
            }
            break;
        }
        case Kind::Quiver: {
            ImDrawList* dl = ImPlot::GetPlotDrawList();
            ImPlot::PushPlotClipRect();
            for (size_t i : p.q_drawn) {
                const ImVec2 a = ImPlot::PlotToPixels(p.qx[i], p.qy[i]);
                const ImVec2 b = ImPlot::PlotToPixels(p.qx[i] + p.qu[i] * p.q_scale, p.qy[i] + p.qv[i] * p.q_scale);
                const float dx = b.x - a.x, dy = b.y - a.y, len = std::sqrt(dx * dx + dy * dy);
                if (len < 0.5f) continue;
                const float ux = dx / len, uy = dy / len, hs = std::min(6.0f, len * 0.45f);
                const ImU32 c = ui::ToU32(GridColourOf(p, std::hypot(p.qu[i], p.qv[i])));
                dl->AddLine(a, b, c, 1.4f);
                dl->AddTriangleFilled(b, ImVec2(b.x - ux * hs - uy * hs * 0.6f, b.y - uy * hs + ux * hs * 0.6f),
                                      ImVec2(b.x - ux * hs + uy * hs * 0.6f, b.y - uy * hs - ux * hs * 0.6f), c);
            }
            ImPlot::PopPlotClipRect();
            break;
        }
        case Kind::Stream: {
            ImDrawList* dl = ImPlot::GetPlotDrawList();
            ImPlot::PushPlotClipRect();
            std::vector<ImVec2> pts;
            for (size_t l = 0; l < p.stream_lines.size(); ++l) {
                const auto& ln = p.stream_lines[l];
                pts.clear();
                for (size_t k = 0; k + 1 < ln.size(); k += 2) pts.push_back(ImPlot::PlotToPixels(ln[k], ln[k + 1]));
                if (pts.size() < 2) continue;
                const ImU32 c = ui::ToU32(GridColourOf(p, l < p.stream_speed.size() ? p.stream_speed[l] : 0.0));
                dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), c, ImDrawFlags_None, 1.3f);
                // An arrowhead at the middle, pointing along the line.
                const size_t m = pts.size() / 2;
                const ImVec2 a = pts[m - 1], b = pts[m];
                const float dx = b.x - a.x, dy = b.y - a.y, len = std::sqrt(dx * dx + dy * dy);
                if (len > 0.01f) {
                    const float ux = dx / len, uy = dy / len, hs = 5.0f;
                    dl->AddTriangleFilled(b, ImVec2(b.x - ux * hs - uy * hs * 0.6f, b.y - uy * hs + ux * hs * 0.6f),
                                          ImVec2(b.x - ux * hs + uy * hs * 0.6f, b.y - uy * hs - ux * hs * 0.6f), c);
                }
            }
            ImPlot::PopPlotClipRect();
            break;
        }
        case Kind::Hexbin: {
            ImDrawList* dl = ImPlot::GetPlotDrawList();
            ImPlot::PushPlotClipRect();
            static const double vx[6] = {0.5, 0.5, 0.0, -0.5, -0.5, 0.0};
            static const double vy[6] = {-1.0 / 6, 1.0 / 6, 1.0 / 3, 1.0 / 6, -1.0 / 6, -1.0 / 3};
            for (size_t i = 0; i < p.hex_x.size(); ++i) {
                ImVec2 pts[6];
                for (int k = 0; k < 6; ++k)
                    pts[k] = ImPlot::PlotToPixels(p.hex_x[i] + vx[k] * p.hex_sx, p.hex_y[i] + vy[k] * p.hex_sy);
                dl->AddConvexPolyFilled(pts, 6, ui::ToU32(GridColourOf(p, p.hex_v[i])));
            }
            ImPlot::PopPlotClipRect();
            break;
        }
        case Kind::Contour:
        case Kind::FilledContour: {
            if (kind == Kind::FilledContour && p.band_rows > 0) {
                ImPlot::PushColormap(SequentialColormap());
                ImPlot::PlotHeatmap("##bands", p.band_grid.data(), p.band_rows, p.band_cols, p.grid_lo, p.grid_hi, nullptr,
                                    ImPlotPoint(p.x_min, p.y_min), ImPlotPoint(p.x_max, p.y_max));
                ImPlot::PopColormap();
            }
            ImDrawList* dl = ImPlot::GetPlotDrawList();
            ImPlot::PushPlotClipRect();
            for (size_t l = 0; l < p.contour_segments.size(); ++l) {
                const ImU32 c = kind == Kind::FilledContour ? ui::ToU32(ui::WithAlpha(t.plot_bg, 0.8f))
                                                            : ui::ToU32(GridColourOf(p, p.contour_levels[l]));
                const auto& seg = p.contour_segments[l];
                for (size_t k = 0; k + 3 < seg.size(); k += 4)
                    dl->AddLine(ImPlot::PlotToPixels(seg[k], seg[k + 1]), ImPlot::PlotToPixels(seg[k + 2], seg[k + 3]), c,
                                kind == Kind::FilledContour ? 1.0f : 1.8f);
            }
            ImPlot::PopPlotClipRect();
            break;
        }
    }

    // A y = x reference line (ROC chance) across the data's extent.
    if (p.spec.show_diagonal && (kind == Kind::Line || kind == Kind::Scatter)) {
        double lo = 0, hi = 0;
        bool first = true;
        for (const auto& s : p.series)
            for (size_t k = 0; k < std::min(s.x.size(), s.y.size()); ++k) {
                const double a = std::min(s.x[k], s.y[k]), b = std::max(s.x[k], s.y[k]);
                lo = first ? a : std::min(lo, a);
                hi = first ? b : std::max(hi, b);
                first = false;
            }
        if (!first && hi > lo) {
            const double xs[2] = {lo, hi}, ys[2] = {lo, hi};
            ImPlot::SetNextLineStyle(ui::WithAlpha(t.text_dim, 0.8f), 1.2f);
            ImPlot::PlotLine("y = x", xs, ys, 2);
        }
    }

    if (ImPlot::IsPlotHovered() && pending_->action == Pending::Action::None) DrawHover();
    const ImPlotRect lim = ImPlot::GetPlotLimits();
    range_ = AxisRange{lim.X.Min, lim.X.Max, lim.Y.Min, lim.Y.Max, log_y_};
    ImPlot::EndPlot();
    frame_min_ = ImGui::GetItemRectMin();
    frame_max_ = ImGui::GetItemRectMax();

    if (scale_bar && p.colour_scale) {
        // The colour bar of a scatter coloured by a number column.
        ImGui::SameLine();
        ImPlot::ColormapScale(p.colour_label.c_str(), p.colour_min, p.colour_max, ImVec2(bar_w, plot_size.y), "%g", 0,
                              p.colour_diverging ? DivergingColormap() : SequentialColormap());
    } else if (scale_bar) {
        double lo = p.grid_lo, hi = p.grid_hi > p.grid_lo ? p.grid_hi : p.grid_lo + 1.0;
        const bool log = kind == Kind::Hexbin && p.spec.log_colour;
        if (log) {
            lo = std::log1p(std::max(0.0, lo));
            hi = std::log1p(std::max(0.0, hi));
        }
        ImGui::SameLine();
        const char* scale_label = log ? "log(1 + count)##scale" : kind == Kind::Quiver ? "length##scale" : kind == Kind::Stream ? "speed##scale" : "##scale";
        ImPlot::ColormapScale(scale_label, lo, hi, ImVec2(60, plot_size.y), "%g", 0,
                              p.grid_diverging ? DivergingColormap() : SequentialColormap());
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
        case Kind::Stem:
        case Kind::Kde: {
            // The value of each series at the x nearest the mouse (all points).
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                const auto& xs = s.all_x.empty() ? s.x : s.all_x;
                const auto& ys = s.all_y.empty() ? s.y : s.all_y;
                if (xs.empty()) continue;
                const size_t k = s.x_sorted ? NearestSorted(xs, m.x) : NearestScan(xs, m.x);
                begin(XLabel(p) + " " + Value(xs[k]));
                TooltipRow(ColourOf((i)), s.label, Value(ys[k]));
                if (!s.smooth_y.empty()) {
                    const size_t j = NearestScan(s.smooth_x, xs[k]);
                    TooltipRow(ui::Mix(ColourOf((i)), t.text_bright, 0.35f),
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
                TooltipRow(ColourOf((bs)), XLabel(p), Value(s.x[bk]));
                TooltipRow(ColourOf((bs)), p.spec.y_columns.empty() ? "y" : p.spec.y_columns.front(), Value(s.y[bk]));
                if (p.colour_scale && bk < s.c.size())
                    TooltipRow(ScaleColourOf(p, s.c[bk]), p.colour_label,
                               std::isfinite(s.c[bk]) ? Value(s.c[bk]) : std::string("missing"));
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
                TooltipRow(ColourOf((i)), p.series[i].label,
                           Value(p.series[i].y[b]) + (p.spec.density ? std::string() : std::string(share)));
            }
            break;
        }
        case Kind::Bar:
        case Kind::ErrorBars: {
            const int i = category(p.categories, m.x);
            if (i < 0 || p.series.empty()) break;
            begin(p.categories[static_cast<size_t>(i)]);
            if (p.series.size() > 1) {  // Colour by: every group in this category
                const bool percent = p.spec.bar_layout == PlotSpec::BarLayout::Percent;
                for (size_t g = 0; g < p.series.size(); ++g)
                    TooltipRow(SeriesColour(g), p.series[g].label,
                               Value(p.series[g].y[static_cast<size_t>(i)]) + (percent ? "%" : ""));
                break;
            }
            const auto& s = p.series[0];
            TooltipRow(ColourOf((0)), s.label, Value(s.y[static_cast<size_t>(i)]) +
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
                    TooltipRow(ColourOf((k)), p.series[0].label, Value(p.series[0].y[k]) + share);
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
            const ImVec4 c = ColourOf((static_cast<size_t>(i)));
            begin(p.series[static_cast<size_t>(i)].label);
            TooltipRow(c, "high whisker", Value(b.high));
            TooltipRow(c, "q3", Value(b.q3));
            TooltipRow(c, "median", Value(b.median));
            TooltipRow(c, "q1", Value(b.q1));
            TooltipRow(c, "low whisker", Value(b.low));
            TooltipRow(c, "mean", Value(b.mean));
            break;
        }
        case Kind::Heatmap:
        case Kind::Matrix: {
            const int c = static_cast<int>(std::floor(m.x));
            const int r = p.grid_rows - 1 - static_cast<int>(std::floor(m.y));
            if (c < 0 || r < 0 || c >= p.grid_cols || r >= p.grid_rows) break;
            // "Actual 0 · Predicted 1": each value with its axis name.
            const std::string yl = YLabel(p), xl = XLabel(p);
            begin((yl.empty() ? std::string() : yl + " ") + p.row_names[static_cast<size_t>(r)] + " \xC2\xB7 " +
                  (xl.empty() ? std::string() : xl + " ") + p.col_names[static_cast<size_t>(c)]);
            const double v = p.grid[static_cast<size_t>(r * p.grid_cols + c)];
            const std::string what = kind == Kind::Matrix
                                         ? (p.spec.matrix_values == PlotSpec::MatrixValues::Values ? "value" : "correlation")
                                         : (p.spec.value_column.empty() ? std::string("rows") : p.spec.value_column);
            TooltipRow(GridColourOf(p, v), what, std::isfinite(v) ? Value(v) : std::string("missing"));
            break;
        }
        case Kind::Histogram2D: {
            if (m.x < p.x_min || m.x > p.x_max || m.y < p.y_min || m.y > p.y_max || p.grid_cols == 0) break;
            const double dx = (p.x_max - p.x_min) / p.grid_cols, dy = (p.y_max - p.y_min) / p.grid_rows;
            const int c = std::min(p.grid_cols - 1, static_cast<int>((m.x - p.x_min) / dx));
            const int rb = std::min(p.grid_rows - 1, static_cast<int>((m.y - p.y_min) / dy));
            const int r = p.grid_rows - 1 - rb;
            begin(XLabel(p) + " " + Value(p.x_min + dx * c) + " to " + Value(p.x_min + dx * (c + 1)));
            TooltipRow(ColourOf((0)), YLabel(p) + " " + Value(p.y_min + dy * rb) + " to " + Value(p.y_min + dy * (rb + 1)),
                       Value(p.grid[static_cast<size_t>(r * p.grid_cols + c)]) + " rows");
            break;
        }
        case Kind::Polar: {
            // The nearest point (12 px).
            const ImVec2 mouse = ImGui::GetMousePos();
            const double rmax = p.polar_rmax > 0 ? p.polar_rmax : 1.0;
            float best = 144.0f;
            size_t bs = 0, bk = 0;
            bool found = false;
            for (size_t i = 0; i < p.series.size(); ++i)
                for (size_t k = 0; k < p.series[i].x.size(); ++k) {
                    const double a = p.series[i].x[k], r = p.series[i].y[k] / rmax;
                    const ImVec2 q = ImPlot::PlotToPixels(r * std::sin(a), r * std::cos(a));
                    const float d = (q.x - mouse.x) * (q.x - mouse.x) + (q.y - mouse.y) * (q.y - mouse.y);
                    if (d < best) {
                        best = d;
                        bs = i;
                        bk = k;
                        found = true;
                    }
                }
            if (!found) break;
            begin(PolarAngleText(p, p.series[bs].x[bk]));
            TooltipRow(ColourOf(bs), p.series[bs].label, Value(p.series[bs].y[bk]));
            break;
        }
        case Kind::Quiver: {
            // The nearest arrow's start (all rows, 12 px).
            const ImVec2 mouse = ImGui::GetMousePos();
            float best = 144.0f;
            size_t bi = 0;
            bool found = false;
            for (size_t i = 0; i < p.qx.size(); ++i) {
                const ImVec2 q = ImPlot::PlotToPixels(p.qx[i], p.qy[i]);
                const float d = (q.x - mouse.x) * (q.x - mouse.x) + (q.y - mouse.y) * (q.y - mouse.y);
                if (d < best) {
                    best = d;
                    bi = i;
                    found = true;
                }
            }
            if (!found) break;
            const double len = std::hypot(p.qu[bi], p.qv[bi]);
            begin(XLabel(p) + " " + Value(p.qx[bi]) + " \xC2\xB7 " + YLabel(p) + " " + Value(p.qy[bi]));
            TooltipRow(GridColourOf(p, len), "length", Value(len));
            TooltipRow(GridColourOf(p, len), "towards", Value(CompassOf(p.qu[bi], p.qv[bi])) + "\xC2\xB0 from north");
            TooltipRow(GridColourOf(p, len), "u, v", Value(p.qu[bi]) + ", " + Value(p.qv[bi]));
            break;
        }
        case Kind::Stream: {
            // The field at the mouse (bilinear between the grid points).
            if (m.x < p.x_min || m.x > p.x_max || m.y < p.y_min || m.y > p.y_max || p.grid_cols < 2 || p.grid_rows < 2) break;
            const double gx = (m.x - p.x_min) / (p.x_max - p.x_min) * (p.grid_cols - 1);
            const double gy = (p.y_max - m.y) / (p.y_max - p.y_min) * (p.grid_rows - 1);  // row 0 at the top
            const int c0 = std::min(static_cast<int>(gx), p.grid_cols - 2), r0 = std::min(static_cast<int>(gy), p.grid_rows - 2);
            const double fx = gx - c0, fy = gy - r0;
            const auto lerp2 = [&](const std::vector<double>& F) {
                const auto at = [&](int r, int c) { return F[static_cast<size_t>(r * p.grid_cols + c)]; };
                return (1 - fy) * ((1 - fx) * at(r0, c0) + fx * at(r0, c0 + 1)) + fy * ((1 - fx) * at(r0 + 1, c0) + fx * at(r0 + 1, c0 + 1));
            };
            const double u = lerp2(p.field_u), v = lerp2(p.field_v), sp = std::hypot(u, v);
            begin(XLabel(p) + " " + Value(m.x) + " \xC2\xB7 " + YLabel(p) + " " + Value(m.y));
            TooltipRow(GridColourOf(p, sp), "speed", Value(sp));
            TooltipRow(GridColourOf(p, sp), "towards", Value(CompassOf(u, v)) + "\xC2\xB0 from north");
            break;
        }
        case Kind::Hexbin: {
            // The hexagon whose centre is nearest in lattice units.
            size_t best = 0;
            double best_d = 1e300;
            for (size_t i = 0; i < p.hex_x.size(); ++i) {
                const double ux = (m.x - p.hex_x[i]) / p.hex_sx, uy = (m.y - p.hex_y[i]) / p.hex_sy;
                const double dd = ux * ux + 3.0 * uy * uy;
                if (dd < best_d) {
                    best_d = dd;
                    best = i;
                }
            }
            if (p.hex_x.empty() || best_d > 0.34) break;
            double total = 0;
            for (double v : p.hex_v) total += v;
            begin(XLabel(p) + " " + Value(p.hex_x[best]) + " \xC2\xB7 " + YLabel(p) + " " + Value(p.hex_y[best]));
            if (p.spec.value_column.empty()) {
                char share[32];
                std::snprintf(share, sizeof(share), "  (%.1f%%)", total > 0 ? 100.0 * p.hex_v[best] / total : 0.0);
                TooltipRow(GridColourOf(p, p.hex_v[best]), "rows", Value(p.hex_v[best]) + share);
            } else {
                TooltipRow(GridColourOf(p, p.hex_v[best]), "mean of " + p.spec.value_column, Value(p.hex_v[best]));
            }
            break;
        }
        case Kind::Contour:
        case Kind::FilledContour: {
            if (m.x < p.x_min || m.x > p.x_max || m.y < p.y_min || m.y > p.y_max || p.grid_cols == 0) break;
            const double dx = (p.x_max - p.x_min) / p.grid_cols, dy = (p.y_max - p.y_min) / p.grid_rows;
            const int c = std::min(p.grid_cols - 1, static_cast<int>((m.x - p.x_min) / dx));
            const int r = p.grid_rows - 1 - std::min(p.grid_rows - 1, static_cast<int>((m.y - p.y_min) / dy));
            const double v = p.grid[static_cast<size_t>(r * p.grid_cols + c)];
            begin(XLabel(p) + " " + Value(m.x) + " \xC2\xB7 " + YLabel(p) + " " + Value(m.y));
            TooltipRow(GridColourOf(p, v), p.spec.value_column.empty() ? std::string("density (rows per cell)")
                                                                       : "mean of " + p.spec.value_column,
                       std::isfinite(v) ? Value(v) : std::string("no rows here"));
            break;
        }
    }
    if (shown) ImGui::EndTooltip();
}

}  // namespace cyxwiz::plot
