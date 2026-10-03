#include "plot_view.h"

#include "plot_style.h"
#include "../../core/plot/plot_prepare.h"
#include "../../core/plot/world_map.h"
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
           k == Kind::FilledContour || k == Kind::Quiver || k == Kind::Stream || k == Kind::Confusion || k == Kind::MapRegions;
}

// Drawn in pixels inside an undecorated plot.
bool IsPixelKind(Kind k) {
    return k == Kind::Sankey || k == Kind::Treemap;
}

// A map region's colour: the theme scale over the plot's range (log10 when asked).
ImVec4 RegionColourOf(const Prepared& p, double v) {
    double lo = p.grid_lo, hi = p.grid_hi;
    if (p.spec.log_colour) {
        v = std::log10(std::max(v, lo));
        lo = std::log10(lo);
        hi = std::log10(hi);
    }
    const float t = static_cast<float>(hi > lo ? std::clamp((v - lo) / (hi - lo), 0.0, 1.0) : 0.0);
    return ImPlot::SampleColormap(t, ScaleColormap(p.spec, false));
}

// Dark or light text on a fill, whichever reads.
ImU32 InkOn(const ImVec4& fill) {
    const ui::Tokens& t = ui::CurrentTokens();
    const float lum = 0.2126f * fill.x + 0.7152f * fill.y + 0.0722f * fill.z;
    return ui::ToU32(lum > 0.55f ? t.plot_bg : t.text_bright);
}

// A point on a cubic band edge from (x0, y0) to (x1, y1) with both control
// points at the middle x; t found from x by bisection.
float BandEdgeY(float x0, float y0, float x1, float y1, float x) {
    const float mx = (x0 + x1) * 0.5f;
    float lo = 0.0f, hi = 1.0f, t = 0.5f;
    for (int i = 0; i < 24; ++i) {
        t = (lo + hi) * 0.5f;
        const float u = 1.0f - t;
        const float bx = u * u * u * x0 + 3 * u * u * t * mx + 3 * u * t * t * mx + t * t * t * x1;
        if (bx < x) lo = t;
        else hi = t;
    }
    const float u = 1.0f - t;
    return u * u * u * y0 + 3 * u * u * t * y0 + 3 * u * t * t * y1 + t * t * t * y1;
}

// Curves on 0..1 both ways (model results).
bool IsUnitKind(Kind k) {
    return k == Kind::Roc || k == Kind::PrCurve || k == Kind::Calibration;
}

}  // namespace

// A model figure as text: shares in percent, counts with separators, small
// figures to three places ("AUC 0.871", "RMSE 20.76").
std::string MetricText(const std::string& name, double v) {
    if (!std::isfinite(v)) return "n/a";
    char buf[48];
    if (name == "accuracy" || name == "positive share") std::snprintf(buf, sizeof(buf), "%.1f%%", v * 100.0);
    else if (name == "rows" || name == "countries" || name == "not matched") return Thousands(static_cast<long long>(std::llround(v)));
    else if (name.rfind("at ", 0) == 0 || std::fabs(v) >= 10.0) std::snprintf(buf, sizeof(buf), "%.4g", v);
    else std::snprintf(buf, sizeof(buf), "%.3f", v);
    return buf;
}

namespace {

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
    if (p.spec.kind == Kind::Importance) return "";  // the features name the rows
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
    return ImPlot::SampleColormap(t, ScaleColormap(p.spec, p.grid_diverging));
}

// Colour of a value on the plot's colour scale (a missing value: faint).
ImVec4 ScaleColourOf(const Prepared& p, double v) {
    if (!std::isfinite(v)) return ui::CurrentTokens().text_faint;
    const double span = p.colour_max - p.colour_min;
    const float t = static_cast<float>(span > 0 ? std::clamp((v - p.colour_min) / span, 0.0, 1.0) : 0.0);
    return ImPlot::SampleColormap(t, ScaleColormap(p.spec, p.colour_diverging));
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
    if (kind_changed || data.tree_levels != data_.tree_levels) tree_zoom_.clear();
    // Network / Tree: a new graph or layout drops the user's moves and folds;
    // turning a tree keeps its folds and fits the new shape.
    if (kind_changed || data.graph_nodes.size() != data_.graph_nodes.size() || data.spec.graph_layout != data_.spec.graph_layout) {
        graph_pos_.clear();
        graph_moved_.clear();
        graph_fold_.clear();
    } else if (data.spec.tree_left_right != data_.spec.tree_left_right) {
        std::fill(graph_moved_.begin(), graph_moved_.end(), 0);
        fit_ = true;
    }
    graph_layout_dirty_ = true;
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
    // A colour picked for this series (P4.4) comes first.
    if (i < data_.spec.series_colours.size()) {
        const std::string& h = data_.spec.series_colours[i];
        unsigned r = 0, g = 0, b = 0;
        if (h.size() == 7 && h[0] == '#' && std::sscanf(h.c_str() + 1, "%02x%02x%02x", &r, &g, &b) == 3)
            return ImVec4(static_cast<float>(r) / 255.0f, static_cast<float>(g) / 255.0f, static_cast<float>(b) / 255.0f, 1.0f);
    }
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
    const bool flat = Info(data_.spec.kind).group != Group::ThreeD;  // SVG is 2D
    if (ImGui::MenuItem("Save image as SVG...", nullptr, false, can_save && flat)) {
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
    const bool three_d = has_data_ && Info(data_.spec.kind).group == Group::ThreeD;
    const bool graph = has_data_ && (data_.spec.kind == Kind::Network || data_.spec.kind == Kind::Tree);
    const char* views[] = {"Turn", "Top", "Front", "Side"};
    float views_w = 0.0f;
    if (three_d)
        for (const char* v : views) views_w += ui::ButtonWidth(v, ui::ButtonSize::Small) + gap;
    const float w = views_w + (graph ? ui::ButtonWidth("Re-layout", ui::ButtonSize::Small) + gap : 0.0f) +
                    (o.tool_fit ? ui::ButtonWidth("Fit", ui::ButtonSize::Small) + gap : 0.0f) +
                    (o.tool_log && !three_d && !graph ? ui::ButtonWidth("Log Y", ui::ButtonSize::Small) + gap : 0.0f) +
                    (o.tool_legend && !graph ? ui::ButtonWidth("Legend", ui::ButtonSize::Small) + gap : 0.0f) +
                    ui::ButtonWidth("Export", ui::ButtonSize::Small) +
                    (o.own_window_button ? ui::ButtonWidth(own, ui::ButtonSize::Small) + gap : 0.0f);
    const float right = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - w;
    if (right > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(right);
    const bool usable = has_data_ && data_.problem.empty();
    if (three_d) {
        // Turn: the default turned view; Top, Front and Side look along an axis.
        const double el[] = {NAN, 90.0, 0.0, 0.0}, az[] = {NAN, 0.0, 0.0, 90.0};
        for (int i = 0; i < 4; ++i) {
            if (ui::GhostButton((std::string(views[i]) + "##view" + id_).c_str(), usable)) SetView(el[i], az[i]);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(i == 0 ? "The default turned view (drag to turn, right-drag to move, wheel to zoom)" : "Look along an axis");
            ImGui::SameLine();
        }
    }
    if (graph) {
        if (ui::GhostButton(("Re-layout##" + id_).c_str(), usable)) ResetGraphLayout();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Back to the computed layout: moved nodes and folded branches return");
        ImGui::SameLine();
    }
    if (o.tool_fit) {
        if (ui::GhostButton(("Fit##" + id_).c_str(), usable)) fit_ = true;
        ImGui::SameLine();
    }
    const bool can_log = usable && data_.spec.kind != Kind::Pie && data_.spec.kind != Kind::Polar && data_.spec.kind != Kind::Image &&
                         data_.spec.kind != Kind::PairPlot && data_.spec.kind != Kind::Parallel && !IsGridKind(data_.spec.kind) &&
                         !IsUnitKind(data_.spec.kind) && data_.spec.kind != Kind::Importance && data_.spec.kind != Kind::Residuals &&
                         !IsPixelKind(data_.spec.kind) && data_.spec.kind != Kind::MapPoints && !three_d;
    if (o.tool_log && !three_d && !graph) {  // 3D and graphs: no log axis
        if (ui::GhostButton(("Log Y##" + id_).c_str(), can_log, "Not for this plot type", log_y_)) {
            log_y_ = !log_y_;
            fit_ = true;
        }
        ImGui::SameLine();
    }
    if (o.tool_legend && !graph) {  // graphs: the groups are in the Colour section
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

void PlotView::DrawImages(ImVec2 size) {
    const Prepared& p = data_;
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##images", size);
    const bool hovered = ImGui::IsItemHovered();
    frame_min_ = origin;
    frame_max_ = ImVec2(origin.x + size.x, origin.y + size.y);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(frame_min_, frame_max_, ui::ToU32(t.plot_bg), 4.0f);
    if (p.pictures.empty() || p.img_w <= 0 || p.img_h <= 0) return;
    const int cols = PictureColumns(p);
    const int rows = static_cast<int>((p.pictures.size() + static_cast<size_t>(cols) - 1) / static_cast<size_t>(cols));
    const float caption = ImGui::GetTextLineHeight() + 4.0f;
    const float cell = std::max(8.0f, std::min((size.x - 8.0f) / cols, (size.y - 8.0f) / rows - caption));
    const float px = cell * 0.94f / static_cast<float>(std::max(p.img_w, p.img_h));
    const ImVec2 mouse = ImGui::GetMousePos();
    int hover_pic = -1, hover_x = -1, hover_y = -1;
    dl->PushClipRect(frame_min_, frame_max_, true);
    for (size_t i = 0; i < p.pictures.size(); ++i) {
        const auto& pic = p.pictures[i];
        const float ox = origin.x + 4.0f + static_cast<float>(static_cast<int>(i) % cols) * cell;
        const float oy = origin.y + 4.0f + static_cast<float>(static_cast<int>(i) / cols) * (cell + caption);
        for (int y = 0; y < p.img_h; ++y)
            for (int x = 0; x < p.img_w; ++x) {
                const size_t k = (static_cast<size_t>(y) * static_cast<size_t>(p.img_w) + static_cast<size_t>(x)) * static_cast<size_t>(p.img_channels);
                if (k >= pic.pix.size()) continue;
                ImVec4 c;
                if (p.img_channels == 3) {
                    const auto ch = [&](size_t j) {
                        const float v = std::isfinite(pic.pix[k + j]) ? pic.pix[k + j] : 0.0f;
                        return p.spec.image_invert ? 1.0f - v : v;
                    };
                    c = ImVec4(ch(0), ch(1), ch(2), 1.0f);
                } else {
                    if (!std::isfinite(pic.pix[k])) continue;
                    const float v = p.spec.image_invert ? 1.0f - pic.pix[k] : pic.pix[k];
                    c = p.spec.image_grey ? ImVec4(v, v, v, 1.0f) : ImPlot::SampleColormap(v, ScaleColormap(data_.spec, false));
                }
                const ImVec2 a(ox + x * px, oy + y * px), b(ox + (x + 1) * px, oy + (y + 1) * px);
                dl->AddRectFilled(a, b, ui::ToU32(c));
                if (hovered && mouse.x >= a.x && mouse.x < b.x && mouse.y >= a.y && mouse.y < b.y) {
                    hover_pic = static_cast<int>(i);
                    hover_x = x;
                    hover_y = y;
                }
            }
        std::string text = pic.label;
        if (pic.row > 0) text = "row " + Thousands(static_cast<long long>(pic.row)) + (text.empty() ? "" : " \xC2\xB7 " + text);
        else text += "  (" + Thousands(static_cast<long long>(pic.count)) + ")";
        dl->AddText(ImVec2(ox, oy + p.img_h * px + 2.0f), ui::ToU32(t.text_dim), text.c_str());
    }
    dl->PopClipRect();
    if (hover_pic >= 0) {
        // The pixel under the mouse, in the table's units.
        const auto& pic = p.pictures[static_cast<size_t>(hover_pic)];
        const size_t k = (static_cast<size_t>(hover_y) * static_cast<size_t>(p.img_w) + static_cast<size_t>(hover_x)) * static_cast<size_t>(p.img_channels);
        const auto raw = [&](size_t j) { return std::isfinite(pic.pix[k + j]) ? Value(p.img_lo + pic.pix[k + j] * (p.img_hi - p.img_lo)) : std::string("missing"); };
        ImGui::BeginTooltip();
        ImGui::TextColored(t.text_bright, "%s", pic.row > 0 ? ("row " + Thousands(static_cast<long long>(pic.row)) + (pic.label.empty() ? "" : " \xC2\xB7 " + pic.label)).c_str()
                                                             : (pic.label + " (mean of " + Thousands(static_cast<long long>(pic.count)) + " rows)").c_str());
        ImGui::TextColored(t.text_dim, "x %d, y %d", hover_x, hover_y);
        if (p.img_channels == 3) ImGui::Text("r %s  g %s  b %s", raw(0).c_str(), raw(1).c_str(), raw(2).c_str());
        else ImGui::Text("value %s", raw(0).c_str());
        ImGui::EndTooltip();
    }
}

void PlotView::DrawPairPlot(ImVec2 size) {
    const Prepared& p = data_;
    const int n = static_cast<int>(p.multi_cols.size());
    if (n < 2) return;
    ImPlotSubplotFlags sflags = ImPlotSubplotFlags_NoTitle | ImPlotSubplotFlags_NoResize | ImPlotSubplotFlags_ShareItems | ImPlotSubplotFlags_NoMenus;
    if (!legend_) sflags |= ImPlotSubplotFlags_NoLegend;
    if (!ImPlot::BeginSubplots("##pairs", n, n, size, sflags)) return;
    // Points per group (sampled rows), built once per frame.
    const size_t groups = std::max<size_t>(1, p.multi_groups.size());
    std::vector<std::vector<size_t>> rows_of(groups);
    for (size_t r = 0; r < p.multi_group.size(); ++r)
        rows_of[static_cast<size_t>(std::clamp(p.multi_group[r], 0, static_cast<int>(groups) - 1))].push_back(r);
    std::vector<double> xs, ys;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const std::string id = "##pair" + std::to_string(i) + "_" + std::to_string(j);
            if (!ImPlot::BeginPlot(id.c_str(), ImVec2(-1, -1), ImPlotFlags_NoMenus | ImPlotFlags_NoMouseText)) continue;
            const ImPlotAxisFlags xf = i == n - 1 ? ImPlotAxisFlags_None : ImPlotAxisFlags_NoTickLabels;
            const ImPlotAxisFlags yf = (j == 0 && i != j) ? ImPlotAxisFlags_None : ImPlotAxisFlags_NoTickLabels;
            ImPlot::SetupAxes(i == n - 1 ? p.multi_cols[static_cast<size_t>(j)].c_str() : nullptr,
                              j == 0 ? p.multi_cols[static_cast<size_t>(i)].c_str() : nullptr, xf, yf);
            const double xlo = p.multi_lo[static_cast<size_t>(j)], xhi = p.multi_hi[static_cast<size_t>(j)];
            const double mx = (xhi - xlo) * 0.04;
            if (i == j) {
                double peak = 0;
                for (const auto& g : p.pair_diag[static_cast<size_t>(i)])
                    for (double v : g) peak = std::max(peak, v);
                ImPlot::SetupAxesLimits(xlo - mx, xhi + mx, 0, peak > 0 ? peak * 1.1 : 1.0, ImPlotCond_Always);
                for (size_t g = 0; g < p.pair_diag[static_cast<size_t>(i)].size(); ++g) {
                    const auto& d = p.pair_diag[static_cast<size_t>(i)][g];
                    xs.clear();
                    const int steps = static_cast<int>(d.size());
                    const bool hist = p.spec.pair_histogram;
                    for (int k = 0; k < steps; ++k)
                        xs.push_back(hist ? xlo + (xhi - xlo) * (k + 0.5) / steps : xlo + (xhi - xlo) * k / std::max(1, steps - 1));
                    const std::string label = g < p.multi_groups.size() && !p.multi_groups[g].empty() ? p.multi_groups[g] : std::string("rows");
                    if (hist) {
                        ImPlot::SetNextFillStyle(SeriesColour(g), 0.55f);
                        ImPlot::PlotBars(label.c_str(), xs.data(), d.data(), steps, (xhi - xlo) / steps * 0.95);
                    } else {
                        ImPlot::SetNextLineStyle(SeriesColour(g), 1.6f);
                        ImPlot::PlotLine(label.c_str(), xs.data(), d.data(), steps);
                    }
                }
            } else {
                const double ylo = p.multi_lo[static_cast<size_t>(i)], yhi = p.multi_hi[static_cast<size_t>(i)];
                const double my = (yhi - ylo) * 0.04;
                ImPlot::SetupAxesLimits(xlo - mx, xhi + mx, ylo - my, yhi + my, ImPlotCond_Always);
                for (size_t g = 0; g < groups; ++g) {
                    xs.clear();
                    ys.clear();
                    for (size_t r : rows_of[g]) {
                        xs.push_back(p.multi_values[static_cast<size_t>(j)][r]);
                        ys.push_back(p.multi_values[static_cast<size_t>(i)][r]);
                    }
                    const std::string label = g < p.multi_groups.size() && !p.multi_groups[g].empty() ? p.multi_groups[g] : std::string("rows");
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1.8f, ui::WithAlpha(SeriesColour(g), 0.7f), 0.0f);
                    ImPlot::PlotScatter(label.c_str(), xs.data(), ys.data(), static_cast<int>(xs.size()));
                }
                if (ImPlot::IsPlotHovered()) {
                    const ImPlotPoint m = ImPlot::GetPlotMousePos();
                    ImGui::BeginTooltip();
                    ImGui::Text("%s %s", p.multi_cols[static_cast<size_t>(j)].c_str(), Value(m.x).c_str());
                    ImGui::Text("%s %s", p.multi_cols[static_cast<size_t>(i)].c_str(), Value(m.y).c_str());
                    ImGui::EndTooltip();
                }
            }
            ImPlot::EndPlot();
        }
    ImPlot::EndSubplots();
    frame_min_ = ImGui::GetItemRectMin();
    frame_max_ = ImGui::GetItemRectMax();
}

void PlotView::DrawPlot(ImVec2 size) {
    const Prepared& p = data_;
    const Kind kind = p.spec.kind;
    if (kind == Kind::Image) {
        DrawImages(size);
        return;
    }
    if (kind == Kind::PairPlot) {
        DrawPairPlot(size);
        return;
    }
    if (Info(kind).group == Group::ThreeD) {
        Draw3D(size);
        return;
    }
    if (kind == Kind::Network || kind == Kind::Tree) {
        DrawGraph(size);
        return;
    }
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
    if (kind == Kind::MapPoints || kind == Kind::MapRegions) flags |= ImPlotFlags_Equal;
    if (IsPixelKind(kind)) flags |= ImPlotFlags_NoLegend;

    // Fixed-layout kinds set their limits; the rest fit to the data.
    const ImPlotCond cond = fit_ ? ImPlotCond_Always : ImPlotCond_Once;
    if (fit_ && !x_range_.on && !y_range_.on && kind != Kind::Pie && kind != Kind::Polar && kind != Kind::Box && kind != Kind::Violin &&
        kind != Kind::Parallel && !IsGridKind(kind) && !IsUnitKind(kind) && !IsPixelKind(kind) && kind != Kind::MapPoints)
        ImPlot::SetNextAxesToFit();

    const std::string plot_id = "##plot";
    if (!ImPlot::BeginPlot(plot_id.c_str(), plot_size, flags)) return;

    const std::string xl = XLabel(p), yl = YLabel(p);
    if (kind == Kind::Pie || IsPixelKind(kind)) {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxesLimits(0, 1, 0, 1, ImPlotCond_Always);
    } else if (kind == Kind::Parallel) {
        // One vertical axis per column at x = 0, 1, ...; y is each column's 0..1.
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks, ImPlotAxisFlags_NoDecorations);
        const double n = static_cast<double>(std::max<size_t>(2, p.multi_cols.size()));
        ImPlot::SetupAxesLimits(-0.25, n - 0.75, -0.06, 1.1, ImPlotCond_Always);
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
        if (kind == Kind::Heatmap || kind == Kind::Matrix || kind == Kind::Confusion)
            xf = yf = ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
        if (kind == Kind::Importance) yf = ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;
        ImPlot::SetupAxes(xl.empty() ? nullptr : xl.c_str(), yl.empty() ? nullptr : yl.c_str(), xf, yf);
        if (log_y_ && !IsGridKind(kind)) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        if (kind == Kind::Calibration) {
            // Rows per bin as bars along the bottom fifth (right axis).
            double most = 1;
            for (double c : p.grid_counts) most = std::max(most, c);
            ImPlot::SetupAxis(ImAxis_Y2, "rows", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_NoGridLines);
            ImPlot::SetupAxisLimits(ImAxis_Y2, 0, most * 5.0, ImPlotCond_Always);
        }
    }
    // Curves that rise to the top right keep their legend out of the way.
    ImPlot::SetupLegend(kind == Kind::Roc || kind == Kind::Calibration || kind == Kind::Importance ? ImPlotLocation_SouthEast
                                                                                                  : ImPlotLocation_NorthEast);
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
        case Kind::Parallel: set_ticks(ImAxis_X1, p.multi_cols, 0.0); break;
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
        case Kind::Roc:
        case Kind::PrCurve:
        case Kind::Calibration: ImPlot::SetupAxesLimits(-0.02, 1.02, -0.02, 1.04, cond); break;
        case Kind::MapRegions: ImPlot::SetupAxesLimits(-180, 180, -60, 85, cond); break;
        case Kind::MapPoints: {
            // The points' extent with a margin, inside the world (a plain fit
            // is stretched by the equal aspect).
            double x0 = 180, x1 = -180, y0 = 90, y1 = -90;
            for (const auto& sr : p.series)
                for (size_t k = 0; k < std::min(sr.x.size(), sr.y.size()); ++k) {
                    x0 = std::min(x0, sr.x[k]);
                    x1 = std::max(x1, sr.x[k]);
                    y0 = std::min(y0, sr.y[k]);
                    y1 = std::max(y1, sr.y[k]);
                }
            if (x1 < x0 || x_range_.on || y_range_.on) break;
            const double mx = std::max(2.0, (x1 - x0) * 0.06), my = std::max(2.0, (y1 - y0) * 0.06);
            ImPlot::SetupAxesLimits(std::max(-180.0, x0 - mx), std::min(180.0, x1 + mx), std::max(-90.0, y0 - my), std::min(90.0, y1 + my), cond);
            break;
        }
        case Kind::Importance: {
            // Features down the side, the largest on top.
            const size_t count = p.categories.size();
            for (size_t i = 0; i < count && count <= 60; ++i) {
                ynames.push_back(p.categories[i].c_str());
                ypos.push_back(static_cast<double>(count - 1 - i));
            }
            if (!ypos.empty()) ImPlot::SetupAxisTicks(ImAxis_Y1, ypos.data(), static_cast<int>(ypos.size()), ynames.data());
            break;
        }
        case Kind::Heatmap:
        case Kind::Matrix:
        case Kind::Confusion: {
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
            // All rows in grey behind the filtered bins (same edges).
            if (background_ && background_->spec.kind == Kind::Histogram && background_->edges == p.edges && background_->series.size() == 1) {
                const auto& bg = background_->series[0];
                ImPlot::SetNextFillStyle(ui::WithAlpha(t.text_dim, 0.35f));
                ImPlot::SetNextLineStyle(ImVec4(0, 0, 0, 0));
                ImPlot::PlotBars("all rows", bg.x.data(), bg.y.data(), n(bg.x), width * 0.94);
            }
            const bool ranged = std::isfinite(highlight_lo_) && std::isfinite(highlight_hi_) && p.series.size() == 1;
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                if (ranged) {
                    // The selected bins in full colour, the rest dimmed.
                    std::vector<double> in_x, in_y;
                    for (size_t k = 0; k < s.x.size(); ++k)
                        if (s.x[k] >= highlight_lo_ && s.x[k] <= highlight_hi_) {
                            in_x.push_back(s.x[k]);
                            in_y.push_back(s.y[k]);
                        }
                    ImPlot::SetNextFillStyle(ColourOf(i), 0.3f);
                    ImPlot::SetNextLineStyle(ImVec4(0, 0, 0, 0));  // no outline on the dimmed bars
                    ImPlot::PlotBars(("##dim" + s.label).c_str(), s.x.data(), s.y.data(), n(s.x), width * 0.94);
                    ImPlot::SetNextFillStyle(ColourOf(i), 0.95f);
                    ImPlot::PlotBars(s.label.c_str(), in_x.data(), in_y.data(), n(in_x), width * 0.94);
                    continue;
                }
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
                const auto& s = p.series[0];
                // All rows in grey behind, matched by category name.
                if (background_ && background_->spec.kind == Kind::Bar && background_->series.size() == 1) {
                    std::vector<double> bx, by;
                    for (size_t k = 0; k < p.categories.size() && k < s.x.size(); ++k)
                        for (size_t j = 0; j < background_->categories.size() && j < background_->series[0].y.size(); ++j)
                            if (background_->categories[j] == p.categories[k]) {
                                bx.push_back(s.x[k]);
                                by.push_back(background_->series[0].y[j]);
                                break;
                            }
                    ImPlot::SetNextFillStyle(ui::WithAlpha(t.text_dim, 0.35f));
                    ImPlot::SetNextLineStyle(ImVec4(0, 0, 0, 0));
                    ImPlot::PlotBars("all rows", bx.data(), by.data(), n(bx), 0.67);
                }
                if (!highlight_.empty()) {
                    // The selected categories in full colour, the rest dimmed.
                    std::vector<double> in_x, in_y;
                    for (size_t k = 0; k < p.categories.size() && k < s.x.size(); ++k)
                        if (std::find(highlight_.begin(), highlight_.end(), p.categories[k]) != highlight_.end()) {
                            in_x.push_back(s.x[k]);
                            in_y.push_back(s.y[k]);
                        }
                    ImPlot::SetNextFillStyle(ColourOf(0), 0.3f);
                    ImPlot::SetNextLineStyle(ImVec4(0, 0, 0, 0));  // no outline on the dimmed bars
                    ImPlot::PlotBars(("##dim" + s.label).c_str(), s.x.data(), s.y.data(), n(s.x), 0.67);
                    ImPlot::SetNextFillStyle(ColourOf(0), 0.95f);
                    ImPlot::PlotBars(s.label.c_str(), in_x.data(), in_y.data(), n(in_x), 0.67);
                } else {
                    ImPlot::SetNextFillStyle(ColourOf((0)), 0.9f);
                    ImPlot::PlotBars(s.label.c_str(), s.x.data(), s.y.data(), n(s.x), 0.67);
                }
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
            ImPlot::PushColormap(ScaleColormap(p.spec, p.grid_diverging));
            // Numbers in the cells when they fit (about 40 px a cell).
            const bool numbers = named && p.grid.size() <= 400 && plot_size.x / static_cast<float>(std::max(1, p.grid_cols)) >= 40.0f;
            const char* format = kind == Kind::Matrix && p.spec.matrix_values != PlotSpec::MatrixValues::Values ? "%.2f" : "%g";
            const ImPlotPoint lo = named ? ImPlotPoint(0, 0) : ImPlotPoint(p.x_min, p.y_min);
            const ImPlotPoint hi = named ? ImPlotPoint(p.grid_cols, p.grid_rows) : ImPlotPoint(p.x_max, p.y_max);
            ImPlot::PlotHeatmap("##grid", p.grid.data(), p.grid_rows, p.grid_cols, p.grid_lo,
                                p.grid_hi > p.grid_lo ? p.grid_hi : p.grid_lo + 1.0, numbers ? format : nullptr, lo, hi);
            ImPlot::PopColormap();
            break;
        }
        case Kind::Parallel: {
            // A line per sampled row, coloured by group; the hovered one on top.
            ImDrawList* dl = ImPlot::GetPlotDrawList();
            const size_t cols = p.multi_cols.size();
            const auto at = [&](size_t c, double v) {
                return ImPlot::PlotToPixels(static_cast<double>(c), (v - p.multi_lo[c]) / (p.multi_hi[c] - p.multi_lo[c]));
            };
            // The line nearest the mouse (6 px).
            int hot = -1;
            if (ImPlot::IsPlotHovered()) {
                const ImVec2 mouse = ImGui::GetMousePos();
                float best = 36.0f;
                for (size_t r = 0; r < p.multi_group.size(); ++r)
                    for (size_t c = 0; c + 1 < cols; ++c) {
                        const ImVec2 a = at(c, p.multi_values[c][r]), b = at(c + 1, p.multi_values[c + 1][r]);
                        if (mouse.x < a.x || mouse.x > b.x) continue;
                        const float u = (mouse.x - a.x) / std::max(1.0f, b.x - a.x);
                        const float y = a.y + (b.y - a.y) * u, d = (y - mouse.y) * (y - mouse.y);
                        if (d < best) {
                            best = d;
                            hot = static_cast<int>(r);
                        }
                    }
            }
            hovered_row_ = hot;
            ImPlot::PushPlotClipRect();
            std::vector<ImVec2> pts(cols);
            for (size_t r = 0; r < p.multi_group.size(); ++r) {
                for (size_t c = 0; c < cols; ++c) pts[c] = at(c, p.multi_values[c][r]);
                const ImVec4 colour = SeriesColour(static_cast<size_t>(std::max(0, p.multi_group[r])));
                dl->AddPolyline(pts.data(), static_cast<int>(cols), ui::ToU32(ui::WithAlpha(colour, hot >= 0 ? 0.18f : 0.4f)), ImDrawFlags_None, 1.0f);
            }
            if (hot >= 0) {
                for (size_t c = 0; c < cols; ++c) pts[c] = at(c, p.multi_values[c][static_cast<size_t>(hot)]);
                dl->AddPolyline(pts.data(), static_cast<int>(cols), ui::ToU32(t.text_bright), ImDrawFlags_None, 2.2f);
            }
            // The axes, each with its top and bottom value.
            for (size_t c = 0; c < cols; ++c) {
                const ImVec2 top = at(c, p.multi_hi[c]), bottom = at(c, p.multi_lo[c]);
                dl->AddLine(top, bottom, ui::ToU32(t.text_dim), 1.2f);
                const std::string hi = Value(p.multi_hi[c]), lo = Value(p.multi_lo[c]);
                const ImVec2 hs = ImGui::CalcTextSize(hi.c_str());
                dl->AddText(ImVec2(top.x - hs.x * 0.5f, top.y - hs.y - 2.0f), ui::ToU32(t.text_dim), hi.c_str());
                const ImVec2 ls = ImGui::CalcTextSize(lo.c_str());
                dl->AddText(ImVec2(bottom.x - ls.x * 0.5f, bottom.y + 2.0f), ui::ToU32(t.text_dim), lo.c_str());
            }
            ImPlot::PopPlotClipRect();
            // Legend entries for the groups.
            for (size_t g = 0; g < p.multi_groups.size() && p.multi_groups.size() > 1; ++g) {
                ImPlot::SetNextLineStyle(SeriesColour(g), 2.0f);
                ImPlot::PlotDummy(p.multi_groups[g].c_str());
            }
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
                ImPlot::PushColormap(ScaleColormap(data_.spec, false));
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

    // P2b group 4: model results.
    switch (kind) {
        case Kind::Confusion: {
            // Cells coloured by the share (or count); each shows the count and the share.
            ImPlot::PushColormap(ScaleColormap(data_.spec, false));
            ImPlot::PlotHeatmap("##grid", p.grid.data(), p.grid_rows, p.grid_cols, p.grid_lo, p.grid_hi > p.grid_lo ? p.grid_hi : p.grid_lo + 1.0,
                                nullptr, ImPlotPoint(0, 0), ImPlotPoint(p.grid_cols, p.grid_rows));
            ImPlot::PopColormap();
            if (p.grid_rows > 20) break;
            const bool share = p.spec.confusion_show != PlotSpec::ConfusionShow::Counts;
            ImDrawList* dl = ImPlot::GetPlotDrawList();
            ImPlot::PushPlotClipRect();
            const float line = ImGui::GetTextLineHeight();
            for (int r = 0; r < p.grid_rows; ++r)
                for (int c = 0; c < p.grid_cols; ++c) {
                    const size_t k = static_cast<size_t>(r * p.grid_cols + c);
                    const ImVec2 at = ImPlot::PlotToPixels(c + 0.5, p.grid_rows - r - 0.5);
                    const double tv = p.grid_hi > p.grid_lo ? (p.grid[k] - p.grid_lo) / (p.grid_hi - p.grid_lo) : 0.0;
                    const ImVec4 ink_colour = tv > 0.55 ? t.plot_bg : t.text_bright;
                    const ImU32 ink = ui::ToU32(ink_colour);
                    const std::string count = Thousands(static_cast<long long>(std::llround(p.grid_counts[k])));
                    const ImVec2 cs = ImGui::CalcTextSize(count.c_str());
                    dl->AddText(ImVec2(at.x - cs.x * 0.5f, at.y - (share ? line : line * 0.5f)), ink, count.c_str());
                    if (share) {
                        char buf[24];
                        std::snprintf(buf, sizeof(buf), "%.1f%%", p.grid[k] * 100.0);
                        const ImVec2 ss = ImGui::CalcTextSize(buf);
                        dl->AddText(ImVec2(at.x - ss.x * 0.5f, at.y), ui::ToU32(ui::WithAlpha(ink_colour, 0.75f)), buf);
                    }
                }
            ImPlot::PopPlotClipRect();
            break;
        }
        case Kind::Roc:
        case Kind::PrCurve:
            if (!p.series.empty()) {
                const auto& sr = p.series[0];
                if (kind == Kind::PrCurve && std::isfinite(p.baseline)) {
                    const std::string bl = "positive share " + MetricText("positive share", p.baseline);
                    ImPlot::SetNextLineStyle(ui::WithAlpha(t.text_dim, 0.8f), 1.2f);
                    ImPlot::PlotInfLines(bl.c_str(), &p.baseline, 1, ImPlotInfLinesFlags_Horizontal);
                }
                ImPlot::SetNextFillStyle(ColourOf(0), 0.12f);
                ImPlot::PlotShaded(("##fill" + sr.label).c_str(), sr.x.data(), sr.y.data(), n(sr.x), 0.0);
                ImPlot::SetNextLineStyle(ColourOf(0), 2.0f);
                if (kind == Kind::PrCurve) {
                    // Precision holds until recall reaches the next point.
                    std::vector<double> sx{0.0}, sy{sr.y.empty() ? 1.0 : sr.y.front()};
                    sx.insert(sx.end(), sr.x.begin(), sr.x.end());
                    sy.insert(sy.end(), sr.y.begin(), sr.y.end());
                    ImPlot::PlotStairs(sr.label.c_str(), sx.data(), sy.data(), n(sx), ImPlotStairsFlags_PreStep);
                } else {
                    ImPlot::PlotLine(sr.label.c_str(), sr.x.data(), sr.y.data(), n(sr.x));
                }
            }
            break;
        case Kind::Calibration:
            if (!p.series.empty()) {
                const auto& sr = p.series[0];
                // Rows per bin (all bins, on the right axis).
                if (p.edges.size() == p.grid_counts.size() + 1 && !p.grid_counts.empty()) {
                    std::vector<double> mid;
                    for (size_t b = 0; b < p.grid_counts.size(); ++b) mid.push_back((p.edges[b] + p.edges[b + 1]) * 0.5);
                    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                    ImPlot::SetNextFillStyle(t.text_dim, 0.35f);
                    ImPlot::PlotBars("rows", mid.data(), p.grid_counts.data(), n(mid), (p.edges[1] - p.edges[0]) * 0.8);
                    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
                }
                ImPlot::SetNextLineStyle(ColourOf(0), 2.0f);
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.5f, ColourOf(0), 0.0f);
                ImPlot::PlotLine(sr.label.c_str(), sr.x.data(), sr.y.data(), n(sr.x));
            }
            break;
        case Kind::Residuals:
            if (!p.series.empty()) {
                const auto& sr = p.series[0];
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 2.5f, ui::WithAlpha(ColourOf(0), 0.6f), 0.0f);
                ImPlot::PlotScatter(sr.label.c_str(), sr.x.data(), sr.y.data(), n(sr.x));
                const double zero = 0.0;
                ImPlot::SetNextLineStyle(ui::WithAlpha(t.text_dim, 0.9f), 1.4f);
                ImPlot::PlotInfLines("##zero", &zero, 1, ImPlotInfLinesFlags_Horizontal);
            }
            break;
        case Kind::LearningCurve:
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& sr = p.series[i];
                const ImVec4 c = ColourOf(i);
                if (sr.low.size() == sr.x.size() && !sr.x.empty()) {
                    ImPlot::SetNextFillStyle(c, 0.18f);
                    ImPlot::PlotShaded(("##band" + sr.label).c_str(), sr.x.data(), sr.low.data(), sr.high.data(), n(sr.x));
                }
                ImPlot::SetNextLineStyle(c, 2.0f);
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3.5f, c, 0.0f);
                ImPlot::PlotLine(sr.label.c_str(), sr.x.data(), sr.y.data(), n(sr.x));
                if (static_cast<int>(i) == p.best_series && p.best_index < sr.x.size()) {
                    ImDrawList* dl = ImPlot::GetPlotDrawList();
                    ImPlot::PushPlotClipRect();
                    dl->AddCircle(ImPlot::PlotToPixels(sr.x[p.best_index], sr.y[p.best_index]), 7.0f, ui::ToU32(t.text_bright), 16, 1.8f);
                    ImPlot::PopPlotClipRect();
                }
            }
            break;
        case Kind::Importance:
            if (!p.series.empty()) {
                const auto& sr = p.series[0];
                std::vector<double> pos;
                for (size_t i = 0; i < sr.y.size(); ++i) pos.push_back(static_cast<double>(sr.y.size() - 1 - i));
                ImPlot::SetNextFillStyle(ColourOf(0), 0.9f);
                ImPlot::PlotBars(sr.label.c_str(), sr.y.data(), pos.data(), n(pos), 0.67, ImPlotBarsFlags_Horizontal);
                if (sr.low.size() == sr.y.size()) {
                    ImPlot::SetNextErrorBarStyle(t.text_bright, 1.2f, 5.0f);
                    ImPlot::PlotErrorBars(("##spread" + sr.label).c_str(), sr.y.data(), pos.data(), sr.low.data(), sr.high.data(), n(pos),
                                          ImPlotErrorBarsFlags_Horizontal);
                }
            }
            break;
        case Kind::Sankey: DrawSankey(); break;
        case Kind::Treemap: DrawTreemap(); break;
        case Kind::MapRegions: DrawWorld(true); break;
        case Kind::MapPoints: {
            DrawWorld(false);
            ImDrawList* dl = ImPlot::GetPlotDrawList();
            const double span = p.size_max - p.size_min;
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& sr = p.series[i];
                const ImVec4 c = ColourOf(i);
                // A small marker per point keeps the fit and the legend; the
                // sized and coloured circles are drawn over it.
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1.0f, p.colour_scale ? ui::WithAlpha(t.text, 0.0f) : c, 0.0f);
                ImPlot::PlotScatter((p.series.size() > 1 ? LegendLabel(p, i, sr.label) : std::string("##points")).c_str(), sr.x.data(), sr.y.data(), n(sr.x));
                ImPlot::PushPlotClipRect();
                for (size_t k = 0; k < sr.x.size(); ++k) {
                    float r = 3.0f;
                    if (k < sr.z.size() && std::isfinite(sr.z[k]) && span > 0)
                        r = 2.0f + 9.0f * static_cast<float>(std::sqrt(std::clamp((sr.z[k] - p.size_min) / span, 0.0, 1.0)));
                    ImVec4 fill = c;
                    if (p.colour_scale) {
                        // A quarter up the scale at least: its low end is the plot colour.
                        const double cv = k < sr.c.size() ? sr.c[k] : NAN;
                        const double span_c = p.colour_max - p.colour_min;
                        const float tc = static_cast<float>(span_c > 0 && std::isfinite(cv) ? std::clamp((cv - p.colour_min) / span_c, 0.0, 1.0) : 0.0);
                        fill = std::isfinite(cv) ? ImPlot::SampleColormap(0.25f + 0.75f * tc, ScaleColormap(p.spec, p.colour_diverging))
                                                 : t.text_faint;
                    }
                    dl->AddCircleFilled(ImPlot::PlotToPixels(sr.x[k], sr.y[k]), r, ui::ToU32(ui::WithAlpha(fill, 0.72f)), 12);
                }
                ImPlot::PopPlotClipRect();
            }
            break;
        }
        default: break;
    }

    // Model results: their figures in the top left corner (maps: bottom left, over the ocean).
    if (!p.metrics.empty() && kind != Kind::Confusion) {
        std::vector<std::string> lines;
        if (!p.positive_label.empty()) lines.push_back("positive: " + p.positive_label);
        for (const auto& [name, v] : p.metrics) lines.push_back(name + "  " + MetricText(name, v));
        const ImVec2 corner = ImPlot::GetPlotPos();
        float wide = 0;
        for (const auto& l : lines) wide = std::max(wide, ImGui::CalcTextSize(l.c_str()).x);
        const float line = ImGui::GetTextLineHeightWithSpacing();
        ImDrawList* dl = ImPlot::GetPlotDrawList();
        ImVec2 a(corner.x + 10.0f, corner.y + 10.0f);
        if (kind == Kind::MapRegions) a.y = corner.y + ImPlot::GetPlotSize().y - 18.0f - line * static_cast<float>(lines.size());
        dl->AddRectFilled(a, ImVec2(a.x + wide + 16.0f, a.y + line * static_cast<float>(lines.size()) + 8.0f),
                          ui::ToU32(ui::WithAlpha(t.plot_bg, 0.88f)), 6.0f);
        for (size_t i = 0; i < lines.size(); ++i)
            dl->AddText(ImVec2(a.x + 8.0f, a.y + 4.0f + line * static_cast<float>(i)),
                        ui::ToU32(i == 0 && !p.positive_label.empty() ? t.text_dim : t.text_bright), lines[i].c_str());
    }

    // A y = x reference line (ROC chance, perfect calibration).
    if (p.spec.show_diagonal && IsUnitKind(kind)) {
        const double xs[2] = {0.0, 1.0}, ys[2] = {0.0, 1.0};
        ImPlot::SetNextLineStyle(ui::WithAlpha(t.text_dim, 0.8f), 1.2f);
        ImPlot::PlotLine(kind == Kind::Roc ? "chance" : "perfectly calibrated", xs, ys, 2);
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
    // A click (not a drag to pan) on a bar, slice or bin.
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && ImGui::GetIO().MouseDragMaxDistanceSqr[0] < 16.0f)
        DetectClick();
    const ImPlotRect lim = ImPlot::GetPlotLimits();
    range_ = AxisRange{lim.X.Min, lim.X.Max, lim.Y.Min, lim.Y.Max, log_y_};
    ImPlot::EndPlot();
    frame_min_ = ImGui::GetItemRectMin();
    frame_max_ = ImGui::GetItemRectMax();

    if (scale_bar && p.colour_scale) {
        // The colour bar of a scatter coloured by a number column.
        ImGui::SameLine();
        ImPlot::ColormapScale(p.colour_label.c_str(), p.colour_min, p.colour_max, ImVec2(bar_w, plot_size.y), "%g", 0,
                              ScaleColormap(p.spec, p.colour_diverging));
    } else if (scale_bar) {
        double lo = p.grid_lo, hi = p.grid_hi > p.grid_lo ? p.grid_hi : p.grid_lo + 1.0;
        const bool log = kind == Kind::Hexbin && p.spec.log_colour;
        if (log) {
            lo = std::log1p(std::max(0.0, lo));
            hi = std::log1p(std::max(0.0, hi));
        }
        const bool log10 = kind == Kind::MapRegions && p.spec.log_colour;
        if (log10) {
            lo = std::log10(p.grid_lo);
            hi = std::log10(p.grid_hi);
        }
        ImGui::SameLine();
        const char* scale_label = log ? "log(1 + count)##scale" : log10 ? "log10##scale" : kind == Kind::Quiver ? "length##scale" : kind == Kind::Stream ? "speed##scale" : "##scale";
        ImPlot::ColormapScale(scale_label, lo, hi, ImVec2(60, plot_size.y), "%g", 0,
                              ScaleColormap(p.spec, p.grid_diverging));
    }
}

// Sankey: steps left to right, a node per category, bands between them.
void PlotView::DrawSankey() {
    const Prepared& p = data_;
    const ui::Tokens& t = ui::CurrentTokens();
    ImDrawList* dl = ImPlot::GetPlotDrawList();
    const ImVec2 pp = ImPlot::GetPlotPos(), ps = ImPlot::GetPlotSize();
    const float line = ImGui::GetTextLineHeight(), node_w = 12.0f, pad = 8.0f;
    const size_t steps = p.sankey_steps.size();
    if (steps < 2) return;
    // Room for the labels of the last step.
    float label_w = 0;
    for (const auto& nd : p.sankey_nodes)
        if (static_cast<size_t>(nd.step) + 1 == steps)
            label_w = std::max(label_w, ImGui::CalcTextSize((nd.name + "  " + Thousands(static_cast<long long>(std::llround(nd.value)))).c_str()).x);
    label_w = std::min(label_w + 10.0f, ps.x * 0.3f);
    const float top = pp.y + line + 10.0f, h = ps.y - (line + 10.0f) - pad;
    const auto sx = [&](int step) { return pp.x + pad + static_cast<float>(step) / static_cast<float>(steps - 1) * (ps.x - 2 * pad - node_w - label_w); };
    const auto sy = [&](double y) { return top + static_cast<float>(y) * h; };
    const ImVec2 mouse = ImGui::GetMousePos();
    const bool hovered = ImPlot::IsPlotHovered();
    hot_node_ = hot_link_ = -1;
    for (size_t i = 0; i < p.sankey_nodes.size() && hovered; ++i) {
        const auto& nd = p.sankey_nodes[i];
        const float x = sx(nd.step);
        if (mouse.x >= x - 2 && mouse.x <= x + node_w + 2 && mouse.y >= sy(nd.y0) && mouse.y <= std::max(sy(nd.y1), sy(nd.y0) + 2)) hot_node_ = static_cast<int>(i);
    }
    // First node of each step: band colours by the source's place in its step.
    std::vector<int> first(steps, -1);
    for (size_t i = 0; i < p.sankey_nodes.size(); ++i)
        if (first[static_cast<size_t>(p.sankey_nodes[i].step)] < 0) first[static_cast<size_t>(p.sankey_nodes[i].step)] = static_cast<int>(i);
    for (size_t li = 0; li < p.sankey_links.size(); ++li) {
        const auto& l = p.sankey_links[li];
        const auto& a = p.sankey_nodes[static_cast<size_t>(l.from)];
        const auto& b = p.sankey_nodes[static_cast<size_t>(l.to)];
        const float x0 = sx(a.step) + node_w, x1 = sx(b.step), y0 = sy(l.y_from), y1 = sy(l.y_to);
        const float th = std::max(1.0f, static_cast<float>(l.thickness) * h);
        if (hovered && hot_node_ < 0 && mouse.x > x0 && mouse.x < x1) {
            const float e = BandEdgeY(x0, y0, x1, y1, mouse.x);
            if (mouse.y >= e && mouse.y <= e + th) hot_link_ = static_cast<int>(li);
        }
        const size_t colour = a.step == 0 ? static_cast<size_t>(l.from) : static_cast<size_t>(l.from - first[static_cast<size_t>(a.step)]) + 3;
        const bool hot = hot_link_ == static_cast<int>(li) || hot_node_ == l.from || hot_node_ == l.to;
        const float mx = (x0 + x1) * 0.5f;
        dl->PathLineTo(ImVec2(x0, y0));
        dl->PathBezierCubicCurveTo(ImVec2(mx, y0), ImVec2(mx, y1), ImVec2(x1, y1), 24);
        dl->PathLineTo(ImVec2(x1, y1 + th));
        dl->PathBezierCubicCurveTo(ImVec2(mx, y1 + th), ImVec2(mx, y0 + th), ImVec2(x0, y0 + th), 24);
        dl->PathFillConcave(ui::ToU32(ui::WithAlpha(SeriesColour(colour), hot ? 0.62f : 0.32f)));
    }
    std::vector<float> label_floor(steps, -FLT_MAX);  // the bottom of the last label drawn per step
    for (size_t i = 0; i < p.sankey_nodes.size(); ++i) {
        const auto& nd = p.sankey_nodes[i];
        const float x = sx(nd.step), y0 = sy(nd.y0), y1 = std::max(sy(nd.y1), y0 + 1.0f);
        dl->AddRectFilled(ImVec2(x, y0), ImVec2(x + node_w, y1), ui::ToU32(hot_node_ == static_cast<int>(i) ? t.text_bright : t.text), 2.0f);
        const ImVec2 at(x + node_w + 5.0f, (y0 + y1) * 0.5f - line * 0.5f);
        float& floor = label_floor[static_cast<size_t>(nd.step)];
        if (at.y < floor + 1.0f) continue;  // would overlap the label above (hover shows it)
        floor = at.y + line;
        const std::string value = Thousands(static_cast<long long>(std::llround(nd.value)));
        const float name_w = ImGui::CalcTextSize(nd.name.c_str()).x, value_w = ImGui::CalcTextSize(value.c_str()).x;
        dl->AddRectFilled(ImVec2(at.x - 3.0f, at.y - 1.0f), ImVec2(at.x + name_w + 6.0f + value_w + 3.0f, at.y + line + 1.0f),
                          ui::ToU32(ui::WithAlpha(t.plot_bg, 0.72f)), 3.0f);
        dl->AddText(at, ui::ToU32(t.text_bright), nd.name.c_str());
        dl->AddText(ImVec2(at.x + name_w + 6.0f, at.y), ui::ToU32(t.text_dim), value.c_str());
    }
    for (size_t k = 0; k < steps; ++k)
        dl->AddText(ImVec2(sx(static_cast<int>(k)), pp.y + 4.0f), ui::ToU32(t.text_dim), p.sankey_steps[k].c_str());
}

// Treemap: nested rectangles; click a group to zoom into it, right-click or
// the path at the top to go back.
void PlotView::DrawTreemap() {
    const Prepared& p = data_;
    const ui::Tokens& t = ui::CurrentTokens();
    ImDrawList* dl = ImPlot::GetPlotDrawList();
    const ImVec2 pp = ImPlot::GetPlotPos(), ps = ImPlot::GetPlotSize();
    const float line = ImGui::GetTextLineHeight();
    const ImVec2 mouse = ImGui::GetMousePos();
    const bool hovered = ImPlot::IsPlotHovered();
    // The path zoomed into: "All > Asia", each part clickable.
    float crumb_h = 0;
    if (!tree_zoom_.empty()) {
        crumb_h = line + 8.0f;
        float x = pp.x + 6.0f;
        for (size_t k = 0; k <= tree_zoom_.size(); ++k) {
            const std::string part = k == 0 ? std::string("All") : tree_zoom_[k - 1];
            const ImVec2 sz = ImGui::CalcTextSize(part.c_str());
            const bool last = k == tree_zoom_.size();
            const bool over = hovered && !last && mouse.x >= x && mouse.x <= x + sz.x && mouse.y >= pp.y + 4 && mouse.y <= pp.y + 4 + sz.y;
            dl->AddText(ImVec2(x, pp.y + 4.0f), ui::ToU32(last ? t.text_bright : over ? t.accent_text : t.text_dim), part.c_str());
            if (over && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                tree_zoom_.resize(k);
                return;
            }
            x += sz.x;
            if (!last) {
                dl->AddText(ImVec2(x + 4.0f, pp.y + 4.0f), ui::ToU32(t.text_faint), ">");
                x += ImGui::CalcTextSize(">").x + 8.0f;
            }
        }
    }
    const float header = line + 6.0f;
    tree_rects_ = TreemapLayout(p, tree_zoom_, ps.x - 2.0f, ps.y - crumb_h - 2.0f, header);
    for (auto& r : tree_rects_) {
        r.x += pp.x + 1.0;
        r.y += pp.y + crumb_h + 1.0;
    }
    hot_rect_ = -1;
    for (size_t i = 0; i < tree_rects_.size() && hovered; ++i) {
        const auto& r = tree_rects_[i];
        if (mouse.x >= r.x && mouse.x < r.x + r.w && mouse.y >= r.y && mouse.y < r.y + r.h) hot_rect_ = static_cast<int>(i);  // deepest wins
    }
    const ImVec4 group_fill = ui::Mix(t.plot_bg, t.text_dim, 0.14f);
    for (size_t i = 0; i < tree_rects_.size(); ++i) {
        const auto& r = tree_rects_[i];
        const ImVec2 a(static_cast<float>(r.x), static_cast<float>(r.y)), b(static_cast<float>(r.x + r.w), static_cast<float>(r.y + r.h));
        if (!r.leaf) {
            dl->AddRectFilled(a, b, ui::ToU32(group_fill), 3.0f);
            if (r.h > header * 2.6 && r.w > header * 3.0) {
                const std::string text = r.path.back() + "  " + Thousands(static_cast<long long>(std::llround(r.size)));
                dl->PushClipRect(a, b, true);
                dl->AddText(ImVec2(a.x + 5.0f, a.y + 3.0f), ui::ToU32(t.text_bright), r.path.back().c_str());
                dl->AddText(ImVec2(a.x + 5.0f + ImGui::CalcTextSize(r.path.back().c_str()).x + 8.0f, a.y + 3.0f), ui::ToU32(t.text_dim),
                            Thousands(static_cast<long long>(std::llround(r.size))).c_str());
                dl->PopClipRect();
            }
            continue;
        }
        ImVec4 fill = p.colour_scale ? ScaleColourOf(p, r.colour) : SeriesColour(static_cast<size_t>(r.top));
        if (hot_rect_ == static_cast<int>(i)) fill = ui::Mix(fill, t.text_bright, 0.18f);
        dl->AddRectFilled(ImVec2(a.x + 0.5f, a.y + 0.5f), ImVec2(b.x - 0.5f, b.y - 0.5f), ui::ToU32(fill), 2.0f);
        if (r.w > 40 && r.h > line + 6) {
            dl->PushClipRect(a, b, true);
            dl->AddText(ImVec2(a.x + 4.0f, a.y + 3.0f), InkOn(fill), r.path.back().c_str());
            if (r.h > line * 2 + 8)
                dl->AddText(ImVec2(a.x + 4.0f, a.y + 3.0f + line), InkOn(fill), Thousands(static_cast<long long>(std::llround(r.size))).c_str());
            dl->PopClipRect();
        }
    }
    // Click: zoom into the group under the mouse; right-click: back one level.
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hot_rect_ >= 0) {
        const auto& r = tree_rects_[static_cast<size_t>(hot_rect_)];
        std::vector<std::string> target = r.path;
        if (r.leaf) target.pop_back();
        if (target.size() > tree_zoom_.size() && target.size() < p.tree_levels.size()) tree_zoom_ = target;
        else if (r.leaf && r.path.size() > tree_zoom_.size() + 1 && r.path.size() >= 2) tree_zoom_.assign(r.path.begin(), r.path.begin() + static_cast<std::ptrdiff_t>(tree_zoom_.size()) + 1);
    }
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !tree_zoom_.empty()) tree_zoom_.pop_back();
}

// Maps: the countries (regions coloured by their value), separated by thin
// gaps in the plot colour.
void PlotView::DrawWorld(bool regions) {
    const Prepared& p = data_;
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& countries = WorldCountries();
    ImDrawList* dl = ImPlot::GetPlotDrawList();
    const ImPlotRect lim = ImPlot::GetPlotLimits();
    const ImVec4 land = ui::Mix(t.plot_bg, t.text_dim, 0.32f);
    const ImPlotPoint m = ImPlot::GetPlotMousePos();
    hot_country_ = regions && ImPlot::IsPlotHovered() ? CountryAt(m.x, m.y) : -1;
    std::vector<ImVec2> pts;
    ImPlot::PushPlotClipRect();
    // The sea: the world's extent a shade off the plot colour.
    dl->AddRectFilled(ImPlot::PlotToPixels(-180, 90), ImPlot::PlotToPixels(180, -90), ui::ToU32(ui::Mix(t.plot_bg, t.text_dim, 0.07f)));
    for (size_t i = 0; i < countries.size(); ++i) {
        const Country& c = countries[i];
        if (c.lon_max < lim.X.Min || c.lon_min > lim.X.Max || c.lat_max < lim.Y.Min || c.lat_min > lim.Y.Max) continue;
        ImVec4 fill = land;
        if (regions && i < p.region_value.size() && std::isfinite(p.region_value[i])) fill = RegionColourOf(p, p.region_value[i]);
        for (const auto& ring : c.rings) {
            pts.clear();
            for (size_t k = 0; k + 1 < ring.size(); k += 2) pts.push_back(ImPlot::PlotToPixels(ring[k], ring[k + 1]));
            if (pts.size() < 3) continue;
            dl->AddConcavePolyFilled(pts.data(), static_cast<int>(pts.size()), ui::ToU32(fill));
            const bool hot = static_cast<int>(i) == hot_country_;
            dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), ui::ToU32(hot ? t.text_bright : t.plot_bg), ImDrawFlags_Closed, hot ? 1.6f : 0.8f);
        }
    }
    ImPlot::PopPlotClipRect();
}

void PlotView::DetectClick() {
    const Prepared& p = data_;
    const ImPlotPoint m = ImPlot::GetPlotMousePos();
    Click c;
    c.field = p.spec.x_column;
    if (c.field.empty()) return;
    switch (p.spec.kind) {
        case Kind::Bar: {
            const int i = static_cast<int>(std::lround(m.x));
            if (i < 0 || i >= static_cast<int>(p.categories.size()) || std::fabs(m.x - i) > 0.45) return;
            if (p.categories[static_cast<size_t>(i)] == "other") return;  // not one value
            c.value = p.categories[static_cast<size_t>(i)];
            break;
        }
        case Kind::Pie: {
            if (p.series.empty()) return;
            const double dx = m.x - 0.5, dy = m.y - 0.5;
            if (dx * dx + dy * dy > 0.16) return;
            double angle = std::atan2(dy, dx) * 180.0 / 3.141592653589793 - 90.0;
            while (angle < 0) angle += 360.0;
            double total = 0, acc = 0;
            for (double v : p.series[0].y) total += std::max(0.0, v);
            for (size_t k = 0; k < p.series[0].y.size() && total > 0; ++k) {
                acc += std::max(0.0, p.series[0].y[k]) / total * 360.0;
                if (angle <= acc) {
                    if (k >= p.categories.size() || p.categories[k] == "other") return;
                    c.value = p.categories[k];
                    break;
                }
            }
            if (c.value.empty()) return;
            break;
        }
        case Kind::Histogram: {
            if (p.edges.size() < 2 || m.x < p.edges.front() || m.x > p.edges.back()) return;
            const double width = p.edges[1] - p.edges[0];
            const size_t b = std::min(p.edges.size() - 2, static_cast<size_t>((m.x - p.edges.front()) / width));
            c.what = Click::What::Range;
            c.lo = p.edges[b];
            c.hi = p.edges[b + 1];
            break;
        }
        default: return;
    }
    click_ = c;
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
        case Kind::Parallel: {
            if (hovered_row_ < 0 || static_cast<size_t>(hovered_row_) >= p.multi_group.size()) break;
            const size_t r = static_cast<size_t>(hovered_row_);
            const int g = p.multi_group[r];
            begin(g >= 0 && static_cast<size_t>(g) < p.multi_groups.size() && !p.multi_groups[static_cast<size_t>(g)].empty()
                      ? p.multi_groups[static_cast<size_t>(g)]
                      : std::string("row"));
            for (size_t c = 0; c < p.multi_cols.size(); ++c)
                TooltipRow(SeriesColour(static_cast<size_t>(std::max(0, g))), p.multi_cols[c], Value(p.multi_values[c][r]));
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
        case Kind::Confusion: {
            const int c = static_cast<int>(std::floor(m.x));
            const int r = p.grid_rows - 1 - static_cast<int>(std::floor(m.y));
            if (c < 0 || r < 0 || c >= p.grid_cols || r >= p.grid_rows) break;
            const size_t k = static_cast<size_t>(r * p.grid_cols + c);
            const std::string& actual = p.row_names[static_cast<size_t>(r)];
            const std::string& predicted = p.col_names[static_cast<size_t>(c)];
            begin("Actual " + actual + " \xC2\xB7 Predicted " + predicted);
            const ImVec4 colour = GridColourOf(p, p.grid[k]);
            TooltipRow(colour, "rows", Thousands(static_cast<long long>(std::llround(p.grid_counts[k]))));
            char buf[64];
            switch (p.spec.confusion_show) {
                case PlotSpec::ConfusionShow::ByActual: std::snprintf(buf, sizeof(buf), "%.1f%% of actual ", p.grid[k] * 100.0); break;
                case PlotSpec::ConfusionShow::ByPredicted: std::snprintf(buf, sizeof(buf), "%.1f%% of predicted ", p.grid[k] * 100.0); break;
                case PlotSpec::ConfusionShow::All: std::snprintf(buf, sizeof(buf), "%.1f%% of all rows", p.grid[k] * 100.0); break;
                case PlotSpec::ConfusionShow::Counts: buf[0] = '\0'; break;
            }
            if (buf[0]) {
                std::string text = buf;
                if (p.spec.confusion_show == PlotSpec::ConfusionShow::ByActual) text += actual;
                if (p.spec.confusion_show == PlotSpec::ConfusionShow::ByPredicted) text += predicted;
                TooltipRow(colour, "share", text);
            }
            TooltipRow(colour, "prediction", r == c ? "right" : "wrong");
            break;
        }
        case Kind::Roc:
        case Kind::PrCurve:
        case Kind::Calibration:
        case Kind::Residuals: {
            // The nearest point within 14 px.
            if (p.series.empty()) break;
            const auto& sr = p.series[0];
            const ImVec2 mouse = ImGui::GetMousePos();
            float best = 196.0f;
            size_t bk = 0;
            bool found = false;
            for (size_t k = 0; k < std::min(sr.x.size(), sr.y.size()); ++k) {
                const ImVec2 q = ImPlot::PlotToPixels(sr.x[k], sr.y[k]);
                const float d = (q.x - mouse.x) * (q.x - mouse.x) + (q.y - mouse.y) * (q.y - mouse.y);
                if (d < best) {
                    best = d;
                    bk = k;
                    found = true;
                }
            }
            if (!found) break;
            const ImVec4 c = ColourOf(0);
            if (kind == Kind::Roc || kind == Kind::PrCurve) {
                const double th = bk < sr.c.size() ? sr.c[bk] : NAN;
                begin(std::isinf(th) ? std::string("threshold above every score") : "threshold " + Value(th));
                TooltipRow(c, kind == Kind::Roc ? "false positive rate" : "recall", Value(sr.x[bk]));
                TooltipRow(c, kind == Kind::Roc ? "true positive rate" : "precision", Value(sr.y[bk]));
            } else if (kind == Kind::Calibration) {
                begin("mean predicted " + Value(sr.x[bk]));
                TooltipRow(c, "share positive", Value(sr.y[bk]));
                TooltipRow(c, "rows", bk < sr.low.size() ? Thousands(static_cast<long long>(sr.low[bk])) : std::string());
            } else {
                begin("predicted " + Value(sr.x[bk]));
                TooltipRow(c, "actual", Value(sr.x[bk] + sr.y[bk]));
                TooltipRow(c, "residual", Value(sr.y[bk]));
            }
            break;
        }
        case Kind::LearningCurve: {
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& sr = p.series[i];
                if (sr.x.empty()) continue;
                const size_t k = sr.x_sorted ? NearestSorted(sr.x, m.x) : NearestScan(sr.x, m.x);
                begin(XLabel(p) + " " + Value(sr.x[k]));
                std::string v = Value(sr.y[k]);
                if (k < sr.high.size()) v += " \xC2\xB1 " + Value(sr.high[k] - sr.y[k]);
                if (static_cast<int>(i) == p.best_series && k == p.best_index) v += "  (best)";
                TooltipRow(ColourOf(i), sr.label, v);
            }
            break;
        }
        case Kind::Importance: {
            if (p.series.empty()) break;
            const auto& sr = p.series[0];
            const int at = static_cast<int>(std::lround(m.y));
            const int i = static_cast<int>(sr.y.size()) - 1 - at;
            if (at < 0 || i < 0 || i >= static_cast<int>(sr.y.size()) || std::fabs(m.y - at) > 0.45) break;
            const size_t k = static_cast<size_t>(i);
            begin(p.categories[k]);
            std::string v = Value(sr.y[k]);
            if (k < sr.low.size() && sr.low[k] > 0) v += " \xC2\xB1 " + Value(sr.low[k]);
            TooltipRow(ColourOf(0), sr.label, v);
            TooltipRow(ColourOf(0), "rank", std::to_string(k + 1) + " of " + std::to_string(sr.y.size()));
            break;
        }
        case Kind::Sankey: {
            const auto value_text = [&](double v, double of) {
                char share[32];
                std::snprintf(share, sizeof(share), "  (%.1f%%)", of > 0 ? 100.0 * v / of : 0.0);
                return Thousands(static_cast<long long>(std::llround(v))) + share;
            };
            const std::string what = p.spec.value_column.empty() ? "rows" : p.spec.value_column;
            if (hot_node_ >= 0) {
                const auto& nd = p.sankey_nodes[static_cast<size_t>(hot_node_)];
                begin(p.sankey_steps[static_cast<size_t>(nd.step)] + ": " + nd.name);
                TooltipRow(t.text, what, value_text(nd.value, p.sankey_total));
            } else if (hot_link_ >= 0) {
                const auto& l = p.sankey_links[static_cast<size_t>(hot_link_)];
                const auto& a = p.sankey_nodes[static_cast<size_t>(l.from)];
                const auto& b = p.sankey_nodes[static_cast<size_t>(l.to)];
                begin(a.name + " > " + b.name);
                TooltipRow(t.text, what, value_text(l.value, a.value));
                TooltipRow(t.text, "", "of " + a.name + " (" + p.sankey_steps[static_cast<size_t>(a.step)] + ")");
            }
            break;
        }
        case Kind::Treemap: {
            if (hot_rect_ < 0 || static_cast<size_t>(hot_rect_) >= tree_rects_.size()) break;
            const auto& r = tree_rects_[static_cast<size_t>(hot_rect_)];
            std::string path;
            for (const auto& part : r.path) path += (path.empty() ? "" : " > ") + part;
            begin(path);
            double total = 0;
            for (const auto& q : tree_rects_)
                if (q.depth == 0) total += q.size;
            char share[32];
            std::snprintf(share, sizeof(share), "  (%.1f%%)", total > 0 ? 100.0 * r.size / total : 0.0);
            const ImVec4 c = p.colour_scale && r.leaf ? ScaleColourOf(p, r.colour) : SeriesColour(static_cast<size_t>(r.top));
            TooltipRow(c, p.spec.value_column.empty() ? std::string("rows") : p.spec.value_column,
                       Thousands(static_cast<long long>(std::llround(r.size))) + share);
            if (p.colour_scale && r.leaf) TooltipRow(c, "mean of " + p.colour_label, std::isfinite(r.colour) ? Value(r.colour) : std::string("missing"));
            if (!r.leaf) ImGui::TextColored(t.text_dim, "Click to zoom in");
            else if (!tree_zoom_.empty()) ImGui::TextColored(t.text_dim, "Right-click to go back");
            break;
        }
        case Kind::MapPoints: {
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
            if (!found) break;
            const auto& sr = p.series[bs];
            begin(p.series.size() > 1 ? sr.label : Value(sr.y[bk]) + ", " + Value(sr.x[bk]));
            const ImVec4 c = p.colour_scale ? ScaleColourOf(p, bk < sr.c.size() ? sr.c[bk] : NAN) : ColourOf(bs);
            TooltipRow(c, XLabel(p), Value(sr.x[bk]));
            TooltipRow(c, YLabel(p), Value(sr.y[bk]));
            if (bk < sr.z.size()) TooltipRow(c, p.size_label, std::isfinite(sr.z[bk]) ? Value(sr.z[bk]) : std::string("missing"));
            if (p.colour_scale && bk < sr.c.size()) TooltipRow(c, p.colour_label, std::isfinite(sr.c[bk]) ? Value(sr.c[bk]) : std::string("missing"));
            const int country = CountryAt(sr.x[bk], sr.y[bk]);
            if (country >= 0) ImGui::TextColored(t.text_dim, "%s", WorldCountries()[static_cast<size_t>(country)].name.c_str());
            break;
        }
        case Kind::MapRegions: {
            if (hot_country_ < 0) break;
            const size_t i = static_cast<size_t>(hot_country_);
            const Country& c = WorldCountries()[i];
            begin(c.name + (c.iso_a3.empty() ? std::string() : " (" + c.iso_a3 + ")"));
            if (i < p.region_value.size() && std::isfinite(p.region_value[i])) {
                const std::string what = (p.spec.region_agg == PlotSpec::RegionAgg::Mean ? "mean of " : "") +
                                         (p.spec.y_columns.empty() ? std::string("value") : p.spec.y_columns.front());
                TooltipRow(RegionColourOf(p, p.region_value[i]), what, Value(p.region_value[i]));
                TooltipRow(RegionColourOf(p, p.region_value[i]), "rows", Thousands(static_cast<long long>(p.region_rows[i])));
            } else {
                ImGui::TextColored(t.text_dim, "No rows name this country.");
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
