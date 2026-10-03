#pragma once

// One plot view for every screen (TOFIX134 P1 step 1.4): draws a prepared
// plot with ImPlot in the theme colours, with a toolbar (data label, Fit,
// Log Y, Legend, Export, own window), hover values and a legend. Screens
// hand it a Prepared snapshot (core/plot) and only call Draw.

#include "../../core/plot/plot_export.h"
#include "../../core/plot/plot_model.h"

#include <imgui.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "../../core/plot/plot_layout.h"
#include <vector>

namespace cyxwiz::plot {

// Things only the application can do (GL read-back, OS dialogs, the
// clipboard image). Set once by the application; screens built without it
// (tests) show those export items disabled.
struct ViewHooks {
    // Reads a screen rectangle as PNG after this frame is drawn.
    std::function<void(ImVec2 min, ImVec2 max, std::function<void(std::vector<unsigned char>)> done)> capture_png;
    // Save dialog: title, extension ("png"), default name -> chosen path.
    std::function<std::optional<std::string>(const char*, const char*, const std::string&)> save_path;
    std::function<bool(const std::vector<unsigned char>&)> copy_png;
};
void SetViewHooks(ViewHooks hooks);
const ViewHooks& Hooks();

// A model figure as text ("AUC 0.871", accuracy "77.5%", rows "2,575").
std::string MetricText(const std::string& name, double value);

class PlotView {
public:
    explicit PlotView(std::string id);

    struct Options {
        bool toolbar = true;
        bool show_title = false;          // the spec title above the plot
        std::string export_name = "plot"; // default file name
        // Text for "Plot with Python (copy script)"; the item is hidden
        // when not set.
        std::function<std::string()> python_script;
        // Extra toolbar content drawn at the left (a screen's own controls).
        std::function<void()> toolbar_left;
        // First theme series colour (a screen with one chart per metric
        // gives each its own colour).
        size_t colour_offset = 0;
        // Fixed axis ranges (a screen that follows its own window, e.g. the
        // Training Dashboard's current epochs); off: fit to the data.
        struct Range {
            bool on = false;
            double lo = 0, hi = 1;
            bool once = false;  // set the first time only (the user may pan after)
        };
        Range x_range, y_range;
        // Toolbar items a screen already has its own controls for.
        bool tool_fit = true;
        bool tool_log = true;
        bool tool_legend = true;
        bool own_window_button = true;
    };

    // A click on a bar, a pie slice or a histogram bin (not a drag): what
    // was clicked, for a dashboard's cross-filter. TakeClick() hands it over once.
    struct Click {
        enum class What { Category, Range } what = What::Category;
        std::string field;   // the column it is a value of
        std::string value;   // Category
        double lo = 0, hi = 0;  // Range
    };
    // A dashboard's context: the same plot over all rows drawn in grey
    // behind this one (when filters apply), and the values or range this
    // widget selected (drawn in full colour, the rest dimmed).
    void SetBackground(std::shared_ptr<const Prepared> all_rows) { background_ = std::move(all_rows); }
    void SetHighlight(std::vector<std::string> categories, double lo = NAN, double hi = NAN) {
        highlight_ = std::move(categories);
        highlight_lo_ = lo;
        highlight_hi_ = hi;
    }
    std::optional<Click> TakeClick() {
        auto c = click_;
        click_.reset();
        return c;
    }

    void SetData(Prepared data);
    const Prepared& Data() const { return data_; }
    bool HasData() const { return has_data_; }
    void Clear();
    // Fit the axes to the data on the next frame.
    void RequestFit() { fit_ = true; }

    // Draws in the current window. size.y <= 0 fills the remaining height.
    void Draw(ImVec2 size, const Options& options);
    // Own window ("Open in its own window"); call every frame, also when the
    // host screen is hidden, so the window stays.
    void DrawOwnWindow(const Options& options);
    bool own_window = false;

    // 3D: turns the view (degrees; NaN elevation: the default view). The view
    // the user turns to is reported once they let go (to save in the plot).
    void SetView(double elevation, double azimuth);
    // Network / Tree: back to the computed layout (moved nodes and folds undone).
    void ResetGraphLayout();
    // The colour series i is drawn with (a picked colour, else the theme's).
    ImVec4 DrawnColour(size_t i) const { return ColourOf(i); }
    std::function<void(double elevation, double azimuth)> on_view_changed;

private:
    void DrawToolbar(const Options& options);
    void DrawPlot(ImVec2 size);
    void Draw3D(ImVec2 size);         // Scatter 3D, Line 3D, Surface, Mesh (plot_view3d.cpp)
    void DrawGraph(ImVec2 size);      // Network, Tree (plot_view_graph.cpp)
    void DrawImages(ImVec2 size);     // Image: a grid of pictures (ImGui, not ImPlot)
    void DrawPairPlot(ImVec2 size);   // Pair plot: an ImPlot subplot grid
    void DrawSankey();                // Sankey, Treemap: drawn in pixels inside the plot
    void DrawTreemap();
    void DrawWorld(bool regions);     // Maps: the built-in country outlines
    void DrawHover();
    ImVec4 ColourOf(size_t i) const;
    void ExportMenu(const Options& options);
    void FinishCaptures();
    std::string Title() const;

    std::string id_;
    Prepared data_;
    bool has_data_ = false;
    bool fit_ = true;
    bool log_y_ = false;
    bool legend_ = true;
    AxisRange range_;            // visible limits last frame (SVG export)
    ImVec2 frame_min_{}, frame_max_{};
    struct Pending {
        enum class Action { None, Save, Copy } action = Action::None;
        bool ready = false;
        std::vector<unsigned char> png;
    };
    std::shared_ptr<Pending> pending_ = std::make_shared<Pending>();
    std::string note_;           // short result line ("Saved ...")
    double note_until_ = 0;
    std::string export_name_ = "plot";
    int capture_frame_ = 0;      // frame at which a requested image is read
    bool drawing_own_window_ = false;
    size_t colour_offset_ = 0;
    Options::Range x_range_, y_range_;
    int hovered_row_ = -1;      // parallel coordinates: the line under the mouse
    std::optional<Click> click_;
    std::shared_ptr<const Prepared> background_;
    std::vector<std::string> highlight_;
    double highlight_lo_ = NAN, highlight_hi_ = NAN;
    void DetectClick();
    int hot_node_ = -1, hot_link_ = -1;          // sankey: under the mouse
    std::vector<TreeRect> tree_rects_;           // treemap: last frame's layout (pixels)
    int hot_rect_ = -1;
    std::vector<std::string> tree_zoom_;         // treemap: the group zoomed into (empty: all)
    int hot_country_ = -1;                       // map regions: under the mouse
    // 3D: a view preset to apply, the view last reported, and this frame's
    // drawn points (screen position and values) for hover.
    double view_el_ = NAN, view_az_ = NAN;
    bool view_apply_ = false;
    double saved_el_ = NAN, saved_az_ = NAN;
    struct Hover3D {
        ImVec2 pos;
        double x, y, z, c;
    };
    std::vector<Hover3D> hover3d_;
    // Network / Tree: positions (x right, y up) after moves and folds.
    std::vector<ImVec2> graph_pos_;
    std::vector<char> graph_moved_, graph_fold_;
    bool graph_layout_dirty_ = true;
    int hot_graph_ = -1, graph_drag_ = -1;
    ImVec2 graph_press_{};
    ImVec2 hot_min_{}, hot_max_{};   // the node under the mouse last frame (pixels): where a press grabs it
    bool graph_dragging_ = false;
};

}  // namespace cyxwiz::plot
