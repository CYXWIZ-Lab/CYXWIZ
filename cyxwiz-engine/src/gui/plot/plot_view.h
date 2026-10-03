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

private:
    void DrawToolbar(const Options& options);
    void DrawPlot(ImVec2 size);
    void DrawImages(ImVec2 size);     // Image: a grid of pictures (ImGui, not ImPlot)
    void DrawPairPlot(ImVec2 size);   // Pair plot: an ImPlot subplot grid
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
};

}  // namespace cyxwiz::plot
