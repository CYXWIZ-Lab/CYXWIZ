#pragma once

#include "../panel.h"
#include "../../scripting/scripting_engine.h"
#include <vector>
#include <memory>
#include <string>

// Use GLAD for cross-platform OpenGL loading
#include <glad/glad.h>
#include <imgui.h>  // For ImVec2

namespace cyxwiz::plot {
class PlotWindow;
}

namespace cyxwiz {

/**
 * Plot Output - one place for what Python shows (TOFIX134 P5, board 18):
 * matplotlib figures (images: zoom, pan, copy, save) and plots made with the
 * cyxwiz module (live: each owns an Engine Plot window; the same title
 * updates it). UI thread only: both arrive through the ScriptingEngine's
 * inboxes, taken every frame.
 */
class PlotOutputPanel : public Panel {
public:
    PlotOutputPanel();
    ~PlotOutputPanel();

    void Render() override;

    // Set the scripting engine reference for polling results
    void SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine);

    // Manually add a plot (for external use)
    void AddPlot(const scripting::CapturedPlot& plot);

    // A notebook output's "Open in window": the image, selected and shown here.
    void ShowImage(const std::vector<unsigned char>& png_data, const std::string& title);

    // Clear all plots
    void ClearPlots();

    // Check if panel has any plots
    bool HasPlots() const { return !plots_.empty(); }

    // Get plot count
    size_t GetPlotCount() const { return plots_.size(); }

private:
    struct PlotEntry {
        GLuint texture_id = 0;
        int width = 0;
        int height = 0;
        std::string label;
        std::vector<unsigned char> png_data;  // For copy/save operations
        // A cyxwiz plot: its Plot window, what it is and where it came from.
        bool python = false;
        std::string kind_label;
        std::shared_ptr<std::string> source_text;
        std::unique_ptr<plot::PlotWindow> window;
    };

    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;
    std::vector<PlotEntry> plots_;
    int selected_plot_index_ = -1;
    bool auto_scroll_ = true;
    bool show_thumbnails_ = true;
    int filter_ = 0;  // 0 all, 1 figures, 2 Python plots
    int next_window_id_ = 1;
    bool focus_next_ = false;

    // Zoom and pan state
    float zoom_level_ = 1.0f;           // 1.0 = 100%, 2.0 = 200%
    ImVec2 pan_offset_ = ImVec2(0, 0);  // Pan offset in normalized coords (0-1)
    float min_zoom_ = 0.25f;            // 25% minimum
    float max_zoom_ = 10.0f;            // 1000% maximum
    bool is_panning_ = false;           // Currently dragging to pan

    // Create OpenGL texture from PNG data
    GLuint CreateTextureFromPNG(const std::vector<unsigned char>& png_data, int& out_width, int& out_height);

    // Delete texture
    void DeleteTexture(GLuint texture_id);

    // The entries the filter shows, in order.
    std::vector<int> Shown() const;
    void Select(int index);
    void RemoveEntry(int index);

    // A cyxwiz plot arrived: a new window, or the window of the same title updated.
    void AddPythonPlot(scripting::PythonPlot plot);
    void ClosePythonPlot(const std::string& title);

    // Render a single plot at full size
    void RenderSelectedPlot();
    // A cyxwiz plot selected: what it is, Show window, Close plot.
    void RenderPythonCard(PlotEntry& entry);

    // The list of figures and plots
    void RenderThumbnails();

    // Render toolbar
    void RenderToolbar();

    // Context menu for plot actions
    void RenderPlotContextMenu(int plot_index);

    // Zoom/pan controls
    void ResetZoom();
    void ZoomIn();
    void ZoomOut();
    void FitToWindow();
    void ActualSize();
    void HandleZoomPan();  // Mouse scroll/drag handling

    // Copy plot to clipboard
    bool CopyToClipboard(int plot_index);

    // Save plot to file
    bool SaveToFile(int plot_index);

    // Take the figures and plots scripts published (every frame, also while
    // the window is hidden; TOFIX134 P0 item 6, P5)
    void PollForNewPlots();
};

} // namespace cyxwiz
