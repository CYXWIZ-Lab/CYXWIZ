#pragma once

#include "../panel.h"
#include "../../scripting/scripting_engine.h"
#include <vector>
#include <memory>
#include <string>
#include <mutex>

// Use GLAD for cross-platform OpenGL loading
#include <glad/glad.h>

namespace cyxwiz {

/**
 * Plot Output Panel - displays matplotlib figures from script execution
 * Similar to MATLAB's figure window, shows plots captured during script runs
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
        bool selected = false;
    };

    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;
    std::vector<PlotEntry> plots_;
    int selected_plot_index_ = -1;
    bool auto_scroll_ = true;
    bool show_thumbnails_ = true;
    float thumbnail_size_ = 80.0f;

    // For polling script results
    bool was_script_running_ = false;

    // Thread safety
    mutable std::mutex plots_mutex_;

    // Create OpenGL texture from PNG data
    GLuint CreateTextureFromPNG(const std::vector<unsigned char>& png_data, int& out_width, int& out_height);

    // Delete texture
    void DeleteTexture(GLuint texture_id);

    // Render a single plot at full size
    void RenderSelectedPlot();

    // Render thumbnail gallery
    void RenderThumbnails();

    // Render toolbar
    void RenderToolbar();

    // Context menu for plot actions
    void RenderPlotContextMenu(int plot_index);

    // Copy plot to clipboard
    bool CopyToClipboard(int plot_index);

    // Save plot to file
    bool SaveToFile(int plot_index);

    // Poll for new plots from script execution
    void PollForNewPlots();
};

} // namespace cyxwiz
