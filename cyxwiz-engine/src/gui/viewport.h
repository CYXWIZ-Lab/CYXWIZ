#pragma once

#include <memory>

namespace cyxwiz {
    class TrainingPlotPanel;
}

namespace gui {

class Viewport {
public:
    Viewport();
    ~Viewport();

    void Render();

    // Set training panel for live metrics display
    void SetTrainingPanel(cyxwiz::TrainingPlotPanel* panel) { training_panel_ = panel; }

    // Visibility control for sidebar integration
    bool* GetVisiblePtr() { return &show_window_; }

private:
    bool show_window_;
    cyxwiz::TrainingPlotPanel* training_panel_ = nullptr;
};

} // namespace gui
