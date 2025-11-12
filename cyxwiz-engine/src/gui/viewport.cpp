#include "viewport.h"
#include <imgui.h>

namespace gui {

Viewport::Viewport() : show_window_(true) {
}

Viewport::~Viewport() = default;

void Viewport::Render() {
    if (!show_window_) return;

    if (ImGui::Begin("Viewport", &show_window_)) {
        ImGui::Text("Training Visualization");
        ImGui::Separator();

        // TODO: Render real-time training plots
        // - Loss curve
        // - Accuracy curve
        // - Learning rate schedule
        // - GPU utilization
        // Use ImPlot library: https://github.com/epezent/implot

        ImGui::Text("Loss: 0.XXX");
        ImGui::Text("Accuracy: 0.XXX");
        ImGui::Text("Epoch: X/Y");

        // Placeholder for plot
        ImGui::BeginChild("PlotRegion", ImVec2(0, 300), true);
        ImGui::Text("Plot area (integrate ImPlot)");
        ImGui::EndChild();
    }
    ImGui::End();
}

} // namespace gui
