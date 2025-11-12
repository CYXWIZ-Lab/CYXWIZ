#include "properties.h"
#include <imgui.h>

namespace gui {

Properties::Properties() : show_window_(true) {
}

Properties::~Properties() = default;

void Properties::Render() {
    if (!show_window_) return;

    if (ImGui::Begin("Properties", &show_window_)) {
        ImGui::Text("Selected Layer/Node Properties");
        ImGui::Separator();

        // TODO: Show properties of selected node
        // Example properties:
        ImGui::Text("Layer Type: Dense");
        ImGui::InputInt("Units", &units_);
        ImGui::InputFloat("Learning Rate", &learning_rate_, 0.001f, 0.01f, "%.4f");

        ImGui::Spacing();
        if (ImGui::Button("Apply")) {
            // TODO: Apply changes
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            // TODO: Reset to defaults
        }
    }
    ImGui::End();
}

} // namespace gui
