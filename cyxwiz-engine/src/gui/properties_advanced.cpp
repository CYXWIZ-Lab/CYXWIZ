#include "properties_advanced.h"
#include "node_editor.h"
#include <imgui.h>

namespace gui::properties_advanced {

bool RenderAdvancedSection(NodeEditor* node_editor, MLNode& node, bool section_open) {
    ImGui::SetNextItemOpen(section_open, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Advanced")) {
        if (node.has_initial_position) {
            ImGui::Text("Initial Position: (%.1f, %.1f)", node.initial_pos_x, node.initial_pos_y);
        }

        if (node_editor) {
            const auto& links = node_editor->GetLinks();
            int input_count = 0;
            int output_count = 0;
            for (const auto& link : links) {
                if (link.to_node == node.id) input_count++;
                if (link.from_node == node.id) output_count++;
            }
            ImGui::Text("Connections: %d in, %d out", input_count, output_count);
        }

        if (ImGui::TreeNode("Raw Parameters")) {
            for (const auto& [key, value] : node.parameters) {
                ImGui::Text("%s: %s", key.c_str(), value.c_str());
            }
            ImGui::TreePop();
        }

        return true;
    }

    return false;
}

} // namespace gui::properties_advanced
