// Properties panel editors for plugin custom node types.

#include "properties_node_editors.h"
#include "node_editor.h"
#include "../core/file_dialogs.h"
#include "../plugin/registries/plugin_node_registry.h"

#include <imgui.h>

#include <cstring>
#include <string>

namespace gui::properties_node_editors {

void RenderPluginCustomNodeProperties(MLNode& node, RenderNodePropertiesContext context) {
    switch (node.type) {
        case NodeType::PluginCustom: {
            // Get plugin info for display
            auto info_opt = cyxwiz::plugin::PluginNodeRegistry::Instance().GetNodeTypeInfoCopy(
                node.plugin_qualified_name);

            std::string node_type_name;
            if (info_opt.has_value()) {
                const auto& info = info_opt.value();
                node_type_name = info.type_name;
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.8f, 1.0f), "%s", info.display_name.c_str());
                if (!info.description.empty()) {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s", info.description.c_str());
                }
                ImGui::Separator();
            }

            // ===== MuJoCo Plant - Custom Properties UI =====
            if (node_type_name == "MuJoCoPlant") {
                // MJCF File Path with Browse button
                ImGui::Text("MJCF Model:");
                std::string& mjcf_path = node.parameters["mjcf_path"];
                char path_buf[512];
                strncpy(path_buf, mjcf_path.c_str(), sizeof(path_buf) - 1);
                path_buf[sizeof(path_buf) - 1] = '\0';
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 70.0f);
                if (ImGui::InputText("##mjcf_path", path_buf, sizeof(path_buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
                    mjcf_path = path_buf;
                    if (node.has_dynamic_pins && context.node_editor) {
                        context.node_editor->ResolveDynamicPins(node.id);
                    }
                    context.invalidate_shapes();
                }
                ImGui::SameLine();
                if (ImGui::Button("Browse")) {
                    if (auto selected = cyxwiz::FileDialogs::OpenFile(
                            "Select MJCF Model", {{"MJCF Files", "xml"}, {"All Files", "*"}},
                            mjcf_path.empty() ? nullptr : mjcf_path.c_str())) {
                        mjcf_path = *selected;
                        if (node.has_dynamic_pins && context.node_editor) {
                            context.node_editor->ResolveDynamicPins(node.id);
                        }
                        context.invalidate_shapes();
                    }
                }

                // Show loaded model status from Environment Library
                {
                    auto meta_path = node.parameters.find("_meta_loaded_path");
                    if (meta_path != node.parameters.end() && !meta_path->second.empty()) {
                        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.5f, 1.0f),
                            "Loaded from Environment Library:");
                        ImGui::TextWrapped("%s", meta_path->second.c_str());
                    } else if (mjcf_path.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                            "No model set.");
                        if (node.has_dynamic_pins && context.node_editor) {
                            if (ImGui::Button("Sync from Environment Library")) {
                                context.node_editor->ResolveDynamicPins(node.id);
                                context.invalidate_shapes();
                            }
                            ImGui::SameLine();
                            ImGui::TextDisabled("(Load a model in the Env Library first)");
                        }
                    }
                }

                ImGui::Spacing();

                // Interface mode dropdown
                std::string& iface = node.parameters["interface"];
                if (iface.empty()) iface = "bus";
                int iface_idx = (iface == "vector") ? 1 : 0;
                ImGui::Text("Interface:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                const char* iface_items[] = { "Bus (per-actuator)", "Vector (single array)" };
                if (ImGui::Combo("##iface_mode", &iface_idx, iface_items, 2)) {
                    iface = (iface_idx == 1) ? "vector" : "bus";
                    if (node.has_dynamic_pins && context.node_editor) {
                        context.node_editor->ResolveDynamicPins(node.id);
                    }
                    context.invalidate_shapes();
                }

                // Timestep
                std::string& ts = node.parameters["timestep"];
                if (ts.empty()) ts = "0.002";
                float timestep = std::stof(ts);
                ImGui::Text("Timestep:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100.0f);
                if (ImGui::InputFloat("##timestep", &timestep, 0.001f, 0.01f, "%.4f")) {
                    if (timestep < 0.0001f) timestep = 0.0001f;
                    ts = std::to_string(timestep);
                }

                // Frame skip
                std::string& fs = node.parameters["frame_skip"];
                if (fs.empty()) fs = "1";
                int frame_skip = std::stoi(fs);
                ImGui::Text("Frame Skip:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100.0f);
                if (ImGui::InputInt("##frame_skip", &frame_skip)) {
                    if (frame_skip < 1) frame_skip = 1;
                    fs = std::to_string(frame_skip);
                }

                // Model info (from dynamic pin metadata)
                bool has_meta = false;
                for (const auto& [key, value] : node.parameters) {
                    if (key.starts_with("_meta_")) {
                        if (!has_meta) {
                            ImGui::Spacing();
                            ImGui::Separator();
                            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Model Info:");
                            has_meta = true;
                        }
                        std::string display_key = key.substr(6);
                        ImGui::Text("  %s: %s", display_key.c_str(), value.c_str());
                    }
                }

                // Pin summary
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Text("Actuator Inputs: %d", static_cast<int>(node.inputs.size()));
                ImGui::Text("Sensor Outputs: %d", static_cast<int>(node.outputs.size()));

                // Actuator table
                if (!node.inputs.empty() && ImGui::TreeNode("Actuator Pins")) {
                    if (ImGui::BeginTable("##act_table", 2, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Pin", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                        ImGui::TableHeadersRow();
                        for (const auto& pin : node.inputs) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", pin.name.c_str());
                            ImGui::TableNextColumn();
                            ImGui::TextDisabled("Scalar");
                        }
                        ImGui::EndTable();
                    }
                    ImGui::TreePop();
                }

                if (!node.outputs.empty() && ImGui::TreeNode("Sensor Pins")) {
                    if (ImGui::BeginTable("##sens_table", 2, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Pin", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                        ImGui::TableHeadersRow();
                        for (const auto& pin : node.outputs) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", pin.name.c_str());
                            ImGui::TableNextColumn();
                            ImGui::TextDisabled("Tensor");
                        }
                        ImGui::EndTable();
                    }
                    ImGui::TreePop();
                }
            }
            // ===== Generic Plugin Node Properties =====
            else {
                // Render editable parameters (skip internal keys)
                for (auto& [key, value] : node.parameters) {
                    if (key == "plugin_qualified_name") continue;
                    if (key.starts_with("_meta_")) continue;

                    char buf[512];
                    strncpy(buf, value.c_str(), sizeof(buf) - 1);
                    buf[sizeof(buf) - 1] = '\0';

                    ImGui::Text("%s:", key.c_str());
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(200.0f);
                    std::string label = "##plugin_param_" + key;
                    if (ImGui::InputText(label.c_str(), buf, sizeof(buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
                        value = buf;

                        if (node.has_dynamic_pins && key == node.dynamic_pin_trigger && context.node_editor) {
                            context.node_editor->ResolveDynamicPins(node.id);
                        }

                        context.invalidate_shapes();
                    }
                }

                // Show dynamic pin metadata if available
                bool has_meta = false;
                for (const auto& [key, value] : node.parameters) {
                    if (key.starts_with("_meta_")) {
                        if (!has_meta) {
                            ImGui::Separator();
                            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Model Info:");
                            has_meta = true;
                        }
                        std::string display_key = key.substr(6);
                        ImGui::Text("  %s: %s", display_key.c_str(), value.c_str());
                    }
                }

                // Show pin summary
                ImGui::Separator();
                ImGui::Text("Inputs: %d  Outputs: %d",
                            static_cast<int>(node.inputs.size()),
                            static_cast<int>(node.outputs.size()));
            }
            break;
        }
        default:
            break;
    }
}

} // namespace gui::properties_node_editors
