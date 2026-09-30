// Properties panel editors for plugin custom node types.

#include "properties_node_editors.h"
#include "node_editor.h"
#include "../core/file_dialogs.h"
#include "../core/extension_node_registry.h"
#include "../core/extension_node_presentation.h"
#include "icons.h"
#include "ui_buttons.h"

#include <imgui.h>

#include <cstring>
#include <set>
#include <string>

namespace gui::properties_node_editors {

namespace {

// Nodes whose Details are open in the not-installed card, by node id.
std::set<int>& OpenMissingDetails() {
    static std::set<int> open;
    return open;
}

void KeyValueTable(const char* id, const std::vector<std::pair<std::string, std::string>>& rows) {
    if (rows.empty()) return;
    if (ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthStretch, 0.35f);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 0.65f);
        for (const auto& [key, value] : rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", key.c_str());
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(value.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTable();
    }
}

// The extension that provides this node is not loaded (approved mockup,
// "Properties - not installed"). Saved settings are shown, not edited.
void RenderMissingExtension(const MLNode& node, RenderNodePropertiesContext context) {
    const auto card = cyxwiz::BuildExtensionMissingCard(node);
    const ImVec4 warning(0.96f, 0.63f, 0.29f, 1.0f);

    ImGui::TextUnformatted(node.name.c_str());
    ImGui::TextDisabled("Extension node");
    ImGui::Spacing();

    ImGui::PushID("missing_extension");
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.42f, 0.29f, 0.12f, 1.0f));
    if (ImGui::BeginChild("##card", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
                              ImGuiChildFlags_AlwaysUseWindowPadding)) {
        ImGui::TextColored(warning, "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, card.title.c_str());
        ImGui::PushTextWrapPos(0.0f);
        for (const auto& line : card.lines) ImGui::TextUnformatted(line.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Spacing();

        if (cyxwiz::ui::PrimaryButton("Open Plugin Manager", context.node_editor != nullptr,
                                      "The node editor is not available")) {
            context.node_editor->OpenPluginManager();
        }
        auto& open = OpenMissingDetails();
        const bool details_open = open.count(node.id) > 0;
        ImGui::SameLine();
        if (cyxwiz::ui::LinkButton(details_open ? "Hide" : "Details")) {
            if (details_open) open.erase(node.id); else open.insert(node.id);
        }
        if (open.count(node.id) > 0) {
            ImGui::Separator();
            KeyValueTable("##details", card.details);
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopID();

    ImGui::Spacing();
    ImGui::TextUnformatted("Settings saved with the graph");
    ImGui::TextDisabled("Shown as saved. They cannot be edited until the extension is loaded.");
    KeyValueTable("##saved_settings", card.settings);

    ImGui::Spacing();
    ImGui::TextUnformatted("Pins");
    KeyValueTable("##saved_pins", card.pins);
}

}  // namespace

void RenderPluginCustomNodeProperties(MLNode& node, RenderNodePropertiesContext context) {
    switch (node.type) {
        case NodeType::PluginCustom: {
            if (!cyxwiz::ExtensionNodeRegistry::Instance().Has(node.extension_type_id)) {
                RenderMissingExtension(node, context);
                break;
            }
            // Get plugin info for display
            const auto descriptor =
                cyxwiz::ExtensionNodeRegistry::Instance().Find(node.extension_type_id);

            std::string node_type_name;
            if (descriptor.has_value()) {
                node_type_name = descriptor->type_name;
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.8f, 1.0f), "%s",
                                   descriptor->metadata.name.c_str());
                if (!descriptor->metadata.brief_description.empty()) {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s",
                                       descriptor->metadata.brief_description.c_str());
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
                if (cyxwiz::ui::SecondaryButton("Browse")) {
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
