// Properties panel editors for plugin custom node types (rows and tokens
// since TOFIX129 A7).

#include "properties_node_editors.h"
#include "node_editor.h"
#include "properties_rows.h"
#include "../core/file_dialogs.h"
#include "../core/extension_node_registry.h"
#include "../core/extension_node_presentation.h"
#include "icons.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

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
    const auto& t = cyxwiz::ui::CurrentTokens();
    if (ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthStretch, 0.35f);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 0.65f);
        for (const auto& [key, value] : rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", key.c_str());
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(value.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTable();
    }
}

std::string ParamOr(const MLNode& node, const char* key, const char* fallback = "") {
    const auto it = node.parameters.find(key);
    return it != node.parameters.end() ? it->second : fallback;
}

// The extension that provides this node is not loaded (approved mockup,
// "Properties - not installed"). Saved settings are shown, not edited.
void RenderMissingExtension(const MLNode& node, RenderNodePropertiesContext context) {
    const auto card = cyxwiz::BuildExtensionMissingCard(node);
    const auto& t = cyxwiz::ui::CurrentTokens();

    cyxwiz::ui::BeginCard("missing_extension");
    ImGui::TextColored(t.caution, "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, card.title.c_str());
    ImGui::PushTextWrapPos(0.0f);
    for (const auto& line : card.lines) ImGui::TextUnformatted(line.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    if (cyxwiz::ui::PrimaryButton("Open Plugin Manager", context.node_editor != nullptr,
                                  "The node editor is not available", cyxwiz::ui::ButtonSize::Small)) {
        context.node_editor->OpenPluginManager();
    }
    auto& open = OpenMissingDetails();
    const bool details_open = open.count(node.id) > 0;
    ImGui::SameLine();
    if (cyxwiz::ui::LinkButton(details_open ? "Hide" : "Details")) {
        if (details_open) open.erase(node.id); else open.insert(node.id);
    }
    if (open.count(node.id) > 0) {
        ImGui::Spacing();
        KeyValueTable("##details", card.details);
    }
    cyxwiz::ui::EndCard();

    ImGui::TextUnformatted("Settings saved with the graph");
    ImGui::TextColored(t.text_dim, "Shown as saved. They cannot be edited until the extension is loaded.");
    KeyValueTable("##saved_settings", card.settings);

    ImGui::Spacing();
    ImGui::TextUnformatted("Pins");
    KeyValueTable("##saved_pins", card.pins);
}

void PinTable(const char* id, const std::vector<NodePin>& pins, const char* type_name) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_RowBg)) return;
    ImGui::TableSetupColumn("Pin", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableHeadersRow();
    for (const auto& pin : pins) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(pin.name.c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "%s", type_name);
    }
    ImGui::EndTable();
}

// "_meta_*" keys written by the dynamic pin resolver, as a Model info block.
void ModelInfo(const MLNode& node) {
    std::vector<std::pair<std::string, std::string>> rows;
    for (const auto& [key, value] : node.parameters) {
        if (key.starts_with("_meta_")) rows.emplace_back(key.substr(6), value);
    }
    if (rows.empty()) return;
    ImGui::Spacing();
    ImGui::TextUnformatted("Model info");
    KeyValueTable("##model_info", rows);
}

void RenderMuJoCoPlant(MLNode& node, RenderNodePropertiesContext& context) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    const auto resolve_pins = [&]() {
        if (node.has_dynamic_pins && context.node_editor) context.node_editor->ResolveDynamicPins(node.id);
        context.invalidate_shapes();
    };
    std::string mjcf_path = ParamOr(node, "mjcf_path");
    {
        properties_rows::Rows rows("##mujoco");
        if (!rows.ok) return;
        // MJCF file with Browse
        properties_rows::Label("MJCF model");
        char path_buf[512];
        std::strncpy(path_buf, mjcf_path.c_str(), sizeof(path_buf) - 1);
        path_buf[sizeof(path_buf) - 1] = '\0';
        const float browse = cyxwiz::ui::ButtonWidth("Browse", cyxwiz::ui::ButtonSize::Small);
        ImGui::SetNextItemWidth(std::max(40.0f, ImGui::GetContentRegionAvail().x - browse - t.space_sm));
        if (ImGui::InputText("##mjcf_path", path_buf, sizeof(path_buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
            node.parameters["mjcf_path"] = mjcf_path = path_buf;
            resolve_pins();
        }
        ImGui::SameLine(0.0f, t.space_sm);
        if (cyxwiz::ui::SecondaryButton("Browse")) {
            if (auto selected = cyxwiz::FileDialogs::OpenFile(
                    "Select MJCF Model", {{"MJCF Files", "xml"}, {"All Files", "*"}},
                    mjcf_path.empty() ? nullptr : mjcf_path.c_str())) {
                node.parameters["mjcf_path"] = mjcf_path = *selected;
                resolve_pins();
            }
        }
        properties_rows::Status("mjcf_path");
        const std::string loaded = ParamOr(node, "_meta_loaded_path");
        if (!loaded.empty()) {
            properties_rows::Note(("Loaded from the Environment Library: " + loaded).c_str());
        } else if (mjcf_path.empty()) {
            properties_rows::Note("No model set.");
        }

        // Interface mode
        properties_rows::Label("Interface");
        int iface_idx = ParamOr(node, "interface", "bus") == "vector" ? 1 : 0;
        static const char* iface_items[] = {"Bus (per-actuator)", "Vector (single array)"};
        if (ImGui::Combo("##iface_mode", &iface_idx, iface_items, 2)) {
            node.parameters["interface"] = iface_idx == 1 ? "vector" : "bus";
            resolve_pins();
        }
        properties_rows::Status("interface");

        // Timestep
        properties_rows::Label("Timestep");
        float timestep = 0.002f;
        try { timestep = std::stof(ParamOr(node, "timestep", "0.002")); } catch (...) {}
        if (ImGui::InputFloat("##timestep", &timestep, 0.001f, 0.01f, "%.4f")) {
            if (timestep < 0.0001f) timestep = 0.0001f;
            node.parameters["timestep"] = std::to_string(timestep);
            context.invalidate_shapes();
        }
        properties_rows::Status("timestep");

        // Frame skip
        properties_rows::Label("Frame skip");
        int frame_skip = 1;
        try { frame_skip = std::stoi(ParamOr(node, "frame_skip", "1")); } catch (...) {}
        if (ImGui::InputInt("##frame_skip", &frame_skip)) {
            if (frame_skip < 1) frame_skip = 1;
            node.parameters["frame_skip"] = std::to_string(frame_skip);
            context.invalidate_shapes();
        }
        properties_rows::Status("frame_skip");
    }
    if (mjcf_path.empty() && ParamOr(node, "_meta_loaded_path").empty() && node.has_dynamic_pins && context.node_editor) {
        if (cyxwiz::ui::SecondaryButton("Sync from Environment Library")) resolve_pins();
        cyxwiz::ui::Tooltip("Load a model in the Environment Library first");
    }

    ModelInfo(node);

    // Pin summary
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "Actuator inputs: %d \xC2\xB7 Sensor outputs: %d",
                       static_cast<int>(node.inputs.size()), static_cast<int>(node.outputs.size()));
    if (!node.inputs.empty() && ImGui::TreeNode("Actuator pins")) {
        PinTable("##act_table", node.inputs, "Scalar");
        ImGui::TreePop();
    }
    if (!node.outputs.empty() && ImGui::TreeNode("Sensor pins")) {
        PinTable("##sens_table", node.outputs, "Tensor");
        ImGui::TreePop();
    }
}

}  // namespace

void RenderPluginCustomNodeProperties(MLNode& node, RenderNodePropertiesContext context) {
    if (node.type != NodeType::PluginCustom) return;
    const auto& t = cyxwiz::ui::CurrentTokens();
    if (!cyxwiz::ExtensionNodeRegistry::Instance().Has(node.extension_type_id)) {
        RenderMissingExtension(node, context);
        return;
    }
    const auto descriptor = cyxwiz::ExtensionNodeRegistry::Instance().Find(node.extension_type_id);
    std::string node_type_name;
    if (descriptor.has_value()) {
        node_type_name = descriptor->type_name;
        if (!descriptor->metadata.brief_description.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", descriptor->metadata.brief_description.c_str());
            ImGui::PopTextWrapPos();
        }
    }

    if (node_type_name == "MuJoCoPlant") {
        RenderMuJoCoPlant(node, context);
        return;
    }

    // Generic plugin node: one text row per parameter (internal keys hidden).
    {
        properties_rows::Rows rows("##plugin");
        if (!rows.ok) return;
        std::vector<std::string> keys;
        for (const auto& [key, value] : node.parameters) {
            if (!key.starts_with("_meta_")) keys.push_back(key);
        }
        for (const auto& key : keys) {
            char buf[512];
            std::strncpy(buf, node.parameters[key].c_str(), sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            properties_rows::Label(key.c_str());
            ImGui::PushID(key.c_str());
            if (ImGui::InputText("##v", buf, sizeof(buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
                node.parameters[key] = buf;
                if (node.has_dynamic_pins && key == node.dynamic_pin_trigger && context.node_editor) {
                    context.node_editor->ResolveDynamicPins(node.id);
                }
                context.invalidate_shapes();
            }
            ImGui::PopID();
            properties_rows::Status(key.c_str());
        }
    }
    ModelInfo(node);
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "Inputs: %d \xC2\xB7 Outputs: %d",
                       static_cast<int>(node.inputs.size()), static_cast<int>(node.outputs.size()));
}

} // namespace gui::properties_node_editors
