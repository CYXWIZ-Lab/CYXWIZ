// Properties panel editors for plugin custom node types (rows and tokens
// since TOFIX129 A7).

#include "properties_node_editors.h"
#include "node_editor.h"
#include "properties_rows.h"
#include "properties_extension_node.h"
#include "../core/file_dialogs.h"
#include "../core/extension_node_registry.h"
#include "../core/extension_node_presentation.h"
#include "../core/project_manager.h"
#include "../plugin/registries/plugin_panel_registry.h"
#include "file_browser_launch.h"
#include "python_bundle_ui.h"
#include "icons.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include <imgui.h>

#include <cstring>
#include <filesystem>
#include <map>
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

    if (!node.extension_content_hash.empty()) {
        // A Python bundle node (saved with its code hash): show where bundles go.
        const auto roots = cyxwiz::python_ui::CurrentRoots();
        const auto folder = roots.project_nodes.empty() ? roots.user_nodes : roots.project_nodes;
        if (cyxwiz::ui::PrimaryButton("Open nodes folder", true, nullptr, cyxwiz::ui::ButtonSize::Small)) {
            std::error_code ec;
            std::filesystem::create_directories(folder, ec);
            cyxwiz::LaunchAssetInFileBrowser(folder.string(), true);
        }
    } else if (cyxwiz::ui::PrimaryButton("Open Plugin Manager", context.node_editor != nullptr,
                                         "The node editor is not available", cyxwiz::ui::ButtonSize::Small)) {
        context.node_editor->OpenPluginManager();
    }
    auto& open = OpenMissingDetails();
    const bool details_open = open.count(node.id) > 0;
    if (properties_rows::Disclosure("Details##missing", details_open) != details_open) {
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
        if (!key.starts_with("_meta_")) continue;
        // The model's reference, name and where it is show above (the Model row)
        const std::string name = key.substr(6);
        if (name == "model_ref" || name == "model_title" || name == "model_status" || name == "model_note" ||
            name == "loaded_path")
            continue;
        rows.emplace_back(name, value);
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
    // The plugin names the model the same way on every machine (TOFIX219): a
    // picked file is saved by that reference
    const std::string model_ref = ParamOr(node, "_meta_model_ref");
    if (!mjcf_path.empty() && !model_ref.empty() && model_ref != mjcf_path) {
        node.parameters["mjcf_path"] = mjcf_path = model_ref;
        resolve_pins();
    }
    const std::string status = ParamOr(node, "_meta_model_status");
    // A model not on this PC is looked for again every 2 s (a download may bring
    // it); a node opened from a file is resolved the first time it is shown
    if (status == "missing" || (status.empty() && !mjcf_path.empty())) {
        static std::map<int, double> last_check;
        double& last = last_check[node.id];
        if (ImGui::GetTime() - last > 2.0) {
            last = ImGui::GetTime();
            node.resolved_config.clear();
            resolve_pins();
        }
    }
    {
        properties_rows::Rows rows("##mujoco");
        if (!rows.ok) return;
        // The model by name, Change... to pick another file
        properties_rows::Label("Model");
        const std::string title = ParamOr(node, "_meta_model_title");
        std::string shown = !title.empty() ? title : !mjcf_path.empty() ? mjcf_path : std::string("No model set.");
        char shown_buf[512];
        std::strncpy(shown_buf, shown.c_str(), sizeof(shown_buf) - 1);
        shown_buf[sizeof(shown_buf) - 1] = '\0';
        const float change = cyxwiz::ui::ButtonWidth("Change...", cyxwiz::ui::ButtonSize::Small);
        ImGui::SetNextItemWidth(std::max(40.0f, ImGui::GetContentRegionAvail().x - change - t.space_sm));
        ImGui::InputText("##mjcf_model", shown_buf, sizeof(shown_buf), ImGuiInputTextFlags_ReadOnly);
        if (ImGui::IsItemHovered() && !mjcf_path.empty()) ImGui::SetTooltip("%s", mjcf_path.c_str());
        ImGui::SameLine(0.0f, t.space_sm);
        if (cyxwiz::ui::SecondaryButton("Change...")) {
            if (auto selected = cyxwiz::FileDialogs::OpenFile(
                    "Select MJCF Model", {{"MJCF Files", "xml"}, {"All Files", "*"}},
                    mjcf_path.empty() ? nullptr : mjcf_path.c_str())) {
                node.parameters["mjcf_path"] = mjcf_path = *selected;
                resolve_pins();
            }
        }
        properties_rows::Status("mjcf_path");
        const std::string note = ParamOr(node, "_meta_model_note");
        if (status == "missing") {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.caution, "%s The graph cannot run until it is.", note.c_str());
            ImGui::PopTextWrapPos();
            if (mjcf_path.rfind("menagerie:", 0) == 0 && cyxwiz::ui::PrimaryButton("Download it")) {
                // The Environment Library (the node's own plugin) offers the version this graph names
                if (const auto descriptor = cyxwiz::ExtensionNodeRegistry::Instance().Find(node.extension_type_id))
                    cyxwiz::plugin::PluginPanelRegistry::Instance().ShowPanel(descriptor->provider_id, "mujoco_env_browser");
            }
        } else if (status == "not portable") {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", note.c_str());
            ImGui::PopTextWrapPos();
            const std::string project = cyxwiz::ProjectManager::Instance().GetProjectRoot();
            const bool outside = mjcf_path.find(':') == std::string::npos || (mjcf_path.size() > 1 && mjcf_path[1] == ':');
            if (outside && !project.empty() && cyxwiz::ui::LinkButton("Copy into the project")) {
                // The model's folder (its meshes and textures) into the project's models
                namespace fs = std::filesystem;
                const fs::path file(mjcf_path);
                const fs::path folder = file.parent_path();
                const fs::path to = fs::path(project) / "mujoco" / "models" / folder.filename();
                std::error_code ec;
                fs::create_directories(to, ec);
                if (!ec) fs::copy(folder, to, fs::copy_options::recursive | fs::copy_options::skip_existing, ec);
                if (ec) {
                    spdlog::error("MuJoCo Plant: copying {} into the project failed: {}", folder.string(), ec.message());
                } else {
                    node.parameters["mjcf_path"] = mjcf_path =
                        "project:" + (fs::path("mujoco") / "models" / folder.filename() / file.filename()).generic_string();
                    spdlog::info("MuJoCo Plant: copied {} into the project", folder.string());
                    resolve_pins();
                }
            }
        } else if (!note.empty()) {
            properties_rows::Note(note.c_str());
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

    if (!descriptor.has_value()) return;
    RenderInstalledExtensionProperties(node, *descriptor, context);
    ModelInfo(node);
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "Inputs: %d \xC2\xB7 Outputs: %d",
                       static_cast<int>(node.inputs.size()), static_cast<int>(node.outputs.size()));
}

} // namespace gui::properties_node_editors
