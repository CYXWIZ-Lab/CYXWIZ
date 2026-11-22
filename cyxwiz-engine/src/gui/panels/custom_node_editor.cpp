#include "custom_node_editor.h"
#include "../../core/engine_config.h"
#include "../icons.h"
#include "../ui_tokens.h"
#include <imgui.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <regex>

namespace gui {

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

using Definition = CustomNodeEditorPanel::CustomNodeDefinition;

constexpr const char* kDefinitionExtension = ".json";

// The definition as its file holds it.
json DefinitionToJson(const Definition& def) {
    json j;
    j["id"] = def.id;
    j["name"] = def.name;
    j["category"] = def.category;
    j["description"] = def.description;
    j["custom_node"] = true;  // Mark as custom node

    // Color
    j["color"] = {
        {"r", def.color.x},
        {"g", def.color.y},
        {"b", def.color.z},
        {"a", def.color.w}
    };

    // Inputs
    j["inputs"] = json::array();
    for (const auto& pin : def.inputs) {
        json pin_j;
        pin_j["name"] = pin.name;
        pin_j["type"] = static_cast<int>(pin.type);
        pin_j["description"] = pin.description;
        pin_j["default_shape"] = pin.default_shape;
        pin_j["optional"] = pin.is_optional;
        j["inputs"].push_back(pin_j);
    }

    // Outputs
    j["outputs"] = json::array();
    for (const auto& pin : def.outputs) {
        json pin_j;
        pin_j["name"] = pin.name;
        pin_j["type"] = static_cast<int>(pin.type);
        pin_j["description"] = pin.description;
        pin_j["default_shape"] = pin.default_shape;
        pin_j["optional"] = pin.is_optional;
        j["outputs"].push_back(pin_j);
    }

    // Parameters
    j["parameters"] = json::array();
    for (const auto& param : def.parameters) {
        json param_j;
        param_j["name"] = param.name;
        param_j["type"] = param.type;
        param_j["default_value"] = param.default_value;
        param_j["description"] = param.description;
        param_j["min_value"] = param.min_value;
        param_j["max_value"] = param.max_value;
        param_j["options"] = param.choices;
        j["parameters"].push_back(param_j);
    }

    // Code templates
    j["code_templates"] = {
        {"pytorch", def.pytorch_template},
        {"tensorflow", def.tensorflow_template},
        {"cyxwiz", def.cyxwiz_template}
    };

    if (!def.validation_code.empty()) {
        j["validation_code"] = def.validation_code;
    }

    return j;
}

// A definition file read back; false when the file is not a custom node.
bool DefinitionFromJson(const json& j, Definition& def) {
    if (!j.is_object() || !j.value("custom_node", false)) return false;
    Definition d;
    d.id = j.value("id", std::string());
    d.name = j.value("name", std::string());
    d.category = j.value("category", std::string("Custom"));
    d.description = j.value("description", std::string());
    if (const auto colour = j.find("color"); colour != j.end() && colour->is_object()) {
        d.color = ImVec4(colour->value("r", d.color.x), colour->value("g", d.color.y),
                         colour->value("b", d.color.z), colour->value("a", d.color.w));
    }
    const auto read_pins = [&j](const char* key, std::vector<CustomNodeEditorPanel::PinDefinition>& pins) {
        const auto list = j.find(key);
        if (list == j.end() || !list->is_array()) return;
        for (const auto& p : *list) {
            CustomNodeEditorPanel::PinDefinition pin;
            pin.name = p.value("name", std::string());
            pin.type = static_cast<PinType>(p.value("type", static_cast<int>(PinType::Tensor)));
            pin.description = p.value("description", std::string());
            pin.default_shape = p.value("default_shape", std::string());
            pin.is_optional = p.value("optional", false);
            pins.push_back(pin);
        }
    };
    read_pins("inputs", d.inputs);
    read_pins("outputs", d.outputs);
    if (const auto list = j.find("parameters"); list != j.end() && list->is_array()) {
        for (const auto& p : *list) {
            CustomNodeEditorPanel::ParameterDefinition param;
            param.name = p.value("name", std::string());
            param.type = p.value("type", std::string("string"));
            param.default_value = p.value("default_value", std::string());
            param.description = p.value("description", std::string());
            param.min_value = p.value("min_value", std::string());
            param.max_value = p.value("max_value", std::string());
            param.choices = p.value("options", std::vector<std::string>());
            d.parameters.push_back(param);
        }
    }
    if (const auto code = j.find("code_templates"); code != j.end() && code->is_object()) {
        d.pytorch_template = code->value("pytorch", std::string());
        d.tensorflow_template = code->value("tensorflow", std::string());
        d.cyxwiz_template = code->value("cyxwiz", std::string());
    }
    d.validation_code = j.value("validation_code", std::string());
    def = std::move(d);
    return true;
}

// The definitions in the folder, by file, with the name each one shows.
std::vector<std::pair<fs::path, std::string>> ListDefinitions(const fs::path& directory) {
    std::vector<std::pair<fs::path, std::string>> found;
    std::error_code ec;
    for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().extension() != kDefinitionExtension) continue;
        std::ifstream file(it->path());
        const json j = json::parse(file, nullptr, false);
        Definition def;
        if (DefinitionFromJson(j, def)) found.emplace_back(it->path(), def.name);
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    return found;
}

}  // namespace

// Category options (display name, internal value)
const std::vector<std::pair<std::string, std::string>> CustomNodeEditorPanel::kCategoryOptions = {
    {"Custom", "Custom"},
    {"Layer", "Layers"},
    {"Activation", "Activations"},
    {"Loss", "Losses"},
    {"Optimizer", "Optimizers"},
    {"Data", "Data"},
    {"Utility", "Utilities"},
    {"Preprocessing", "Preprocessing"},
    {"Attention", "Attention"},
    {"Normalization", "Normalization"}
};

// Placeholder documentation
const char* CustomNodeEditorPanel::kPlaceholderHelp = R"(Available Placeholders:
  {{node_name}}     - Node display name
  {{param_name}}    - Parameter value (e.g., {{units}}, {{kernel_size}})
  {{input_name}}    - Input tensor (e.g., {{x}}, {{input}})
  {{output_name}}   - Output tensor name

Examples:
  PyTorch:
    nn.Linear({{in_features}}, {{out_features}})

  TensorFlow:
    tf.keras.layers.Dense({{units}}, activation={{activation}})
)";

CustomNodeEditorPanel::CustomNodeEditorPanel()
    : Panel("Custom Node Editor", false),  // Hidden by default
      directory_(cyxwiz::core::EngineConfig::Instance().GetUserConfigDir() / "custom_nodes")
{
    CreateNewDefinition();
}

void CustomNodeEditorPanel::CreateNewDefinition() {
    current_def_ = CustomNodeDefinition();
    current_def_.name = "New Custom Node";
    current_def_.category = "Custom";
    current_def_.description = "A custom node type";
    current_def_.color = cyxwiz::ui::CurrentTokens().accent;  // user data; default from the theme

    // Add default input/output
    PinDefinition default_input;
    default_input.name = "input";
    default_input.type = PinType::Tensor;
    default_input.description = "Input tensor";
    current_def_.inputs.push_back(default_input);

    PinDefinition default_output;
    default_output.name = "output";
    default_output.type = PinType::Tensor;
    default_output.description = "Output tensor";
    current_def_.outputs.push_back(default_output);

    // Default code templates
    current_def_.pytorch_template = "# PyTorch implementation\nself.layer = nn.Identity()\n\ndef forward(self, x):\n    return self.layer(x)";
    current_def_.tensorflow_template = "# TensorFlow implementation\nself.layer = tf.keras.layers.Lambda(lambda x: x)";
    current_def_.cyxwiz_template = "# CyxWiz implementation\n# Implement using cyxwiz operations";

    has_unsaved_changes_ = false;
}

bool CustomNodeEditorPanel::LoadDefinition(const fs::path& file) {
    std::ifstream in(file);
    const json j = json::parse(in, nullptr, false);
    Definition def;
    if (!DefinitionFromJson(j, def)) {
        spdlog::error("Not a custom node definition: {}", file.string());
        return false;
    }
    // The file name is the definition's id: a save goes back to the same file.
    def.id = file.stem().string();
    current_def_ = std::move(def);
    has_unsaved_changes_ = false;
    return true;
}

bool CustomNodeEditorPanel::SaveDefinition() {
    std::string error;
    if (!ValidateDefinition(error)) {
        spdlog::error("Validation failed: {}", error);
        return false;
    }

    // Generate ID if new
    if (current_def_.id.empty()) {
        current_def_.id = GenerateUniqueId(current_def_.name);
    }

    std::error_code ec;
    fs::create_directories(directory_, ec);
    if (ec) {
        spdlog::error("Could not create {}: {}", directory_.string(), ec.message());
        return false;
    }
    return SaveDefinitionAs((directory_ / (current_def_.id + kDefinitionExtension)).string());
}

bool CustomNodeEditorPanel::SaveDefinitionAs(const std::string& path) {
    std::string error;
    if (!ValidateDefinition(error)) {
        spdlog::error("Validation failed: {}", error);
        return false;
    }

    // Generate ID if empty
    if (current_def_.id.empty()) {
        current_def_.id = GenerateUniqueId(current_def_.name);
    }

    const json j = DefinitionToJson(current_def_);

    // Write to file
    std::ofstream file(path);
    if (!file.is_open()) {
        spdlog::error("Failed to open file for writing: {}", path);
        return false;
    }

    file << j.dump(2);
    file.close();

    has_unsaved_changes_ = false;
    spdlog::info("Saved custom node: {} to {}", current_def_.name, path);
    return true;
}

void CustomNodeEditorPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Custom Node Editor", &visible_, ImGuiWindowFlags_MenuBar)) {
        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem(ICON_FA_FILE " New", "Ctrl+N")) {
                    if (has_unsaved_changes_) {
                        // TODO: Prompt to save
                    }
                    CreateNewDefinition();
                }
                if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save", "Ctrl+S")) {
                    SaveDefinition();
                }
                if (ImGui::MenuItem(ICON_FA_DOWNLOAD " Save As...")) {
                    show_save_dialog_ = true;
                }
                ImGui::Separator();
                if (ImGui::BeginMenu("Load Existing")) {
                    const auto definitions = ListDefinitions(directory_);
                    if (definitions.empty()) {
                        ImGui::TextDisabled("No custom nodes found");
                    }
                    for (const auto& [file, name] : definitions) {
                        ImGui::PushID(file.string().c_str());
                        if (ImGui::MenuItem(name.c_str())) {
                            LoadDefinition(file);
                        }
                        ImGui::PopID();
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Title and status
        ImGui::Text(ICON_FA_CUBE " Editing: %s", current_def_.name.c_str());
        if (has_unsaved_changes_) {
            ImGui::SameLine();
            ImGui::TextColored(cyxwiz::ui::CurrentTokens().warning, "(unsaved)");
        }
        ImGui::Separator();

        // Tab bar
        if (ImGui::BeginTabBar("CustomNodeTabs")) {
            if (ImGui::BeginTabItem("General")) {
                current_tab_ = 0;
                RenderGeneralTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Pins")) {
                current_tab_ = 1;
                RenderPinsTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Parameters")) {
                current_tab_ = 2;
                RenderParametersTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Code")) {
                current_tab_ = 3;
                RenderCodeTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Preview")) {
                current_tab_ = 4;
                RenderPreviewTab();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();

    // Save As dialog
    if (show_save_dialog_) {
        ImGui::OpenPopup("Save Custom Node As");
    }

    if (ImGui::BeginPopupModal("Save Custom Node As", &show_save_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Save custom node to file:");
        ImGui::InputText("Path", save_path_buffer_, sizeof(save_path_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Browse...")) {
            // TODO: Open file dialog
        }

        ImGui::Separator();

        if (ImGui::Button("Save", ImVec2(120, 0))) {
            if (strlen(save_path_buffer_) > 0) {
                SaveDefinitionAs(save_path_buffer_);
                show_save_dialog_ = false;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_save_dialog_ = false;
        }
        ImGui::EndPopup();
    }
}

void CustomNodeEditorPanel::RenderGeneralTab() {
    ImGui::BeginChild("GeneralContent", ImVec2(0, 0), true);

    // Name
    char name_buf[256];
    strncpy(name_buf, current_def_.name.c_str(), sizeof(name_buf) - 1);
    name_buf[sizeof(name_buf) - 1] = '\0';
    if (ImGui::InputText("Name", name_buf, sizeof(name_buf))) {
        current_def_.name = name_buf;
        has_unsaved_changes_ = true;
    }

    // ID (auto-generated or manual)
    char id_buf[128];
    strncpy(id_buf, current_def_.id.c_str(), sizeof(id_buf) - 1);
    id_buf[sizeof(id_buf) - 1] = '\0';
    if (ImGui::InputText("ID", id_buf, sizeof(id_buf))) {
        current_def_.id = id_buf;
        has_unsaved_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Auto")) {
        current_def_.id = GenerateUniqueId(current_def_.name);
        has_unsaved_changes_ = true;
    }

    // Category
    if (ImGui::BeginCombo("Category", current_def_.category.c_str())) {
        for (const auto& [display, value] : kCategoryOptions) {
            bool is_selected = (current_def_.category == value);
            if (ImGui::Selectable(display.c_str(), is_selected)) {
                current_def_.category = value;
                has_unsaved_changes_ = true;
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    // Description
    char desc_buf[1024];
    strncpy(desc_buf, current_def_.description.c_str(), sizeof(desc_buf) - 1);
    desc_buf[sizeof(desc_buf) - 1] = '\0';
    if (ImGui::InputTextMultiline("Description", desc_buf, sizeof(desc_buf), ImVec2(-1, 80))) {
        current_def_.description = desc_buf;
        has_unsaved_changes_ = true;
    }

    // Color
    ImGui::Separator();
    ImGui::Text("Node Appearance");
    float color[4] = {current_def_.color.x, current_def_.color.y, current_def_.color.z, current_def_.color.w};
    if (ImGui::ColorEdit4("Node Color", color)) {
        current_def_.color = ImVec4(color[0], color[1], color[2], color[3]);
        has_unsaved_changes_ = true;
    }

    // Icon (optional)
    char icon_buf[64];
    strncpy(icon_buf, current_def_.icon.c_str(), sizeof(icon_buf) - 1);
    icon_buf[sizeof(icon_buf) - 1] = '\0';
    if (ImGui::InputText("Icon (FontAwesome)", icon_buf, sizeof(icon_buf))) {
        current_def_.icon = icon_buf;
        has_unsaved_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Enter a FontAwesome icon name like 'brain' or 'cube'");
        ImGui::EndTooltip();
    }

    ImGui::EndChild();
}

void CustomNodeEditorPanel::RenderPinsTab() {
    // Split view: inputs on left, outputs on right
    float width = ImGui::GetContentRegionAvail().x;

    // Inputs section
    ImGui::BeginChild("InputPins", ImVec2(width * 0.5f - 5, 0), true);
    ImGui::Text(ICON_FA_ARROW_RIGHT " Input Pins");
    ImGui::Separator();

    if (ImGui::Button(ICON_FA_PLUS " Add Input")) {
        PinDefinition new_pin;
        new_pin.name = "input_" + std::to_string(current_def_.inputs.size());
        new_pin.type = PinType::Tensor;
        new_pin.description = "Input tensor";
        current_def_.inputs.push_back(new_pin);
        has_unsaved_changes_ = true;
    }

    RenderPinEditor(current_def_.inputs, "input", true);
    ImGui::EndChild();

    ImGui::SameLine();

    // Outputs section
    ImGui::BeginChild("OutputPins", ImVec2(0, 0), true);
    ImGui::Text(ICON_FA_ARROW_LEFT " Output Pins");
    ImGui::Separator();

    if (ImGui::Button(ICON_FA_PLUS " Add Output")) {
        PinDefinition new_pin;
        new_pin.name = "output_" + std::to_string(current_def_.outputs.size());
        new_pin.type = PinType::Tensor;
        new_pin.description = "Output tensor";
        current_def_.outputs.push_back(new_pin);
        has_unsaved_changes_ = true;
    }

    RenderPinEditor(current_def_.outputs, "output", false);
    ImGui::EndChild();
}

void CustomNodeEditorPanel::RenderPinEditor(std::vector<PinDefinition>& pins, const char* label, bool is_input) {
    (void)label;     // Reserved for future use (section headers)
    (void)is_input;  // Reserved for future use (direction-specific UI)
    int to_delete = -1;

    for (size_t i = 0; i < pins.size(); ++i) {
        auto& pin = pins[i];
        ImGui::PushID(static_cast<int>(i));

        bool open = ImGui::TreeNodeEx(pin.name.c_str(), ImGuiTreeNodeFlags_DefaultOpen);

        // Delete button
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20);
        if (ImGui::SmallButton(ICON_FA_TRASH)) {
            to_delete = static_cast<int>(i);
        }

        if (open) {
            // Name
            char name_buf[128];
            strncpy(name_buf, pin.name.c_str(), sizeof(name_buf) - 1);
            name_buf[sizeof(name_buf) - 1] = '\0';
            if (ImGui::InputText("Name", name_buf, sizeof(name_buf))) {
                pin.name = name_buf;
                has_unsaved_changes_ = true;
            }

            // Type
            RenderPinTypeCombo(pin.type);

            // Description
            char desc_buf[256];
            strncpy(desc_buf, pin.description.c_str(), sizeof(desc_buf) - 1);
            desc_buf[sizeof(desc_buf) - 1] = '\0';
            if (ImGui::InputText("Description", desc_buf, sizeof(desc_buf))) {
                pin.description = desc_buf;
                has_unsaved_changes_ = true;
            }

            // Shape hint
            char shape_buf[128];
            strncpy(shape_buf, pin.default_shape.c_str(), sizeof(shape_buf) - 1);
            shape_buf[sizeof(shape_buf) - 1] = '\0';
            if (ImGui::InputText("Shape Hint", shape_buf, sizeof(shape_buf))) {
                pin.default_shape = shape_buf;
                has_unsaved_changes_ = true;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("e.g., [batch, channels, height, width]");
                ImGui::EndTooltip();
            }

            // Optional
            if (ImGui::Checkbox("Optional", &pin.is_optional)) {
                has_unsaved_changes_ = true;
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    // Delete after iteration
    if (to_delete >= 0 && pins.size() > 1) {
        pins.erase(pins.begin() + to_delete);
        has_unsaved_changes_ = true;
    }
}

void CustomNodeEditorPanel::RenderPinTypeCombo(PinType& type) {
    const char* type_names[] = {
        "Tensor", "Scalar", "Integer", "Float", "String", "Bool", "Any", "Flow"
    };
    int current = static_cast<int>(type);

    if (ImGui::Combo("Type", &current, type_names, IM_ARRAYSIZE(type_names))) {
        type = static_cast<PinType>(current);
        has_unsaved_changes_ = true;
    }
}

void CustomNodeEditorPanel::RenderParametersTab() {
    ImGui::BeginChild("ParametersContent", ImVec2(0, 0), true);

    if (ImGui::Button(ICON_FA_PLUS " Add Parameter")) {
        ParameterDefinition param;
        param.name = "param_" + std::to_string(current_def_.parameters.size());
        param.type = "int";
        param.default_value = "0";
        param.description = "Parameter description";
        current_def_.parameters.push_back(param);
        has_unsaved_changes_ = true;
    }

    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Parameters are configurable values shown in the Properties panel.");
        ImGui::Text("They can be referenced in code templates using {{param_name}}.");
        ImGui::EndTooltip();
    }

    ImGui::Separator();

    RenderParameterEditor();

    ImGui::EndChild();
}

void CustomNodeEditorPanel::RenderParameterEditor() {
    int to_delete = -1;

    for (size_t i = 0; i < current_def_.parameters.size(); ++i) {
        auto& param = current_def_.parameters[i];
        ImGui::PushID(static_cast<int>(i));

        bool open = ImGui::TreeNodeEx(param.name.c_str(), ImGuiTreeNodeFlags_DefaultOpen);

        // Delete button
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20);
        if (ImGui::SmallButton(ICON_FA_TRASH)) {
            to_delete = static_cast<int>(i);
        }

        if (open) {
            // Name
            char name_buf[128];
            strncpy(name_buf, param.name.c_str(), sizeof(name_buf) - 1);
            name_buf[sizeof(name_buf) - 1] = '\0';
            if (ImGui::InputText("Name", name_buf, sizeof(name_buf))) {
                param.name = name_buf;
                has_unsaved_changes_ = true;
            }

            // Type
            RenderParameterTypeCombo(param.type);

            // Default value
            char default_buf[256];
            strncpy(default_buf, param.default_value.c_str(), sizeof(default_buf) - 1);
            default_buf[sizeof(default_buf) - 1] = '\0';
            if (ImGui::InputText("Default Value", default_buf, sizeof(default_buf))) {
                param.default_value = default_buf;
                has_unsaved_changes_ = true;
            }

            // Description
            char desc_buf[256];
            strncpy(desc_buf, param.description.c_str(), sizeof(desc_buf) - 1);
            desc_buf[sizeof(desc_buf) - 1] = '\0';
            if (ImGui::InputText("Description", desc_buf, sizeof(desc_buf))) {
                param.description = desc_buf;
                has_unsaved_changes_ = true;
            }

            // Min/Max for numeric types
            if (param.type == "int" || param.type == "float") {
                char min_buf[64], max_buf[64];
                strncpy(min_buf, param.min_value.c_str(), sizeof(min_buf) - 1);
                strncpy(max_buf, param.max_value.c_str(), sizeof(max_buf) - 1);
                min_buf[sizeof(min_buf) - 1] = '\0';
                max_buf[sizeof(max_buf) - 1] = '\0';

                ImGui::SetNextItemWidth(100);
                if (ImGui::InputText("Min", min_buf, sizeof(min_buf))) {
                    param.min_value = min_buf;
                    has_unsaved_changes_ = true;
                }
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100);
                if (ImGui::InputText("Max", max_buf, sizeof(max_buf))) {
                    param.max_value = max_buf;
                    has_unsaved_changes_ = true;
                }
            }

            // Choices for choice type
            if (param.type == "choice") {
                ImGui::Text("Choices:");
                for (size_t j = 0; j < param.choices.size(); ++j) {
                    ImGui::PushID(static_cast<int>(j));
                    char choice_buf[128];
                    strncpy(choice_buf, param.choices[j].c_str(), sizeof(choice_buf) - 1);
                    choice_buf[sizeof(choice_buf) - 1] = '\0';
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::InputText("##choice", choice_buf, sizeof(choice_buf))) {
                        param.choices[j] = choice_buf;
                        has_unsaved_changes_ = true;
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("-")) {
                        param.choices.erase(param.choices.begin() + j);
                        has_unsaved_changes_ = true;
                    }
                    ImGui::PopID();
                }
                if (ImGui::SmallButton("+ Add Choice")) {
                    param.choices.push_back("option_" + std::to_string(param.choices.size()));
                    has_unsaved_changes_ = true;
                }
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    // Delete after iteration
    if (to_delete >= 0) {
        current_def_.parameters.erase(current_def_.parameters.begin() + to_delete);
        has_unsaved_changes_ = true;
    }
}

void CustomNodeEditorPanel::RenderParameterTypeCombo(std::string& type) {
    const char* type_names[] = {"int", "float", "string", "bool", "choice"};
    int current = 0;
    for (int i = 0; i < IM_ARRAYSIZE(type_names); ++i) {
        if (type == type_names[i]) {
            current = i;
            break;
        }
    }

    if (ImGui::Combo("Type", &current, type_names, IM_ARRAYSIZE(type_names))) {
        type = type_names[current];
        has_unsaved_changes_ = true;
    }
}

void CustomNodeEditorPanel::RenderCodeTab() {
    // Framework tabs
    if (ImGui::BeginTabBar("CodeFrameworkTabs")) {
        if (ImGui::BeginTabItem("PyTorch")) {
            code_tab_ = 0;
            RenderCodeTemplateEditor("PyTorch Template", current_def_.pytorch_template, "python");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("TensorFlow")) {
            code_tab_ = 1;
            RenderCodeTemplateEditor("TensorFlow Template", current_def_.tensorflow_template, "python");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("CyxWiz")) {
            code_tab_ = 2;
            RenderCodeTemplateEditor("CyxWiz Template", current_def_.cyxwiz_template, "python");
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Validation")) {
            ImGui::Text("Validation Code (Optional)");
            ImGui::TextDisabled("Python code to validate node configuration");

            char val_buf[4096];
            strncpy(val_buf, current_def_.validation_code.c_str(), sizeof(val_buf) - 1);
            val_buf[sizeof(val_buf) - 1] = '\0';
            if (ImGui::InputTextMultiline("##validation", val_buf, sizeof(val_buf),
                    ImVec2(-1, ImGui::GetContentRegionAvail().y - 50))) {
                current_def_.validation_code = val_buf;
                has_unsaved_changes_ = true;
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void CustomNodeEditorPanel::RenderCodeTemplateEditor(const char* label, std::string& code, const char* language) {
    (void)language;  // Reserved for syntax highlighting

    // Placeholder help
    if (ImGui::CollapsingHeader("Placeholder Reference")) {
        ImGui::TextWrapped("%s", kPlaceholderHelp);

        // Insert buttons for common placeholders
        ImGui::Text("Quick Insert:");
        if (ImGui::SmallButton("{{input}}")) InsertPlaceholder(code, "{{input}}");
        ImGui::SameLine();
        if (ImGui::SmallButton("{{output}}")) InsertPlaceholder(code, "{{output}}");
        ImGui::SameLine();

        // Parameter placeholders
        for (const auto& param : current_def_.parameters) {
            if (ImGui::SmallButton(("{{" + param.name + "}}").c_str())) {
                InsertPlaceholder(code, ("{{" + param.name + "}}").c_str());
            }
            ImGui::SameLine();
        }
        ImGui::NewLine();
    }

    // Code editor
    ImGui::Text("%s", label);

    char code_buf[8192];
    strncpy(code_buf, code.c_str(), sizeof(code_buf) - 1);
    code_buf[sizeof(code_buf) - 1] = '\0';

    // Calculate available height
    float height = ImGui::GetContentRegionAvail().y - 10;
    if (height < 200) height = 200;

    if (ImGui::InputTextMultiline("##code", code_buf, sizeof(code_buf),
            ImVec2(-1, height), ImGuiInputTextFlags_AllowTabInput)) {
        code = code_buf;
        has_unsaved_changes_ = true;
    }
}

void CustomNodeEditorPanel::InsertPlaceholder(std::string& code, const char* placeholder) {
    // For now, just append at the end
    // A more sophisticated version would insert at cursor position
    code += placeholder;
    has_unsaved_changes_ = true;
}

void CustomNodeEditorPanel::RenderPreviewTab() {
    float width = ImGui::GetContentRegionAvail().x;

    // Node preview on left
    ImGui::BeginChild("NodePreview", ImVec2(width * 0.4f - 5, 0), true);
    ImGui::Text(ICON_FA_CUBE " Node Preview");
    ImGui::Separator();
    RenderNodePreview();
    ImGui::EndChild();

    ImGui::SameLine();

    // Code preview on right
    ImGui::BeginChild("CodePreview", ImVec2(0, 0), true);
    ImGui::Text(ICON_FA_CODE " Generated Code Preview");
    ImGui::Separator();
    RenderCodePreview();
    ImGui::EndChild();
}

void CustomNodeEditorPanel::RenderNodePreview() {
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    // Draw a simple representation of the node
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    float node_width = 180.0f;
    float header_height = 25.0f;
    float pin_spacing = 22.0f;

    int num_inputs = static_cast<int>(current_def_.inputs.size());
    int num_outputs = static_cast<int>(current_def_.outputs.size());
    int max_pins = std::max(num_inputs, num_outputs);
    float body_height = std::max(60.0f, max_pins * pin_spacing + 10);
    float total_height = header_height + body_height;

    // Node background
    ImU32 bg_color = cyxwiz::ui::ToU32(t.bg_panel);
    ImU32 header_color = ImGui::ColorConvertFloat4ToU32(current_def_.color);
    ImU32 border_color = cyxwiz::ui::ToU32(t.border);

    // Draw header
    draw_list->AddRectFilled(pos, ImVec2(pos.x + node_width, pos.y + header_height), header_color, 5.0f, ImDrawFlags_RoundCornersTop);

    // Draw body
    draw_list->AddRectFilled(ImVec2(pos.x, pos.y + header_height), ImVec2(pos.x + node_width, pos.y + total_height), bg_color, 5.0f, ImDrawFlags_RoundCornersBottom);

    // Draw border
    draw_list->AddRect(pos, ImVec2(pos.x + node_width, pos.y + total_height), border_color, 5.0f, 0, 1.5f);

    // Draw title
    ImGui::SetCursorScreenPos(ImVec2(pos.x + 10, pos.y + 5));
    ImGui::TextColored(cyxwiz::ui::ReadableOn(t, current_def_.color), "%s", current_def_.name.c_str());

    // Draw input pins
    float pin_y = pos.y + header_height + 10;
    ImU32 pin_color = cyxwiz::ui::ToU32(t.series[3]);

    for (const auto& pin : current_def_.inputs) {
        // Pin circle
        draw_list->AddCircleFilled(ImVec2(pos.x, pin_y + 5), 5.0f, pin_color);
        draw_list->AddCircle(ImVec2(pos.x, pin_y + 5), 5.0f, border_color);

        // Pin name
        ImGui::SetCursorScreenPos(ImVec2(pos.x + 10, pin_y));
        ImGui::TextColored(t.text, "%s", pin.name.c_str());

        pin_y += pin_spacing;
    }

    // Draw output pins
    pin_y = pos.y + header_height + 10;
    ImU32 output_pin_color = cyxwiz::ui::ToU32(t.series[4]);

    for (const auto& pin : current_def_.outputs) {
        // Pin circle
        draw_list->AddCircleFilled(ImVec2(pos.x + node_width, pin_y + 5), 5.0f, output_pin_color);
        draw_list->AddCircle(ImVec2(pos.x + node_width, pin_y + 5), 5.0f, border_color);

        // Pin name (right-aligned)
        ImVec2 text_size = ImGui::CalcTextSize(pin.name.c_str());
        ImGui::SetCursorScreenPos(ImVec2(pos.x + node_width - text_size.x - 10, pin_y));
        ImGui::TextColored(t.text, "%s", pin.name.c_str());

        pin_y += pin_spacing;
    }

    // Reserve space for the node drawing
    ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y + total_height + 10));
    ImGui::Dummy(ImVec2(node_width, 0));

    // Show parameter summary below
    ImGui::Separator();
    ImGui::Text("Parameters: %zu", current_def_.parameters.size());
    for (const auto& param : current_def_.parameters) {
        ImGui::BulletText("%s: %s = %s", param.name.c_str(), param.type.c_str(), param.default_value.c_str());
    }
}

void CustomNodeEditorPanel::RenderCodePreview() {
    // Show what the generated code would look like with sample values
    std::string preview_code;

    switch (code_tab_) {
        case 0: preview_code = current_def_.pytorch_template; break;
        case 1: preview_code = current_def_.tensorflow_template; break;
        case 2: preview_code = current_def_.cyxwiz_template; break;
    }

    // Replace placeholders with sample values
    for (const auto& param : current_def_.parameters) {
        std::string placeholder = "{{" + param.name + "}}";
        size_t pos = 0;
        while ((pos = preview_code.find(placeholder, pos)) != std::string::npos) {
            preview_code.replace(pos, placeholder.length(), param.default_value);
            pos += param.default_value.length();
        }
    }

    // Replace input/output placeholders
    if (!current_def_.inputs.empty()) {
        std::string input_placeholder = "{{input}}";
        size_t pos = 0;
        while ((pos = preview_code.find(input_placeholder, pos)) != std::string::npos) {
            preview_code.replace(pos, input_placeholder.length(), current_def_.inputs[0].name);
        }
    }

    if (!current_def_.outputs.empty()) {
        std::string output_placeholder = "{{output}}";
        size_t pos = 0;
        while ((pos = preview_code.find(output_placeholder, pos)) != std::string::npos) {
            preview_code.replace(pos, output_placeholder.length(), current_def_.outputs[0].name);
        }
    }

    ImGui::TextWrapped("%s", preview_code.c_str());
}

bool CustomNodeEditorPanel::ValidateDefinition(std::string& error_message) {
    if (current_def_.name.empty()) {
        error_message = "Node name is required";
        return false;
    }

    if (current_def_.inputs.empty()) {
        error_message = "At least one input pin is required";
        return false;
    }

    if (current_def_.outputs.empty()) {
        error_message = "At least one output pin is required";
        return false;
    }

    // Check for duplicate pin names
    std::vector<std::string> all_pins;
    for (const auto& pin : current_def_.inputs) {
        if (std::find(all_pins.begin(), all_pins.end(), pin.name) != all_pins.end()) {
            error_message = "Duplicate pin name: " + pin.name;
            return false;
        }
        all_pins.push_back(pin.name);
    }
    for (const auto& pin : current_def_.outputs) {
        if (std::find(all_pins.begin(), all_pins.end(), pin.name) != all_pins.end()) {
            error_message = "Duplicate pin name: " + pin.name;
            return false;
        }
        all_pins.push_back(pin.name);
    }

    // Check for duplicate parameter names
    std::vector<std::string> param_names;
    for (const auto& param : current_def_.parameters) {
        if (std::find(param_names.begin(), param_names.end(), param.name) != param_names.end()) {
            error_message = "Duplicate parameter name: " + param.name;
            return false;
        }
        param_names.push_back(param.name);
    }

    return true;
}

std::string CustomNodeEditorPanel::GenerateUniqueId(const std::string& name) {
    // Convert name to lowercase with underscores
    std::string id;
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            id += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (c == ' ' || c == '-') {
            if (!id.empty() && id.back() != '_') {
                id += '_';
            }
        }
    }

    // Add prefix
    id = "custom_" + id;

    // Unique among the definitions saved: a new one never overwrites another's file.
    std::string base_id = id;
    int counter = 1;
    std::error_code ec;
    while (fs::exists(directory_ / (id + kDefinitionExtension), ec)) {
        id = base_id + "_" + std::to_string(counter++);
    }

    return id;
}

} // namespace gui
