// Include Windows headers first, then undef conflicting macros.
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#ifdef CreateDialog
#undef CreateDialog
#endif
#ifdef CreateDialogA
#undef CreateDialogA
#endif
#ifdef CreateDialogW
#undef CreateDialogW
#endif
#endif

#include "properties_metadata_editor.h"
#include "properties_parameter_rules.h"
#include "node_editor.h"
#include "../core/file_dialogs.h"
#include <imgui.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace gui::properties_metadata {

namespace {

std::string HumanizeParameterName(const std::string& name) {
    std::string label;
    label.reserve(name.size());

    bool capitalize_next = true;
    for (char ch : name) {
        if (ch == '_' || ch == '-') {
            label.push_back(' ');
            capitalize_next = true;
            continue;
        }

        unsigned char c = static_cast<unsigned char>(ch);
        if (capitalize_next) {
            label.push_back(static_cast<char>(std::toupper(c)));
            capitalize_next = false;
        } else {
            label.push_back(ch);
        }
    }

    return label.empty() ? name : label;
}

std::string GetParameterLabel(const cyxwiz::ParameterDefinition& param) {
    return param.display_name.empty() ? HumanizeParameterName(param.name) : param.display_name;
}

bool ShouldUseIntSlider(
    const cyxwiz::ParameterDefinition& param,
    const properties_rules::NumericRange& range) {
    // Epoch counts require exact entry; a 1-10000 slider is difficult to
    // control and duplicates the Data Loader dialog's manual input behavior.
    if (param.name == "epochs") {
        return false;
    }

    if (!range.has_range) {
        return false;
    }

    const double span = range.max_value - range.min_value;
    return span >= 1.0 && span <= 10000.0;
}

std::string ToLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

bool ContainsAny(const std::string& text, const std::vector<const char*>& needles) {
    const std::string lower_text = ToLower(text);
    for (const char* needle : needles) {
        if (lower_text.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool IsFolderParameterType(const cyxwiz::ParameterDefinition& param) {
    return param.type == "directory" || param.type == "folder";
}

bool ShouldUseMultilineText(const cyxwiz::ParameterDefinition& param) {
    if (param.type == "multiline" || param.type == "text") {
        return true;
    }

    return param.type == "string" &&
           ContainsAny(param.name, {"query", "sql", "body", "prompt", "template", "expression"});
}

bool ShouldUsePasswordText(const cyxwiz::ParameterDefinition& param) {
    if (param.type == "password") {
        return true;
    }

    return param.type == "string" &&
           ContainsAny(param.name, {"password", "api_key", "token", "secret", "credential"});
}

void RenderParameter(
    MLNode& node,
    const cyxwiz::ParameterDefinition& param,
    std::map<std::string, std::string>& validation_errors,
    const InvalidateCallback& invalidate) {
    ImGui::PushID(param.name.c_str());

    std::string& value = node.parameters[param.name];

    if (value.empty() && !param.default_value.empty()) {
        value = param.default_value;
    }

    std::string initial_error;
    if (!properties_rules::ValidateParameter(value, param, initial_error)) {
        validation_errors[param.name] = initial_error;
    } else {
        validation_errors.erase(param.name);
    }

    bool has_error = validation_errors.count(param.name) > 0;
    if (has_error) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
    }

    const std::string label = GetParameterLabel(param);

    ImGui::Text("%s%s:", label.c_str(), param.required ? " *" : "");
    if (has_error) {
        ImGui::PopStyleColor();
    }

    if (!param.description.empty() && ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(param.description.c_str());
        if (!param.default_value.empty()) {
            ImGui::TextDisabled("Default: %s", param.default_value.c_str());
        }
        ImGui::TextDisabled(param.required ? "Required" : "Optional");
        ImGui::EndTooltip();
    }

    ImGui::SameLine();

    bool changed = false;

    if (param.type == "int") {
        int int_val = 0;
        properties_rules::TryParseIntStrict(value, int_val);
        ImGui::SetNextItemWidth(120.0f);
        properties_rules::NumericRange range = properties_rules::ParseNumericRange(param.validation);
        if (ShouldUseIntSlider(param, range)) {
            int min_v = static_cast<int>(range.min_value);
            int max_v = static_cast<int>(range.max_value);
            int_val = std::clamp(int_val, min_v, max_v);
            if (ImGui::SliderInt("##value", &int_val, min_v, max_v)) {
                value = std::to_string(int_val);
                changed = true;
            }
        } else if (ImGui::InputInt("##value", &int_val)) {
            if (range.has_range) {
                int_val = std::clamp(
                    int_val,
                    static_cast<int>(range.min_value),
                    static_cast<int>(range.max_value));
            }
            value = std::to_string(int_val);
            changed = true;
        }
    }
    else if (param.type == "float") {
        double parsed_float = 0.0;
        properties_rules::TryParseDoubleStrict(value, parsed_float);
        float float_val = static_cast<float>(parsed_float);
        ImGui::SetNextItemWidth(120.0f);

        properties_rules::NumericRange range = properties_rules::ParseNumericRange(param.validation);
        if (range.has_range) {
            float min_v = static_cast<float>(range.min_value);
            float max_v = static_cast<float>(range.max_value);
            float_val = std::clamp(float_val, min_v, max_v);
            if (ImGui::SliderFloat("##value", &float_val, min_v, max_v, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", float_val);
                value = buf;
                changed = true;
            }
        } else {
            if (ImGui::InputFloat("##value", &float_val, 0.01f, 0.1f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", float_val);
                value = buf;
                changed = true;
            }
        }
    }
    else if (param.type == "bool") {
        bool bool_val = (value == "true" || value == "1");
        if (ImGui::Checkbox("##value", &bool_val)) {
            value = bool_val ? "true" : "false";
            changed = true;
        }
    }
    else if ((param.type == "enum" || param.type == "dropdown") && !param.enum_values.empty()) {
        int current_idx = 0;
        for (size_t i = 0; i < param.enum_values.size(); i++) {
            if (param.enum_values[i] == value) {
                current_idx = static_cast<int>(i);
                break;
            }
        }

        std::vector<const char*> items;
        for (const auto& ev : param.enum_values) {
            items.push_back(ev.c_str());
        }

        ImGui::SetNextItemWidth(150.0f);
        if (ImGui::Combo("##value", &current_idx, items.data(), static_cast<int>(items.size()))) {
            value = param.enum_values[current_idx];
            changed = true;
        }
    }
    else if (param.type == "file") {
        char file_buf[512];
        strncpy(file_buf, value.c_str(), sizeof(file_buf) - 1);
        file_buf[sizeof(file_buf) - 1] = '\0';

        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::InputText("##value", file_buf, sizeof(file_buf))) {
            value = file_buf;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse")) {
#ifdef _WIN32
            OPENFILENAMEA ofn = {};
            char file[512] = {};
            strncpy(file, value.c_str(), sizeof(file) - 1);
            ofn.lStructSize = sizeof(ofn);
            ofn.lpstrFilter = "All Files\0*.*\0";
            ofn.lpstrFile = file;
            ofn.nMaxFile = sizeof(file);
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
            if (GetOpenFileNameA(&ofn)) {
                value = file;
                changed = true;
            }
#endif
        }
    }
    else if (IsFolderParameterType(param)) {
        char folder_buf[512];
        strncpy(folder_buf, value.c_str(), sizeof(folder_buf) - 1);
        folder_buf[sizeof(folder_buf) - 1] = '\0';

        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::InputText("##value", folder_buf, sizeof(folder_buf))) {
            value = folder_buf;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse")) {
            auto selected_folder = cyxwiz::FileDialogs::SelectFolder(
                "Select Folder",
                value.empty() ? nullptr : value.c_str());
            if (selected_folder) {
                value = *selected_folder;
                changed = true;
            }
        }
    }
    else if (ShouldUseMultilineText(param)) {
        char text_buf[2048];
        strncpy(text_buf, value.c_str(), sizeof(text_buf) - 1);
        text_buf[sizeof(text_buf) - 1] = '\0';

        ImGui::SetNextItemWidth(260.0f);
        if (ImGui::InputTextMultiline("##value", text_buf, sizeof(text_buf), ImVec2(260.0f, 72.0f))) {
            value = text_buf;
            changed = true;
        }
    }
    else {
        char str_buf[512];
        strncpy(str_buf, value.c_str(), sizeof(str_buf) - 1);
        str_buf[sizeof(str_buf) - 1] = '\0';

        ImGuiInputTextFlags flags = ShouldUsePasswordText(param) ? ImGuiInputTextFlags_Password : 0;
        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::InputText("##value", str_buf, sizeof(str_buf), flags)) {
            value = str_buf;
            changed = true;
        }
    }

    if (!param.default_value.empty() && value != param.default_value) {
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset")) {
            value = param.default_value;
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Reset to default: %s", param.default_value.c_str());
        }
    }

    if (changed) {
        std::string error;
        if (!properties_rules::ValidateParameter(value, param, error)) {
            validation_errors[param.name] = error;
        } else {
            validation_errors.erase(param.name);
        }
        invalidate();
        has_error = validation_errors.count(param.name) > 0;
    }

    if (has_error) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", validation_errors[param.name].c_str());
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", validation_errors[param.name].c_str());
            ImGui::EndTooltip();
        }
    }

    ImGui::PopID();
}

} // namespace

void RenderParametersContent(
    MLNode& node,
    const cyxwiz::NodeMetadata* metadata,
    std::map<std::string, std::string>& validation_errors,
    const FallbackRenderer& render_fallback,
    const InvalidateCallback& invalidate) {
    if (!metadata || metadata->parameters.empty()) {
        render_fallback(node);
        return;
    }

    using ParameterGroup = std::pair<std::string, std::vector<const cyxwiz::ParameterDefinition*>>;

    std::vector<ParameterGroup> groups;
    std::vector<const cyxwiz::ParameterDefinition*> advanced_params;
    bool rendered_any = false;

    auto add_to_group = [&](const std::string& group_name, const cyxwiz::ParameterDefinition& param) {
        for (auto& group : groups) {
            if (group.first == group_name) {
                group.second.push_back(&param);
                return;
            }
        }
        groups.push_back({group_name, {&param}});
    };

    for (const auto& param : metadata->parameters) {
        if (properties_rules::ShouldHideGenericParameter(node, param)) {
            validation_errors.erase(param.name);
            continue;
        }

        if (param.advanced) {
            advanced_params.push_back(&param);
        } else {
            add_to_group(param.group, param);
        }
    }

    auto render_params = [&](const std::vector<const cyxwiz::ParameterDefinition*>& params) {
        for (const auto* param : params) {
            RenderParameter(node, *param, validation_errors, invalidate);
            rendered_any = true;
        }
    };

    for (const auto& group : groups) {
        if (group.first.empty()) {
            render_params(group.second);
        } else if (ImGui::TreeNodeEx(group.first.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            render_params(group.second);
            ImGui::TreePop();
        }
    }

    if (!advanced_params.empty() &&
        ImGui::TreeNodeEx("Advanced Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        render_params(advanced_params);
        ImGui::TreePop();
    }

    if (!rendered_any) {
        ImGui::TextDisabled("Configure this node from its dialog.");
    }
}

} // namespace gui::properties_metadata
