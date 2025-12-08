#include "variable_explorer.h"
#include "../../scripting/scripting_engine.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>

namespace cyxwiz {

VariableExplorerPanel::VariableExplorerPanel()
    : Panel("Variable Explorer", true)
{
    last_refresh_ = std::chrono::steady_clock::now();
}

const char* VariableExplorerPanel::GetIcon() const {
    return ICON_FA_LIST_UL;
}

void VariableExplorerPanel::SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine) {
    scripting_engine_ = engine;
}

void VariableExplorerPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_LIST_UL " Variable Explorer", &visible_)) {
        focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

        RenderToolbar();
        ImGui::Separator();
        RenderVariableTable();

        // Auto-refresh logic
        if (auto_refresh_ && scripting_engine_ && scripting_engine_->IsInitialized()) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<float>(now - last_refresh_).count();
            if (elapsed >= refresh_interval_) {
                RefreshVariables();
            }
        }
    }
    ImGui::End();
}

void VariableExplorerPanel::RenderToolbar() {
    // Refresh button
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshVariables();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Refresh variables (F5)");
    }

    ImGui::SameLine();

    // Auto-refresh toggle
    if (ImGui::Checkbox("Auto", &auto_refresh_)) {
        if (auto_refresh_) {
            last_refresh_ = std::chrono::steady_clock::now();
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Auto-refresh every %.1f seconds", refresh_interval_);
    }

    ImGui::SameLine();

    // Refresh interval slider (only visible when auto-refresh is on)
    if (auto_refresh_) {
        ImGui::SetNextItemWidth(80);
        ImGui::SliderFloat("##interval", &refresh_interval_, 0.5f, 10.0f, "%.1fs");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Refresh interval");
        }
        ImGui::SameLine();
    }

    // Clear button
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_TRASH " Clear")) {
        Clear();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Clear variable list");
    }

    // Filter input
    ImGui::SameLine();
    float remaining_width = ImGui::GetContentRegionAvail().x;
    ImGui::SetNextItemWidth(remaining_width > 100 ? remaining_width : 100);
    if (ImGui::InputTextWithHint("##filter", ICON_FA_FILTER " Filter...", filter_buffer_, sizeof(filter_buffer_))) {
        filter_ = filter_buffer_;
    }
}

void VariableExplorerPanel::RenderVariableTable() {
    if (variables_.empty()) {
        ImGui::TextDisabled("No variables in Python namespace");
        ImGui::TextDisabled("Run some code to see variables here");
        return;
    }

    // Table flags
    ImGuiTableFlags flags = ImGuiTableFlags_Resizable
                          | ImGuiTableFlags_Reorderable
                          | ImGuiTableFlags_Sortable
                          | ImGuiTableFlags_RowBg
                          | ImGuiTableFlags_Borders
                          | ImGuiTableFlags_ScrollY;

    if (ImGui::BeginTable("##variables", 4, flags)) {
        // Setup columns
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 120);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Shape", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        // Handle sorting
        if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
            if (sort_specs->SpecsDirty) {
                // Sort the variables
                if (sort_specs->SpecsCount > 0) {
                    const ImGuiTableColumnSortSpecs& spec = sort_specs->Specs[0];
                    bool ascending = (spec.SortDirection == ImGuiSortDirection_Ascending);

                    std::sort(variables_.begin(), variables_.end(),
                        [&spec, ascending](const PythonVariable& a, const PythonVariable& b) {
                            int cmp = 0;
                            switch (spec.ColumnIndex) {
                                case 0: cmp = a.name.compare(b.name); break;
                                case 1: cmp = a.type.compare(b.type); break;
                                case 2: cmp = a.shape.compare(b.shape); break;
                                case 3: cmp = a.value_repr.compare(b.value_repr); break;
                            }
                            return ascending ? (cmp < 0) : (cmp > 0);
                        });
                }
                sort_specs->SpecsDirty = false;
            }
        }

        // Render rows
        for (const auto& var : variables_) {
            // Filter check
            if (!filter_.empty()) {
                std::string lower_name = var.name;
                std::string lower_filter = filter_;
                std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
                std::transform(lower_filter.begin(), lower_filter.end(), lower_filter.begin(), ::tolower);
                if (lower_name.find(lower_filter) == std::string::npos) {
                    continue;
                }
            }

            RenderVariable(var);
        }

        ImGui::EndTable();
    }

    // Status line
    ImGui::Text("%zu variables", variables_.size());
}

void VariableExplorerPanel::RenderVariable(const PythonVariable& var, int depth) {
    ImGui::TableNextRow();

    // Name column
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%s", var.name.c_str());

    // Type column with color
    ImGui::TableSetColumnIndex(1);
    ImVec4 type_color = GetTypeColor(var.type);
    ImGui::TextColored(type_color, "%s", var.type.c_str());

    // Shape column
    ImGui::TableSetColumnIndex(2);
    if (!var.shape.empty()) {
        ImGui::TextDisabled("%s", var.shape.c_str());
    }

    // Value column
    ImGui::TableSetColumnIndex(3);

    // Truncate value for display
    std::string display_value = var.value_repr;
    if (display_value.length() > 80) {
        display_value = display_value.substr(0, 77) + "...";
    }
    ImGui::TextWrapped("%s", display_value.c_str());

    // Tooltip with full value
    if (ImGui::IsItemHovered() && var.value_repr.length() > 80) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(400.0f);
        ImGui::TextUnformatted(var.value_repr.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void VariableExplorerPanel::RefreshVariables() {
    last_refresh_ = std::chrono::steady_clock::now();

    if (!scripting_engine_ || !scripting_engine_->IsInitialized()) {
        return;
    }

    // Don't refresh while a script is running
    if (scripting_engine_->IsScriptRunning()) {
        return;
    }

    variables_ = FetchVariablesFromPython();
}

std::vector<PythonVariable> VariableExplorerPanel::FetchVariablesFromPython() {
    std::vector<PythonVariable> result;

    if (!scripting_engine_) {
        return result;
    }

    // Python code to introspect the namespace
    std::string py_code = R"(
import json
import sys

def _cyxwiz_get_variables():
    result = []
    main_globals = __import__('__main__').__dict__

    for name, value in main_globals.items():
        # Skip private/magic variables and modules
        if name.startswith('_'):
            continue
        if name in ('__name__', '__doc__', '__package__', '__loader__', '__spec__', '__builtins__', '__file__'):
            continue

        try:
            type_name = type(value).__name__

            # Skip functions, modules, and classes by default
            if type_name in ('function', 'module', 'type', 'builtin_function_or_method'):
                continue

            var_info = {
                'name': name,
                'type': type_name,
                'value': '',
                'shape': '',
                'size': 0
            }

            # Get string representation (truncated)
            try:
                repr_val = repr(value)
                if len(repr_val) > 200:
                    repr_val = repr_val[:200] + '...'
                var_info['value'] = repr_val
            except:
                var_info['value'] = '<error getting repr>'

            # Get size
            try:
                var_info['size'] = sys.getsizeof(value)
            except:
                pass

            # Handle numpy arrays
            if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                try:
                    var_info['shape'] = str(value.shape)
                    var_info['type'] = f"{type_name}[{value.dtype}]"
                except:
                    pass

            # Handle torch tensors
            elif type_name == 'Tensor' and hasattr(value, 'shape'):
                try:
                    var_info['shape'] = str(tuple(value.shape))
                    if hasattr(value, 'dtype'):
                        var_info['type'] = f"Tensor[{value.dtype}]"
                except:
                    pass

            # Handle lists/tuples
            elif isinstance(value, (list, tuple)):
                try:
                    var_info['shape'] = f'({len(value)},)'
                except:
                    pass

            # Handle dicts
            elif isinstance(value, dict):
                try:
                    var_info['shape'] = f'{{{len(value)} items}}'
                except:
                    pass

            # Handle sets
            elif isinstance(value, (set, frozenset)):
                try:
                    var_info['shape'] = f'{{{len(value)} items}}'
                except:
                    pass

            # Handle strings
            elif isinstance(value, str):
                var_info['shape'] = f'len={len(value)}'

            result.append(var_info)
        except Exception as e:
            pass  # Skip variables that cause errors

    return json.dumps(result)

_cyxwiz_result = _cyxwiz_get_variables()
del _cyxwiz_get_variables
)";

    try {
        auto exec_result = scripting_engine_->ExecuteScript(py_code);
        if (!exec_result.success) {
            spdlog::warn("Variable introspection failed: {}", exec_result.error_message);
            return result;
        }

        // Get the result variable
        std::string get_result_code = "_cyxwiz_result";
        auto value_result = scripting_engine_->ExecuteCommand("print(_cyxwiz_result); del _cyxwiz_result");

        if (value_result.success && !value_result.output.empty()) {
            result = ParseVariableJson(value_result.output);
        }
    } catch (const std::exception& e) {
        spdlog::error("Exception during variable introspection: {}", e.what());
    }

    return result;
}

std::vector<PythonVariable> VariableExplorerPanel::ParseVariableJson(const std::string& json_str) {
    std::vector<PythonVariable> result;

    try {
        // Trim whitespace
        std::string trimmed = json_str;
        size_t start = trimmed.find_first_not_of(" \t\n\r");
        size_t end = trimmed.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            trimmed = trimmed.substr(start, end - start + 1);
        }

        auto json = nlohmann::json::parse(trimmed);

        for (const auto& item : json) {
            PythonVariable var;
            var.name = item.value("name", "");
            var.type = item.value("type", "");
            var.value_repr = item.value("value", "");
            var.shape = item.value("shape", "");
            var.size_bytes = item.value("size", 0);
            result.push_back(var);
        }
    } catch (const nlohmann::json::exception& e) {
        spdlog::warn("Failed to parse variable JSON: {}", e.what());
    }

    return result;
}

void VariableExplorerPanel::Clear() {
    variables_.clear();
}

ImVec4 VariableExplorerPanel::GetTypeColor(const std::string& type_name) const {
    // Color coding by type
    if (type_name == "int" || type_name == "float" || type_name == "complex") {
        return ImVec4(0.4f, 0.7f, 1.0f, 1.0f);  // Blue for numbers
    }
    if (type_name == "str" || type_name == "bytes") {
        return ImVec4(0.6f, 0.9f, 0.6f, 1.0f);  // Green for strings
    }
    if (type_name == "bool") {
        return ImVec4(0.9f, 0.6f, 0.9f, 1.0f);  // Purple for booleans
    }
    if (type_name == "list" || type_name == "tuple" || type_name == "set") {
        return ImVec4(0.9f, 0.7f, 0.4f, 1.0f);  // Orange for sequences
    }
    if (type_name == "dict") {
        return ImVec4(1.0f, 0.8f, 0.4f, 1.0f);  // Yellow for dicts
    }
    if (type_name.find("ndarray") != std::string::npos) {
        return ImVec4(1.0f, 0.5f, 0.3f, 1.0f);  // Orange-red for numpy arrays
    }
    if (type_name.find("Tensor") != std::string::npos) {
        return ImVec4(0.9f, 0.4f, 0.4f, 1.0f);  // Red for tensors
    }
    if (type_name.find("DataFrame") != std::string::npos || type_name.find("Series") != std::string::npos) {
        return ImVec4(0.5f, 0.8f, 0.8f, 1.0f);  // Teal for pandas
    }
    if (type_name == "NoneType") {
        return ImVec4(0.6f, 0.6f, 0.6f, 1.0f);  // Gray for None
    }

    return ImVec4(0.8f, 0.8f, 0.8f, 1.0f);  // Default gray
}

std::string VariableExplorerPanel::FormatSize(size_t bytes) const {
    if (bytes < 1024) {
        return std::to_string(bytes) + " B";
    } else if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + " KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    } else {
        return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
    }
}

} // namespace cyxwiz
