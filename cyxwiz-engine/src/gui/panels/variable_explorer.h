#pragma once

#include "../panel.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <imgui.h>

namespace scripting {
class ScriptingEngine;
}

namespace cyxwiz {

/**
 * Python variable information for display
 */
struct PythonVariable {
    std::string name;           // Variable name
    std::string type;           // Type name (int, str, list, ndarray, etc.)
    std::string value_repr;     // String representation (truncated)
    std::string shape;          // For arrays: "(10, 20)"
    size_t size_bytes = 0;      // Memory usage estimate
    bool is_expandable = false; // Has children (dict, list, object)
    std::vector<PythonVariable> children;
};

/**
 * VariableExplorerPanel - Display Python namespace variables
 * Shows variables from the current Python session with type, shape, and value
 * Supports auto-refresh and filtering
 */
class VariableExplorerPanel : public Panel {
public:
    VariableExplorerPanel();
    ~VariableExplorerPanel() override = default;

    void Render() override;
    const char* GetIcon() const override;

    // Set scripting engine reference (for Python introspection)
    void SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine);

    // Refresh variables from Python
    void RefreshVariables();

    // Clear all variables
    void Clear();

private:
    void RenderToolbar();
    void RenderVariableTable();
    void RenderVariable(const PythonVariable& var, int depth = 0);

    // Fetch variables from Python namespace
    std::vector<PythonVariable> FetchVariablesFromPython();

    // Parse JSON response from Python introspection
    std::vector<PythonVariable> ParseVariableJson(const std::string& json_str);

    // Get color for variable type
    ImVec4 GetTypeColor(const std::string& type_name) const;

    // Format size for display
    std::string FormatSize(size_t bytes) const;

    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;
    std::vector<PythonVariable> variables_;

    // UI state
    char filter_buffer_[256] = {0};
    std::string filter_;
    bool auto_refresh_ = true;
    float refresh_interval_ = 2.0f;  // seconds
    std::chrono::steady_clock::time_point last_refresh_;

    // Sorting
    enum class SortColumn { Name, Type, Shape, Value };
    SortColumn sort_column_ = SortColumn::Name;
    bool sort_ascending_ = true;

    // Expand/collapse state
    std::vector<std::string> expanded_variables_;
};

} // namespace cyxwiz
