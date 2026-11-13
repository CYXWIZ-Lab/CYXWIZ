#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace gui {

/**
 * JSON tree view with collapsible nodes and color-coded values
 */
class JSONTreeRenderer {
public:
    JSONTreeRenderer() = default;

    // Set JSON content from string
    bool SetContent(const std::string& json_str);


    // Render the tree
    void Render();

    // Check if valid JSON is loaded
    bool IsValid() const { return is_valid_; }

    // Error message if parse failed
    const std::string& GetError() const { return error_; }

private:
    nlohmann::ordered_json json_;  // keys in the document's order
    bool is_valid_ = false;
    std::string error_;
    int node_id_ = 0;  // Unique ID counter for ImGui

    // Recursive rendering
    void RenderNode(const std::string& key, const nlohmann::ordered_json& value, bool is_root = false);
    void RenderValue(const nlohmann::ordered_json& value);
    void RenderContextMenu(const nlohmann::ordered_json& value);
};

} // namespace gui
