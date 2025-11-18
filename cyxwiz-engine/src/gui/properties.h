#pragma once

#include <string>
#include <map>

namespace gui {

// Forward declarations
enum class NodeType;
struct MLNode;

class Properties {
public:
    Properties();
    ~Properties();

    void Render();

    // Set the currently selected node to display properties for
    void SetSelectedNode(MLNode* node);
    void ClearSelection();

private:
    void RenderNodeProperties(MLNode& node);

    bool show_window_;
    MLNode* selected_node_ = nullptr;
};

} // namespace gui
