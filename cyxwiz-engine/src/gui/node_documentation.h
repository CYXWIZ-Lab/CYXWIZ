#pragma once

#include <string>
#include <vector>
#include <map>
#include "node_editor.h"

namespace cyxwiz { struct NodeMetadata; }

namespace gui {

// Documentation structure for a node type
struct NodeDocumentation {
    std::string title;
    std::string description;
    std::string usage;
    std::vector<std::pair<std::string, std::string>> parameters;  // name, description
    std::vector<std::string> tips;
    std::string category;
};

// Singleton manager for node documentation
class NodeDocumentationManager {
public:
    static NodeDocumentationManager& Instance();

    // Get documentation for a node type (returns nullptr if not found)
    const NodeDocumentation* GetDocumentation(NodeType type) const;

    // Render a tooltip for the given node type
    // Call this when hovering over a node
    void RenderTooltip(NodeType type);

    // Render the tooltip for a node on the canvas. Extension nodes describe
    // their own type (or say that it is not installed), not PluginCustom.
    void RenderTooltip(const MLNode& node);

    // Render a help marker (?) icon with tooltip
    // Returns true if hovered
    bool RenderHelpMarker(NodeType type);

    // Render compact tooltip (just title and description)
    void RenderCompactTooltip(NodeType type);

    // Get category name for a node type
    static const char* GetCategoryName(NodeType type);


private:
    NodeDocumentationManager();
    ~NodeDocumentationManager() = default;
    NodeDocumentationManager(const NodeDocumentationManager&) = delete;
    NodeDocumentationManager& operator=(const NodeDocumentationManager&) = delete;

    void InitializeDocumentation();
    void RenderTooltip(const cyxwiz::NodeMetadata* metadata, const NodeDocumentation* doc);

    std::map<NodeType, NodeDocumentation> docs_;
};

} // namespace gui
