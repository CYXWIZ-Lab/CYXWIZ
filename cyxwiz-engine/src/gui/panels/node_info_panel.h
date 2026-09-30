#pragma once

#include "../../core/extension_node_presentation.h"
#include <optional>
#include <string>
#include "../panel.h"
#include "../../core/node_metadata.h"
#include <imgui.h>
#include <functional>

namespace cyxwiz {

/**
 * NodeInfoPanel - Context-sensitive documentation panel for nodes
 * 
 * Displays detailed information about the currently selected/hovered node:
 * - Node name and category
 * - Brief and detailed description
 * - Input/output ports with types
 * - Parameters with defaults
 * - Example usage
 * - "View Full Docs" link
 * 
 * Updates automatically when:
 * - Node is hovered in Node Browser
 * - Node is selected in Node Editor canvas
 */
class NodeInfoPanel : public Panel {
public:
    NodeInfoPanel();
    ~NodeInfoPanel() override = default;

    void Render() override;
    const char* GetIcon() const override;

    /**
     * Set the node to display info for
     * @param type The node type to display (Unknown to clear)
     */
    void SetSelectedNode(NodeType type);

    /**
     * Hover preview: shown only while the pointer stays on a node (call it
     * every frame while hovering). Without it the pinned node is shown.
     */
    void PreviewNode(NodeType type);

    /**
     * Extension nodes (TOFIX125) share one NodeType; they are identified by
     * their type id "<provider_id>:<TypeName>".
     */
    void SetSelectedExtension(const std::string& type_id);
    void PreviewExtension(const std::string& type_id);

    /**
     * Clear the current selection (show placeholder)
     */
    void ClearSelection();

    /**
     * Get the currently displayed node type
     */
    NodeType GetSelectedNode() const { return selected_type_; }

    /**
     * Check if a node is currently selected
     */
    bool HasSelection() const { return selected_type_ != NodeType::Unknown; }

private:
    // Render sections
    void RenderHeader();
    void RenderDescription();
    void RenderSupport();
    void RenderPorts();
    void RenderParameters();
    void RenderExamples();
    void RenderPlaceholder();

    // Get color for pin type
    ImU32 GetPinTypeColor(PinType type) const;
    const char* GetPinTypeName(PinType type) const;

    // Pinned node (clicked) and the node being displayed this frame.
    NodeType selected_type_ = NodeType::Unknown;
    const NodeMetadata* pinned_metadata_ = nullptr;
    const NodeMetadata* metadata_ = nullptr;
    // Hover preview, valid while PreviewNode is called each frame.
    NodeType preview_type_ = NodeType::Unknown;
    const NodeMetadata* preview_metadata_ = nullptr;
    int preview_frame_ = -10;

    // Extension nodes: copies taken when selected, so an unloaded plugin
    // leaves nothing dangling.
    struct ExtensionView {
        std::string type_id;
        NodeMetadata metadata;
        ExtensionInfoFacts facts;
    };
    static std::optional<ExtensionView> BuildExtensionView(const std::string& type_id);
    void RenderProvidedBy();
    std::optional<ExtensionView> pinned_extension_;
    std::optional<ExtensionView> preview_extension_;
    int preview_extension_frame_ = -10;
    const ExtensionInfoFacts* facts_ = nullptr;  // this frame's, when an extension node is shown
};

} // namespace cyxwiz
