#pragma once

#include <functional>

namespace gui {

class NodeEditor;
struct MLNode;

namespace properties_node_editors {

struct RenderNodePropertiesContext {
    NodeEditor* node_editor;
    std::function<void()> invalidate_shapes;
};

void RenderNodeProperties(MLNode& node, RenderNodePropertiesContext context);
void RenderPluginCustomNodeProperties(MLNode& node, RenderNodePropertiesContext context);

} // namespace properties_node_editors

} // namespace gui
