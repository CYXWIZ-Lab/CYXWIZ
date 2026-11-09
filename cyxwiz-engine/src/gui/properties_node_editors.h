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
// Policy Head, Value Head, PPO: what the environment makes of them (the
// head's output layer, PPO's tuned settings), above their settings
void RenderRlNodeFacts(MLNode& node, RenderNodePropertiesContext context);

} // namespace properties_node_editors

} // namespace gui
