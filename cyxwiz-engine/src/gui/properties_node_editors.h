#pragma once

#include <deque>
#include <functional>
#include <map>

namespace gui {

class NodeEditor;
struct MLNode;

namespace properties_node_editors {

struct ScopeBuffer {
    std::deque<float> times;
    std::deque<float> values;
    int max_samples = 500;

    void Push(float t, float v);
    void Clear();
};

struct RenderNodePropertiesContext {
    NodeEditor* node_editor;
    std::map<int, ScopeBuffer>& scope_buffers;
    std::function<void()> invalidate_shapes;
};

void RenderNodeProperties(MLNode& node, RenderNodePropertiesContext context);
void RenderDataPipelineNodeProperties(MLNode& node, RenderNodePropertiesContext context);
void RenderPluginCustomNodeProperties(MLNode& node, RenderNodePropertiesContext context);

} // namespace properties_node_editors

} // namespace gui
