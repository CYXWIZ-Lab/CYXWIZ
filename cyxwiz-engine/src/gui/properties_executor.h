#pragma once

namespace gui {

class NodeEditor;
struct MLNode;

namespace properties_executor {

void RenderExecutorSection(NodeEditor* node_editor, MLNode& node);

} // namespace properties_executor

} // namespace gui
