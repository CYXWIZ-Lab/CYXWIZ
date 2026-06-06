#pragma once

namespace gui {

class NodeEditor;
struct MLNode;

namespace properties_advanced {

bool RenderAdvancedSection(NodeEditor* node_editor, MLNode& node, bool section_open);

} // namespace properties_advanced

} // namespace gui
