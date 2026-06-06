#pragma once

#include "node_editor.h"
#include <string>
#include <vector>

namespace gui::properties_presets {

void SavePreset(const MLNode& node, const std::string& name);
void LoadPreset(MLNode& node, const std::string& name);
std::vector<std::string> GetPresetsForNodeType(NodeType type);

} // namespace gui::properties_presets
