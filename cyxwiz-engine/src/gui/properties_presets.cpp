#include "properties_presets.h"
#include <spdlog/spdlog.h>

namespace gui::properties_presets {

void SavePreset(const MLNode& node, const std::string& name) {
    // Placeholder until preset persistence is backed by project/user storage.
    spdlog::info("Saved preset '{}' for node type {}", name, static_cast<int>(node.type));
}

void LoadPreset(MLNode& node, const std::string& name) {
    // Placeholder until preset persistence is backed by project/user storage.
    spdlog::info("Loading preset '{}' for node type {}", name, static_cast<int>(node.type));
}

std::vector<std::string> GetPresetsForNodeType(NodeType type) {
    switch (type) {
        case NodeType::Conv2D:
            return {"VGG-style", "ResNet-style", "MobileNet-style"};
        case NodeType::Dense:
            return {"Small (64)", "Medium (256)", "Large (1024)"};
        case NodeType::Adam:
            return {"Default", "Fast Learning", "Fine-tuning"};
        default:
            return {};
    }
}

} // namespace gui::properties_presets
