#!/usr/bin/env python3
"""Implement preset persistence for Phase 3 Properties Panel"""

# New implementation for SavePreset, LoadPreset, GetPresetsForNodeType

preset_implementation = '''
void Properties::SavePreset(const MLNode& node, const std::string& name) {
    // Create presets directory if it doesn't exist
    std::filesystem::path presets_dir = std::filesystem::current_path() / "presets";
    if (!std::filesystem::exists(presets_dir)) {
        std::filesystem::create_directories(presets_dir);
    }

    // Create filename: NodeType_PresetName.json
    std::string type_name = std::to_string(static_cast<int>(node.type));
    std::string safe_name = name;
    // Replace spaces with underscores
    std::replace(safe_name.begin(), safe_name.end(), ' ', '_');

    std::filesystem::path preset_file = presets_dir / (type_name + "_" + safe_name + ".json");

    // Build JSON
    nlohmann::json j;
    j["node_type"] = static_cast<int>(node.type);
    j["preset_name"] = name;
    j["parameters"] = nlohmann::json::object();

    for (const auto& [key, value] : node.parameters) {
        j["parameters"][key] = value;
    }

    // Write to file
    std::ofstream file(preset_file);
    if (file.is_open()) {
        file << j.dump(2);
        file.close();
        spdlog::info("Saved preset '{}' to {}", name, preset_file.string());
    } else {
        spdlog::error("Failed to save preset '{}' to {}", name, preset_file.string());
    }
}

void Properties::LoadPreset(MLNode& node, const std::string& name) {
    std::filesystem::path presets_dir = std::filesystem::current_path() / "presets";

    std::string type_name = std::to_string(static_cast<int>(node.type));
    std::string safe_name = name;
    std::replace(safe_name.begin(), safe_name.end(), ' ', '_');

    std::filesystem::path preset_file = presets_dir / (type_name + "_" + safe_name + ".json");

    if (!std::filesystem::exists(preset_file)) {
        spdlog::warn("Preset file not found: {}", preset_file.string());
        return;
    }

    try {
        std::ifstream file(preset_file);
        nlohmann::json j = nlohmann::json::parse(file);

        // Load parameters
        if (j.contains("parameters") && j["parameters"].is_object()) {
            for (auto& [key, value] : j["parameters"].items()) {
                node.parameters[key] = value.get<std::string>();
            }
        }

        spdlog::info("Loaded preset '{}' from {}", name, preset_file.string());
        InvalidateShapes();

    } catch (const std::exception& e) {
        spdlog::error("Failed to load preset '{}': {}", name, e.what());
    }
}

std::vector<std::string> Properties::GetPresetsForNodeType(NodeType type) {
    std::vector<std::string> presets;

    std::filesystem::path presets_dir = std::filesystem::current_path() / "presets";
    if (!std::filesystem::exists(presets_dir)) {
        return presets;
    }

    std::string type_prefix = std::to_string(static_cast<int>(type)) + "_";

    try {
        for (const auto& entry : std::filesystem::directory_iterator(presets_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                std::string filename = entry.path().stem().string();

                // Check if this preset is for our node type
                if (filename.substr(0, type_prefix.size()) == type_prefix) {
                    // Extract preset name (after the type prefix)
                    std::string preset_name = filename.substr(type_prefix.size());
                    // Replace underscores with spaces
                    std::replace(preset_name.begin(), preset_name.end(), '_', ' ');
                    presets.push_back(preset_name);
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("Error scanning presets directory: {}", e.what());
    }

    // Also add built-in presets (hardcoded defaults)
    switch (type) {
        case NodeType::Conv2D:
            if (std::find(presets.begin(), presets.end(), "VGG-style") == presets.end())
                presets.push_back("VGG-style");
            if (std::find(presets.begin(), presets.end(), "ResNet-style") == presets.end())
                presets.push_back("ResNet-style");
            break;
        case NodeType::Dense:
            if (std::find(presets.begin(), presets.end(), "Small (64)") == presets.end())
                presets.push_back("Small (64)");
            if (std::find(presets.begin(), presets.end(), "Medium (256)") == presets.end())
                presets.push_back("Medium (256)");
            break;
        case NodeType::Adam:
            if (std::find(presets.begin(), presets.end(), "Default") == presets.end())
                presets.push_back("Default");
            if (std::find(presets.begin(), presets.end(), "Fine-tuning") == presets.end())
                presets.push_back("Fine-tuning");
            break;
        default:
            break;
    }

    return presets;
}
'''

# Read the current file
with open("cyxwiz-engine/src/gui/properties.cpp", "r") as f:
    content = f.read()

# Check if filesystem header is included
if "#include <filesystem>" not in content:
    # Add filesystem include after other includes
    content = content.replace(
        "#include <algorithm>",
        "#include <algorithm>\n#include <filesystem>\n#include <fstream>\n#include <nlohmann/json.hpp>"
    )

# Replace the stub implementations
old_save = '''void Properties::SavePreset(const MLNode& node, const std::string& name) {
    \\ TODO: Implement preset persistence (save to JSON file)
    spdlog::info("Saved preset '{}' for node type {}", name, static_cast<int>(node.type));
}'''

old_load = '''void Properties::LoadPreset(MLNode& node, const std::string& name) {
    \\ TODO: Implement preset loading (read from JSON file)
    spdlog::info("Loading preset '{}' for node type {}", name, static_cast<int>(node.type));
    InvalidateShapes();
}'''

old_get = '''std::vector<std::string> Properties::GetPresetsForNodeType(NodeType type) {
    \\ TODO: Implement preset discovery (scan presets directory)
    std::vector<std::string> presets;

    // Return some built-in presets for common node types
    switch (type) {
        case NodeType::Conv2D:
            presets = {"VGG-style", "ResNet-style", "MobileNet-style"};
            break;
        case NodeType::Dense:
            presets = {"Small (64)", "Medium (256)", "Large (1024)"};
            break;
        case NodeType::Adam:
            presets = {"Default", "Fast Learning", "Fine-tuning"};
            break;
        default:
            break;
    }

    return presets;
}'''

# Check if we can find the patterns
if "TODO: Implement preset persistence" in content:
    print("Found SavePreset stub")
else:
    print("SavePreset stub not found - may already be implemented")

if "TODO: Implement preset loading" in content:
    print("Found LoadPreset stub")
else:
    print("LoadPreset stub not found - may already be implemented")

if "TODO: Implement preset discovery" in content:
    print("Found GetPresetsForNodeType stub")
else:
    print("GetPresetsForNodeType stub not found - may already be implemented")

# Write the file with includes added
with open("cyxwiz-engine/src/gui/properties.cpp", "w") as f:
    f.write(content)

print("Added filesystem includes")
print("NOTE: Manual replacement of stub functions needed")
