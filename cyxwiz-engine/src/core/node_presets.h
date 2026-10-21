#pragma once

// Node presets (TOFIX129 A7, decision A7-2): a named set of parameter values
// for a node type, applied from the Properties panel's Actions menu. Built-in
// presets are fixed here; saved ones live in one JSON file in the user's
// CyxWiz folder (node_presets.json), keyed by the node type's serialized
// name, so they survive projects and sessions.

#include "graph_model.h"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::node_presets {

struct Preset {
    std::string name;
    std::map<std::string, std::string> parameters;
    bool builtin = false;
};

// Built-in presets of a type (empty for most types).
std::vector<Preset> BuiltinPresets(gui::NodeType type);

// A store of saved presets: pure data, the file is read and written through
// Load/Save so tests use their own path.
class Store {
public:
    // Reads the file; a missing file is an empty store, a broken one is
    // reported through `error` and treated as empty.
    bool Load(const std::filesystem::path& file, std::string* error = nullptr);
    bool Save(const std::filesystem::path& file, std::string* error = nullptr) const;

    // Saved presets of a type, in the order they were added.
    std::vector<Preset> PresetsFor(const std::string& type_name) const;
    // Adds or replaces a preset by name.
    void Put(const std::string& type_name, const std::string& name,
             const std::map<std::string, std::string>& parameters);
    bool Remove(const std::string& type_name, const std::string& name);
    size_t Size() const;

private:
    // type name -> presets
    std::map<std::string, std::vector<Preset>> presets_;
};

// Built-in first, then saved; a saved preset with a built-in's name replaces it.
std::vector<Preset> AllPresets(gui::NodeType type, const std::string& type_name, const Store& store);

// Writes the preset's values into the node (keys the preset does not name
// are left alone). Returns the number of parameters written.
size_t Apply(const Preset& preset, gui::MLNode& node);

// The values a preset saves from a node: every stored parameter except the
// internal "_meta_" keys and the saved view specs.
std::map<std::string, std::string> ParametersToSave(const gui::MLNode& node);

// %APPDATA%/CyxWiz/node_presets.json (or ~/.cyxwiz/node_presets.json).
std::filesystem::path DefaultStoreFile();

}  // namespace cyxwiz::node_presets
