#pragma once

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

// What a removal request for one pack ID removes: exactly one of the two is set.
struct BackendPackRemovalTarget {
    std::string backend;    // a backend pack active in active-runtime.json
    std::string plugin_id;  // a plugin pack, active or only installed
};

// Resolves a pack ID for removal. A backend pack active in
// active-runtime.json resolves from that file alone, so a damaged
// active-plugins.json never blocks its removal. Otherwise the ID is a plugin
// pack: active in active-plugins.json, or installed (not active) under
// runtime/plugins/<plugin-id>/<pack-id>/. False with a plain message when the
// state the ID needs cannot be read, or the ID is neither.
bool ResolveBackendPackRemovalTarget(
    const std::filesystem::path& runtime_root,
    const std::string& pack_id,
    BackendPackRemovalTarget& target,
    std::string& error);

}  // namespace cyxwiz::runtime
