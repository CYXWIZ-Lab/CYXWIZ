#include "project_session.h"

#include <filesystem>

namespace cyxwiz::project_session {

namespace fs = std::filesystem;

std::string FirstExisting(const std::vector<std::string>& paths, const std::function<bool(const std::string&)>& exists) {
    for (const auto& path : paths)
        if (!path.empty() && exists(path)) return path;
    return {};
}

std::string StoredGraph(const std::string& graph_path, const std::string& project_root) {
    if (graph_path.empty()) return {};
    const fs::path graph = fs::path(graph_path).lexically_normal();
    if (project_root.empty()) return graph.generic_string();
    const fs::path relative = graph.lexically_relative(fs::path(project_root).lexically_normal());
    const bool inside = !relative.empty() && relative.is_relative() && *relative.begin() != "..";
    return inside ? relative.generic_string() : graph.generic_string();
}

std::string ResolvedGraph(const std::string& stored, const std::string& project_root) {
    if (stored.empty()) return {};
    const fs::path path(stored);
    if (path.is_absolute() || project_root.empty()) return path.lexically_normal().string();
    return (fs::path(project_root) / path).lexically_normal().string();
}

}  // namespace cyxwiz::project_session
