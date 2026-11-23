#pragma once

// Reopening the last project (Preferences > General > When the Engine
// starts, TOFIX129 B9-3): which recent project to open, and the graph a
// project had open, kept relative to the project when it lies inside it so
// a moved project still finds it. Pure: paths in, paths out.

#include <functional>
#include <string>
#include <vector>

namespace cyxwiz::project_session {

// The first path, newest first, that still exists; empty when none does
std::string FirstExisting(const std::vector<std::string>& paths, const std::function<bool(const std::string&)>& exists);

// The graph path to keep in the project file: relative to the project root
// ("cyxgraph/model.cyxgraph", forward slashes) inside it, absolute outside
// it, empty for none
std::string StoredGraph(const std::string& graph_path, const std::string& project_root);

// A kept graph path back to a full path (relative ones under the root)
std::string ResolvedGraph(const std::string& stored, const std::string& project_root);

}  // namespace cyxwiz::project_session
