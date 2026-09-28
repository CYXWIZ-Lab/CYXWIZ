#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace cyxwiz {

// The folder of the graph file open in the editor (set on load and save).
// A relative data path that does not exist where the project rule puts it
// is also looked up there and in its parent folders, so an example saved
// with repo-relative paths loads wherever the Engine was started from.
inline std::string& GraphDataSearchDirectory() {
    static std::string directory;
    return directory;
}

// The folder the Engine was started from (main.cpp sets CYXWIZ_LAUNCH_CWD
// before it changes to the executable's folder); empty when unset.
inline std::string EngineLaunchDirectory() {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, "CYXWIZ_LAUNCH_CWD") != 0 || value == nullptr) {
        return {};
    }
    std::string directory(value);
    std::free(value);
    return directory;
#else
    const char* value = std::getenv("CYXWIZ_LAUNCH_CWD");
    return value ? std::string(value) : std::string{};
#endif
}

inline void SetGraphDataSearchDirectoryFromGraphFile(const std::string& graph_file) {
    GraphDataSearchDirectory() =
        graph_file.empty() ? std::string{}
                           : std::filesystem::path(graph_file).parent_path().string();
}

// How a data source path in a saved graph is interpreted (TOFIX101 package B).
//
// An absolute path is used as written. A relative path is taken relative to
// the open project's root, so a graph that says `datasets/raw/web.zip` works
// on any machine and from any working directory. With no project root the
// path is returned unchanged, which keeps the previous behaviour (relative to
// the process working directory) for headless callers that have no project.
//
// Export nodes resolve a relative path against the project root too when
// their path_base is "project" (new nodes); graphs saved before that setting
// load with path_base "exports" (the project exports folder), unchanged.
//
// When that path does not exist, the same relative path is tried under the
// open graph's folder and each of its parents, then under the folder the
// Engine was launched from (CYXWIZ_LAUNCH_CWD); the first that exists wins.
// Paths that already resolve are unchanged.
inline std::string ResolveProjectDataPath(const std::string& path,
                                          const std::string& project_root) {
    if (path.empty()) {
        return path;
    }
    const std::filesystem::path candidate(path);
    // A drive- or root-relative Windows path is not a project-relative path;
    // leave it to the OS rather than guess.
    if (candidate.is_absolute() || candidate.has_root_name() ||
        candidate.has_root_directory()) {
        return path;
    }
    const std::string primary = project_root.empty()
        ? path
        : (std::filesystem::path(project_root) / candidate).lexically_normal().string();
    std::error_code ec;
    if (std::filesystem::exists(primary, ec)) {
        return primary;
    }
    const auto found = [&](const std::filesystem::path& base) {
        return !base.empty() && std::filesystem::exists(base / candidate, ec);
    };
    for (std::filesystem::path dir = GraphDataSearchDirectory(); !dir.empty();
         dir = dir.parent_path()) {
        if (found(dir)) return (dir / candidate).lexically_normal().string();
        if (dir == dir.parent_path()) break;
    }
    const std::string launch_directory = EngineLaunchDirectory();
    if (found(launch_directory)) {
        return (std::filesystem::path(launch_directory) / candidate).lexically_normal().string();
    }
    return primary;
}

// The inverse, used when a user picks a file: a path inside the project is
// stored relative to the project root (forward slashes), so the saved graph
// is portable. Paths outside the project, or with no project, stay as given.
inline std::string MakeProjectRelativePath(const std::string& path,
                                           const std::string& project_root) {
    if (path.empty() || project_root.empty()) {
        return path;
    }
    const std::filesystem::path candidate(path);
    if (!candidate.is_absolute()) {
        return path;
    }
    std::error_code ec;
    const auto root = std::filesystem::weakly_canonical(project_root, ec);
    if (ec) return path;
    const auto target = std::filesystem::weakly_canonical(candidate, ec);
    if (ec) return path;
    const auto relative = target.lexically_relative(root);
    if (relative.empty() || relative.native().empty() || *relative.begin() == "..") {
        return path;
    }
    return relative.generic_string();
}

}  // namespace cyxwiz
