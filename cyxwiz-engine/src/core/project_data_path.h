#pragma once

#include <filesystem>
#include <string>

namespace cyxwiz {

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
inline std::string ResolveProjectDataPath(const std::string& path,
                                          const std::string& project_root) {
    if (path.empty() || project_root.empty()) {
        return path;
    }
    const std::filesystem::path candidate(path);
    // A drive- or root-relative Windows path ("\\data", "C:data") is not a
    // project-relative path; leave it to the OS rather than guess.
    if (candidate.is_absolute() || candidate.has_root_name() ||
        candidate.has_root_directory()) {
        return path;
    }
    return (std::filesystem::path(project_root) / candidate)
        .lexically_normal()
        .string();
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
