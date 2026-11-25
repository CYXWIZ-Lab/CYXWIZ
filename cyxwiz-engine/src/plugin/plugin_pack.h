#pragma once

// Installed add-ons (plugin packs) as the plugin loader and the PluginManager
// see them: plain data plus the file checks, filled by
// ConfigureInstalledPluginPacks (plugin_pack_trust.h) from the pack's signed
// inventory. The loader and the manager never link the pack service (its
// OpenSSL hashing): the checks come in as callbacks.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

// One file of a pack's signed inventory.
struct PackComponent {
    std::string path;  // relative, '/' separated, as signed
    std::uint64_t size = 0;
    std::string sha256;
};

enum class PackFileProblem {
    Missing,         // missing, a link, or not a regular file
    SizeDiffers,
    ContentDiffers,  // same size, other contents
    Unreadable,
    NotListed        // in the folder, not in the signed inventory
};

struct PackFileIssue {
    std::string path;
    PackFileProblem problem = PackFileProblem::ContentDiffers;
};

// A pack folder against its inventory: every file, not only the first that
// differs (the refusal says which files).
struct PackFolderCheck {
    std::size_t files_checked = 0;
    std::uint64_t bytes_checked = 0;
    std::vector<PackFileIssue> issues;
    std::string error;  // a check that could not run (a link in the folder)
    bool ok() const { return issues.empty() && error.empty(); }
};

struct PackFileChecks {
    // The folder's file set and every component's size and SHA-256.
    std::function<PackFolderCheck(const std::filesystem::path& folder,
                                  const std::vector<PackComponent>& components)> check_folder;
    // One file (a loaded copy) against its component.
    std::function<bool(const std::filesystem::path& file, const PackComponent& component,
                       PackFileProblem& problem, std::string& error)> check_file;
};

// Why the loader refused an add-on: its files differ from the signed package
// (the issues), or they could not be checked.
struct PackLoadResult {
    bool refused = false;
    std::vector<PackFileIssue> issues;
};

// What the loader checks before it loads an add-on's library.
struct PackLoadCheck {
    std::string pack_id;
    std::string package_version;
    std::vector<PackComponent> components;
    PackFileChecks checks;
};

// An active add-on whose signed, installed metadata verified.
struct PackPlugin {
    std::string plugin_id;
    std::string pack_id;
    std::string title;             // the add-on's name ("Simulation (MuJoCo)")
    std::string package_version;
    std::string companion_base_id;
    std::filesystem::path directory;
    std::vector<PackComponent> components;
};

// An active add-on the Engine does not load, and why.
struct PackRefusal {
    std::string plugin_id;
    std::string pack_id;
    std::string title;
    std::string package_version;
    std::filesystem::path directory;
    std::string reason;                // "bin\mujoco.dll does not match its signed package"
    std::string detail;                // the check's own words, for Details
    std::vector<PackFileIssue> issues;
    std::size_t files_listed = 0;      // files in the signed inventory (0: metadata not read)
    std::chrono::system_clock::time_point checked_at{};
};

struct InstalledPluginPacks {
    std::filesystem::path runtime_root;
    std::vector<PackPlugin> packs;
    std::vector<PackRefusal> refusals;  // refused before any file was read (metadata)
    PackFileChecks checks;
};

// A pack-relative path in this platform's spelling (bin\mujoco.dll on Windows).
inline std::string PackDisplayPath(const std::string& path) {
    return std::filesystem::path(path).make_preferred().string();
}

// Why an add-on was refused, in one clause: the first file that differs and,
// when more do, how many.
inline std::string PackRefusalReason(const std::vector<PackFileIssue>& issues) {
    if (issues.empty()) return "its files could not be checked against its signed package";
    const auto& first = issues.front();
    const std::string file = PackDisplayPath(first.path);
    std::string reason;
    switch (first.problem) {
        case PackFileProblem::Missing: reason = file + " is missing from its signed package"; break;
        case PackFileProblem::NotListed: reason = file + " is not part of its signed package"; break;
        case PackFileProblem::Unreadable: reason = file + " could not be read"; break;
        case PackFileProblem::SizeDiffers:
        case PackFileProblem::ContentDiffers: reason = file + " does not match its signed package"; break;
    }
    if (issues.size() > 1) {
        reason += " (and " + std::to_string(issues.size() - 1) +
                  (issues.size() == 2 ? " other file)" : " other files)");
    }
    return reason;
}

}  // namespace cyxwiz::plugin
