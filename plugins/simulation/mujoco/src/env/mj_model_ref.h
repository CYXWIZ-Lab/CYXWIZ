#pragma once

// A MuJoCo model named the same way on every machine (TOFIX219): what a
// project saves instead of a path into one user's home folder.
//
//   builtin:half_cheetah.xml                          the plugin's assets
//   menagerie:franka_emika_panda/scene.xml@<commit>   a MuJoCo Menagerie robot at one commit
//   project:mujoco/models/arm/arm.xml                 a file in the project
//   D:/robots/arm.xml                                 any other file (works on this machine only)
//
// A Menagerie reference is found in the project's copy of that robot first
// (<project>/mujoco/models/<robot>/), then in the shared download of that
// commit (<menagerie>/<robot>@<commit12>/). Each download or copy keeps a
// manifest (.cyxwiz_model.json) naming its robot and commit. No ImGui, no
// network.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

enum class MjRefKind { Builtin, Menagerie, Project, Path };

struct MjModelRef {
    MjRefKind kind = MjRefKind::Path;
    std::string robot;   // Menagerie: the robot's folder in the repository ("franka_emika_panda")
    std::string file;    // Builtin: the asset file; Menagerie: the file in the robot's folder; Project: the
                         // path under the project ("mujoco/models/arm/arm.xml"), with forward slashes
    std::string commit;  // Menagerie: the commit (empty: downloaded before versions were kept)
    std::string path;    // Path: the file as given

    bool operator==(const MjModelRef&) const = default;
};

// Empty text: nullopt. Text without a known prefix is a Path.
std::optional<MjModelRef> ParseModelRef(const std::string& text);
std::string FormatModelRef(const MjModelRef& ref);

// Where models are found on this machine (empty: none)
struct MjModelRoots {
    std::filesystem::path builtin;    // the plugin's assets
    std::filesystem::path menagerie;  // shared downloads
    std::filesystem::path project;    // the open project
};

// What a download or copy says about itself
constexpr const char* kModelManifestName = ".cyxwiz_model.json";
struct MjModelManifest {
    std::string source = "menagerie";  // "menagerie" or "url"
    std::string robot;
    std::string commit;
    std::string url;      // source "url": where it came from
    std::string date;     // when it was downloaded or copied (local time, "2026-11-13 15:40")
    uint64_t bytes = 0;   // the folder's files
    int files = 0;

    bool operator==(const MjModelManifest&) const = default;
};
std::optional<MjModelManifest> ReadModelManifest(const std::filesystem::path& folder);
bool WriteModelManifest(const std::filesystem::path& folder, const MjModelManifest& manifest, std::string& error);

// The folder a pinned shared download lives in: <root>/<robot>@<commit12>
std::filesystem::path MenagerieFolder(const std::filesystem::path& root, const std::string& robot,
                                      const std::string& commit);
// The project's own models: <project>/mujoco/models
std::filesystem::path ProjectModelsFolder(const std::filesystem::path& project);
// A commit as shown ("2f1a9c3"); empty: "version unknown"
std::string ShortCommit(const std::string& commit);

enum class MjModelWhere { ProjectCopy, InProject, SharedDownload, Builtin, Elsewhere, Missing };
struct MjResolvedModel {
    std::filesystem::path file;  // empty when missing
    MjModelWhere where = MjModelWhere::Missing;
    bool portable = false;       // another machine finds or fetches the same model
    std::string why;             // a missing model: what is missing, in words
};
MjResolvedModel ResolveModelRef(const MjModelRef& ref, const MjModelRoots& roots);

// The reference a file on this machine gets: a Menagerie robot's file (shared
// download or the project's copy, by its manifest) -> menagerie:; another file
// in the project -> project:; the plugin's assets -> builtin:; else its path
MjModelRef RefForFile(const std::filesystem::path& file, const MjModelRoots& roots);

// One line about where the model is: "copy in this project", "shared download",
// "built in", "not in the project: other PCs will not find it", or the missing reason
std::string DescribeWhere(const MjResolvedModel& resolved);

}  // namespace cyxwiz::plugin::mujoco
