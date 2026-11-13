#include "mj_model_ref.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <system_error>

namespace cyxwiz::plugin::mujoco {

namespace fs = std::filesystem;

namespace {

constexpr const char* kBuiltin = "builtin:";
constexpr const char* kMenagerie = "menagerie:";
constexpr const char* kProject = "project:";

bool StartsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

// `file` inside `dir`: its path below it with forward slashes, else nullopt.
// Compared as written by the OS (case-insensitive on Windows).
std::optional<std::string> Below(const fs::path& file, const fs::path& dir) {
    if (dir.empty() || file.empty()) return std::nullopt;
    std::error_code ec;
    fs::path f = fs::weakly_canonical(file, ec);
    if (ec) f = file.lexically_normal();
    fs::path d = fs::weakly_canonical(dir, ec);
    if (ec) d = dir.lexically_normal();
    std::string fs_ = f.generic_string();
    std::string ds = d.generic_string();
    if (!ds.empty() && ds.back() != '/') ds += '/';
#ifdef _WIN32
    std::string fl = fs_, dl = ds;
    std::transform(fl.begin(), fl.end(), fl.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(dl.begin(), dl.end(), dl.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (fl.rfind(dl, 0) != 0) return std::nullopt;
#else
    if (fs_.rfind(ds, 0) != 0) return std::nullopt;
#endif
    return fs_.substr(ds.size());
}

bool IsFile(const fs::path& p) {
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

// A path with no ".." and no root: safe to join under a folder
bool SafeRelative(const std::string& rel) {
    if (rel.empty()) return false;
    const fs::path p(rel);
    if (p.is_absolute() || p.has_root_name() || p.has_root_directory()) return false;
    for (const auto& part : p)
        if (part == "..") return false;
    return true;
}

}  // namespace

std::optional<MjModelRef> ParseModelRef(const std::string& text) {
    if (text.empty()) return std::nullopt;
    MjModelRef ref;
    if (StartsWith(text, kBuiltin)) {
        ref.kind = MjRefKind::Builtin;
        ref.file = text.substr(std::string(kBuiltin).size());
        return ref;
    }
    if (StartsWith(text, kMenagerie)) {
        ref.kind = MjRefKind::Menagerie;
        std::string rest = text.substr(std::string(kMenagerie).size());
        const size_t at = rest.rfind('@');
        if (at != std::string::npos) {
            ref.commit = rest.substr(at + 1);
            rest.resize(at);
        }
        const size_t slash = rest.find('/');
        ref.robot = slash == std::string::npos ? rest : rest.substr(0, slash);
        ref.file = slash == std::string::npos ? std::string("scene.xml") : rest.substr(slash + 1);
        return ref;
    }
    if (StartsWith(text, kProject)) {
        ref.kind = MjRefKind::Project;
        ref.file = text.substr(std::string(kProject).size());
        return ref;
    }
    ref.kind = MjRefKind::Path;
    ref.path = text;
    return ref;
}

std::string FormatModelRef(const MjModelRef& ref) {
    switch (ref.kind) {
        case MjRefKind::Builtin: return kBuiltin + ref.file;
        case MjRefKind::Menagerie:
            return kMenagerie + ref.robot + "/" + ref.file + (ref.commit.empty() ? "" : "@" + ref.commit);
        case MjRefKind::Project: return kProject + ref.file;
        case MjRefKind::Path: return ref.path;
    }
    return ref.path;
}

std::optional<MjModelManifest> ReadModelManifest(const fs::path& folder) {
    std::ifstream in(folder / kModelManifestName);
    if (!in) return std::nullopt;
    std::stringstream buffer;
    buffer << in.rdbuf();
    const nlohmann::json j = nlohmann::json::parse(buffer.str(), nullptr, false);
    if (!j.is_object()) return std::nullopt;
    MjModelManifest m;
    const auto text = [&j](const char* key, std::string& out) {
        if (j.contains(key) && j[key].is_string()) out = j[key].get<std::string>();
    };
    text("source", m.source);
    text("robot", m.robot);
    text("commit", m.commit);
    text("url", m.url);
    text("date", m.date);
    if (j.contains("bytes") && j["bytes"].is_number_unsigned()) m.bytes = j["bytes"].get<uint64_t>();
    if (j.contains("files") && j["files"].is_number_integer()) m.files = j["files"].get<int>();
    if (m.source == "menagerie" && m.robot.empty()) return std::nullopt;
    return m;
}

bool WriteModelManifest(const fs::path& folder, const MjModelManifest& m, std::string& error) {
    nlohmann::json j;
    j["source"] = m.source;
    j["robot"] = m.robot;
    j["commit"] = m.commit;
    if (!m.url.empty()) j["url"] = m.url;
    j["date"] = m.date;
    j["bytes"] = m.bytes;
    j["files"] = m.files;
    std::error_code ec;
    fs::create_directories(folder, ec);
    std::ofstream out(folder / kModelManifestName, std::ios::binary | std::ios::trunc);
    if (!out) {
        error = "Could not write " + (folder / kModelManifestName).string() + ".";
        return false;
    }
    out << j.dump(2) << "\n";
    return static_cast<bool>(out);
}

fs::path MenagerieFolder(const fs::path& root, const std::string& robot, const std::string& commit) {
    if (commit.empty()) return root / robot;  // downloaded before versions were kept
    return root / (robot + "@" + commit.substr(0, std::min<size_t>(12, commit.size())));
}

fs::path ProjectModelsFolder(const fs::path& project) { return project / "mujoco" / "models"; }

std::string ShortCommit(const std::string& commit) {
    return commit.empty() ? std::string("version unknown") : commit.substr(0, std::min<size_t>(7, commit.size()));
}

MjResolvedModel ResolveModelRef(const MjModelRef& ref, const MjModelRoots& roots) {
    MjResolvedModel r;
    switch (ref.kind) {
        case MjRefKind::Builtin: {
            if (SafeRelative(ref.file) && !roots.builtin.empty() && IsFile(roots.builtin / ref.file)) {
                r.file = roots.builtin / ref.file;
                r.where = MjModelWhere::Builtin;
                r.portable = true;
            } else {
                r.why = "The built-in model " + ref.file + " is not in this MuJoCo plugin.";
            }
            return r;
        }
        case MjRefKind::Menagerie: {
            if (!SafeRelative(ref.robot) || !SafeRelative(ref.file)) {
                r.why = "The model reference is not valid.";
                return r;
            }
            // The project's copy of that robot first
            if (!roots.project.empty()) {
                const fs::path copy = ProjectModelsFolder(roots.project) / ref.robot;
                const auto manifest = ReadModelManifest(copy);
                if (manifest && manifest->robot == ref.robot && IsFile(copy / ref.file)) {
                    r.file = copy / ref.file;
                    r.where = MjModelWhere::ProjectCopy;
                    r.portable = true;
                    return r;
                }
            }
            if (!roots.menagerie.empty()) {
                const fs::path shared = MenagerieFolder(roots.menagerie, ref.robot, ref.commit) / ref.file;
                if (IsFile(shared)) {
                    r.file = shared;
                    r.where = MjModelWhere::SharedDownload;
                    r.portable = !ref.commit.empty();
                    return r;
                }
            }
            r.why = "Not on this PC: MuJoCo Menagerie " + ref.robot + ", " +
                    (ref.commit.empty() ? std::string("version unknown") : "version " + ShortCommit(ref.commit)) + ".";
            return r;
        }
        case MjRefKind::Project: {
            if (SafeRelative(ref.file) && !roots.project.empty() && IsFile(roots.project / ref.file)) {
                r.file = roots.project / ref.file;
                r.where = MjModelWhere::InProject;
                r.portable = true;
            } else {
                r.why = roots.project.empty() ? "Open the project that holds " + ref.file + "."
                                              : "Not in this project: " + ref.file + ".";
            }
            return r;
        }
        case MjRefKind::Path: {
            if (IsFile(ref.path)) {
                r.file = ref.path;
                r.where = MjModelWhere::Elsewhere;
            } else {
                r.why = "Not found: " + ref.path + ".";
            }
            return r;
        }
    }
    return r;
}

MjModelRef RefForFile(const fs::path& file, const MjModelRoots& roots) {
    MjModelRef ref;
    // A Menagerie robot's file: in a shared download, or the project's copy of one
    const auto menagerie_ref = [&](const fs::path& folder, const std::string& rel) -> std::optional<MjModelRef> {
        const auto manifest = ReadModelManifest(folder);
        if (!manifest || manifest->source != "menagerie") return std::nullopt;
        MjModelRef m;
        m.kind = MjRefKind::Menagerie;
        m.robot = manifest->robot;
        m.commit = manifest->commit;
        m.file = rel;
        return m;
    };
    const auto split_first = [](const std::string& rel, std::string& first, std::string& rest) {
        const size_t slash = rel.find('/');
        if (slash == std::string::npos) return false;
        first = rel.substr(0, slash);
        rest = rel.substr(slash + 1);
        return true;
    };
    std::string first, rest;
    if (const auto rel = Below(file, roots.menagerie); rel && split_first(*rel, first, rest)) {
        if (auto m = menagerie_ref(roots.menagerie / first, rest)) return *m;
        // Downloaded before versions were kept: the robot's folder, version unknown
        MjModelRef m;
        m.kind = MjRefKind::Menagerie;
        m.robot = first.substr(0, first.find('@'));
        m.file = rest;
        return m;
    }
    if (!roots.project.empty()) {
        if (const auto rel = Below(file, ProjectModelsFolder(roots.project)); rel && split_first(*rel, first, rest))
            if (auto m = menagerie_ref(ProjectModelsFolder(roots.project) / first, rest)) return *m;
        if (const auto rel = Below(file, roots.project)) {
            ref.kind = MjRefKind::Project;
            ref.file = *rel;
            return ref;
        }
    }
    if (const auto rel = Below(file, roots.builtin); rel && rel->find('/') == std::string::npos) {
        ref.kind = MjRefKind::Builtin;
        ref.file = *rel;
        return ref;
    }
    ref.kind = MjRefKind::Path;
    ref.path = file.string();
    return ref;
}

std::string DescribeWhere(const MjResolvedModel& resolved) {
    switch (resolved.where) {
        case MjModelWhere::ProjectCopy: return "copy in this project";
        case MjModelWhere::InProject: return "in this project";
        case MjModelWhere::SharedDownload:
            return resolved.portable ? "shared download" : "shared download, version unknown: download it again to pin it";
        case MjModelWhere::Builtin: return "built in";
        case MjModelWhere::Elsewhere: return "not in the project: other PCs will not find it";
        case MjModelWhere::Missing: return resolved.why;
    }
    return resolved.why;
}

}  // namespace cyxwiz::plugin::mujoco
