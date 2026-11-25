#include "plugin_presentation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz {

namespace {

constexpr int kStartupListVersion = 1;

std::string Count(size_t n, const char* one, const char* many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

void Detail(std::vector<std::pair<std::string, std::string>>& details, const char* key, std::string value) {
    if (!value.empty()) details.emplace_back(key, std::move(value));
}

constexpr const char* kSignedWith = "A pack key in this install's trust file";

PluginStartupEntry* Find(PluginStartupList& list, const std::string& id) {
    for (auto& entry : list.entries) {
        if (entry.id == id) return &entry;
    }
    return nullptr;
}

}  // namespace

PluginStartupList ParsePluginStartupList(const std::string& json_text, std::string* error) {
    PluginStartupList list;
    const auto fail = [&](const std::string& reason) {
        if (error) *error = reason;
        return PluginStartupList{};
    };
    const auto json = nlohmann::json::parse(json_text, nullptr, false);
    if (json.is_discarded() || !json.is_object()) return fail("not a JSON object");
    if (json.value("version", 0) != kStartupListVersion) return fail("unknown version");
    const auto plugins = json.find("plugins");
    if (plugins == json.end() || !plugins->is_array()) return fail("no plugin list");
    for (const auto& item : *plugins) {
        if (!item.is_object()) continue;
        PluginStartupEntry entry;
        entry.id = item.value("id", std::string());
        entry.folder = item.value("folder", std::string());
        entry.load_at_startup = item.value("load_at_startup", true);
        if (entry.id.empty() || Find(list, entry.id)) continue;
        list.entries.push_back(std::move(entry));
    }
    if (error) error->clear();
    return list;
}

std::string SerializePluginStartupList(const PluginStartupList& list) {
    nlohmann::json plugins = nlohmann::json::array();
    for (const auto& entry : list.entries) {
        plugins.push_back({{"id", entry.id}, {"folder", entry.folder}, {"load_at_startup", entry.load_at_startup}});
    }
    return nlohmann::json{{"version", kStartupListVersion}, {"plugins", plugins}}.dump(2);
}

void RecordPluginLoaded(PluginStartupList& list, const std::string& id, const std::string& folder) {
    if (id.empty()) return;
    if (auto* entry = Find(list, id)) {
        entry->folder = folder;
        return;
    }
    list.entries.push_back({id, folder, true});
}

void RecordPluginRemoved(PluginStartupList& list, const std::string& id) {
    list.entries.erase(std::remove_if(list.entries.begin(), list.entries.end(),
                                      [&](const PluginStartupEntry& e) { return e.id == id; }),
                       list.entries.end());
}

void SetPluginLoadAtStartup(PluginStartupList& list, const std::string& id, bool on) {
    if (auto* entry = Find(list, id)) entry->load_at_startup = on;
}

bool PluginLoadsAtStartup(const PluginStartupList& list, const std::string& id, bool installed_add_on) {
    for (const auto& entry : list.entries) {
        if (entry.id == id) return entry.load_at_startup;
    }
    return installed_add_on;
}

std::vector<PluginStartupEntry> PluginsToLoadAtStartup(
    const PluginStartupList& list, const std::vector<PluginStartupEntry>& pack_plugins) {
    const auto pack_folder = [&](const std::string& id) -> const std::string* {
        for (const auto& pack : pack_plugins)
            if (pack.id == id) return &pack.folder;
        return nullptr;
    };
    std::vector<PluginStartupEntry> result;
    for (const auto& entry : list.entries) {
        if (!entry.load_at_startup) continue;
        auto chosen = entry;
        if (const auto* folder = pack_folder(entry.id)) chosen.folder = *folder;
        if (!chosen.folder.empty()) result.push_back(std::move(chosen));
    }
    for (const auto& pack : pack_plugins) {
        const bool listed = std::any_of(list.entries.begin(), list.entries.end(),
                                        [&](const PluginStartupEntry& e) { return e.id == pack.id; });
        if (!listed && !pack.folder.empty()) result.push_back({pack.id, pack.folder, true});
    }
    return result;
}

std::string PluginStartupNote(bool on, const PluginContribution& c) {
    if (c.trainable_nodes + c.simulation_nodes > 0) {
        return on ? "Graphs that use its nodes open ready after a restart."
                  : "After a restart its nodes show as not installed until you load it here.";
    }
    std::vector<std::string> what;
    if (c.dataset_formats > 0) what.push_back("dataset formats");
    if (c.analytics > 0) what.push_back("analytics");
    if (c.windows > 0) what.push_back("windows");
    if (what.empty()) {
        return on ? "It loads again after a restart." : "After a restart it stays unloaded until you load it here.";
    }
    std::string list = what[0];
    for (size_t i = 1; i < what.size(); ++i) list += (i + 1 == what.size() ? " and " : ", ") + what[i];
    return on ? "Its " + list + " are there after a restart."
              : "After a restart its " + list + " are missing until you load it here.";
}

std::string PluginAddsLine(const PluginContribution& c) {
    std::vector<std::string> parts;
    if (c.windows > 0) parts.push_back(Count(c.windows, "window", "windows"));
    if (c.trainable_nodes > 0) parts.push_back(Count(c.trainable_nodes, "trainable node", "trainable nodes"));
    if (c.simulation_nodes > 0) parts.push_back(Count(c.simulation_nodes, "simulation node", "simulation nodes"));
    if (!parts.empty() && !c.node_categories.empty()) {  // the categories follow the nodes
        std::string categories = " (";
        for (size_t i = 0; i < c.node_categories.size(); ++i) {
            if (i > 0) categories += ", ";
            categories += c.node_categories[i];
        }
        parts.back() += categories + ")";
    }
    if (c.dataset_formats > 0) parts.push_back(Count(c.dataset_formats, "dataset format", "dataset formats"));
    if (c.analytics > 0) parts.push_back(Count(c.analytics, "analytic", "analytics"));
    if (parts.empty()) return {};
    std::string text = "Adds: ";
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) text += ", ";
        text += parts[i];
    }
    return text;
}

std::string PluginMenuDetail(const PluginContribution& c) {
    const size_t nodes = c.trainable_nodes + c.simulation_nodes;
    std::string text;
    if (c.windows > 0) text = Count(c.windows, "window", "windows");
    if (nodes > 0) text += (text.empty() ? "" : ", ") + Count(nodes, "node", "nodes");
    if (c.dataset_formats > 0) text += (text.empty() ? "" : ", ") + Count(c.dataset_formats, "format", "formats");
    if (c.analytics > 0) text += (text.empty() ? "" : ", ") + Count(c.analytics, "analytic", "analytics");
    return text;
}

std::string FormatPackBytes(std::uint64_t bytes) {
    char text[32];
    if (bytes >= 1000ull * 1000ull) {
        std::snprintf(text, sizeof(text), "%.1f MB", static_cast<double>(bytes) / 1e6);
    } else {
        std::snprintf(text, sizeof(text), "%.0f KB", std::ceil(static_cast<double>(bytes) / 1e3));
    }
    return text;
}

std::string PackFileProblemText(plugin::PackFileProblem problem) {
    switch (problem) {
        case plugin::PackFileProblem::Missing: return "missing";
        case plugin::PackFileProblem::SizeDiffers: return "its size differs from the signed list";
        case plugin::PackFileProblem::ContentDiffers: return "same size, different contents";
        case plugin::PackFileProblem::Unreadable: return "could not be read";
        case plugin::PackFileProblem::NotListed: return "not in the signed list";
    }
    return "differs";
}

PluginTrustView PluginTrustLine(const PluginTrustFacts& facts) {
    PluginTrustView view;
    const std::string version =
        facts.package_version.empty() ? std::string{} : "package version " + facts.package_version;
    switch (facts.kind) {
        case PluginTrustKind::Developer:
            view.line = "Developer plugin, not signed";
            view.note = facts.folder.empty() ? std::string{} : "loaded from " + facts.folder;
            view.tone = PluginTrustTone::Muted;
            Detail(view.details, "Folder", facts.folder);
            Detail(view.details, "Loaded from", "a copy in the Engine's plugin cache");
            return view;
        case PluginTrustKind::SignedAddOn: {
            view.line = facts.at_start ? "Signed add-on, verified at start" : "Signed add-on, verified when it loaded";
            view.note = version;
            view.tone = PluginTrustTone::Verified;
            Detail(view.details, "Pack", facts.pack_id);
            std::string checked;
            if (facts.files_checked > 0) {
                checked = Count(facts.files_checked, "file", "files") + ", " + FormatPackBytes(facts.bytes_checked) +
                          ", in " + std::to_string(static_cast<long long>(std::llround(facts.check_ms))) + " ms";
            }
            Detail(view.details, "Checked", checked);
            Detail(view.details, "Signed with", kSignedWith);
            Detail(view.details, "Built for",
                   facts.companion_base_id.empty() ? std::string{} : "CPU Engine " + facts.companion_base_id);
            Detail(view.details, "Folder", facts.folder);
            Detail(view.details, "Loaded from", "a checked copy in the Engine's plugin cache");
            return view;
        }
        case PluginTrustKind::RefusedAddOn:
            break;
    }
    view.line = facts.at_start ? "Signed add-on, check failed at start" : "Signed add-on, check failed when it loaded";
    view.note = version;
    view.tone = PluginTrustTone::Failed;
    std::string reason = facts.reason.empty() ? std::string("its files differ from its signed package") : facts.reason;
    // A sentence starts with a capital; a file name keeps its spelling.
    if (reason.rfind("its ", 0) == 0) reason[0] = 'I';
    view.body = reason +
                ", so the Engine did not load this add-on. Graphs that use its nodes show them as not installed.";
    view.hint = "Repair it to put the signed files back: the installer applies the repair once this Engine is "
                "closed. It also opens from Preferences > Devices > Installed packs.";
    Detail(view.details, "Pack", facts.pack_id);
    std::size_t differing = 0;
    std::size_t unlisted = 0;
    for (const auto& issue : facts.issues) {
        (issue.problem == plugin::PackFileProblem::NotListed ? unlisted : differing) += 1;
    }
    std::string check;
    if (facts.files_listed == 0) {
        check = "its signed package record could not be verified";
    } else if (differing > 0) {
        check = std::to_string(differing) + " of " + Count(facts.files_listed, "file", "files") +
                (differing == 1 ? " differs" : " differ") + " from the signed list";
    }
    if (unlisted > 0) {
        if (!check.empty()) check += "; ";
        check += Count(unlisted, "file", "files") + (unlisted == 1 ? " is" : " are") + " not in the signed list";
    }
    Detail(view.details, "Check", check);
    std::string files;
    for (const auto& issue : facts.issues) {
        if (!files.empty()) files += "\n";
        files += plugin::PackDisplayPath(issue.path) + ": " + PackFileProblemText(issue.problem);
    }
    Detail(view.details, facts.issues.size() > 1 ? "Files" : "File", files);
    Detail(view.details, "Why", facts.detail);
    Detail(view.details, "Checked",
           facts.checked_at.empty() ? std::string{}
                                    : (facts.at_start ? "at this start, " : "when it loaded, ") + facts.checked_at);
    Detail(view.details, "Folder", facts.folder);
    Detail(view.details, "Signed with", facts.files_listed == 0 ? std::string{} : kSignedWith);
    return view;
}

std::vector<std::string> PluginListSummary(const PluginListCounts& counts) {
    std::string line = Count(counts.plugins, "plugin", "plugins") + ": " + std::to_string(counts.active) + " active, ";
    if (counts.refused > 0 && counts.failed == 0) {
        line += std::to_string(counts.refused) + " not loaded (check failed), ";
    } else {
        line += std::to_string(counts.failed) + " failed, ";
        if (counts.refused > 0) line += std::to_string(counts.refused) + " not loaded (check failed), ";
    }
    line += std::to_string(counts.disabled) + " disabled";
    std::vector<std::string> lines{line};
    if (counts.verified > 0) {
        lines.push_back(Count(counts.verified, "signed add-on", "signed add-ons") + " verified at start");
    }
    return lines;
}

}  // namespace cyxwiz
