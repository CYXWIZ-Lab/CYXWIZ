#include "plugin_presentation.h"

#include <nlohmann/json.hpp>

#include <algorithm>

namespace cyxwiz {

namespace {

constexpr int kStartupListVersion = 1;

std::string Count(size_t n, const char* one, const char* many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

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

bool PluginLoadsAtStartup(const PluginStartupList& list, const std::string& id) {
    for (const auto& entry : list.entries) {
        if (entry.id == id) return entry.load_at_startup;
    }
    return false;
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

// ---- The Plugin Manager card (TOFIX129 C6)

namespace {

std::string Sentence(std::string text) {
    if (text.empty()) return text;
    if (text[0] >= 'a' && text[0] <= 'z') text[0] = static_cast<char>(text[0] - 'a' + 'A');
    const char last = text.back();
    if (last != '.' && last != '!' && last != '?' && last != ')') text += '.';
    return text;
}

std::string Lower(const std::string& text) {
    std::string out = text;
    for (char& c : out)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return out;
}

}  // namespace

PluginCardView DescribePluginCard(const PluginCardFacts& f) {
    PluginCardView v;
    switch (f.state) {
        case PluginRunState::Running:
            v.state_word = "Active";
            v.tone = PluginTone::Good;
            v.action = PluginAction::Disable;
            v.action_label = "Disable";
            break;
        case PluginRunState::Loaded:
            if (f.awaiting_answer) {
                v.state_word = "Waiting";
                v.tone = PluginTone::Waiting;
                v.reason = "Waiting for your answer to its permission request.";
            } else if (!f.error.empty()) {
                v.state_word = "Waiting";
                v.tone = PluginTone::Waiting;
                v.reason = Sentence(f.error) + " It starts by itself when that one does.";
                v.action = PluginAction::Start;
                v.action_label = "Start";
            } else {
                v.state_word = "Not started";
                v.tone = PluginTone::Waiting;
                v.reason = "Loaded but not started: its nodes and windows are not there yet.";
                v.action = PluginAction::Start;
                v.action_label = "Start";
            }
            break;
        case PluginRunState::Failed:
            v.state_word = "Failed";
            v.tone = PluginTone::Problem;
            v.reason = f.error.empty() ? "Failed to start; the Console has the details."
                                       : "Failed to start: " + Sentence(f.error);
            v.action = PluginAction::TryAgain;
            v.action_label = "Try again";
            break;
        case PluginRunState::Disabled:
            v.state_word = "Disabled";
            v.tone = PluginTone::Off;
            v.reason = "You turned it off. Its nodes and windows are hidden until you enable it.";
            v.action = PluginAction::Enable;
            v.action_label = "Enable";
            break;
        case PluginRunState::Unloaded:
            v.state_word = "Unloaded";
            v.tone = PluginTone::Off;
            v.reason = "Unloaded. Refresh or Install loads it again.";
            break;
    }
    return v;
}

std::vector<size_t> PluginCardOrder(const std::vector<PluginCardFacts>& cards) {
    std::vector<size_t> order(cards.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&cards](size_t a, size_t b) {
        const std::string la = Lower(cards[a].name), lb = Lower(cards[b].name);
        if (la != lb) return la < lb;
        return cards[a].id < cards[b].id;
    });
    return order;
}

std::string PluginFooter(const std::vector<PluginCardFacts>& cards) {
    if (cards.empty()) return "No plugins";
    size_t active = 0, not_started = 0, waiting = 0, failed = 0, disabled = 0, unloaded = 0;
    for (const PluginCardFacts& c : cards) {
        const PluginCardView v = DescribePluginCard(c);
        if (v.state_word == "Active") ++active;
        else if (v.state_word == "Not started") ++not_started;
        else if (v.state_word == "Waiting") ++waiting;
        else if (v.state_word == "Failed") ++failed;
        else if (v.state_word == "Disabled") ++disabled;
        else ++unloaded;
    }
    std::string text = Count(cards.size(), "plugin", "plugins") + ":";
    bool first = true;
    const auto add = [&](size_t n, const char* word) {
        if (n == 0) return;
        text += (first ? " " : ", ") + std::to_string(n) + " " + word;
        first = false;
    };
    add(active, "active");
    add(not_started, "not started");
    add(waiting, "waiting");
    add(failed, "failed");
    add(disabled, "disabled");
    add(unloaded, "unloaded");
    return text;
}

std::vector<PluginPermissionRow> PluginPermissionRows(const std::vector<PluginPermissionFacts>& permissions) {
    std::vector<PluginPermissionRow> rows;
    for (const PluginPermissionFacts& p : permissions) {
        PluginPermissionRow row;
        row.name = p.name;
        row.description = p.description;
        row.can_change = p.needs_answer;
        if (!p.needs_answer) {
            row.answer = "Allowed (low risk, not asked)";
        } else {
            row.answer = p.answer == PermissionAnswer::Allowed  ? "Allowed"
                         : p.answer == PermissionAnswer::Denied ? "Denied"
                                                                : "Not asked yet";
        }
        rows.push_back(row);
    }
    return rows;
}

std::string PluginPermissionsLink(const std::vector<PluginPermissionFacts>& permissions) {
    if (permissions.empty()) return {};
    size_t allowed = 0, denied = 0, not_asked = 0;
    for (const PluginPermissionFacts& p : permissions) {
        if (!p.needs_answer || p.answer == PermissionAnswer::Allowed) ++allowed;
        else if (p.answer == PermissionAnswer::Denied) ++denied;
        else ++not_asked;
    }
    std::vector<std::string> parts;
    if (allowed) parts.push_back(std::to_string(allowed) + " allowed");
    if (denied) parts.push_back(std::to_string(denied) + " denied");
    if (not_asked) parts.push_back(std::to_string(not_asked) + " not asked yet");
    std::string text = "Permissions (";
    for (size_t i = 0; i < parts.size(); ++i) text += (i ? ", " : "") + parts[i];
    return text + ")";
}

std::string PluginUnloadQuestion(const std::string& name, const PluginContribution& c) {
    std::string text = "Unload " + name + "?";
    const size_t nodes = c.trainable_nodes + c.simulation_nodes;
    if (c.windows > 0) text += " Its " + Count(c.windows, "window closes", "windows close") + ".";
    if (nodes > 0) text += " Graphs that use its nodes cannot run until it is loaded again.";
    if (c.dataset_formats > 0) text += " Data Inputs that read its formats stop loading.";
    if (c.analytics > 0) text += " Its analytics leave Explore > Plugins.";
    return text;
}

}  // namespace cyxwiz
