#include "permission_store.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fstream>

namespace cyxwiz::plugin::security {

PermissionStore& PermissionStore::Instance() {
    static PermissionStore instance;
    return instance;
}

std::string PermissionStore::MakeKey(const std::string& id, const std::string& version) {
    return id + ":" + version;
}

std::filesystem::path PermissionStore::GetStorePath() const {
#ifdef _WIN32
    if (auto* appdata = std::getenv("APPDATA"))
        return std::filesystem::path(appdata) / "cyxwiz" / "plugin_permissions.json";
#else
    if (auto* home = std::getenv("HOME"))
        return std::filesystem::path(home) / ".cyxwiz" / "plugin_permissions.json";
#endif
    return "plugin_permissions.json";
}

void PermissionStore::Load() {
    std::lock_guard lock(mutex_);
    records_.clear();

    auto path = GetStorePath();
    if (!std::filesystem::exists(path)) return;

    try {
        std::ifstream f(path);
        nlohmann::json j;
        f >> j;

        for (auto& [key, record_json] : j.items()) {
            PermissionRecord rec;
            for (auto& [perm_str, decision_str] : record_json.items()) {
                uint32_t perm_flag = std::stoul(perm_str);
                PermissionDecision decision = PermissionDecision::Undecided;
                if (decision_str == "allowed") decision = PermissionDecision::Allowed;
                else if (decision_str == "denied") decision = PermissionDecision::Denied;
                rec.decisions[perm_flag] = decision;
            }
            records_[key] = std::move(rec);
        }
        spdlog::info("PermissionStore: Loaded {} plugin permission records", records_.size());
    } catch (const std::exception& e) {
        spdlog::warn("PermissionStore: Failed to load {}: {}", path.string(), e.what());
    }
}

void PermissionStore::Save() {
    std::lock_guard lock(mutex_);

    auto path = GetStorePath();
    try {
        std::filesystem::create_directories(path.parent_path());

        nlohmann::json j;
        for (const auto& [key, rec] : records_) {
            nlohmann::json record_json;
            for (const auto& [perm_flag, decision] : rec.decisions) {
                const char* decision_str = "undecided";
                if (decision == PermissionDecision::Allowed) decision_str = "allowed";
                else if (decision == PermissionDecision::Denied) decision_str = "denied";
                record_json[std::to_string(perm_flag)] = decision_str;
            }
            j[key] = record_json;
        }

        std::ofstream f(path);
        f << j.dump(2);
        spdlog::debug("PermissionStore: Saved to {}", path.string());
    } catch (const std::exception& e) {
        spdlog::error("PermissionStore: Failed to save {}: {}", path.string(), e.what());
    }
}

PermissionDecision PermissionStore::GetDecision(
    const std::string& plugin_id,
    const std::string& plugin_version,
    PluginPermission perm) const
{
    std::lock_guard lock(mutex_);
    auto key = MakeKey(plugin_id, plugin_version);
    auto it = records_.find(key);
    if (it == records_.end()) return PermissionDecision::Undecided;

    auto dit = it->second.decisions.find(static_cast<uint32_t>(perm));
    if (dit == it->second.decisions.end()) return PermissionDecision::Undecided;
    return dit->second;
}

void PermissionStore::SetDecision(
    const std::string& plugin_id,
    const std::string& plugin_version,
    PluginPermission perm,
    PermissionDecision decision)
{
    std::lock_guard lock(mutex_);
    auto key = MakeKey(plugin_id, plugin_version);
    records_[key].decisions[static_cast<uint32_t>(perm)] = decision;
}

std::vector<PluginPermission> PermissionStore::GetUndecidedDangerous(
    const std::string& plugin_id,
    const std::string& plugin_version,
    PluginPermissionFlags requested) const
{
    std::vector<PluginPermission> result;
    static const PluginPermission dangerous[] = {
        PluginPermission::FileSystem,
        PluginPermission::Network,
        PluginPermission::SystemCommands,
        PluginPermission::Python,
    };

    for (auto perm : dangerous) {
        if (HasPermission(requested, perm)) {
            if (GetDecision(plugin_id, plugin_version, perm) == PermissionDecision::Undecided) {
                result.push_back(perm);
            }
        }
    }
    return result;
}

PluginPermissionFlags PermissionStore::GetGrantedPermissions(
    const std::string& plugin_id,
    const std::string& plugin_version,
    PluginPermissionFlags requested) const
{
    PluginPermissionFlags granted = PluginPermissionFlags{0};

    // Safe permissions are auto-granted
    static const PluginPermission safe[] = {
        PluginPermission::GPU,
        PluginPermission::DataRegistry,
        PluginPermission::Training,
        PluginPermission::UIModify,
    };
    for (auto perm : safe) {
        if (HasPermission(requested, perm))
            granted = static_cast<PluginPermissionFlags>(
                static_cast<uint32_t>(granted) | static_cast<uint32_t>(perm));
    }

    // Dangerous permissions only if explicitly allowed
    static const PluginPermission dangerous[] = {
        PluginPermission::FileSystem,
        PluginPermission::Network,
        PluginPermission::SystemCommands,
        PluginPermission::Python,
    };
    for (auto perm : dangerous) {
        if (HasPermission(requested, perm) &&
            GetDecision(plugin_id, plugin_version, perm) == PermissionDecision::Allowed) {
            granted = static_cast<PluginPermissionFlags>(
                static_cast<uint32_t>(granted) | static_cast<uint32_t>(perm));
        }
    }

    return granted;
}

void PermissionStore::ClearPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    for (auto it = records_.begin(); it != records_.end();) {
        // Key format: "plugin_id:version" — extract exact ID before first colon
        auto colon_pos = it->first.find(':');
        if (colon_pos != std::string::npos &&
            it->first.substr(0, colon_pos) == plugin_id) {
            it = records_.erase(it);
        } else {
            ++it;
        }
    }
}

bool PermissionStore::IsDangerousPermission(PluginPermission perm) {
    return perm == PluginPermission::FileSystem ||
           perm == PluginPermission::Network ||
           perm == PluginPermission::SystemCommands ||
           perm == PluginPermission::Python;
}

const char* PermissionStore::PermissionDisplayName(PluginPermission perm) {
    switch (perm) {
        case PluginPermission::FileSystem:      return "File System";
        case PluginPermission::Network:         return "Network Access";
        case PluginPermission::SystemCommands:  return "System Commands";
        case PluginPermission::Python:          return "Python Execution";
        case PluginPermission::GPU:             return "GPU Compute";
        case PluginPermission::DataRegistry:    return "Dataset Access";
        case PluginPermission::Training:        return "Training Hooks";
        case PluginPermission::UIModify:        return "UI Modification";
        default:                                return "Unknown";
    }
}

const char* PermissionStore::PermissionDescription(PluginPermission perm) {
    switch (perm) {
        case PluginPermission::FileSystem:
            return "Read and write files on your computer";
        case PluginPermission::Network:
            return "Make network connections to remote servers";
        case PluginPermission::SystemCommands:
            return "Execute system commands and processes";
        case PluginPermission::Python:
            return "Run arbitrary Python code in the embedded interpreter";
        case PluginPermission::GPU:
            return "Access GPU compute resources";
        case PluginPermission::DataRegistry:
            return "Access and modify loaded datasets";
        case PluginPermission::Training:
            return "Hook into the training pipeline";
        case PluginPermission::UIModify:
            return "Add custom panels and nodes to the UI";
        default:
            return "";
    }
}

} // namespace cyxwiz::plugin::security
