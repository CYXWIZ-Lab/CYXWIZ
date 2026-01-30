#pragma once

#include "../plugin_types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <filesystem>

namespace cyxwiz::plugin::security {

enum class PermissionDecision {
    Undecided,
    Allowed,
    Denied,
};

class PermissionStore {
public:
    static PermissionStore& Instance();

    PermissionStore(const PermissionStore&) = delete;
    PermissionStore& operator=(const PermissionStore&) = delete;

    // Load/save from ~/.cyxwiz/plugin_permissions.json (or %APPDATA%\cyxwiz\)
    void Load();
    void Save();

    // Get decision for a specific plugin + permission
    PermissionDecision GetDecision(const std::string& plugin_id,
                                    const std::string& plugin_version,
                                    PluginPermission perm) const;

    // Store a decision
    void SetDecision(const std::string& plugin_id,
                     const std::string& plugin_version,
                     PluginPermission perm,
                     PermissionDecision decision);

    // Get list of dangerous permissions that haven't been decided yet
    std::vector<PluginPermission> GetUndecidedDangerous(
        const std::string& plugin_id,
        const std::string& plugin_version,
        PluginPermissionFlags requested) const;

    // Compute effective granted permissions = (requested & allowed) | safe_auto_granted
    PluginPermissionFlags GetGrantedPermissions(
        const std::string& plugin_id,
        const std::string& plugin_version,
        PluginPermissionFlags requested) const;

    // Clear all decisions for a plugin
    void ClearPlugin(const std::string& plugin_id);

    // Check if a permission is considered "dangerous" (requires user approval)
    static bool IsDangerousPermission(PluginPermission perm);

    // Permission name for display
    static const char* PermissionDisplayName(PluginPermission perm);
    static const char* PermissionDescription(PluginPermission perm);

private:
    PermissionStore() = default;

    std::filesystem::path GetStorePath() const;

    // Key: "plugin_id:version"
    struct PermissionRecord {
        std::unordered_map<uint32_t, PermissionDecision> decisions; // perm flag -> decision
    };

    static std::string MakeKey(const std::string& id, const std::string& version);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, PermissionRecord> records_;
};

} // namespace cyxwiz::plugin::security
