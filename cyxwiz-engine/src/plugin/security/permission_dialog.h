#pragma once

#include "../plugin_types.h"
#include "permission_store.h"
#include <string>
#include <vector>
#include <queue>
#include <optional>
#include <functional>
#include <mutex>

namespace cyxwiz::plugin::security {

class PermissionDialog {
public:
    struct PendingApproval;
    using DecisionCallback = std::function<void(const PendingApproval&)>;

    struct PendingApproval {
        std::string plugin_id;
        std::string plugin_name;
        std::string plugin_version;
        std::string plugin_author;
        std::vector<PluginPermission> permissions;  // undecided dangerous perms
        std::vector<bool> allowed;                   // user's per-permission choices
        bool resolved = false;
        bool accepted = false;  // true = Allow Selected, false = Deny All
        DecisionCallback callback;                   // per-request callback
    };

    // Queue a plugin for permission approval.
    // Returns true if there are permissions to approve (dialog will show).
    bool RequestApproval(const PluginManifest& manifest,
                         const std::vector<PluginPermission>& undecided_perms,
                         DecisionCallback callback);

    // Render the ImGui modal. Call from main render loop.
    // Returns true if a decision was made this frame.
    bool Render();

    // Check if there's a pending dialog
    bool HasPending() const;

    // Clear any pending request for a specific plugin (call on unload)
    void ClearPendingForPlugin(const std::string& plugin_id);

private:
    mutable std::mutex mutex_;
    std::queue<PendingApproval> pending_queue_;
    std::optional<PendingApproval> current_;
    bool open_requested_ = false;
};

} // namespace cyxwiz::plugin::security
