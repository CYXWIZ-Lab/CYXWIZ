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

// The request a plugin makes when it starts with a dangerous permission the
// user has not answered: one per plugin at a time, answered with "Allow
// selected" or "Start without these" (also Esc). The plugin starts either way.
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
        DecisionCallback callback;                   // per-request callback
    };

    // Queue a plugin for permission approval. Returns true when a request is
    // shown or waiting for it (a plugin already asking is not asked twice).
    bool RequestApproval(const PluginManifest& manifest,
                         const std::vector<PluginPermission>& undecided_perms,
                         DecisionCallback callback);

    // Render the ImGui modal. Call from main render loop.
    // Returns true if a decision was made this frame.
    bool Render();

    // Check if there's a pending dialog
    bool HasPending() const;
    // A request for this plugin is shown or queued.
    bool IsPendingFor(const std::string& plugin_id) const;

    // Clear any pending request for a specific plugin (call on unload)
    void ClearPendingForPlugin(const std::string& plugin_id);

private:
    bool IsPendingForLocked(const std::string& plugin_id) const;

    mutable std::mutex mutex_;
    std::queue<PendingApproval> pending_queue_;
    std::optional<PendingApproval> current_;
    bool open_requested_ = false;
};

} // namespace cyxwiz::plugin::security
