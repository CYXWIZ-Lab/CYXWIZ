#pragma once

#include "../plugin_types.h"
#include "permission_store.h"
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <vector>

namespace cyxwiz::plugin::security {

// The permission requests plugins make when they start with a dangerous
// permission the user has not answered: one per plugin at a time, shown one
// after another. No drawing here (the Engine's tools and tests link this
// without a GUI): gui/plugin_permission_request draws the one shown.
class PermissionRequests {
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

    // Queue a plugin's request. Returns true when a request is shown or
    // waiting for it (a plugin already asking is not asked twice).
    bool RequestApproval(const PluginManifest& manifest,
                         const std::vector<PluginPermission>& undecided_perms,
                         DecisionCallback callback);

    // The request to show now (the next queued one once the last is
    // answered), copied; `first_time` is set on the frame it becomes shown.
    std::optional<PendingApproval> Shown(bool* first_time = nullptr);
    // The user ticked or unticked permissions of the shown request.
    void SetChoices(const std::string& plugin_id, const std::vector<bool>& allowed);
    // The user answered: the ticked permissions (`allow_selected`) or none.
    // Runs the request's callback outside the lock.
    void Answer(const std::string& plugin_id, bool allow_selected);

    // A request for this plugin is shown or queued.
    bool IsPendingFor(const std::string& plugin_id) const;
    // Clear any pending request for a specific plugin (call on unload)
    void ClearPendingForPlugin(const std::string& plugin_id);

private:
    bool IsPendingForLocked(const std::string& plugin_id) const;

    mutable std::mutex mutex_;
    std::queue<PendingApproval> pending_queue_;
    std::optional<PendingApproval> current_;
    bool first_time_ = false;
};

} // namespace cyxwiz::plugin::security
