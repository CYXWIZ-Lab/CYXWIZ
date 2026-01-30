#pragma once

#include "../plugin_types.h"
#include "permission_store.h"
#include <string>
#include <vector>
#include <optional>
#include <functional>

namespace cyxwiz::plugin::security {

class PermissionDialog {
public:
    struct PendingApproval {
        std::string plugin_id;
        std::string plugin_name;
        std::string plugin_version;
        std::string plugin_author;
        std::vector<PluginPermission> permissions;  // undecided dangerous perms
        std::vector<bool> allowed;                   // user's per-permission choices
        bool resolved = false;
        bool accepted = false;  // true = Allow Selected, false = Deny All
    };

    using DecisionCallback = std::function<void(const PendingApproval&)>;

    // Queue a plugin for permission approval.
    // Returns true if there are permissions to approve (dialog will show).
    bool RequestApproval(const PluginManifest& manifest,
                         const std::vector<PluginPermission>& undecided_perms);

    // Render the ImGui modal. Call from main render loop.
    // Returns true if a decision was made this frame.
    bool Render();

    // Check if there's a pending dialog
    bool HasPending() const { return current_.has_value() && !current_->resolved; }

    // Set callback for when user makes a decision
    void SetCallback(DecisionCallback cb) { callback_ = std::move(cb); }

private:
    std::optional<PendingApproval> current_;
    DecisionCallback callback_;
    bool open_requested_ = false;
};

} // namespace cyxwiz::plugin::security
