#include "permission_requests.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz::plugin::security {

bool PermissionRequests::RequestApproval(const PluginManifest& manifest,
                                         const std::vector<PluginPermission>& undecided_perms,
                                         DecisionCallback callback) {
    if (undecided_perms.empty()) return false;

    std::lock_guard lock(mutex_);
    // One request per plugin: a second Start or Refresh while it asks waits
    // for the same answer instead of opening another request over it.
    if (IsPendingForLocked(manifest.id)) return true;

    PendingApproval pending;
    pending.plugin_id = manifest.id;
    pending.plugin_name = manifest.name;
    pending.plugin_version = manifest.version.ToString();
    pending.plugin_author = manifest.author;
    pending.permissions = undecided_perms;
    pending.allowed.resize(undecided_perms.size(), false);
    pending.callback = std::move(callback);
    pending_queue_.push_back(std::move(pending));

    spdlog::info("PermissionRequests: Queued approval for plugin '{}' ({} permissions)", manifest.name,
                 undecided_perms.size());
    return true;
}

std::optional<PermissionRequests::PendingApproval> PermissionRequests::Shown() {
    std::lock_guard lock(mutex_);
    if ((!current_.has_value() || current_->resolved) && !pending_queue_.empty()) {
        current_ = std::move(pending_queue_.front());
        pending_queue_.pop_front();
    }
    if (!current_.has_value() || current_->resolved) return std::nullopt;
    return current_;
}

void PermissionRequests::SetChoices(const std::string& plugin_id, const std::vector<bool>& allowed) {
    std::lock_guard lock(mutex_);
    if (current_.has_value() && !current_->resolved && current_->plugin_id == plugin_id &&
        allowed.size() == current_->allowed.size())
        current_->allowed = allowed;
}

void PermissionRequests::Answer(const std::string& plugin_id, bool allow_selected) {
    PendingApproval answer;
    {
        std::lock_guard lock(mutex_);
        if (!current_.has_value() || current_->resolved || current_->plugin_id != plugin_id) return;
        if (!allow_selected) std::fill(current_->allowed.begin(), current_->allowed.end(), false);
        current_->resolved = true;
        answer = *current_;
    }
    spdlog::info("PermissionRequests: '{}' {}", answer.plugin_name,
                 allow_selected ? "starts with the selected permissions" : "starts without the permissions it asked for");
    // Outside the lock: the callback starts the plugin, which may ask again
    if (answer.callback) answer.callback(answer);
}

bool PermissionRequests::IsPendingFor(const std::string& plugin_id) const {
    std::lock_guard lock(mutex_);
    return IsPendingForLocked(plugin_id);
}

bool PermissionRequests::IsPendingForLocked(const std::string& plugin_id) const {
    if (current_.has_value() && !current_->resolved && current_->plugin_id == plugin_id) return true;
    return std::any_of(pending_queue_.begin(), pending_queue_.end(),
                       [&plugin_id](const PendingApproval& p) { return p.plugin_id == plugin_id; });
}

void PermissionRequests::ClearPendingForPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    if (current_.has_value() && current_->plugin_id == plugin_id) current_.reset();
    pending_queue_.erase(std::remove_if(pending_queue_.begin(), pending_queue_.end(),
                                        [&plugin_id](const PendingApproval& p) { return p.plugin_id == plugin_id; }),
                         pending_queue_.end());
}

} // namespace cyxwiz::plugin::security
