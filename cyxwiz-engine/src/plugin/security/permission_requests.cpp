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
    pending_queue_.push(std::move(pending));

    spdlog::info("PermissionRequests: Queued approval for plugin '{}' ({} permissions)", manifest.name,
                 undecided_perms.size());
    return true;
}

std::optional<PermissionRequests::PendingApproval> PermissionRequests::Shown(bool* first_time) {
    std::lock_guard lock(mutex_);
    if ((!current_.has_value() || current_->resolved) && !pending_queue_.empty()) {
        current_ = std::move(pending_queue_.front());
        pending_queue_.pop();
        first_time_ = true;
    }
    if (first_time) *first_time = first_time_;
    first_time_ = false;
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
    std::queue<PendingApproval> copy = pending_queue_;
    while (!copy.empty()) {
        if (copy.front().plugin_id == plugin_id) return true;
        copy.pop();
    }
    return false;
}

void PermissionRequests::ClearPendingForPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    if (current_.has_value() && current_->plugin_id == plugin_id) {
        current_.reset();
        first_time_ = false;
    }
    std::queue<PendingApproval> filtered;
    while (!pending_queue_.empty()) {
        auto item = std::move(pending_queue_.front());
        pending_queue_.pop();
        if (item.plugin_id != plugin_id) filtered.push(std::move(item));
    }
    pending_queue_ = std::move(filtered);
}

} // namespace cyxwiz::plugin::security
