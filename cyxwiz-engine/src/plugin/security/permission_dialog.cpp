#include "permission_dialog.h"
#include "../../gui/ui_tokens.h"
#include "../../gui/ui_widgets.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz::plugin::security {

namespace {
constexpr const char* kPopupId = "###PluginPermDialog";
}

bool PermissionDialog::RequestApproval(
    const PluginManifest& manifest,
    const std::vector<PluginPermission>& undecided_perms,
    DecisionCallback callback)
{
    if (undecided_perms.empty()) return false;

    std::lock_guard lock(mutex_);
    // One request per plugin: a second Start or Refresh while it asks waits
    // for the same answer instead of opening another dialog over it.
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

    spdlog::info("PermissionDialog: Queued approval for plugin '{}' ({} permissions)",
                 manifest.name, undecided_perms.size());
    return true;
}

bool PermissionDialog::HasPending() const {
    std::lock_guard lock(mutex_);
    return (current_.has_value() && !current_->resolved) || !pending_queue_.empty();
}

bool PermissionDialog::IsPendingFor(const std::string& plugin_id) const {
    std::lock_guard lock(mutex_);
    return IsPendingForLocked(plugin_id);
}

bool PermissionDialog::IsPendingForLocked(const std::string& plugin_id) const {
    if (current_.has_value() && !current_->resolved && current_->plugin_id == plugin_id) return true;
    std::queue<PendingApproval> copy = pending_queue_;
    while (!copy.empty()) {
        if (copy.front().plugin_id == plugin_id) return true;
        copy.pop();
    }
    return false;
}

void PermissionDialog::ClearPendingForPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);

    // Clear current if it matches
    if (current_.has_value() && current_->plugin_id == plugin_id) {
        current_.reset();
        open_requested_ = false;
    }

    // Remove from queue
    std::queue<PendingApproval> filtered;
    while (!pending_queue_.empty()) {
        auto item = std::move(pending_queue_.front());
        pending_queue_.pop();
        if (item.plugin_id != plugin_id) {
            filtered.push(std::move(item));
        }
    }
    pending_queue_ = std::move(filtered);
}

bool PermissionDialog::Render() {
    // The request shown this frame, copied under the lock: an unload may clear
    // it meanwhile, and the answer's callback may queue another request.
    std::optional<PendingApproval> shown;
    bool open_now = false;
    {
        std::lock_guard lock(mutex_);
        if ((!current_.has_value() || current_->resolved) && !pending_queue_.empty()) {
            current_ = std::move(pending_queue_.front());
            pending_queue_.pop();
            open_requested_ = true;
        }
        if (!current_.has_value() || current_->resolved) return false;
        shown = *current_;
        open_now = open_requested_;
        open_requested_ = false;
    }

    const std::string title = shown->plugin_name + " asks for permission" + kPopupId;
    if (open_now) ImGui::OpenPopup(title.c_str());

    ui::DialogOptions options;
    options.size = ImVec2(560.0f, 380.0f);
    options.min_size = ImVec2(460.0f, 300.0f);
    if (!ui::BeginDialog(title.c_str(), options)) return false;

    const ui::Tokens& t = ui::CurrentTokens();
    std::string who = shown->plugin_id + "  \xC2\xB7  v" + shown->plugin_version;
    if (!shown->plugin_author.empty()) who += "  \xC2\xB7  " + shown->plugin_author;
    ImGui::TextColored(t.text_dim, "%s", who.c_str());
    ImGui::Spacing();
    ImGui::TextUnformatted("To work fully it wants to:");
    ImGui::Spacing();
    bool changed = false;
    for (size_t i = 0; i < shown->permissions.size(); ++i) {
        const PluginPermission perm = shown->permissions[i];
        ImGui::PushID(static_cast<int>(static_cast<uint32_t>(perm)));
        bool checked = shown->allowed[i];  // std::vector<bool> hands out proxies
        if (ImGui::Checkbox(PermissionStore::PermissionDisplayName(perm), &checked)) {
            shown->allowed[i] = checked;
            changed = true;
        }
        ImGui::Indent(ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x);
        ImGui::TextColored(t.text_dim, "%s", PermissionStore::PermissionDescription(perm));
        ImGui::Unindent(ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x);
        ImGui::PopID();
    }
    ImGui::Spacing();
    ui::NoticeLine("The plugin starts either way. Without a permission, the parts that need it stay off. "
                   "Your answer is kept; change it later on the plugin's card in Plugin Manager.",
                   t.text_dim);

    const bool any = std::any_of(shown->allowed.begin(), shown->allowed.end(), [](bool b) { return b; });
    const ui::DialogResult result =
        ui::EndDialog("Allow selected", "Start without these", any, "Tick at least one, or choose Start without these");

    PendingApproval answer;
    bool decided = false;
    {
        std::lock_guard lock(mutex_);
        if (!current_.has_value() || current_->plugin_id != shown->plugin_id || current_->resolved) return false;
        if (changed) current_->allowed = shown->allowed;
        if (result != ui::DialogResult::None) {
            if (result == ui::DialogResult::Secondary)
                std::fill(current_->allowed.begin(), current_->allowed.end(), false);
            current_->resolved = true;
            answer = *current_;
            decided = true;
        }
    }
    if (!decided) return false;
    spdlog::info("PermissionDialog: '{}' {}", answer.plugin_name,
                 result == ui::DialogResult::Primary ? "starts with the selected permissions"
                                                     : "starts without the permissions it asked for");
    if (answer.callback) answer.callback(answer);
    return true;
}

} // namespace cyxwiz::plugin::security
