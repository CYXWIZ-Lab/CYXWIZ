#include "plugin_permission_request.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "../plugin/plugin_manager.h"
#include "../plugin/security/permission_store.h"
#include <imgui.h>
#include <algorithm>
#include <string>

namespace gui {

void RenderPluginPermissionRequest() {
    using cyxwiz::plugin::security::PermissionStore;
    namespace ui = cyxwiz::ui;
    auto& requests = cyxwiz::plugin::PluginManager::Instance().PermissionRequestQueue();
    bool first_time = false;
    auto shown = requests.Shown(&first_time);
    if (!shown) return;

    // The title names the plugin; the ###id keeps one popup across plugins
    const std::string title = shown->plugin_name + " asks for permission###PluginPermDialog";
    if (first_time) ImGui::OpenPopup(title.c_str());

    ui::DialogOptions options;
    options.size = ImVec2(560.0f, 380.0f);
    options.min_size = ImVec2(460.0f, 300.0f);
    if (!ui::BeginDialog(title.c_str(), options)) return;

    const ui::Tokens& t = ui::CurrentTokens();
    std::string who = shown->plugin_id + "  \xC2\xB7  v" + shown->plugin_version;
    if (!shown->plugin_author.empty()) who += "  \xC2\xB7  " + shown->plugin_author;
    ImGui::TextColored(t.text_dim, "%s", who.c_str());
    ImGui::Spacing();
    ImGui::TextUnformatted("To work fully it wants to:");
    ImGui::Spacing();
    bool changed = false;
    const float indent = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x;
    for (size_t i = 0; i < shown->permissions.size(); ++i) {
        const auto perm = shown->permissions[i];
        ImGui::PushID(static_cast<int>(static_cast<uint32_t>(perm)));
        bool checked = shown->allowed[i];  // std::vector<bool> hands out proxies
        if (ImGui::Checkbox(PermissionStore::PermissionDisplayName(perm), &checked)) {
            shown->allowed[i] = checked;
            changed = true;
        }
        ImGui::Indent(indent);
        ImGui::TextColored(t.text_dim, "%s", PermissionStore::PermissionDescription(perm));
        ImGui::Unindent(indent);
        ImGui::PopID();
    }
    ImGui::Spacing();
    ui::NoticeLine("The plugin starts either way. Without a permission, the parts that need it stay off. "
                   "Your answer is kept; change it later on the plugin's card in Plugin Manager.",
                   t.text_dim);

    const bool any = std::any_of(shown->allowed.begin(), shown->allowed.end(), [](bool b) { return b; });
    const ui::DialogResult result =
        ui::EndDialog("Allow selected", "Start without these", any, "Tick at least one, or choose Start without these");

    if (changed) requests.SetChoices(shown->plugin_id, shown->allowed);
    if (result == ui::DialogResult::Primary) requests.Answer(shown->plugin_id, true);
    if (result == ui::DialogResult::Secondary) requests.Answer(shown->plugin_id, false);
}

}  // namespace gui
