#include "plugin_manager_panel.h"
#include "../icons.h"
#include "../../plugin/plugin_manager.h"
#include "../../plugin/security/permission_store.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>

namespace cyxwiz {

// Case-insensitive substring search
static bool ContainsCI(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return true;
    auto it = std::search(haystack.begin(), haystack.end(),
                          needle.begin(), needle.end(),
                          [](char a, char b) { return std::tolower(a) == std::tolower(b); });
    return it != haystack.end();
}

static ImVec4 StateColor(plugin::PluginState state) {
    switch (state) {
        case plugin::PluginState::Initialized:
        case plugin::PluginState::Active:
            return ImVec4(0.3f, 0.9f, 0.3f, 1.0f);  // green
        case plugin::PluginState::Loaded:
            return ImVec4(0.9f, 0.9f, 0.3f, 1.0f);  // yellow
        case plugin::PluginState::Failed:
            return ImVec4(0.9f, 0.3f, 0.3f, 1.0f);  // red
        case plugin::PluginState::Disabled:
            return ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // gray
        default:
            return ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    }
}

PluginManagerPanel::PluginManagerPanel()
    : Panel("Plugin Manager", false) {
}

const char* PluginManagerPanel::GetIcon() const {
    return ICON_FA_PLUG;
}

void PluginManagerPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(GetName(), &visible_)) {
        ImGui::End();
        return;
    }

    RenderToolbar();
    ImGui::Separator();
    RenderPluginList();
    ImGui::Separator();
    RenderStatusBar();

    ImGui::End();

    RenderInstallPopup();
}

// ============================================================================
// Toolbar
// ============================================================================

void PluginManagerPanel::RenderToolbar() {
    float avail = ImGui::GetContentRegionAvail().x;

    // Search box
    ImGui::SetNextItemWidth(avail - 200.0f);
    ImGui::InputTextWithHint("##PluginSearch", ICON_FA_MAGNIFYING_GLASS " Search plugins...",
                             search_buf_, sizeof(search_buf_));

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Refresh")) {
        auto& mgr = plugin::PluginManager::Instance();
        mgr.LoadAllFromSearchPaths();
        spdlog::info("PluginManagerPanel: Refreshed plugin list");
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_FOLDER_PLUS " Install...")) {
        show_install_popup_ = true;
        install_path_buf_[0] = '\0';
    }
}

// ============================================================================
// Plugin List
// ============================================================================

void PluginManagerPanel::RenderPluginList() {
    auto& mgr = plugin::PluginManager::Instance();
    auto plugins = mgr.GetAllPlugins();
    std::string filter(search_buf_);

    float height = ImGui::GetContentRegionAvail().y - 30.0f; // reserve for status bar
    if (ImGui::BeginChild("PluginList", ImVec2(0, height), ImGuiChildFlags_None)) {
        if (plugins.empty()) {
            ImGui::TextDisabled("No plugins loaded.");
            ImGui::TextDisabled("Use the search paths or Install button to add plugins.");
        } else {
            for (const auto* p : plugins) {
                if (!p) continue;

                // Filter
                if (!filter.empty()) {
                    bool match = ContainsCI(p->manifest.name, filter) ||
                                 ContainsCI(p->manifest.id, filter) ||
                                 ContainsCI(p->manifest.author, filter) ||
                                 ContainsCI(p->manifest.description, filter);
                    if (!match) continue;
                }

                RenderPluginCard(p);
                ImGui::Spacing();
            }
        }
    }
    ImGui::EndChild();
}

// ============================================================================
// Plugin Card
// ============================================================================

void PluginManagerPanel::RenderPluginCard(const plugin::LoadedPlugin* p) {
    ImGui::PushID(p->manifest.id.c_str());

    // Card background
    ImVec2 cursor = ImGui::GetCursorScreenPos();
    float card_width = ImGui::GetContentRegionAvail().x;
    float card_height = 0; // auto

    // State dot + Name + Version
    ImVec4 color = StateColor(p->state);
    ImGui::TextColored(color, ICON_FA_CIRCLE);
    ImGui::SameLine();
    ImGui::Text("%s", p->manifest.name.c_str());
    ImGui::SameLine(card_width - 60.0f);
    ImGui::TextDisabled("v%s", p->manifest.version.ToString().c_str());

    // Author
    if (!p->manifest.author.empty()) {
        ImGui::TextDisabled("  Author: %s", p->manifest.author.c_str());
    }

    // Description
    if (!p->manifest.description.empty()) {
        ImGui::TextWrapped("  %s", p->manifest.description.c_str());
    }

    // Error message for failed plugins
    if (p->state == plugin::PluginState::Failed && !p->error_message.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "  ERROR: %s", p->error_message.c_str());
    }

    // Permission badges + action buttons on same line
    ImGui::Text("  ");
    ImGui::SameLine();

    // Permission badges
    static const plugin::PluginPermission all_perms[] = {
        plugin::PluginPermission::FileSystem,
        plugin::PluginPermission::Network,
        plugin::PluginPermission::SystemCommands,
        plugin::PluginPermission::Python,
        plugin::PluginPermission::GPU,
        plugin::PluginPermission::DataRegistry,
        plugin::PluginPermission::Training,
        plugin::PluginPermission::UIModify,
    };

    for (auto perm : all_perms) {
        if (plugin::HasPermission(p->manifest.permissions, perm)) {
            bool dangerous = plugin::security::PermissionStore::IsDangerousPermission(perm);
            RenderPermissionBadge(perm, dangerous);
            ImGui::SameLine();
        }
    }

    // Action buttons (right-aligned)
    float buttons_width = 140.0f;
    ImGui::SameLine(card_width - buttons_width);

    auto& mgr = plugin::PluginManager::Instance();

    switch (p->state) {
        case plugin::PluginState::Initialized:
        case plugin::PluginState::Active:
            if (ImGui::SmallButton("Disable")) {
                mgr.DisablePlugin(p->manifest.id);
            }
            break;
        case plugin::PluginState::Loaded:
            if (ImGui::SmallButton("Initialize")) {
                mgr.InitializePlugin(p->manifest.id);
            }
            break;
        case plugin::PluginState::Failed:
            if (ImGui::SmallButton("Retry")) {
                mgr.SetPluginState(p->manifest.id, plugin::PluginState::Loaded);
                mgr.InitializePlugin(p->manifest.id);
            }
            break;
        case plugin::PluginState::Disabled:
            if (ImGui::SmallButton("Enable")) {
                if (mgr.EnablePlugin(p->manifest.id)) {
                    mgr.InitializePlugin(p->manifest.id);
                }
            }
            break;
        default:
            break;
    }

    ImGui::SameLine();
    if (ImGui::SmallButton("Unload")) {
        mgr.UnloadPlugin(p->manifest.id);
    }

    // Separator between cards
    ImGui::Separator();

    ImGui::PopID();
}

// ============================================================================
// Permission Badge
// ============================================================================

void PluginManagerPanel::RenderPermissionBadge(plugin::PluginPermission perm, bool is_dangerous) {
    const char* name = plugin::security::PermissionStore::PermissionDisplayName(perm);
    ImVec4 bg = is_dangerous ? ImVec4(0.7f, 0.2f, 0.2f, 0.8f) : ImVec4(0.2f, 0.5f, 0.2f, 0.8f);
    ImVec4 text_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

    ImGui::PushStyleColor(ImGuiCol_Button, bg);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bg);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, bg);
    ImGui::PushStyleColor(ImGuiCol_Text, text_col);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);

    ImGui::SmallButton(name);

    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(plugin::security::PermissionStore::PermissionDescription(perm));
        ImGui::EndTooltip();
    }

    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(4);
}

// ============================================================================
// Status Bar
// ============================================================================

void PluginManagerPanel::RenderStatusBar() {
    auto& mgr = plugin::PluginManager::Instance();
    auto plugins = mgr.GetAllPlugins();

    int active = 0, disabled = 0, failed = 0;
    for (const auto* p : plugins) {
        if (!p) continue;
        switch (p->state) {
            case plugin::PluginState::Initialized:
            case plugin::PluginState::Active:
                active++;
                break;
            case plugin::PluginState::Disabled:
                disabled++;
                break;
            case plugin::PluginState::Failed:
                failed++;
                break;
            default:
                break;
        }
    }

    ImGui::TextDisabled("%d active, %d failed, %d disabled", active, failed, disabled);
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 80.0f);
    ImGui::TextDisabled("%zu plugins", plugins.size());
}

// ============================================================================
// Install Popup
// ============================================================================

void PluginManagerPanel::RenderInstallPopup() {
    if (show_install_popup_) {
        ImGui::OpenPopup("Install Plugin###InstallPluginPopup");
        show_install_popup_ = false;
    }

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(500, 0), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Install Plugin###InstallPluginPopup", nullptr,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Enter the path to a plugin directory (containing plugin.json):");
        ImGui::Spacing();

        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##InstallPath", install_path_buf_, sizeof(install_path_buf_));

        ImGui::Spacing();

        float button_width = 120.0f;
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float total = button_width * 2 + spacing;
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - total) * 0.5f);

        if (ImGui::Button("Install", ImVec2(button_width, 0))) {
            std::string path_str(install_path_buf_);
            if (!path_str.empty()) {
                auto& mgr = plugin::PluginManager::Instance();
                std::filesystem::path dir(path_str);
                if (mgr.LoadPlugin(dir)) {
                    // Find the newly loaded plugin by matching its path
                    for (const auto* p : mgr.GetAllPlugins()) {
                        if (p && p->state == plugin::PluginState::Loaded) {
                            mgr.InitializePlugin(p->manifest.id);
                        }
                    }
                    spdlog::info("PluginManagerPanel: Installed plugin from {}", path_str);
                }
            }
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

} // namespace cyxwiz
