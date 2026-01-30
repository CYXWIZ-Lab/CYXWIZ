#include "plugin_manager_panel.h"
#include "../icons.h"
#include "../../plugin/plugin_manager.h"
#include "../../plugin/security/permission_store.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <filesystem>

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

static const char* StateLabel(plugin::PluginState state) {
    switch (state) {
        case plugin::PluginState::Initialized: return "Initialized";
        case plugin::PluginState::Active:      return "Active";
        case plugin::PluginState::Loaded:      return "Loaded";
        case plugin::PluginState::Failed:      return "Failed";
        case plugin::PluginState::Disabled:    return "Disabled";
        case plugin::PluginState::Unloaded:    return "Unloaded";
        default:                               return "Unknown";
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

    // Fetch plugins once per frame (MEDIUM-9)
    auto& mgr = plugin::PluginManager::Instance();
    auto plugins = mgr.GetAllPlugins();

    RenderToolbar();
    ImGui::Separator();
    RenderPluginList(plugins);
    ImGui::Separator();
    RenderStatusBar(plugins);

    // Apply deferred unloads after iteration is complete (CRITICAL-1/HIGH-4)
    for (const auto& id : deferred_unloads_) {
        mgr.UnloadPlugin(id);
        spdlog::info("PluginManagerPanel: Unloaded plugin {}", id);
    }
    deferred_unloads_.clear();

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

void PluginManagerPanel::RenderPluginList(const std::vector<const plugin::LoadedPlugin*>& plugins) {
    std::string filter(search_buf_);

    float height = ImGui::GetContentRegionAvail().y - 30.0f; // reserve for status bar
    if (ImGui::BeginChild("PluginList", ImVec2(0, height), ImGuiChildFlags_None)) {
        if (plugins.empty()) {
            ImGui::TextDisabled("No plugins loaded.");
            ImGui::TextDisabled("Use the search paths or Install button to add plugins.");
        } else {
            size_t match_count = 0;
            for (size_t i = 0; i < plugins.size(); ++i) {
                const auto* p = plugins[i];
                if (!p) continue;

                // Filter
                if (!filter.empty()) {
                    bool match = ContainsCI(p->manifest.name, filter) ||
                                 ContainsCI(p->manifest.id, filter) ||
                                 ContainsCI(p->manifest.author, filter) ||
                                 ContainsCI(p->manifest.description, filter);
                    if (!match) continue;
                }

                match_count++;
                ImGui::PushID(static_cast<int>(i));
                RenderPluginCard(p);
                ImGui::Spacing();
                ImGui::PopID();
            }

            // No results message (LOW-15)
            if (match_count == 0 && !filter.empty()) {
                ImGui::TextDisabled("No plugins match '%s'", filter.c_str());
            }
        }
    }
    ImGui::EndChild();
}

// ============================================================================
// Plugin Card
// ============================================================================

void PluginManagerPanel::RenderPluginCard(const plugin::LoadedPlugin* p) {
    float card_width = ImGui::GetContentRegionAvail().x;

    // State dot + Name + Version
    ImVec4 color = StateColor(p->state);
    ImGui::TextColored(color, ICON_FA_CIRCLE);

    // State tooltip (LOW-12)
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("State: %s", StateLabel(p->state));
        ImGui::EndTooltip();
    }

    ImGui::SameLine();
    ImGui::Text("%s", p->manifest.name.c_str());

    std::string version_str = "v" + p->manifest.version.ToString();
    float version_width = ImGui::CalcTextSize(version_str.c_str()).x;
    ImGui::SameLine(card_width - version_width - 10.0f);
    ImGui::TextDisabled("%s", version_str.c_str());

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
    static constexpr plugin::PluginPermission all_perms[] = {
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
    std::string plugin_id = p->manifest.id;  // Copy ID before any mutation

    switch (p->state) {
        case plugin::PluginState::Initialized:
        case plugin::PluginState::Active:
            if (ImGui::SmallButton("Disable")) {
                mgr.DisablePlugin(plugin_id);
            }
            break;
        case plugin::PluginState::Loaded:
            if (ImGui::SmallButton("Initialize")) {
                mgr.InitializePlugin(plugin_id);
            }
            break;
        case plugin::PluginState::Failed:
            if (ImGui::SmallButton("Retry")) {
                mgr.SetPluginState(plugin_id, plugin::PluginState::Loaded);
                mgr.InitializePlugin(plugin_id);
            }
            break;
        case plugin::PluginState::Disabled:
            if (ImGui::SmallButton("Enable")) {
                if (mgr.EnablePlugin(plugin_id)) {
                    mgr.InitializePlugin(plugin_id);
                }
            }
            break;
        default:
            break;
    }

    ImGui::SameLine();
    if (ImGui::SmallButton("Unload")) {
        // Defer unload to after iteration (CRITICAL-1/HIGH-4)
        deferred_unloads_.push_back(plugin_id);
    }

    // Separator between cards
    ImGui::Separator();
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

void PluginManagerPanel::RenderStatusBar(const std::vector<const plugin::LoadedPlugin*>& plugins) {
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
    std::string count_text = std::to_string(plugins.size()) + " plugins";
    float count_width = ImGui::CalcTextSize(count_text.c_str()).x;
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - count_width - 10.0f);
    ImGui::TextDisabled("%s", count_text.c_str());
}

// ============================================================================
// Install Popup
// ============================================================================

void PluginManagerPanel::RenderInstallPopup() {
    if (show_install_popup_) {
        ImGui::OpenPopup("Install Plugin###InstallPluginPopup");
        show_install_popup_ = false;
    }

    // Install error popup (HIGH-6)
    if (show_install_error_) {
        ImGui::OpenPopup("Install Error###InstallErrorPopup");
        show_install_error_ = false;
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
            // Trim whitespace (HIGH-3)
            path_str.erase(0, path_str.find_first_not_of(" \t\n\r"));
            if (!path_str.empty())
                path_str.erase(path_str.find_last_not_of(" \t\n\r") + 1);

            if (path_str.empty()) {
                install_error_ = "Please enter a path.";
                show_install_error_ = true;
            } else {
                std::filesystem::path dir(path_str);
                std::error_code ec;

                // Validate path exists (HIGH-3)
                if (!std::filesystem::exists(dir, ec)) {
                    install_error_ = "Path does not exist: " + path_str;
                    show_install_error_ = true;
                } else if (!std::filesystem::is_directory(dir, ec)) {
                    install_error_ = "Not a directory: " + path_str;
                    show_install_error_ = true;
                } else if (!std::filesystem::exists(dir / "plugin.json", ec)) {
                    install_error_ = "No plugin.json found in: " + path_str;
                    show_install_error_ = true;
                } else {
                    auto& mgr = plugin::PluginManager::Instance();
                    if (mgr.LoadPlugin(dir)) {
                        // Initialize only the plugin we just loaded by matching directory (CRITICAL-2)
                        for (const auto* p : mgr.GetAllPlugins()) {
                            if (p && p->plugin_dir == dir) {
                                mgr.InitializePlugin(p->manifest.id);
                                break;
                            }
                        }
                        spdlog::info("PluginManagerPanel: Installed plugin from {}", path_str);
                    } else {
                        install_error_ = "Failed to load plugin. Check console for details.";
                        show_install_error_ = true;
                    }
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

    // Error popup (HIGH-6)
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Install Error###InstallErrorPopup", nullptr,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", install_error_.c_str());
        ImGui::Spacing();
        float ok_width = 80.0f;
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ok_width) * 0.5f);
        if (ImGui::Button("OK", ImVec2(ok_width, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

} // namespace cyxwiz
