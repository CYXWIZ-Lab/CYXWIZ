#include "plugin_manager_panel.h"
#include "../icons.h"
#include "../../core/file_dialogs.h"
#include "../../plugin/plugin_manager.h"
#include "../../plugin/plugin_session.h"
#include "../../plugin/security/permission_store.h"
#include "../../core/async_task_manager.h"
#include "../ui_tokens.h"
#include "../ui_buttons.h"
#include "../python_bundle_ui.h"
#include "../../core/user_folders.h"
#include "../../core/window_manager.h"
#include "../../extension/python/python_node_provider.h"
#include "backend_pack_platform.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
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
    const ui::Tokens& t = ui::CurrentTokens();
    switch (state) {
        case plugin::PluginState::Initialized:
        case plugin::PluginState::Active:
            return t.success;  // green
        case plugin::PluginState::Loaded:
            return t.warning;  // yellow
        case plugin::PluginState::Failed:
            return t.error;  // red
        case plugin::PluginState::Disabled:
            return t.pending;  // gray
        default:
            return t.pending;
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

static std::string TrimCopy(std::string value) {
    value.erase(0, value.find_first_not_of(" \t\n\r"));
    if (!value.empty()) {
        value.erase(value.find_last_not_of(" \t\n\r") + 1);
    }
    return value;
}

static std::filesystem::path InstallerPath() {
    std::filesystem::path path = core::WindowManager::GetExecutablePath();
    path.replace_filename(std::string(runtime::CurrentInstallerManagerExecutableName()));
    return path;
}

// A folder as the card shows it: from the install (runtime\plugins\...) or the
// Engine's folder (plugins\examples\...) when inside one, else in full.
static std::string ShownFolder(const std::filesystem::path& folder) {
    std::error_code ec;
    auto absolute = std::filesystem::absolute(folder, ec).lexically_normal();
    std::vector<std::filesystem::path> bases;
    if (const auto runtime = ActiveRuntimeRootFromEnvironment(); !runtime.empty())
        bases.push_back(runtime.parent_path());
    bases.push_back(std::filesystem::path(core::WindowManager::GetExecutablePath()).parent_path());
    for (const auto& base : bases) {
        auto relative = absolute.lexically_relative(std::filesystem::absolute(base, ec).lexically_normal());
        if (!relative.empty() && *relative.begin() != "..") return relative.make_preferred().string();
    }
    return absolute.make_preferred().string();
}

static std::string ClockTime(std::chrono::system_clock::time_point when) {
    if (when == std::chrono::system_clock::time_point{}) return {};
    const std::time_t time = std::chrono::system_clock::to_time_t(when);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &time);
#else
    localtime_r(&time, &local);
#endif
    char text[16];
    std::strftime(text, sizeof(text), "%H:%M", &local);
    return text;
}

static PluginTrustFacts TrustFacts(const plugin::LoadedPlugin& p) {
    PluginTrustFacts facts;
    facts.folder = ShownFolder(p.plugin_dir);
    if (p.origin != plugin::LoadedPlugin::Origin::SignedPack) return facts;
    facts.kind = PluginTrustKind::SignedAddOn;
    facts.package_version = p.package_version;
    facts.pack_id = p.pack_id;
    for (const auto& pack : plugin::PluginManager::Instance().GetPackPlugins())
        if (pack.plugin_id == p.manifest.id) facts.companion_base_id = pack.companion_base_id;
    facts.files_checked = p.pack_files_checked;
    facts.bytes_checked = p.pack_bytes_checked;
    facts.check_ms = p.pack_check_ms;
    facts.checked_at = ClockTime(p.pack_checked_at);
    return facts;
}

static PluginTrustFacts TrustFacts(const plugin::PackRefusal& refusal) {
    PluginTrustFacts facts;
    facts.kind = PluginTrustKind::RefusedAddOn;
    facts.package_version = refusal.package_version;
    facts.pack_id = refusal.pack_id;
    facts.folder = ShownFolder(refusal.directory);
    facts.reason = refusal.reason;
    facts.detail = refusal.detail;
    facts.files_listed = refusal.files_listed;
    facts.issues = refusal.issues;
    facts.checked_at = ClockTime(refusal.checked_at);
    return facts;
}

static ImVec4 TrustColour(PluginTrustTone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
        case PluginTrustTone::Verified: return t.success;
        case PluginTrustTone::Failed: return t.error;
        case PluginTrustTone::Muted: break;
    }
    return t.text_dim;
}

static const char* TrustIcon(PluginTrustTone tone) {
    switch (tone) {
        case PluginTrustTone::Verified: return ICON_FA_SHIELD_HALVED;
        case PluginTrustTone::Failed: return ICON_FA_TRIANGLE_EXCLAMATION;
        case PluginTrustTone::Muted: break;
    }
    return ICON_FA_CODE;
}

// The trust line under a card's name: icon and words in the tone, the note muted.
static void RenderTrustLine(const PluginTrustView& view) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec4 colour = TrustColour(view.tone);
    ImGui::TextColored(colour, "%s", TrustIcon(view.tone));
    ImGui::SameLine();
    ImGui::TextColored(colour, "%s", view.line.c_str());
    if (!view.note.empty()) {
        ImGui::SameLine();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "\xc2\xb7 %s", view.note.c_str());
        ImGui::PopTextWrapPos();
    }
}

// A card's first line: state dot, name, id muted, and on the right `right`.
static void RenderCardHeader(const ImVec4& dot, const char* state_tip, const std::string& name,
                             const std::string& id, const std::string& right, const ImVec4& right_colour) {
    const ui::Tokens& t = ui::CurrentTokens();
    const float right_edge = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
    ImGui::TextColored(dot, ICON_FA_CIRCLE);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("State: %s", state_tip);
    ImGui::SameLine();
    ImGui::TextColored(t.text_bright, "%s", name.c_str());
    const float right_width = ImGui::CalcTextSize(right.c_str()).x;
    const float id_width = ImGui::CalcTextSize(id.c_str()).x;
    ImGui::SameLine();
    // The id only where it fits before the right-hand text.
    if (ImGui::GetCursorPosX() + id_width + ImGui::GetStyle().ItemSpacing.x * 2 + right_width < right_edge) {
        ImGui::TextColored(t.text_dim, "%s", id.c_str());
        ImGui::SameLine();
    }
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), right_edge - right_width));
    ImGui::TextColored(right_colour, "%s", right.c_str());
}

static void ShowInstallSpinner(const std::string& status) {
    const ui::Tokens& t = ui::CurrentTokens();
    float time = static_cast<float>(ImGui::GetTime());
    const char* spinner_chars[] = {"|", "/", "-", "\\"};
    int spinner_idx = static_cast<int>(time * 8) % 4;
    ImGui::TextColored(t.running, "%s %s",
                       spinner_chars[spinner_idx], status.c_str());
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
    const auto refusals = mgr.GetPackRefusals();

    RenderToolbar();
    ImGui::Spacing();
    RenderPluginList(plugins, refusals);
    RenderStatusBar(plugins, refusals);

    // Apply deferred unloads after iteration is complete (CRITICAL-1/HIGH-4)
    for (const auto& id : deferred_unloads_) {
        mgr.UnloadPlugin(id);
        plugin::RecordPluginUnloaded(id);
        spdlog::info("PluginManagerPanel: Unloaded plugin {}", id);
    }
    if (!deferred_unloads_.empty()) ReloadStartupList();
    deferred_unloads_.clear();

    ImGui::End();

    RenderInstallPopup();
}

// ============================================================================
// Toolbar
// ============================================================================

void PluginManagerPanel::RenderToolbar() {
    float avail = ImGui::GetContentRegionAvail().x;

    // Search box, then the two actions at their own width (never clipped)
    const char* refresh_label = ICON_FA_ARROWS_ROTATE " Refresh";
    const char* install_label = ICON_FA_FOLDER_PLUS " Install...";
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float buttons = ui::ButtonWidth(refresh_label, ui::ButtonSize::Regular) +
                          ui::ButtonWidth(install_label, ui::ButtonSize::Regular) + spacing * 2;
    ImGui::SetNextItemWidth(std::max(120.0f, avail - buttons));
    ImGui::InputTextWithHint("##PluginSearch", ICON_FA_MAGNIFYING_GLASS " Search plugins...",
                             search_buf_, sizeof(search_buf_));

    ImGui::SameLine();

    if (ui::SecondaryButton(refresh_label, true, nullptr, ui::ButtonSize::Regular)) {
        auto& mgr = plugin::PluginManager::Instance();
        mgr.LoadAllFromSearchPaths();
        mgr.InitializeAll();
        plugin::RecordLoadedPlugins(mgr);
        ReloadStartupList();
        if (auto* bundles = python_ui::Provider()) bundles->Rescan();  // Python node bundles too
        spdlog::info("PluginManagerPanel: Refreshed plugin list");
    }

    ImGui::SameLine();

    if (ui::SecondaryButton(install_label, true, nullptr, ui::ButtonSize::Regular)) {
        show_install_popup_ = true;
        install_path_buf_[0] = '\0';
    }
}

// ============================================================================
// Plugin List
// ============================================================================

void PluginManagerPanel::RenderPluginList(const std::vector<const plugin::LoadedPlugin*>& plugins,
                                          const std::vector<plugin::PackRefusal>& refusals) {
    std::string filter(search_buf_);

    // Reserve the status bar's lines (two when an add-on was verified)
    const float status_height = ImGui::GetTextLineHeightWithSpacing() * 2.0f + ImGui::GetStyle().ItemSpacing.y;
    float height = ImGui::GetContentRegionAvail().y - status_height;
    if (ImGui::BeginChild("PluginList", ImVec2(0, height), ImGuiChildFlags_None)) {
        // Installed add-ons the Engine refused come first: they need the user.
        for (const auto& refusal : refusals) {
            if (!filter.empty() && !ContainsCI(refusal.title, filter) && !ContainsCI(refusal.plugin_id, filter))
                continue;
            ImGui::PushID(("refused:" + refusal.plugin_id).c_str());
            RenderRefusedCard(refusal);
            ImGui::PopID();
            ImGui::Spacing();
        }
        if (plugins.empty()) {
            if (refusals.empty()) {
                ImGui::TextDisabled("No plugins loaded.");
                ImGui::TextDisabled("Use the search paths or Install button to add plugins.");
            }
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
                ImGui::PushID(p->manifest.id.c_str());
                RenderPluginCard(p);
                ImGui::Spacing();
                ImGui::PopID();
            }

            // No results message (LOW-15)
            if (match_count == 0 && !filter.empty()) {
                ImGui::TextDisabled("No plugins match '%s'", filter.c_str());
            }
        }
        // Python node bundles (TOFIX125 P4 step 4.5, mockup row 6).
        ImGui::Spacing();
        ImGui::Spacing();
        python_ui::RenderBundleSection(filter);
    }
    ImGui::EndChild();
}

// ============================================================================
// Plugin Card
// ============================================================================


void PluginManagerPanel::RenderPluginCard(const plugin::LoadedPlugin* p) {
    const ui::Tokens& t = ui::CurrentTokens();
    // A card: tinted, no outline; its height follows the content.
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
    ImGui::BeginChild("card", ImVec2(0.0f, 0.0f),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar);

    RenderCardHeader(StateColor(p->state), StateLabel(p->state), p->manifest.name, p->manifest.id,
                     "v" + p->manifest.version.ToString(), t.text_dim);
    const float indent = ImGui::CalcTextSize(ICON_FA_CIRCLE).x + ImGui::GetStyle().ItemSpacing.x;
    ImGui::Indent(indent);
    const auto trust = PluginTrustLine(TrustFacts(*p));
    RenderTrustLine(trust);

    ImGui::PushTextWrapPos(0.0f);
    if (!p->manifest.description.empty()) ImGui::TextUnformatted(p->manifest.description.c_str());
    if (!p->manifest.author.empty()) ImGui::TextColored(t.text_dim, "Author: %s", p->manifest.author.c_str());
    if (p->state == plugin::PluginState::Failed && !p->error_message.empty())
        ImGui::TextColored(t.error, "%s", p->error_message.c_str());
    ImGui::PopTextWrapPos();

    // What it adds and whether it loads at startup (TOFIX125 steps 2.9-2.10).
    {
        const auto contribution = plugin::DescribeContribution(p->manifest.id);
        const std::string adds = cyxwiz::PluginAddsLine(contribution);
        if (!adds.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.accent_text, "%s", adds.c_str());
            ImGui::PopTextWrapPos();
        }
        if (!startup_list_read_) ReloadStartupList();
        bool at_startup = cyxwiz::PluginLoadsAtStartup(
            startup_list_, p->manifest.id, p->origin == plugin::LoadedPlugin::Origin::SignedPack);
        if (ImGui::Checkbox("Load when the Engine starts", &at_startup)) {
            plugin::SetLoadAtStartup(p->manifest.id, p->plugin_dir.generic_string(), at_startup);
            ReloadStartupList();
        }
        ImGui::Indent(ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", cyxwiz::PluginStartupNote(at_startup, contribution).c_str());
        ImGui::PopTextWrapPos();
        ImGui::Unindent(ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x);
    }

    // Permission badges wrap within the card; the actions get their own line below them
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

    const float right_edge = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    bool first_badge = true;
    for (auto perm : all_perms) {
        if (plugin::HasPermission(p->manifest.permissions, perm)) {
            const char* name = plugin::security::PermissionStore::PermissionDisplayName(perm);
            const float badge_width = ui::StatusPillWidth(name);
            if (!first_badge) {
                ImGui::SameLine();
                if (ImGui::GetCursorScreenPos().x + badge_width > right_edge) ImGui::NewLine();
            }
            bool dangerous = plugin::security::PermissionStore::IsDangerousPermission(perm);
            RenderPermissionBadge(perm, dangerous);
            first_badge = false;
        }
    }

    auto& mgr = plugin::PluginManager::Instance();
    std::string plugin_id = p->manifest.id;  // Copy ID before any mutation

    // Row actions: the one the state allows, Unload, then Details.
    ImGui::Spacing();
    switch (p->state) {
        case plugin::PluginState::Initialized:
        case plugin::PluginState::Active:
            if (ui::SecondaryButton("Disable", true, nullptr, ui::ButtonSize::Regular)) {
                mgr.DisablePlugin(plugin_id);
            }
            ImGui::SameLine();
            break;
        case plugin::PluginState::Loaded:
            if (ui::SecondaryButton("Initialize", true, nullptr, ui::ButtonSize::Regular)) {
                mgr.InitializePlugin(plugin_id);
            }
            ImGui::SameLine();
            break;
        case plugin::PluginState::Failed:
            if (ui::SecondaryButton("Retry", true, nullptr, ui::ButtonSize::Regular)) {
                mgr.SetPluginState(plugin_id, plugin::PluginState::Loaded);
                mgr.InitializePlugin(plugin_id);
            }
            ImGui::SameLine();
            break;
        case plugin::PluginState::Disabled:
            if (ui::SecondaryButton("Enable", true, nullptr, ui::ButtonSize::Regular)) {
                if (mgr.EnablePlugin(plugin_id)) {
                    mgr.InitializePlugin(plugin_id);
                }
            }
            ImGui::SameLine();
            break;
        default:
            break;
    }

    if (ui::SecondaryButton("Unload", true, nullptr, ui::ButtonSize::Regular)) {
        // Defer unload to after iteration (CRITICAL-1/HIGH-4)
        deferred_unloads_.push_back(plugin_id);
    }
    const bool open = details_open_.count(plugin_id) > 0;
    if (!trust.details.empty()) {
        ImGui::SameLine();
        if (ui::LinkButton(open ? "Hide" : "Details")) {
            if (open) details_open_.erase(plugin_id); else details_open_.insert(plugin_id);
        }
        if (open) RenderTrustDetails(trust);
    }
    ImGui::Unindent(indent);
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void PluginManagerPanel::RenderRefusedCard(const plugin::PackRefusal& refusal) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
    ImGui::BeginChild("card", ImVec2(0.0f, 0.0f),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar);
    RenderCardHeader(t.error, "Not loaded", refusal.title, refusal.plugin_id, "Not loaded", t.error);
    const float indent = ImGui::CalcTextSize(ICON_FA_CIRCLE).x + ImGui::GetStyle().ItemSpacing.x;
    ImGui::Indent(indent);
    const auto trust = PluginTrustLine(TrustFacts(refusal));
    RenderTrustLine(trust);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(trust.body.c_str());
    ImGui::TextColored(t.caution, "%s", trust.hint.c_str());
    ImGui::PopTextWrapPos();

    // Repair opens the CyxWiz Installer with this add-on ticked for repair,
    // the same action as its card in Installed packs.
    ImGui::Spacing();
    const auto runtime_root = ActiveRuntimeRootFromEnvironment();
    const auto installer = InstallerPath();
    std::error_code ec;
    const bool can_repair = !runtime_root.empty() && std::filesystem::is_regular_file(installer, ec);
    if (ui::PrimaryButton("Repair add-on...", can_repair,
                          "The CyxWiz Installer is part of an installed Engine; it is not in this build",
                          ui::ButtonSize::Regular)) {
        core::WindowManager::LaunchExecutable(
            installer.string(), {"--runtime-root", runtime_root.string(), "--select", refusal.pack_id});
        spdlog::info("Opened the CyxWiz Installer to repair add-on {}", refusal.title);
    }
    if (can_repair && ImGui::IsItemHovered())
        ImGui::SetTooltip("Opens the CyxWiz Installer with this add-on selected for repair.");
    ImGui::SameLine();
    if (ui::SecondaryButton("Open Installed packs", static_cast<bool>(open_installed_packs_), nullptr,
                            ui::ButtonSize::Regular)) {
        open_installed_packs_();
    }
    const bool open = details_open_.count(refusal.plugin_id) > 0;
    ImGui::SameLine();
    if (ui::LinkButton(open ? "Hide" : "Details")) {
        if (open) details_open_.erase(refusal.plugin_id); else details_open_.insert(refusal.plugin_id);
    }
    if (open) RenderTrustDetails(trust);
    ImGui::Unindent(indent);
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

// Details: a full-width key/value table, two pairs per line, on a raised tone.
void PluginManagerPanel::RenderTrustDetails(const PluginTrustView& view) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_bar);
    ImGui::BeginChild("details", ImVec2(0.0f, 0.0f),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar);
    if (ImGui::BeginTable("details", 4, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("key1", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("value1", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        ImGui::TableSetupColumn("key2", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("value2", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        for (std::size_t index = 0; index < view.details.size(); ++index) {
            if (index % 2 == 0) ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", view.details[index].first.c_str());
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(view.details[index].second.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void PluginManagerPanel::ReloadStartupList() {
    startup_list_ = plugin::ReadPluginStartupList();
    startup_list_read_ = true;
}

// ============================================================================
// Permission Badge
// ============================================================================

void PluginManagerPanel::RenderPermissionBadge(plugin::PluginPermission perm, bool is_dangerous) {
    const ui::Tokens& t = ui::CurrentTokens();
    const char* name = plugin::security::PermissionStore::PermissionDisplayName(perm);
    // A label, not an action: the shared status pill, its dot in the risk's tone
    ui::StatusPill((std::string("##perm_") + name).c_str(), name, is_dangerous ? t.error : t.success);
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(plugin::security::PermissionStore::PermissionDescription(perm));
        ImGui::EndTooltip();
    }
}

// ============================================================================
// Status Bar
// ============================================================================

void PluginManagerPanel::RenderStatusBar(const std::vector<const plugin::LoadedPlugin*>& plugins,
                                         const std::vector<plugin::PackRefusal>& refusals) {
    PluginListCounts counts;
    counts.refused = refusals.size();
    for (const auto* p : plugins) {
        if (!p) continue;
        ++counts.plugins;
        if (p->origin == plugin::LoadedPlugin::Origin::SignedPack) ++counts.verified;
        switch (p->state) {
            case plugin::PluginState::Initialized:
            case plugin::PluginState::Active:
                ++counts.active;
                break;
            case plugin::PluginState::Disabled:
                ++counts.disabled;
                break;
            case plugin::PluginState::Failed:
                ++counts.failed;
                break;
            default:
                break;
        }
    }
    counts.plugins += counts.refused;

    // One line when it fits, else one line each.
    const auto lines = PluginListSummary(counts);
    for (std::size_t index = 0; index < lines.size(); ++index) {
        if (index > 0) {
            const float width = ImGui::CalcTextSize(lines[index].c_str()).x;
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.x * 4);
            if (ImGui::GetContentRegionAvail().x < width) ImGui::NewLine();
        }
        ImGui::TextDisabled("%s", lines[index].c_str());
    }
}

// ============================================================================
// Install Popup
// ============================================================================

void PluginManagerPanel::RenderInstallPopup() {
    const ui::Tokens& t = ui::CurrentTokens();
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
        if (close_install_popup_) {
            close_install_popup_ = false;
            ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
            return;
        }

        ImGui::Text("Choose a plugin directory containing plugin.json:");
        ImGui::Spacing();

        ImGui::BeginDisabled(is_installing_plugin_);

        const float browse_width = 100.0f;
        ImGui::SetNextItemWidth(-browse_width - ImGui::GetStyle().ItemSpacing.x);
        bool enter_pressed = ImGui::InputTextWithHint(
            "##InstallPath",
            "D:\\path\\to\\plugin",
            install_path_buf_,
            sizeof(install_path_buf_),
            ImGuiInputTextFlags_EnterReturnsTrue);

        ImGui::SameLine();
        if (ui::SecondaryButton(ICON_FA_FOLDER_OPEN " Browse", true, nullptr, ui::ButtonSize::Regular,
                                browse_width)) {
            const std::string current_path = TrimCopy(install_path_buf_);
            const char* default_path = current_path.empty() ? nullptr : current_path.c_str();
            if (auto selected = cyxwiz::FileDialogs::SelectFolder(
                    "Select Plugin Directory", default_path)) {
                std::snprintf(install_path_buf_, sizeof(install_path_buf_),
                              "%s", selected->c_str());
            }
        }

        ImGui::EndDisabled();

        ImGui::Spacing();

        float button_width = 120.0f;
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float total = button_width * 2 + spacing;
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - total) * 0.5f);

        auto start_install = [this]() -> bool {
            if (is_installing_plugin_) {
                return false;
            }

            std::string path_str = TrimCopy(install_path_buf_);
            if (path_str.empty()) {
                install_error_ = "Please enter a path.";
                show_install_error_ = true;
                return false;
            }

            std::filesystem::path dir(path_str);
            std::error_code ec;
            if (!std::filesystem::exists(dir, ec)) {
                install_error_ = "Path does not exist: " + path_str;
                show_install_error_ = true;
                return false;
            }
            if (!std::filesystem::is_directory(dir, ec)) {
                install_error_ = "Not a directory: " + path_str;
                show_install_error_ = true;
                return false;
            }
            if (!std::filesystem::exists(dir / "plugin.json", ec)) {
                install_error_ = "No plugin.json found in: " + path_str;
                show_install_error_ = true;
                return false;
            }

            plugin::PluginManifest manifest;
            std::string manifest_error;
            if (!plugin::PluginLoader::ParseManifest(
                    dir, manifest, manifest_error)) {
                install_error_ = "Invalid plugin manifest: " + manifest_error;
                show_install_error_ = true;
                return false;
            }

            // A plugin id loads once. Installing it again would keep the code
            // already in memory and look like a fresh install, so say what to
            // do instead.
            if (const auto* existing = plugin::PluginManager::Instance().GetLoadedPlugin(manifest.id)) {
                if (existing->state == plugin::PluginState::Disabled) {
                    install_error_ = "Plugin is already installed but disabled. "
                        "Close this dialog and use Enable.";
                } else if (existing->state == plugin::PluginState::Failed) {
                    install_error_ = "Plugin is already installed but failed. "
                        "Close this dialog and use Retry.";
                } else {
                    install_error_ = manifest.name + " " + existing->manifest.version.ToString() +
                        " is already loaded from " + existing->plugin_dir.string() +
                        ". To load a new build: Unload it, rebuild, then Install again. "
                        "If a model still uses its nodes, restart the Engine instead of Unload.";
                }
                show_install_error_ = true;
                return false;
            }

            is_installing_plugin_ = true;
            install_status_ = "Installing...";
            const std::string task_name = "Installing plugin: " + dir.filename().string();

            install_task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
                task_name,
                [dir, path_str](cyxwiz::LambdaTask& task) {
                    task.ReportProgress(0.30f, "Loading plugin...");
                    if (task.ShouldStop()) return;

                    auto& mgr = plugin::PluginManager::Instance();
                    std::string load_error;
                    if (!mgr.LoadPlugin(dir, &load_error)) {
                        task.MarkFailed("Failed to load plugin from " + path_str +
                                        (load_error.empty() ? "" : ": " + load_error));
                        return;
                    }

                    task.MarkCompleted("Plugin loaded");
                },
                nullptr,
                [this, path_str, plugin_id = manifest.id](
                    bool success, const std::string& error) {
                    is_installing_plugin_ = false;
                    install_task_id_ = 0;
                    install_status_.clear();

                    if (!success) {
                        install_error_ = error.empty()
                            ? "Failed to install plugin. Check console for details."
                            : error;
                        show_install_error_ = true;
                        return;
                    }

                    auto& mgr = plugin::PluginManager::Instance();
                    plugin::RecordLoadedPlugins(mgr);
                    ReloadStartupList();
                    const bool initialized = mgr.InitializePlugin(plugin_id);
                    const auto final_state = mgr.GetPluginState(plugin_id);
                    const auto* loaded = mgr.GetLoadedPlugin(plugin_id);
                    if (!initialized && final_state != plugin::PluginState::Loaded) {
                        install_error_ =
                            "Plugin loaded but could not be initialized: " + path_str;
                        if (loaded && !loaded->error_message.empty())
                            install_error_ += "\n" + loaded->error_message;
                        show_install_error_ = true;
                        return;
                    }
                    // Nodes the plugin offered but the Engine refused stay
                    // visible here, not only in the Console.
                    if (loaded && !loaded->skipped_nodes.empty()) {
                        install_error_ = "Plugin installed, but " +
                            std::to_string(loaded->skipped_nodes.size()) +
                            " of its nodes were not added:";
                        for (const auto& reason : loaded->skipped_nodes) install_error_ += "\n- " + reason;
                        show_install_error_ = true;
                        return;
                    }

                    close_install_popup_ = true;
                    spdlog::info(
                        "PluginManagerPanel: Plugin ready after selecting {}",
                        path_str);
                }
            );
            return true;
        };

        if (enter_pressed) {
            start_install();
        }

        ImGui::BeginDisabled(is_installing_plugin_);

        if (ui::PrimaryButton(is_installing_plugin_ ? ICON_FA_SPINNER " Installing..." : "Install", true, nullptr,
                              ui::ButtonSize::Regular, button_width)) {
            start_install();
        }

        ImGui::EndDisabled();

        ImGui::SameLine();

        if (ui::SecondaryButton(is_installing_plugin_ ? "Cancel" : "Close", true, nullptr, ui::ButtonSize::Regular,
                                button_width)) {
            if (is_installing_plugin_) {
                install_status_ = "Cancelling...";
                if (install_task_id_ != 0) {
                    cyxwiz::AsyncTaskManager::Instance().Cancel(install_task_id_);
                }
            } else {
                ImGui::CloseCurrentPopup();
            }
        }

        if (is_installing_plugin_) {
            ImGui::Spacing();
            ShowInstallSpinner(install_status_.empty() ? "Installing plugin..." : install_status_);
            ImGui::TextDisabled("The engine stays responsive while the install runs.");
        }

        ImGui::EndPopup();
    }

    // Error popup (HIGH-6)
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Install Error###InstallErrorPopup", nullptr,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextColored(t.error, "%s", install_error_.c_str());
        ImGui::Spacing();
        float ok_width = 80.0f;
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ok_width) * 0.5f);
        if (ui::PrimaryButton("OK", true, nullptr, ui::ButtonSize::Regular, ok_width)) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

} // namespace cyxwiz
