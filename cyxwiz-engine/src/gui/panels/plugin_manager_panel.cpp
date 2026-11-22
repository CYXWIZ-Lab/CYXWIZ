#include "plugin_manager_panel.h"
#include "../icons.h"
#include "../file_browser_launch.h"
#include "../../core/file_dialogs.h"
#include "../../plugin/plugin_manager.h"
#include "../../plugin/plugin_session.h"
#include "../../plugin/security/permission_store.h"
#include "../../core/async_task_manager.h"
#include "../ui_tokens.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"
#include "../python_bundle_ui.h"
#include "../../extension/python/python_node_provider.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>

namespace cyxwiz {

namespace {

constexpr const char* kInstallTitle = "Install a plugin###InstallPluginDialog";

// Case-insensitive substring search
bool ContainsCI(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return true;
    auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
                          [](char a, char b) { return std::tolower(a) == std::tolower(b); });
    return it != haystack.end();
}

std::string TrimCopy(std::string value) {
    value.erase(0, value.find_first_not_of(" \t\n\r"));
    if (!value.empty()) value.erase(value.find_last_not_of(" \t\n\r") + 1);
    return value;
}

PluginRunState RunStateOf(plugin::PluginState state) {
    switch (state) {
        case plugin::PluginState::Loaded: return PluginRunState::Loaded;
        case plugin::PluginState::Initialized:
        case plugin::PluginState::Active: return PluginRunState::Running;
        case plugin::PluginState::Failed: return PluginRunState::Failed;
        case plugin::PluginState::Disabled: return PluginRunState::Disabled;
        case plugin::PluginState::Unloaded: return PluginRunState::Unloaded;
    }
    return PluginRunState::Unloaded;
}

ImVec4 ToneColour(PluginTone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
        case PluginTone::Good: return t.success;
        case PluginTone::Waiting: return t.pending;
        case PluginTone::Problem: return t.caution;
        case PluginTone::Off: return t.text_dim;
    }
    return t.text_dim;
}

constexpr plugin::PluginPermission kAllPermissions[] = {
    plugin::PluginPermission::FileSystem, plugin::PluginPermission::Network,
    plugin::PluginPermission::SystemCommands, plugin::PluginPermission::Python,
    plugin::PluginPermission::GPU, plugin::PluginPermission::DataRegistry,
    plugin::PluginPermission::Training, plugin::PluginPermission::UIModify,
};

std::vector<PluginPermissionFacts> PermissionFactsOf(const plugin::LoadedPlugin& p,
                                                     std::vector<plugin::PluginPermission>* perms_out = nullptr) {
    using plugin::security::PermissionStore;
    auto& store = PermissionStore::Instance();
    const std::string version = p.manifest.version.ToString();
    std::vector<PluginPermissionFacts> facts;
    for (const auto perm : kAllPermissions) {
        if (!plugin::HasPermission(p.manifest.permissions, perm)) continue;
        PluginPermissionFacts f;
        f.name = PermissionStore::PermissionDisplayName(perm);
        f.description = PermissionStore::PermissionDescription(perm);
        f.needs_answer = PermissionStore::IsDangerousPermission(perm);
        const auto decision = store.GetDecision(p.manifest.id, version, perm);
        f.answer = decision == plugin::security::PermissionDecision::Allowed  ? PermissionAnswer::Allowed
                   : decision == plugin::security::PermissionDecision::Denied ? PermissionAnswer::Denied
                                                                              : PermissionAnswer::NotAsked;
        facts.push_back(f);
        if (perms_out) perms_out->push_back(perm);
    }
    return facts;
}

}  // namespace

PluginManagerPanel::PluginManagerPanel()
    : Panel("Plugin Manager", false) {
}

const char* PluginManagerPanel::GetIcon() const {
    return ICON_FA_PLUG;
}

void PluginManagerPanel::ShowPlugin(const std::string& plugin_id) {
    visible_ = true;
    scroll_to_ = plugin_id;
    search_buf_[0] = '\0';  // a filter must not hide the card asked for
}

const PluginContribution& PluginManagerPanel::ContributionOf(const std::string& plugin_id) {
    const auto now = std::chrono::steady_clock::now();
    if (now - contributions_read_at_ > std::chrono::seconds(1)) {
        contributions_.clear();
        contributions_read_at_ = now;
    }
    auto it = contributions_.find(plugin_id);
    if (it == contributions_.end()) it = contributions_.emplace(plugin_id, plugin::DescribeContribution(plugin_id)).first;
    return it->second;
}

void PluginManagerPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(640, 560), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(GetName(), &visible_)) {
        ImGui::End();
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    auto& mgr = plugin::PluginManager::Instance();
    const auto plugins = mgr.GetAllPlugins();
    if (!startup_list_read_) ReloadStartupList();

    std::vector<PluginCardFacts> facts;
    facts.reserve(plugins.size());
    for (const auto* p : plugins) {
        PluginCardFacts f;
        if (p) {
            f.id = p->manifest.id;
            f.name = p->manifest.name;
            f.state = RunStateOf(p->state);
            f.error = p->error_message;
            f.awaiting_answer = p->state == plugin::PluginState::Loaded && mgr.IsAwaitingPermission(p->manifest.id);
        }
        facts.push_back(f);
    }

    RenderToolbar();
    const std::string footer = PluginFooter(facts);
    const float footer_h = ImGui::GetTextLineHeightWithSpacing() * 2.0f;
    if (ImGui::BeginChild("##plugin_list", ImVec2(0, -footer_h), ImGuiChildFlags_None)) {
        const std::string filter(search_buf_);
        ImGui::TextColored(t.text_dim, "Plugins");
        size_t shown = 0;
        for (const size_t i : PluginCardOrder(facts)) {
            const auto* p = plugins[i];
            if (!p) continue;
            if (!filter.empty() && !ContainsCI(p->manifest.name, filter) && !ContainsCI(p->manifest.id, filter) &&
                !ContainsCI(p->manifest.author, filter) && !ContainsCI(p->manifest.description, filter))
                continue;
            ++shown;
            RenderPluginCard(*p, facts[i]);
        }
        if (plugins.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "No plugins loaded. Install... adds one from its folder; Refresh looks in "
                                           "the plugin folders again.");
            ImGui::PopTextWrapPos();
        } else if (shown == 0) {
            ImGui::TextColored(t.text_dim, "No plugins match '%s'", filter.c_str());
        }
        ImGui::Spacing();
        python_ui::RenderBundleSection(filter);
    }
    ImGui::EndChild();
    ImGui::TextColored(t.text_dim, "%s", footer.c_str());

    // Unloads after the list is drawn: the cards hold pointers into it
    for (const auto& id : deferred_unloads_) {
        mgr.UnloadPlugin(id);
        plugin::RecordPluginUnloaded(id);
        spdlog::info("PluginManagerPanel: Unloaded plugin {}", id);
    }
    if (!deferred_unloads_.empty()) {
        ReloadStartupList();
        contributions_.clear();
    }
    deferred_unloads_.clear();

    ImGui::End();

    RenderInstallDialog();
}

// ============================================================================
// Toolbar
// ============================================================================

void PluginManagerPanel::RenderToolbar() {
    const char* kRefresh = ICON_FA_ARROWS_ROTATE " Refresh";
    const char* kInstall = ICON_FA_FOLDER_PLUS " Install...";
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float buttons = ui::ButtonWidth(kRefresh, ui::ButtonSize::Small) +
                          ui::ButtonWidth(kInstall, ui::ButtonSize::Small) + gap * 2.0f;
    const float search_w = std::max(120.0f, ImGui::GetContentRegionAvail().x - buttons);
    ui::SearchField("##PluginSearch", search_buf_, sizeof(search_buf_), "Search plugins", search_w);
    ui::Tooltip("Searches the plugins and the Python node bundles: name, id, author, description");
    ImGui::SameLine();
    if (ui::SecondaryButton(kRefresh)) {
        auto& mgr = plugin::PluginManager::Instance();
        mgr.LoadAllFromSearchPaths();
        mgr.InitializeAll();
        plugin::RecordLoadedPlugins(mgr);
        ReloadStartupList();
        contributions_.clear();
        if (auto* bundles = python_ui::Provider()) bundles->Rescan();  // Python node bundles too
        spdlog::info("PluginManagerPanel: Refreshed plugin list");
    }
    ui::Tooltip("Looks in the plugin folders again and starts what is not running");
    ImGui::SameLine();
    if (ui::SecondaryButton(kInstall)) {
        open_install_ = true;
        install_path_buf_[0] = '\0';
        install_message_.clear();
        install_tone_ = InstallTone::None;
    }
}

// ============================================================================
// Plugin card
// ============================================================================

void PluginManagerPanel::RenderPluginCard(const plugin::LoadedPlugin& p, const PluginCardFacts& facts) {
    const ui::Tokens& t = ui::CurrentTokens();
    auto& mgr = plugin::PluginManager::Instance();
    const std::string plugin_id = p.manifest.id;  // copied: an action below may change the plugin
    const PluginCardView view = DescribePluginCard(facts);
    const PluginContribution& contribution = ContributionOf(plugin_id);

    ImGui::PushID(plugin_id.c_str());
    if (scroll_to_ == plugin_id) {
        ImGui::SetScrollHereY(0.0f);
        scroll_to_.clear();
    }
    ui::BeginCard(plugin_id.c_str());

    // Header: state dot, name, version and author; the state word at the right
    ImGui::TextColored(ToneColour(view.tone), ICON_FA_CIRCLE);
    ImGui::SameLine();
    ImGui::TextUnformatted(p.manifest.name.c_str());
    ImGui::SameLine();
    std::string by = "v" + p.manifest.version.ToString();
    if (!p.manifest.author.empty()) by += " \xC2\xB7 " + p.manifest.author;
    ImGui::TextColored(t.text_dim, "%s", by.c_str());
    ui::SameLineRight(ImGui::CalcTextSize(view.state_word.c_str()).x);
    ImGui::TextColored(ToneColour(view.tone), "%s", view.state_word.c_str());

    ImGui::PushTextWrapPos(0.0f);
    if (!p.manifest.description.empty()) ImGui::TextColored(t.text_dim, "%s", p.manifest.description.c_str());
    if (!view.reason.empty())
        ImGui::TextColored(view.tone == PluginTone::Problem ? t.caution : t.text, "%s", view.reason.c_str());
    const std::string adds = PluginAddsLine(contribution);
    if (!adds.empty()) ImGui::TextColored(t.accent_text, "%s", adds.c_str());
    ImGui::PopTextWrapPos();

    bool at_startup = PluginLoadsAtStartup(startup_list_, plugin_id);
    if (ImGui::Checkbox("Load when the Engine starts", &at_startup)) {
        plugin::SetLoadAtStartup(plugin_id, p.plugin_dir.generic_string(), at_startup);
        ReloadStartupList();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", PluginStartupNote(at_startup, contribution).c_str());
    ImGui::PopTextWrapPos();

    // The main action for its state, the Permissions link, the Actions menu
    bool drew = false;
    if (view.action != PluginAction::None && ui::SecondaryButton(view.action_label.c_str())) {
        switch (view.action) {
            case PluginAction::Disable: mgr.DisablePlugin(plugin_id); break;
            case PluginAction::Start: mgr.InitializePlugin(plugin_id); break;
            case PluginAction::TryAgain:
                mgr.SetPluginState(plugin_id, plugin::PluginState::Loaded);
                mgr.InitializePlugin(plugin_id);
                break;
            case PluginAction::Enable:
                if (mgr.EnablePlugin(plugin_id)) mgr.InitializePlugin(plugin_id);
                break;
            case PluginAction::None: break;
        }
        contributions_.clear();
    }
    drew = view.action != PluginAction::None;
    const std::vector<PluginPermissionFacts> perms = PermissionFactsOf(p);
    const bool perms_open = open_permissions_.count(plugin_id) > 0;
    if (!perms.empty()) {
        if (drew) ImGui::SameLine();
        const std::string link = perms_open ? std::string("Hide permissions") : PluginPermissionsLink(perms);
        if (ui::LinkButton(link.c_str())) {
            if (perms_open) open_permissions_.erase(plugin_id);
            else open_permissions_.insert(plugin_id);
        }
        drew = true;
    }
    const char* kActions = "Actions";
    if (drew) ui::SameLineRight(ui::ButtonWidth(kActions, ui::ButtonSize::Small));
    if (ui::SecondaryButton(kActions)) ImGui::OpenPopup("##plugin_actions");
    if (ImGui::BeginPopup("##plugin_actions")) {
        if (ImGui::MenuItem("Open the plugin folder")) LaunchAssetInFileBrowser(p.plugin_dir.string(), true);
        if (ImGui::MenuItem("Copy its id")) ImGui::SetClipboardText(plugin_id.c_str());
        ImGui::PushStyleColor(ImGuiCol_Text, t.error);
        if (ImGui::MenuItem("Unload...")) unload_asked_ = plugin_id;
        ImGui::PopStyleColor();
        ImGui::EndPopup();
    }

    if (unload_asked_ == plugin_id) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(PluginUnloadQuestion(p.manifest.name, contribution).c_str());
        ImGui::PopTextWrapPos();
        if (ui::SecondaryButton("Keep it")) unload_asked_.clear();
        ImGui::SameLine();
        if (ui::DangerButton("Unload")) {
            deferred_unloads_.push_back(plugin_id);
            unload_asked_.clear();
            open_permissions_.erase(plugin_id);
        }
    }

    if (perms_open && !perms.empty()) RenderPermissions(p);

    ui::EndCard();
    ImGui::PopID();
}

void PluginManagerPanel::RenderPermissions(const plugin::LoadedPlugin& p) {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<plugin::PluginPermission> perms;
    const std::vector<PluginPermissionFacts> facts = PermissionFactsOf(p, &perms);
    const std::vector<PluginPermissionRow> rows = PluginPermissionRows(facts);
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "What it asked for when it started, and your answer (kept for this version).");
    ImGui::PopTextWrapPos();
    static const char* const kAnswers[] = {"Allowed", "Denied", "Ask again"};
    // The answer column: the switch when an answer can change, else the widest answer's text
    float answers_w = 0.0f;
    for (const PluginPermissionRow& row : rows)
        answers_w = std::max(answers_w, row.can_change ? ui::SegmentedControlWidth(kAnswers, 3)
                                                       : ImGui::CalcTextSize(row.answer.c_str()).x);
    if (ImGui::BeginTable("##permissions", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("permission", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("answer", ImGuiTableColumnFlags_WidthFixed, answers_w);
        for (size_t i = 0; i < rows.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(rows[i].name.c_str());
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", rows[i].description.c_str());
            ImGui::PopTextWrapPos();
            ImGui::TableNextColumn();
            if (!rows[i].can_change) {
                ImGui::TextColored(t.text_dim, "%s", rows[i].answer.c_str());
            } else {
                int chosen = facts[i].answer == PermissionAnswer::Allowed  ? 0
                             : facts[i].answer == PermissionAnswer::Denied ? 1
                                                                           : 2;
                if (ui::SegmentedControl("##answer", kAnswers, 3, &chosen)) {
                    using plugin::security::PermissionDecision;
                    auto& store = plugin::security::PermissionStore::Instance();
                    store.SetDecision(p.manifest.id, p.manifest.version.ToString(), perms[i],
                                      chosen == 0   ? PermissionDecision::Allowed
                                      : chosen == 1 ? PermissionDecision::Denied
                                                    : PermissionDecision::Undecided);
                    store.Save();
                    spdlog::info("PluginManagerPanel: {} permission '{}' set to {}", p.manifest.id, rows[i].name,
                                 kAnswers[chosen]);
                }
                if (facts[i].answer == PermissionAnswer::NotAsked)
                    ui::Tooltip("Not asked yet: the plugin asks when it next starts");
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (std::any_of(rows.begin(), rows.end(), [](const PluginPermissionRow& r) { return r.can_change; })) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "A change takes effect the next time the plugin starts (Disable, then Enable).");
        ImGui::PopTextWrapPos();
    }
}

void PluginManagerPanel::ReloadStartupList() {
    startup_list_ = plugin::ReadPluginStartupList();
    startup_list_read_ = true;
}

// ============================================================================
// Install dialog
// ============================================================================

void PluginManagerPanel::StartInstall() {
    if (is_installing_plugin_) return;
    const auto problem = [this](std::string message) {
        install_message_ = std::move(message);
        install_tone_ = InstallTone::Problem;
    };
    const std::string path_str = TrimCopy(install_path_buf_);
    if (path_str.empty()) return problem("Choose the plugin's folder first.");
    const std::filesystem::path dir(path_str);
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return problem("There is no folder " + path_str + ".");
    if (!std::filesystem::is_directory(dir, ec)) return problem(path_str + " is a file. Choose the folder that holds plugin.json.");
    if (!std::filesystem::exists(dir / "plugin.json", ec))
        return problem("No plugin.json in " + path_str + ". Choose the folder that holds plugin.json.");

    plugin::PluginManifest manifest;
    std::string manifest_error;
    if (!plugin::PluginLoader::ParseManifest(dir, manifest, manifest_error))
        return problem("Its plugin.json cannot be read: " + manifest_error);

    // A plugin id loads once. Installing it again would keep the code already
    // in memory and look like a fresh install, so say what to do instead.
    if (const auto* existing = plugin::PluginManager::Instance().GetLoadedPlugin(manifest.id)) {
        if (existing->state == plugin::PluginState::Disabled)
            return problem(manifest.name + " is installed but disabled. Close this and press Enable on its card.");
        if (existing->state == plugin::PluginState::Failed)
            return problem(manifest.name + " is installed but failed to start. Close this and press Try again on its card.");
        return problem(manifest.name + " " + existing->manifest.version.ToString() + " is already loaded from " +
                       existing->plugin_dir.string() +
                       ". To load a new build: Unload it, rebuild, then Install again. If a model still uses its "
                       "nodes, restart the Engine instead of Unload.");
    }

    is_installing_plugin_ = true;
    install_message_ = "Installing " + manifest.name + "... the Engine stays responsive. Cancel stops it before loading starts.";
    install_tone_ = InstallTone::None;
    const std::string task_name = "Installing plugin: " + dir.filename().string();
    install_task_id_ = AsyncTaskManager::Instance().RunAsync(
        task_name,
        [dir, path_str](LambdaTask& task) {
            task.ReportProgress(0.30f, "Loading plugin...");
            if (task.ShouldStop()) return;
            std::string load_error;
            if (!plugin::PluginManager::Instance().LoadPlugin(dir, &load_error)) {
                task.MarkFailed("Failed to load plugin from " + path_str + (load_error.empty() ? "" : ": " + load_error));
                return;
            }
            task.MarkCompleted("Plugin loaded");
        },
        nullptr,
        [this, path_str, plugin_id = manifest.id, name = manifest.name](bool success, const std::string& error) {
            is_installing_plugin_ = false;
            install_task_id_ = 0;
            if (!success) {
                install_message_ = error.empty() ? "The install failed; the Console has the details." : error;
                install_tone_ = InstallTone::Problem;
                return;
            }
            auto& mgr = plugin::PluginManager::Instance();
            plugin::RecordLoadedPlugins(mgr);
            ReloadStartupList();
            contributions_.clear();
            const bool initialized = mgr.InitializePlugin(plugin_id);
            const auto* loaded = mgr.GetLoadedPlugin(plugin_id);
            if (!initialized && mgr.GetPluginState(plugin_id) != plugin::PluginState::Loaded) {
                install_message_ = name + " is installed but could not start.";
                if (loaded && !loaded->error_message.empty()) install_message_ += " " + loaded->error_message;
                install_tone_ = InstallTone::Problem;
                return;
            }
            // Nodes the plugin offered but the Engine refused stay visible
            // here, not only in the Console.
            if (loaded && !loaded->skipped_nodes.empty()) {
                install_message_ = name + " is installed, but " + std::to_string(loaded->skipped_nodes.size()) +
                                   " of its nodes " + (loaded->skipped_nodes.size() == 1 ? "was" : "were") +
                                   " not added:";
                for (const auto& reason : loaded->skipped_nodes) install_message_ += "\n- " + reason;
                install_tone_ = InstallTone::Partial;
                return;
            }
            close_install_ = true;
            spdlog::info("PluginManagerPanel: Plugin ready after selecting {}", path_str);
        });
}

void PluginManagerPanel::RenderInstallDialog() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (open_install_) {
        ImGui::OpenPopup(kInstallTitle);
        open_install_ = false;
    }
    ui::DialogOptions options;
    options.size = ImVec2(600.0f, 300.0f);
    options.min_size = ImVec2(460.0f, 240.0f);
    if (!ui::BeginDialog(kInstallTitle, options)) return;
    if (close_install_) {
        close_install_ = false;
        ImGui::CloseCurrentPopup();
    }

    ImGui::TextUnformatted("The plugin's folder (the one that holds plugin.json)");
    ImGui::BeginDisabled(is_installing_plugin_);
    const char* kBrowse = ICON_FA_FOLDER_OPEN " Browse...";
    ImGui::SetNextItemWidth(-(ui::ButtonWidth(kBrowse, ui::ButtonSize::Small) + ImGui::GetStyle().ItemSpacing.x));
    const bool enter = ImGui::InputTextWithHint("##InstallPath", "D:\\path\\to\\plugin", install_path_buf_,
                                                sizeof(install_path_buf_), ImGuiInputTextFlags_EnterReturnsTrue);
    if (ImGui::IsItemEdited() && install_tone_ != InstallTone::None) {
        install_message_.clear();  // a new path: the old message no longer applies
        install_tone_ = InstallTone::None;
    }
    ImGui::SameLine();
    if (ui::SecondaryButton(kBrowse)) {
        const std::string current = TrimCopy(install_path_buf_);
        if (auto selected = FileDialogs::SelectFolder("Select Plugin Directory", current.empty() ? nullptr : current.c_str())) {
            std::snprintf(install_path_buf_, sizeof(install_path_buf_), "%s", selected->c_str());
            install_message_.clear();
            install_tone_ = InstallTone::None;
        }
    }
    ImGui::EndDisabled();
    if (enter) StartInstall();

    if (!install_message_.empty()) {
        ImGui::Spacing();
        const ImVec4 tone = install_tone_ == InstallTone::Problem   ? t.caution
                            : install_tone_ == InstallTone::Partial ? t.warning
                                                                    : t.running;
        ui::NoticeLine(install_message_, tone);
    }

    const bool installed = install_tone_ == InstallTone::Partial;
    const char* primary = is_installing_plugin_ ? ICON_FA_SPINNER " Installing..." : "Install";
    const char* reason = is_installing_plugin_ ? "Already installing" : installed ? "Installed" : "Choose a folder first";
    const bool can_install = !is_installing_plugin_ && !installed && TrimCopy(install_path_buf_) != "";
    const ui::DialogResult result =
        ui::EndDialog(primary, is_installing_plugin_ ? "Cancel" : "Close", can_install, reason, false, false);
    if (result == ui::DialogResult::Primary) StartInstall();
    if (result == ui::DialogResult::Secondary && is_installing_plugin_ && install_task_id_ != 0)
        AsyncTaskManager::Instance().Cancel(install_task_id_);
}

} // namespace cyxwiz
