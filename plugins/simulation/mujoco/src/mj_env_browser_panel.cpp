#include "mj_env_browser_panel.h"
#include "mj_menagerie_downloader.h"
#include "mj_ui_helpers.h"
#include "mj_viewport_settings.h"

#include "plugin/plugin_context.h"
#include "plugin/plugin_ui.h"

#include <imgui.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>

namespace cyxwiz::plugin::mujoco {
namespace {

constexpr const char* kViewportPanelId = "mujoco_viewport";
constexpr const char* kSources[] = {"All", "Built-in", "Menagerie", "In this project", "Imported"};
constexpr int kSourceCount = 5;
constexpr float kCardMinWidth = 280.0f;

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string WithThousands(int value) {
    std::string digits = std::to_string(value);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return digits;
}

// "2.1 MB"
std::string Megabytes(uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f MB", static_cast<double>(bytes) / 1.0e6);
    return text;
}

// Where the model is and which version: "Menagerie · shared download · version
// 2f1a9c3 (2026-11-13)", "Menagerie · copy in this project · version 2f1a9c3",
// "Menagerie · not downloaded", "In this project · mujoco/models/arm", ...
std::string SourceLine(const EnvInfo& info, const MjEnvLibrary& library) {
    const std::string dot = " \xC2\xB7 ";
    switch (info.source) {
        case EnvSource::Menagerie: {
            if (info.where == MjModelWhere::Missing) return "Menagerie" + dot + "not downloaded";
            const std::string place = info.where == MjModelWhere::ProjectCopy ? "copy in this project" : "shared download";
            if (info.commit.empty()) return "Menagerie" + dot + place + dot + "version unknown";
            return "Menagerie" + dot + place + dot + "version " + ShortCommit(info.commit) +
                   (info.date.empty() ? "" : " (" + info.date.substr(0, 10) + ")");
        }
        case EnvSource::Project: {
            const auto rel = RefForFile(info.custom_asset_path, library.Roots());
            return "In this project" + dot + (rel.kind == MjRefKind::Project ? rel.file : info.filename);
        }
        case EnvSource::Imported:
            return "Imported from URL, outside any project";
        default:
            return "Built-in" + dot + info.category;
    }
}

std::string Now() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char date[32];
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M", std::localtime(&now));
    return date;
}

// "1 action · 4 observations · 1,000 steps" (the episode length)
std::string Facts(const EnvInfo& info) {
    if (info.act_dim == 0 && info.obs_dim == 0) return "Actions and observations known after loading";
    std::string facts = std::to_string(info.act_dim) + (info.act_dim == 1 ? " action" : " actions") +
                        " \xC2\xB7 " + std::to_string(info.obs_dim) + " observations";
    if (info.source == EnvSource::Builtin) facts += " \xC2\xB7 " + WithThousands(info.max_steps) + " steps";
    return facts;
}

// `text` wrapped to `width` and cut to `lines` lines at the cursor; the whole
// text shows on hover when it was cut.
void ClippedText(const char* text, float width, int lines, ImU32 colour) {
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float height = ImGui::GetTextLineHeight() * static_cast<float>(lines);
    const ImVec4 clip(pos.x, pos.y, pos.x + width, pos.y + height);
    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), ImGui::GetFontSize(), pos, colour, text, nullptr, width,
                                        &clip);
    ImGui::Dummy(ImVec2(width, height));
    if (ImGui::IsItemHovered() && ImGui::CalcTextSize(text, nullptr, false, width).y > height + 1.0f)
        ImGui::SetTooltip("%s", text);
}

}  // namespace

MjEnvBrowserPanel::~MjEnvBrowserPanel() {
    if (copy_thread_.joinable()) copy_thread_.join();
}

void MjEnvBrowserPanel::Render(MjEnvLibrary& library, const MjEnvManager& env, bool* visible) {
    if (!visible || !*visible) return;

    // Another project open: its own models and copies
    if (ctx_) {
        const std::string project = ctx_->GetProjectDir().string();
        if (project != last_project_) {
            last_project_ = project;
            library.SetProjectDir(project);
            library.Rescan();
        }
    }
    ApplyFinishedDownloads(library);

    // Process deferred load BEFORE any ImGui rendering.
    // LoadEnvironment switches GL contexts which corrupts ImGui's GL state mid-frame.
    if (!pending_load_path_.empty()) {
        std::string path = std::move(pending_load_path_);
        pending_load_path_.clear();
        if (load_callback_ && load_callback_(path, pending_load_info_)) {
            loaded_env_id_ = pending_load_info_.id;
        }
    }

    ImGui::SetNextWindowSize(ImVec2(1000, 700), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Environment Library", visible)) {
        ImGui::End();
        return;
    }
    // The Engine's widgets: always there in an Engine with plugin API 1.4
    if (!ctx_ || !ctx_->GetUi()) {
        ImGui::End();
        return;
    }

    RenderTopRow();
    if (import_open_) RenderUrlImport();

    std::vector<const EnvInfo*> by_source;
    for (const auto& info : library.GetAll())
        if (MatchesSource(info)) by_source.push_back(&info);
    // A category the chosen source does not have falls back to All
    if (!selected_category_.empty() &&
        std::none_of(by_source.begin(), by_source.end(),
                     [&](const EnvInfo* e) { return e->category == selected_category_; }))
        selected_category_.clear();
    std::vector<const EnvInfo*> shown;
    for (const EnvInfo* info : by_source)
        if ((selected_category_.empty() || info->category == selected_category_) && MatchesSearch(*info))
            shown.push_back(info);

    // The footer: the counts and the shared folder, then a line about versions or the last copy
    const float footer_h = ImGui::GetTextLineHeightWithSpacing() * 2.0f + ImGui::GetStyle().ItemSpacing.y;
    const float body_h = std::max(0.0f, ImGui::GetContentRegionAvail().y - footer_h);
    ImGui::BeginChild("##categories", ImVec2(200.0f, body_h));
    RenderCategories(by_source);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##cards", ImVec2(0.0f, body_h));
    RenderCards(library, env, shown);
    ImGui::EndChild();
    RenderFooter(library);

    ImGui::End();
}

void MjEnvBrowserPanel::ApplyFinishedDownloads(MjEnvLibrary& library) {
    std::vector<FinishedDownload> done;
    {
        std::lock_guard lock(finished_mutex_);
        done.swap(finished_);
    }
    for (const auto& d : done) {
        if (d.copy) {
            if (copy_thread_.joinable()) copy_thread_.join();
            copying_env_id_.clear();
            copy_failed_ = !d.ok;
            copy_status_ = d.error;  // the outcome in words, either way
            library.Rescan();
            if (d.ok && load_after_copy_) {
                for (const auto& info : library.GetAll()) {
                    if (info.id != d.id) continue;
                    pending_load_path_ = library.GetAssetPath(info);
                    pending_load_info_ = info;
                }
            }
            load_after_copy_ = false;
        } else if (d.url_import) {
            url_importing_ = false;
            if (!d.ok) {
                url_import_error_ = d.error.empty() ? "The download failed." : d.error;
            } else {
                // Where it came from, kept with it
                MjModelManifest manifest;
                manifest.source = "url";
                manifest.url = d.url;
                manifest.date = Now();
                std::string manifest_error;
                WriteModelManifest(d.dir, manifest, manifest_error);
                const bool in_project = !library.Roots().project.empty();
                library.Rescan();
                const bool listed = std::any_of(library.GetAll().begin(), library.GetAll().end(), [&](const EnvInfo& e) {
                    return e.custom_asset_path.rfind(d.dir, 0) == 0;
                });
                if (listed)
                    url_import_status_ = "Imported " + d.id + (in_project ? " into this project. It is listed under In this project."
                                                                         : ". It is listed under Imported.");
                else
                    url_import_error_ = "No .xml model file was found at that address.";
            }
        } else {
            if (downloading_env_id_ == d.id) downloading_env_id_.clear();
            if (d.ok) {
                library.Rescan();
                download_errors_.erase(d.id);
            } else {
                download_errors_[d.id] = d.error.empty() ? "The download failed." : d.error;
            }
        }
    }
}

bool MjEnvBrowserPanel::MatchesSource(const EnvInfo& info) const {
    switch (source_) {
        case 1: return info.source == EnvSource::Builtin;
        case 2: return info.source == EnvSource::Menagerie;
        case 3:
            return info.source == EnvSource::Project ||
                   (info.source == EnvSource::Menagerie && info.where == MjModelWhere::ProjectCopy);
        case 4: return info.source == EnvSource::Imported;
        default: return true;
    }
}

bool MjEnvBrowserPanel::MatchesSearch(const EnvInfo& info) const {
    if (search_buf_[0] == '\0') return true;
    const std::string query = Lower(search_buf_);
    return Lower(info.name).find(query) != std::string::npos ||
           Lower(info.description).find(query) != std::string::npos;
}

// =============================================================================
// Search, source and Import from URL
// =============================================================================

void MjEnvBrowserPanel::RenderTopRow() {
    PluginUi* ui = ctx_->GetUi();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const char* import_label = "Import from URL...";
    const float source_w = ui->SegmentedControlWidth(kSources, kSourceCount);
    const float import_w = ui->ButtonWidth(import_label, UiButtonSize::Regular);
    const float search_w = std::max(160.0f, ImGui::GetContentRegionAvail().x - source_w - import_w - 2.0f * spacing);

    // One row, its items centred on the tallest (the button)
    const float row_y = ImGui::GetCursorPosY();
    const float field_y = row_y + std::max(0.0f, (ui->ButtonHeight(UiButtonSize::Regular) - ImGui::GetFrameHeight()) * 0.5f);
    ImGui::SetCursorPosY(field_y);
    ui->SearchField("##env_search", search_buf_, sizeof(search_buf_), "Search by name or description", search_w);
    SameLineIfFits(source_w);
    ImGui::SetCursorPosY(field_y);
    int source = source_;
    if (ui->SegmentedControl("##env_source", kSources, kSourceCount, &source)) {
        source_ = source;
        selected_category_.clear();
    }
    SameLineIfFits(import_w);
    ImGui::SetCursorPosY(row_y);
    if (ui->SecondaryButton(import_label)) import_open_ = !import_open_;
    ImGui::Spacing();
}

void MjEnvBrowserPanel::RenderUrlImport() {
    PluginUi* ui = ctx_->GetUi();
    const UiColors colors = ui->Colors();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const bool busy = !downloader_ || downloader_->IsDownloading();

    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ToImVec4(colors.text_dim), "Model URL");
    ImGui::SameLine();
    const float buttons_w =
        ui->ButtonWidth("Import", UiButtonSize::Regular) + ui->ButtonWidth("Cancel", UiButtonSize::Regular) + 2.0f * spacing;
    ImGui::SetNextItemWidth(std::max(200.0f, ImGui::GetContentRegionAvail().x - buttons_w - spacing));
    ImGui::InputTextWithHint("##url", "A GitHub folder (github.com/<owner>/<repo>/tree/<branch>/<folder>) or an .xml file",
                             url_buf_, sizeof(url_buf_));
    SameLineIfFits(buttons_w);
    const bool has_url = url_buf_[0] != '\0';
    if (ui->PrimaryButton("Import", has_url && !busy, busy ? "Another download is running" : "Paste a model's address first")) {
        const std::string url = url_buf_;
        // The model's name: the URL's last part, without .xml
        std::string name = "imported_model";
        const auto last_slash = url.find_last_of('/');
        if (last_slash != std::string::npos && last_slash + 1 < url.size()) {
            std::string tail = url.substr(last_slash + 1);
            const auto dot = tail.rfind('.');
            name = dot != std::string::npos ? tail.substr(0, dot) : tail;
        }
        const std::string project = ctx_->GetProjectDir().string();
        const std::string dest =
            (project.empty() ? std::filesystem::path(MjEnvLibrary::GetImportedRoot()) / name
                             : ProjectModelsFolder(project) / name)
                .string();
        url_importing_ = true;
        url_import_status_.clear();
        url_import_error_.clear();
        downloader_->DownloadFromUrlAsync(url, dest, [this, name, dest, url](bool ok, const std::string& error) {
            std::lock_guard lock(finished_mutex_);
            finished_.push_back({true, false, name, dest, url, ok, error});
        });
    }
    ImGui::SameLine();
    if (ui->SecondaryButton("Cancel")) {
        import_open_ = false;
        url_import_status_.clear();
        url_import_error_.clear();
    }

    ImGui::PushTextWrapPos(0.0f);
    const std::string project = ctx_->GetProjectDir().string();
    if (project.empty())
        ImGui::TextColored(ToImVec4(colors.text_dim),
                           "An MJCF .xml file or a GitHub folder. No project is open: it is saved in %s and listed "
                           "under Imported.",
                           (std::filesystem::path(MjEnvLibrary::GetImportedRoot()) / "<name>").string().c_str());
    else
        ImGui::TextColored(ToImVec4(colors.text_dim),
                           "An MJCF .xml file or a GitHub folder. It is saved in this project (%s), travels with it, "
                           "and is listed under In this project.",
                           (ProjectModelsFolder(project) / "<name>").string().c_str());
    if (url_importing_ && downloader_) {
        ImGui::ProgressBar(downloader_->GetProgress(), ImVec2(420.0f, 6.0f), "");
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s", downloader_->GetStatusText().c_str());
    }
    if (!url_import_status_.empty()) ImGui::TextColored(ToImVec4(colors.success), "%s", url_import_status_.c_str());
    if (!url_import_error_.empty()) ImGui::TextColored(ToImVec4(colors.caution), "%s", url_import_error_.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
}

// =============================================================================
// Categories
// =============================================================================

void MjEnvBrowserPanel::RenderCategories(const std::vector<const EnvInfo*>& by_source) {
    const UiColors colors = ctx_->GetUi()->Colors();
    // The chosen category on the raised surface, as the mockup draws it
    ImGui::PushStyleColor(ImGuiCol_Header, ToImVec4(colors.bg_raised));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ToImVec4(colors.bg_raised));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ToImVec4(colors.bg_raised));
    auto row = [&](const std::string& category, size_t count) {
        const bool selected = selected_category_ == category;
        ImGui::PushID(category.c_str());
        if (ImGui::Selectable(category.empty() ? "All" : category.c_str(), selected)) selected_category_ = category;
        char text[16];
        std::snprintf(text, sizeof(text), "%zu", count);
        const float width = ImGui::CalcTextSize(text).x;
        ImGui::GetWindowDrawList()->AddText(ImVec2(ImGui::GetItemRectMax().x - width - 6.0f, ImGui::GetItemRectMin().y),
                                            ToU32(colors.text_dim), text);
        ImGui::PopID();
    };
    row("", by_source.size());
    // Categories in library order, with their counts under the chosen source
    std::vector<std::pair<std::string, size_t>> categories;
    for (const EnvInfo* info : by_source) {
        auto it = std::find_if(categories.begin(), categories.end(),
                               [&](const auto& c) { return c.first == info->category; });
        if (it == categories.end()) categories.emplace_back(info->category, 1);
        else ++it->second;
    }
    for (const auto& [category, count] : categories) row(category, count);
    ImGui::PopStyleColor(3);
}

// =============================================================================
// Cards
// =============================================================================

void MjEnvBrowserPanel::RenderCards(MjEnvLibrary& library, const MjEnvManager& env,
                                    const std::vector<const EnvInfo*>& shown) {
    PluginUi* ui = ctx_->GetUi();
    const UiColors colors = ui->Colors();

    if (shown.empty()) {
        if (search_buf_[0] != '\0') {
            const std::string title = std::string("No environments match \"") + search_buf_ + "\"";
            ui->EmptyState(title.c_str(), "Search looks in names and descriptions.");
            const float width = ui->ButtonWidth("Clear search", UiButtonSize::Regular);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - width) * 0.5f));
            if (ui->LinkButton("Clear search")) search_buf_[0] = '\0';
        } else if (source_ == 3) {
            ui->EmptyState("No models in this project yet",
                           "Copy a downloaded robot into the project from its card, or Import from URL... with the "
                           "project open.");
        } else if (source_ == 4) {
            ui->EmptyState("Nothing imported outside a project", "With no project open, Import from URL... saves here.");
        } else {
            ui->EmptyState("No environments here");
        }
        return;
    }

    ImGui::TextColored(ToImVec4(colors.text_bright), "%s",
                       selected_category_.empty() ? "All" : selected_category_.c_str());
    ImGui::SameLine();
    ImGui::TextColored(ToImVec4(colors.text_dim), "%zu %s", shown.size(),
                       shown.size() == 1 ? "environment" : "environments");

    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const int columns = std::max(1, static_cast<int>((ImGui::GetContentRegionAvail().x + spacing) / (kCardMinWidth + spacing)));
    const float card_h = ImGui::GetTextLineHeightWithSpacing() * 5.0f + ui->ButtonHeight(UiButtonSize::Regular) +
                         ImGui::GetStyle().WindowPadding.y * 2.0f + 6.0f;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(6.0f, 6.0f));
    if (ImGui::BeginTable("##env_grid", columns, ImGuiTableFlags_SizingStretchSame)) {
        for (const EnvInfo* info : shown) {
            ImGui::TableNextColumn();
            RenderCard(library, *info, env.IsLoaded() && loaded_env_id_ == info->id, card_h);
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
    RenderCopyConfirm(library);
}

void MjEnvBrowserPanel::RenderCard(MjEnvLibrary& library, const EnvInfo& info, bool is_loaded, float height) {
    PluginUi* ui = ctx_->GetUi();
    const UiColors colors = ui->Colors();
    const bool downloading = downloader_ && downloader_->IsDownloading() && downloading_env_id_ == info.id;
    const auto error = download_errors_.find(info.id);
    const bool failed = error != download_errors_.end() && !downloading;

    ImGui::PushID(info.id.c_str());
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ToImVec4(is_loaded ? colors.bg_raised : colors.bg_panel));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    ImGui::BeginChild("##card", ImVec2(0.0f, height), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    const float width = ImGui::GetContentRegionAvail().x;
    ImGui::TextColored(ToImVec4(colors.text_bright), "%s", info.name.c_str());
    const bool needed = !info.needed_commit.empty();
    if (needed) {
        const std::string line = "Not on this PC. This project uses version " + ShortCommit(info.needed_commit) + ".";
        ClippedText(line.c_str(), width, 1, ToU32(colors.caution));
    } else {
        ClippedText(SourceLine(info, library).c_str(), width, 1, ToU32(colors.text_dim));
    }
    ClippedText(info.description.c_str(), width, 2, ToU32(colors.text));
    if (failed) ClippedText(error->second.c_str(), width, 1, ToU32(colors.caution));
    else ClippedText(Facts(info).c_str(), width, 1, ToU32(colors.text_dim));
    if (!failed && info.source == EnvSource::Builtin && ImGui::IsItemHovered())
        ImGui::SetTooltip("Actions the model takes, observations it gives, and the steps an episode lasts");

    // The action row sits at the card's bottom; text beside a button is centred on it
    const float line_h = ImGui::GetTextLineHeight();
    const float button_h = ui->ButtonHeight(UiButtonSize::Regular);
    const float row_h = downloading ? 6.0f + ImGui::GetStyle().ItemSpacing.y + line_h : button_h;
    const float row_y = std::max(ImGui::GetCursorPosY(), ImGui::GetWindowHeight() - ImGui::GetStyle().WindowPadding.y - row_h);
    ImGui::SetCursorPosY(row_y);
    const float right = ImGui::GetCursorPosX() + width;
    auto text_beside = [&](const UiColor& colour, const char* text) {
        ImGui::SetCursorPosY(row_y + (button_h - line_h) * 0.5f);
        ImGui::TextColored(ToImVec4(colour), "%s", text);
        ImGui::SameLine();
    };
    auto right_button_x = [&](float w) {
        ImGui::SetCursorPos(ImVec2(std::max(ImGui::GetCursorPosX(), right - w), row_y));
    };

    const bool busy = !downloader_ || downloader_->IsDownloading();
    const bool copying = copying_env_id_ == info.id;
    const bool project_open = !library.Roots().project.empty();
    if (is_loaded) {
        text_beside(colors.success, "Loaded");
        if (info.source == EnvSource::Menagerie && info.where == MjModelWhere::SharedDownload && project_open &&
            copying_env_id_.empty() && ui->LinkButton("Copy into the project"))
            copy_confirm_id_ = info.id;
        right_button_x(ui->ButtonWidth("Open viewport", UiButtonSize::Regular));
        if (ui->LinkButton("Open viewport")) ctx_->ShowPanel(kViewportPanelId);
    } else if (downloading) {
        ImGui::ProgressBar(downloader_->GetProgress(), ImVec2(-1.0f, 6.0f), "");
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s", downloader_->GetStatusText().c_str());
    } else if (copying) {
        text_beside(colors.text_dim, "Copying into the project...");
    } else if (needed) {
        // The version the project names, downloaded exactly
        text_beside(colors.text_dim, "Downloads that version from GitHub");
        right_button_x(ui->ButtonWidth("Download it", UiButtonSize::Regular));
        if (ui->PrimaryButton("Download it", !busy, "Another download is running"))
            StartDownload(library, info, info.needed_commit);
    } else if (info.source == EnvSource::Menagerie && info.requires_download) {
        const char* label = failed ? "Try again" : "Download";
        if (!failed) text_beside(colors.text_dim, "Downloads the latest version from GitHub");
        right_button_x(ui->ButtonWidth(label, UiButtonSize::Regular));
        if (ui->SecondaryButton(label, !busy, "Another download is running")) StartDownload(library, info, {});
    } else {
        // A shared download: copy it into the project, or pin one downloaded before versions were kept
        if (info.source == EnvSource::Menagerie && info.where == MjModelWhere::SharedDownload) {
            ImGui::SetCursorPosY(row_y + (button_h - line_h) * 0.5f);
            if (info.commit.empty()) {
                if (ui->LinkButton("Download again (pinned)") && !busy) StartDownload(library, info, {});
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Downloaded before versions were kept: download the latest version again so "
                                      "the project can name it and other PCs get the same files.");
            } else if (project_open && copying_env_id_.empty() && ui->LinkButton("Copy into the project")) {
                copy_confirm_id_ = info.id;
            }
            ImGui::SameLine();
        }
        right_button_x(ui->ButtonWidth("Load", UiButtonSize::Regular));
        const std::string blocked = load_blocker_ ? load_blocker_() : std::string();
        if (ui->SecondaryButton("Load", blocked.empty(), blocked.c_str())) {
            // Preferences: a shared download is copied into the project first
            const bool copy_first = info.source == EnvSource::Menagerie &&
                                    info.where == MjModelWhere::SharedDownload && !info.commit.empty() &&
                                    project_open && LoadViewportDefaults(*ctx_).copy_models_into_project;
            if (copy_first && copying_env_id_.empty()) {
                StartCopy(library, info, true);
            } else {
                pending_load_path_ = library.GetAssetPath(info);
                pending_load_info_ = info;
            }
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImGui::PopID();
}

// =============================================================================
// Footer
// =============================================================================

void MjEnvBrowserPanel::RenderFooter(const MjEnvLibrary& library) {
    const UiColors colors = ctx_->GetUi()->Colors();
    int builtin = 0, menagerie = 0, project = 0, imported = 0;
    for (const auto& info : library.GetAll()) {
        if (info.source == EnvSource::Menagerie) ++menagerie;
        else if (info.source == EnvSource::Project) ++project;
        else if (info.source == EnvSource::Imported) ++imported;
        else ++builtin;
    }
    std::string summary = std::to_string(builtin + menagerie + project + imported) + " environments: " +
                          std::to_string(builtin) + " built in, " + std::to_string(menagerie) + " from MuJoCo Menagerie";
    if (project > 0) summary += ", " + std::to_string(project) + " in this project";
    if (imported > 0) summary += ", " + std::to_string(imported) + " imported";
    ImGui::TextColored(ToImVec4(colors.text_dim), "%s", summary.c_str());
    const auto item = [&](const std::string& text, const UiColor& colour) {
        SameLineIfFits(ImGui::CalcTextSize(text.c_str()).x + 16.0f);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 16.0f);
        ImGui::TextColored(ToImVec4(colour), "%s", text.c_str());
    };
    item("Downloads are shared by every project: " + MjEnvLibrary::GetMenagerieRoot(), colors.text_dim);
    if (!copy_status_.empty())
        ImGui::TextColored(ToImVec4(copy_failed_ ? colors.caution : colors.success), "%s", copy_status_.c_str());
    else
        ImGui::TextColored(ToImVec4(colors.text),
                           "A project records each model by name and version, so another PC downloads the same one.");
}

void MjEnvBrowserPanel::StartDownload(MjEnvLibrary& library, const EnvInfo& info, const std::string& commit) {
    if (!downloader_ || downloader_->IsDownloading()) return;
    downloading_env_id_ = info.id;
    download_errors_.erase(info.id);
    const std::string id = info.id;
    const std::string root = library.Roots().menagerie.string();
    downloader_->DownloadModelAsync(info.repo_path, root, commit, [this, id, root](bool ok, const std::string& err) {
        std::lock_guard lock(finished_mutex_);
        finished_.push_back({false, false, id, root, std::string(), ok, err});
    });
}

void MjEnvBrowserPanel::StartCopy(MjEnvLibrary& library, const EnvInfo& info, bool load_after) {
    namespace fs = std::filesystem;
    const fs::path file = library.GetAssetPath(info);
    const fs::path from = MenagerieFolder(library.Roots().menagerie, info.repo_path, info.commit);
    const fs::path to = ProjectModelsFolder(library.Roots().project) / info.repo_path;
    if (file.empty() || copy_thread_.joinable()) return;
    copying_env_id_ = info.id;
    load_after_copy_ = load_after;
    copy_status_.clear();
    const std::string id = info.id, name = info.name, robot = info.repo_path, commit = info.commit;
    copy_thread_ = std::thread([this, from, to, id, name, robot, commit]() {
        std::error_code ec;
        fs::create_directories(to, ec);
        if (!ec) fs::copy(from, to, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
        MjModelManifest manifest;
        manifest.robot = robot;
        manifest.commit = commit;
        manifest.date = Now();
        std::error_code walk;
        for (fs::recursive_directory_iterator it(to, walk), end; !walk && it != end; it.increment(walk)) {
            if (!it->is_regular_file(walk) || it->path().filename() == kModelManifestName) continue;
            manifest.bytes += it->file_size(walk);
            ++manifest.files;
        }
        std::string error;
        const bool ok = !ec && WriteModelManifest(to, manifest, error);
        const std::string outcome = ok ? "Copied " + name + " into the project (" + Megabytes(manifest.bytes) +
                                             "); its nodes use the copy from now on."
                                       : "The copy of " + name + " failed: " + (ec ? ec.message() : error);
        std::lock_guard lock(finished_mutex_);
        finished_.push_back({false, true, id, to.string(), std::string(), ok, outcome});
    });
}

void MjEnvBrowserPanel::RenderCopyConfirm(MjEnvLibrary& library) {
    if (copy_confirm_id_.empty()) return;
    const EnvInfo* info = nullptr;
    for (const auto& e : library.GetAll())
        if (e.id == copy_confirm_id_) info = &e;
    if (!info) {
        copy_confirm_id_.clear();
        return;
    }
    PluginUi* ui = ctx_->GetUi();
    const UiColors colors = ui->Colors();
    const std::string title = "Copy " + info->name + " into the project?###copy_into_project";
    if (!ImGui::IsPopupOpen(title.c_str())) ImGui::OpenPopup(title.c_str());
    ImGui::SetNextWindowSize(ImVec2(520.0f, 0.0f), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(title.c_str(), nullptr, ImGuiWindowFlags_NoResize)) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::Text("Copies the model with its meshes and textures (%s) to", Megabytes(info->bytes).c_str());
    ImGui::TextColored(ToImVec4(colors.text_bright), "%s",
                       (ProjectModelsFolder(library.Roots().project) / info->repo_path).string().c_str());
    ImGui::TextColored(ToImVec4(colors.text_dim),
                       "This project's MuJoCo Plant nodes keep naming %s version %s and use the copy from now on; "
                       "another PC without the copy downloads the same version. The shared download stays for "
                       "other projects. Edit the copy freely: no other project sees it.",
                       info->name.c_str(), ShortCommit(info->commit).c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float buttons = ui->ButtonWidth("Cancel", UiButtonSize::Regular) + ui->ButtonWidth("Copy", UiButtonSize::Regular) + spacing;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - buttons));
    if (ui->SecondaryButton("Cancel")) {
        copy_confirm_id_.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ui->PrimaryButton("Copy", copying_env_id_.empty(), "Another copy is running")) {
        StartCopy(library, *info, false);
        copy_confirm_id_.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

} // namespace cyxwiz::plugin::mujoco
