#include "asset_browser.h"
#include "../icons.h"
#include "../ui_tokens.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_widgets.h"
#include "../editor_fonts.h"
#include "../../core/project_manager.h"
#include "../../core/data_input_formats.h"
#include "../file_browser_launch.h"
#include <imgui.h>
#include <filesystem>
#include <future>
#include <unordered_set>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <spdlog/spdlog.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#elif defined(__APPLE__) || defined(__linux__)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace cyxwiz {

namespace {

// Placement and sizes of the file preview pane.
constexpr float kPreviewBesideMinWidth = 700.0f;  // narrower panels put it below the tree
constexpr float kPreviewMinExtent = 220.0f;
constexpr float kTreeMinExtent = 180.0f;
constexpr float kPreviewHandle = 6.0f;
constexpr int kPreviewRows = 12;

const char* KindIcon(assets::FileKind kind, bool open) {
    using K = assets::FileKind;
    switch (kind) {
        case K::Folder:      return open ? ICON_FA_FOLDER_OPEN : ICON_FA_FOLDER;
        case K::Table:       return ICON_FA_FILE_CSV;
        case K::Spreadsheet: return ICON_FA_FILE_EXCEL;
        case K::Columnar:    return ICON_FA_DATABASE;
        case K::Config:      return ICON_FA_CODE;
        case K::Python:      return ICON_FA_FILE_CODE;
        case K::Notebook:    return ICON_FA_BOOK;
        case K::Graph:       return ICON_FA_DIAGRAM_PROJECT;
        case K::Model:       return ICON_FA_BRAIN;
        case K::Image:       return ICON_FA_IMAGE;
        case K::Audio:       return ICON_FA_MUSIC;
        case K::Video:       return ICON_FA_VIDEO;
        case K::Text:        return ICON_FA_FILE_LINES;
        case K::Plugin:      return ICON_FA_PLUG;
        default:             return ICON_FA_FILE;
    }
}

// One theme series colour per family (series: violet, sky, amber, green,
// coral, pink), so both themes keep readable icons.
ImVec4 KindColour(assets::FileKind kind) {
    using K = assets::FileKind;
    const ui::Tokens& t = ui::CurrentTokens();
    switch (kind) {
        case K::Folder:      return t.series[2];
        case K::Table:
        case K::Spreadsheet:
        case K::Columnar:    return t.series[3];
        case K::Python:
        case K::Notebook:    return t.series[1];
        case K::Graph:
        case K::Model:       return t.series[0];
        case K::Config:      return t.series[5];
        case K::Image:
        case K::Audio:
        case K::Video:       return t.series[4];
        default:             return t.text_dim;
    }
}

bool IsNumericType(const std::string& type) {
    for (const char* name : {"int", "float", "double", "decimal", "num"})
        if (type.find(name) != std::string::npos) return true;
    return false;
}

} // namespace

AssetBrowserPanel::AssetBrowserPanel()
    : Panel("Asset Browser", true)
    , last_clicked_item_(nullptr)
    , context_menu_item_(nullptr)
    , show_rename_dialog_(false)
    , show_delete_confirm_(false)
    , show_new_folder_dialog_(false)
{
    std::memset(search_buffer_, 0, sizeof(search_buffer_));
    std::memset(rename_buffer_, 0, sizeof(rename_buffer_));
    std::memset(new_folder_name_, 0, sizeof(new_folder_name_));

    // Note: ProjectManager callbacks are managed by MainWindow to avoid callback conflicts
    // MainWindow::OnProjectOpened calls SetProjectRoot and MainWindow::OnProjectClosed calls Clear

    // Initialize with current project if exists
    auto& pm = ProjectManager::Instance();
    if (pm.HasActiveProject()) {
        SetProjectRoot(pm.GetProjectRoot());
    }
}

void AssetBrowserPanel::Render() {
    if (!visible_) return;

    // Handle deferred refresh (after drag-drop operations complete)
    if (needs_refresh_) {
        needs_refresh_ = false;
        Refresh();
    }

    // Check if async scan completed and swap in the new tree
    if (scan_completed_.load()) {
        std::unique_ptr<AssetItem> rescanned;
        {
            std::lock_guard<std::mutex> lock(pending_tree_mutex_);
            rescanned = std::move(pending_directory_root_);
        }
        if (rescanned) {
            ApplyRescannedTree(std::move(rescanned));
        }
        scan_completed_.store(false);
        is_scanning_directory_.store(false);
    }

    PollForExternalChanges();

    // Collapsed or behind another dock tab: skip the body (TOFIX129 0.6).
    if (!ImGui::Begin(GetName(), &visible_)) {
        ImGui::End();
        return;
    }

    // Toolbar at top
    RenderToolbar();

    ImGui::Separator();

    // Keyboard shortcuts: when this window has focus, not a dialog it opened
    // (ImGui counts its popups as its children) and not while a text field
    // takes the keys (Ctrl+A or Delete there edit the text, not the files).
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows | ImGuiFocusedFlags_NoPopupHierarchy) &&
        !ImGui::GetIO().WantTextInput) {
        // Get first selected item for single-item operations
        AssetItem* first_selected = selected_items_.empty() ? nullptr : *selected_items_.begin();

        // F2 - Rename (single selection only)
        if (ImGui::IsKeyPressed(ImGuiKey_F2) && selected_items_.size() == 1) {
            context_menu_item_ = first_selected;
            show_rename_dialog_ = true;
            std::strncpy(rename_buffer_, first_selected->name.c_str(), sizeof(rename_buffer_) - 1);
        }
        // Delete - Delete selected (works with multi-select)
        if (ImGui::IsKeyPressed(ImGuiKey_Delete) && !selected_items_.empty()) {
            context_menu_item_ = first_selected;
            show_delete_confirm_ = true;
        }
        // Ctrl+C - Copy (first selected item)
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C) && !selected_items_.empty()) {
            context_menu_item_ = first_selected;
            CopySelectedAsset();
        }
        // Ctrl+X - Cut (first selected item)
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_X) && !selected_items_.empty()) {
            context_menu_item_ = first_selected;
            CutSelectedAsset();
        }
        // Ctrl+V - Paste
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_V) && !clipboard_path_.empty()) {
            if (!selected_items_.empty()) {
                context_menu_item_ = first_selected;
            } else {
                context_menu_item_ = directory_root_.get();
            }
            PasteAsset();
        }
        // Ctrl+N - New script
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_N)) {
            if (first_selected && first_selected->is_directory) {
                context_menu_item_ = first_selected;
            } else {
                context_menu_item_ = directory_root_.get();
            }
            new_file_request_ = NewFileDialog::Mode::Script;
        }
        // Ctrl+A - Select all
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_A) && directory_root_) {
            std::vector<AssetItem*> all_items;
            GetFlatItemList(directory_root_.get(), all_items);
            selected_items_.clear();
            for (auto* item : all_items) {
                selected_items_.insert(item);
            }
        }
        // F5 - Refresh
        if (ImGui::IsKeyPressed(ImGuiKey_F5)) {
            Refresh();
        }
        // Escape - Clear selection
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ClearSelection();
        }
    }

    // Main content: the tree, and the selected file's preview beside it (a
    // wide panel) or below it (a narrow one).
    AssetItem* preview_item = nullptr;
    for (auto* item : selected_items_) {
        if (item && !item->is_directory) {
            preview_item = item;
            break;
        }
    }
    const bool show_preview = show_dataset_preview_ && preview_item != nullptr;
    const float footer = ImGui::GetFrameHeightWithSpacing();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const bool beside = avail.x >= kPreviewBesideMinWidth;
    ImVec2 tree_size(0.0f, -footer);
    if (show_preview && beside) {
        preview_side_width_ = std::clamp(preview_side_width_, kPreviewMinExtent,
                                         std::max(kPreviewMinExtent, avail.x - kTreeMinExtent));
        tree_size.x = avail.x - preview_side_width_ - kPreviewHandle;
    } else if (show_preview) {
        const float spacing = 2.0f * ImGui::GetStyle().ItemSpacing.y;  // around the handle
        preview_below_height_ = std::clamp(preview_below_height_, kPreviewMinExtent,
                                           std::max(kPreviewMinExtent, avail.y - footer - spacing - kTreeMinExtent));
        tree_size.y = std::max(kTreeMinExtent, avail.y - footer - spacing - preview_below_height_ - kPreviewHandle);
    }

    ImGui::BeginChild("AssetContentRegion", tree_size, false);

    RenderDirectoryView();

    // Context menu for empty space - use BeginPopupContextWindow which triggers on right-click on window background
    if (ImGui::BeginPopupContextWindow("AssetEmptyContextMenu", ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
        // Set context to project root for empty space operations
        context_menu_item_ = directory_root_.get();

        ImGui::Text("Project");
        ImGui::Separator();

        // New submenu
        if (ImGui::BeginMenu(ICON_FA_PLUS " New")) {
            if (ImGui::MenuItem(ICON_FA_FILE_CODE " Script...")) {
                new_file_request_ = NewFileDialog::Mode::Script;
            }
            if (ImGui::MenuItem(ICON_FA_FILE_LINES " File...")) {
                new_file_request_ = NewFileDialog::Mode::File;
            }
            if (ImGui::MenuItem(ICON_FA_FOLDER_PLUS " Folder...")) {
                show_new_folder_dialog_ = true;
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        // Refresh
        if (ImGui::MenuItem(ICON_FA_ARROWS_ROTATE " Refresh")) {
            Refresh();
        }

        // Open in Explorer
        if (ImGui::MenuItem(ICON_FA_FOLDER " Show in Explorer")) {
            OpenInExplorer();
        }

        // Open in Terminal
        if (ImGui::MenuItem(ICON_FA_TERMINAL " Open in Terminal")) {
            OpenInTerminal();
        }

        ImGui::EndPopup();
    }

    ImGui::EndChild();

    if (show_preview) {
        if (beside) ImGui::SameLine(0.0f, 0.0f);
        PreviewSplitter(beside);
        if (beside) ImGui::SameLine(0.0f, 0.0f);
        RenderFilePreview(*preview_item, beside ? ImVec2(0.0f, -footer) : ImVec2(0.0f, preview_below_height_));
    }

    // Status bar at bottom
    RenderStatusBar();

    // Dialogs (these are modal popups, render at parent level)
    RenderNewFileDialog();
    RenderNewFolderDialog();
    RenderRenameDialog();
    RenderDeleteConfirmDialog();

    // Quick Preview popup (non-modal, separate window)
    RenderQuickPreviewPopup();

    ImGui::End();
}

void AssetBrowserPanel::SetProjectRoot(const std::string& root) {
    const bool changed = root != project_root_;
    if (changed) {
        // A different project starts with a fresh view and a fresh fingerprint.
        directory_root_.reset();
        ClearSelection();
        context_menu_item_ = nullptr;
        have_disk_signature_ = false;
    }
    project_root_ = root;
    // The same project again (the window was built after the start page opened
    // it, then the project-opened callback ran): one scan is enough (TOFIX159 D3).
    if (changed || (!directory_root_ && !is_scanning_directory_.load())) Refresh();
}

namespace {

// Fingerprint of what the tree shows: every visible entry's path, size and
// modification time, with the scanner's filters (hidden entries and .cyxwiz
// project files skipped). Any add, delete, rename or write changes it.
// The project's cache/ folder and *.tmp files are left out: the Engine writes
// them itself (cache/materialized while training prepares, cache/query_results
// on each query, cache/ingestion), so they are not a change made to the project.
std::uint64_t ProjectDiskSignature(const std::string& root, bool show_hidden) {
    std::uint64_t hash = 1469598103934665603ULL;  // FNV-1a
    const auto mix = [&hash](const void* data, size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    };
    std::error_code ec;
    fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec);
    const fs::recursive_directory_iterator end;
    while (!ec && it != end) {
        const fs::path& path = it->path();
        const std::string name = path.filename().string();
        std::error_code entry_ec;
        std::error_code same_ec;
        const bool engine_cache = name == "cache" && fs::equivalent(path.parent_path(), root, same_ec);
        if ((!show_hidden && !name.empty() && name[0] == '.') || path.extension() == ".cyxwiz" ||
            path.extension() == ".tmp" || engine_cache) {
            if (it->is_directory(entry_ec)) it.disable_recursion_pending();
            it.increment(ec);
            continue;
        }
        const std::string text = path.string();
        mix(text.data(), text.size());
        if (!it->is_directory(entry_ec)) {
            const auto size = it->file_size(entry_ec);
            mix(&size, sizeof(size));
        }
        const auto time = it->last_write_time(entry_ec).time_since_epoch().count();
        mix(&time, sizeof(time));
        it.increment(ec);
    }
    return hash;
}

void CollectOpenFolders(const AssetBrowserPanel::AssetItem& item, std::unordered_set<std::string>& open) {
    if (item.is_directory && item.is_expanded) open.insert(item.absolute_path);
    for (const auto& child : item.children) CollectOpenFolders(*child, open);
}

void RestoreOpenFolders(AssetBrowserPanel::AssetItem& item, const std::unordered_set<std::string>& open) {
    for (auto& child : item.children) {
        if (child->is_directory) child->is_expanded = open.count(child->absolute_path) > 0;
        RestoreOpenFolders(*child, open);
    }
}

}  // namespace

void AssetBrowserPanel::ApplyRescannedTree(std::unique_ptr<AssetItem> new_root) {
    const bool had_tree = directory_root_ != nullptr;
    std::unordered_set<std::string> open_folders;
    std::unordered_set<std::string> selected_paths;
    const std::string context_path = context_menu_item_ ? context_menu_item_->absolute_path : std::string();
    const std::string last_clicked_path = last_clicked_item_ ? last_clicked_item_->absolute_path : std::string();
    if (had_tree) {
        CollectOpenFolders(*directory_root_, open_folders);
        for (const AssetItem* item : selected_items_) selected_paths.insert(item->absolute_path);
    }

    directory_root_ = std::move(new_root);
    if (had_tree) {
        RestoreOpenFolders(*directory_root_, open_folders);
        force_tree_state_ = true;  // apply the restored open state this frame
    }
    SortAssets();

    // Old item pointers died with the old tree: re-point by path, or drop.
    ClearSelection();
    context_menu_item_ = nullptr;
    if (!context_path.empty() && context_path == directory_root_->absolute_path) {
        context_menu_item_ = directory_root_.get();
    }
    std::vector<AssetItem*> items;
    GetFlatItemList(directory_root_.get(), items);
    for (AssetItem* item : items) {
        if (selected_paths.count(item->absolute_path)) selected_items_.insert(item);
        if (!context_path.empty() && item->absolute_path == context_path) context_menu_item_ = item;
        if (!last_clicked_path.empty() && item->absolute_path == last_clicked_path) last_clicked_item_ = item;
    }
}

void AssetBrowserPanel::PollForExternalChanges() {
    if (!visible_ || project_root_.empty()) return;
    if (change_probe_.valid()) {
        if (change_probe_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
        const std::uint64_t signature = change_probe_.get();
        if (change_probe_root_ != project_root_) return;  // project switched meanwhile
        const bool changed = have_disk_signature_ && signature != last_disk_signature_;
        last_disk_signature_ = signature;
        have_disk_signature_ = true;
        if (changed && !is_scanning_directory_.load()) {
            spdlog::info("Asset browser: project files changed on disk; rescanning");
            Refresh();
        }
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < next_change_probe_ || is_scanning_directory_.load()) return;
    next_change_probe_ = now + std::chrono::seconds(2);
    change_probe_root_ = project_root_;
    change_probe_ = std::async(std::launch::async, ProjectDiskSignature, project_root_, show_hidden_files_);
}

void AssetBrowserPanel::Refresh() {
    if (project_root_.empty()) {
        Clear();
        return;
    }

    // If already scanning, cancel and restart
    if (is_scanning_directory_.load()) {
        AsyncTaskManager::Instance().Cancel(scanning_task_id_);
    }

    spdlog::debug("Refreshing asset browser async for project: {}", project_root_);

    is_scanning_directory_.store(true);
    scan_completed_.store(false);

    // Capture necessary values for the async task
    std::string project_root = project_root_;
    bool show_hidden = show_hidden_files_;

    scanning_task_id_ = AsyncTaskManager::Instance().RunAsync(
        "Scanning project files",
        [this, project_root, show_hidden](LambdaTask& task) {
            task.ReportProgress(0.0f, "Scanning directory structure...");

            // Build the tree in background
            auto new_root = std::make_unique<AssetItem>();
            new_root->name = "Project";
            new_root->is_directory = true;
            new_root->is_expanded = true;
            new_root->absolute_path = project_root;
            new_root->relative_path = "";

            // Track processed entries for progress reporting (no pre-counting)
            size_t processed_entries = 0;

            if (task.ShouldStop()) return;

            // Recursively build tree from a directory
            std::function<void(AssetItem&, const fs::path&)> build_tree;
            build_tree = [&](AssetItem& parent, const fs::path& dir_path) {
                if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) return;
                if (task.ShouldStop()) return;

                try {
                    for (const auto& entry : fs::directory_iterator(dir_path)) {
                        if (task.ShouldStop()) return;

                        std::string filename = entry.path().filename().string();

                        // Skip .cyxwiz project files
                        std::string ext = entry.path().extension().string();
                        if (ext == ".cyxwiz") {
                            continue;
                        }

                        // Skip hidden files unless show_hidden is true
                        if (!show_hidden && !filename.empty() && filename[0] == '.') {
                            continue;
                        }

                        auto item = std::make_unique<AssetItem>();
                        item->name = entry.path().filename().string();
                        item->absolute_path = entry.path().string();
                        // Make relative path manually since we don't have ProjectManager in thread
                        item->relative_path = fs::relative(entry.path(), project_root).string();
                        item->is_directory = entry.is_directory();

                        if (entry.is_directory()) {
                            item->type = AssetType::Folder;
                            item->is_expanded = false;
                            // Recursively scan subdirectories
                            build_tree(*item, entry.path());
                        } else {
                            item->type = DetermineAssetType(entry.path().string());
                            try {
                                item->file_size = fs::file_size(entry.path());
                            } catch (...) {
                                item->file_size = 0;
                            }
                        }

                        // Get last modified time
                        try {
                            auto ftime = fs::last_write_time(entry.path());
                            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                                ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                            auto time = std::chrono::system_clock::to_time_t(sctp);
                            char buf[64];
                            std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", std::localtime(&time));
                            item->modified_time = buf;
                        } catch (...) {
                            item->modified_time = "Unknown";
                        }

                        parent.children.push_back(std::move(item));

                        // Update progress periodically (every 100 files) - indeterminate progress
                        processed_entries++;
                        if (processed_entries % 100 == 0) {
                            // Use a pulsing progress without total (0.0 = indeterminate in UI)
                            task.ReportProgress(-1.0f, fmt::format("Scanning files... ({} found)", processed_entries));
                        }
                    }
                } catch (const std::exception& e) {
                    spdlog::warn("Error scanning directory {}: {}", dir_path.string(), e.what());
                }
            };

            // Build tree from project root
            build_tree(*new_root, project_root);

            if (task.ShouldStop()) return;

            task.ReportProgress(0.95f, "Finalizing...");

            // Store the result
            {
                std::lock_guard<std::mutex> lock(pending_tree_mutex_);
                pending_directory_root_ = std::move(new_root);
            }
            scan_completed_.store(true);

            task.MarkCompleted();
        },
        nullptr, // No progress callback needed
        [this](bool success, const std::string& error) {
            // Note: If the task was cancelled (e.g., new scan started), success will be false
            // but that's expected behavior, not an error. Only log actual failures.
            if (!success && !error.empty()) {
                spdlog::error("Directory scan failed: {}", error);
            }
            // Reset scanning state on any completion (success, cancel, or failure)
            is_scanning_directory_.store(false);
        }
    );
}

void AssetBrowserPanel::Clear() {
    // Cancel any ongoing directory scan
    if (is_scanning_directory_.load()) {
        AsyncTaskManager::Instance().Cancel(scanning_task_id_);
        is_scanning_directory_.store(false);
    }
    scan_completed_.store(false);

    // Clear pending tree data
    {
        std::lock_guard<std::mutex> lock(pending_tree_mutex_);
        pending_directory_root_.reset();
    }

    directory_root_.reset();
    selected_items_.clear();
    last_clicked_item_ = nullptr;
    context_menu_item_ = nullptr;
    project_root_.clear();
}

void AssetBrowserPanel::RenderToolbar() {
    // Row 1: Action buttons (always visible, compact)
    // Refresh button
    if (ImGui::Button(ICON_FA_ARROWS_ROTATE)) {
        Refresh();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Refresh (F5)");
    }

    ImGui::SameLine();

    // New folder button
    if (ImGui::Button(ICON_FA_FOLDER_PLUS)) {
        show_new_folder_dialog_ = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("New Folder");
    }

    ImGui::SameLine();

    // Expand all button
    if (ImGui::Button(ICON_FA_ANGLES_DOWN)) {
        ExpandAll();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Expand All");
    }

    ImGui::SameLine();

    // Collapse all button
    if (ImGui::Button(ICON_FA_ANGLES_UP)) {
        CollapseAll();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Collapse All");
    }

    ImGui::SameLine();

    // Hidden files toggle
    if (ImGui::Button(show_hidden_files_ ? ICON_FA_EYE : ICON_FA_EYE_SLASH)) {
        show_hidden_files_ = !show_hidden_files_;
        Refresh();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(show_hidden_files_ ? "Hide hidden files" : "Show hidden files");
    }

    ImGui::SameLine();

    // File preview toggle
    if (ImGui::Button(ICON_FA_TABLE_COLUMNS)) {
        show_dataset_preview_ = !show_dataset_preview_;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(show_dataset_preview_ ? "Hide the file preview" : "Show the file preview");
    }

    ImGui::SameLine();

    // Sort dropdown (on the right)
    const char* sort_names[] = {"Name", "Date", "Size", "Type"};
    ImGui::PushItemWidth(60);
    if (ImGui::BeginCombo("##sort", sort_names[static_cast<int>(sort_mode_)], ImGuiComboFlags_NoArrowButton)) {
        for (int i = 0; i < 4; i++) {
            bool is_selected = (static_cast<int>(sort_mode_) == i);
            if (ImGui::Selectable(sort_names[i], is_selected)) {
                sort_mode_ = static_cast<SortMode>(i);
                SortAssets();
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Sort by");
    }

    // Row 2: Search box (full width)
    ImGui::PushItemWidth(-1);
    if (ImGui::InputTextWithHint("##search", ICON_FA_MAGNIFYING_GLASS " Search...", search_buffer_, sizeof(search_buffer_))) {
        FilterAssets(search_buffer_);
    }
    ImGui::PopItemWidth();
}

void AssetBrowserPanel::RenderDirectoryView() {
    // Show the loading indicator only for the first scan; a rescan keeps the
    // existing tree on screen without shifting it.
    if (is_scanning_directory_.load() && !directory_root_) {
        ImGui::TextColored(ui::CurrentTokens().running, ICON_FA_SPINNER " Scanning project files...");
        return;
    }

    if (!directory_root_) {
        ImGui::TextDisabled("No project loaded");
        return;
    }

    // Render directory tree
    for (auto& child : directory_root_->children) {
        RenderAssetNode(*child);
    }

    // Reset force flag after rendering (only needed for one frame)
    force_tree_state_ = false;

    // Add invisible drop target for the remaining empty space (to drop to project root)
    // Calculate remaining space
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.y > 10.0f) {  // Only if there's enough space
        ImGui::InvisibleButton("##RootDropTarget", ImVec2(-1, avail.y));
        // The project menu (Render's BeginPopupContextWindow, same window and
        // id) skips items, and this button is one: open it from here.
        if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) ImGui::OpenPopup("AssetEmptyContextMenu");

        // Handle drop to project root
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ASSET_PATH")) {
                std::string source_path(static_cast<const char*>(payload->Data));
                fs::path source(source_path);
                fs::path dest = fs::path(directory_root_->absolute_path) / source.filename();

                // Don't drop if already in root
                if (source.parent_path().string() != directory_root_->absolute_path) {
                    try {
                        // Handle name conflicts
                        if (fs::exists(dest)) {
                            std::string stem = dest.stem().string();
                            std::string ext = dest.extension().string();
                            int counter = 1;
                            while (fs::exists(dest)) {
                                dest = fs::path(directory_root_->absolute_path) / (stem + "_" + std::to_string(counter++) + ext);
                            }
                        }
                        fs::rename(source, dest);
                        spdlog::info("Moved to root: {} -> {}", source.string(), dest.string());
                        needs_refresh_ = true;
                    } catch (const std::exception& e) {
                        spdlog::error("Move to root failed: {}", e.what());
                    }
                }
            }
            ImGui::EndDragDropTarget();
        }

        // Show tooltip when hovering with drag payload
        if (ImGui::BeginDragDropTarget()) {
            ImGui::EndDragDropTarget();
        }
    }
}

void AssetBrowserPanel::RenderAssetNode(AssetItem& item, int depth) {
    // Filter: skip items that don't match search and have no matching children
    if (!current_search_query_.empty()) {
        bool matches = MatchesSearch(item, current_search_query_);
        bool has_matching_children = item.is_directory && HasMatchingChildren(item, current_search_query_);

        if (!matches && !has_matching_children) {
            return;  // Skip this item entirely
        }
    }

    // Set up tree node flags
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;

    // Leaf nodes: files without children, or empty directories
    bool is_leaf = item.children.empty() && !item.is_directory;
    if (is_leaf) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    // Add selected flag if this item is in the selection set
    if (IsSelected(&item)) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    // Set expanded state - force when searching or after expand/collapse all buttons
    if (!current_search_query_.empty() && item.is_directory) {
        ImGui::SetNextItemOpen(true, ImGuiCond_Always);  // Force expand during search
    } else if (item.is_directory && force_tree_state_) {
        // Force state when expand/collapse all was clicked
        ImGui::SetNextItemOpen(item.is_expanded, ImGuiCond_Always);
    }

    // For directories, show item count
    std::string label = item.name;
    if (item.is_directory && !item.children.empty()) {
        int count = static_cast<int>(item.children.size());
        label += " (" + std::to_string(count) + ")";
    }

    // Render tree node with icon
    // Keyed by path, not by address: a rescan builds new items, and an address
    // key would make ImGui forget which folders were open.
    // Rows in the theme's selection / hover washes: the Unreal preset's
    // selected Header is a grey barely off the panel while its hovered row is
    // a 50 % orange, so the clicked file glowed and the selection vanished.
    {
        const ui::Tokens& tokens = ui::CurrentTokens();
        ImGui::PushStyleColor(ImGuiCol_Header, ui::WithAlpha(tokens.accent, 0.28f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, tokens.hover);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, ui::WithAlpha(tokens.accent, 0.36f));
    }
    bool node_open = ImGui::TreeNodeEx(item.absolute_path.c_str(), flags, "%s", "");
    ImGui::PopStyleColor(3);
    {
        // Icon in its kind's colour, then the name. Drawn, not submitted, so
        // the tree node stays the last item for the context menu and drag.
        const assets::FileKind kind = assets::KindOfPath(item.absolute_path, item.is_directory);
        const char* icon = KindIcon(kind, item.is_directory && node_open);
        const ImVec2 min = ImGui::GetItemRectMin();
        const float y = min.y + (ImGui::GetItemRectSize().y - ImGui::GetFontSize()) * 0.5f;
        const float icon_width = ImGui::CalcTextSize(ICON_FA_FOLDER_OPEN).x;
        float x = min.x + ImGui::GetTreeNodeToLabelSpacing();
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddText(ImVec2(x + (icon_width - ImGui::CalcTextSize(icon).x) * 0.5f, y), ui::ToU32(KindColour(kind)), icon);
        x += icon_width + ImGui::GetStyle().ItemInnerSpacing.x;
        draw->AddText(ImVec2(x, y), ImGui::GetColorU32(ImGuiCol_Text), label.c_str());
    }

    // Sync our is_expanded state with ImGui's actual state (for directories)
    if (item.is_directory && !is_leaf) {
        item.is_expanded = node_open;
    }

    // Capture every interaction owned by the tree row before rendering any
    // other item. Tooltip contents replace ImGui's last-item state and would
    // otherwise make delayed-hover clicks target the tooltip instead.
    const bool item_hovered =
        ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal);
    const bool item_clicked = ImGui::IsItemClicked(0);
    const bool item_double_clicked =
        item_clicked && ImGui::IsMouseDoubleClicked(0);

    // Handle selection with Ctrl/Shift modifiers for multi-select
    if (item_clicked) {
        bool ctrl_held = ImGui::GetIO().KeyCtrl;
        bool shift_held = ImGui::GetIO().KeyShift;
        SelectItem(&item, ctrl_held, shift_held);
    }

    const auto open_graph_in_studio = [this, &item]() {
        if (!on_open_in_node_editor_) {
            spdlog::error("Asset Browser graph callback is not configured: {}",
                          item.absolute_path);
            return;
        }
        spdlog::info("Asset Browser opening graph in CyxWiz Studio: {}",
                     item.absolute_path);
        on_open_in_node_editor_(item.absolute_path);
    };

    // Handle double-click for files (directories are handled by TreeNode arrow)
    if (!item.is_directory && item_double_clicked) {
        // Special handling for graph files - open in Node Editor
        if (item.type == AssetType::Graph) {
            open_graph_in_studio();
        }
        // Dataset files create a configured Data Input; only that node applies a load.
        else if (IsDatasetFile(item)) {
            CreateDataInputFromItem(item);
        }
        // Other files use the generic open callback. Graph and dataset files
        // already performed their dedicated action above.
        else if (on_double_click_) {
            on_double_click_(item);
        }
    }

    // Handle right-click context menu - use unique ID per item
    ImGui::PushID(&item);
    if (ImGui::BeginPopupContextItem("##ItemContext")) {
        context_menu_item_ = &item;

        ImGui::Text("%s", item.name.c_str());
        ImGui::Separator();

        // New submenu
        if (ImGui::BeginMenu(ICON_FA_PLUS " New")) {
            if (ImGui::MenuItem(ICON_FA_FILE_CODE " Script...")) {
                new_file_request_ = NewFileDialog::Mode::Script;
            }
            if (ImGui::MenuItem(ICON_FA_FILE_LINES " File...")) {
                new_file_request_ = NewFileDialog::Mode::File;
            }
            if (ImGui::MenuItem(ICON_FA_FOLDER_PLUS " Folder...")) {
                show_new_folder_dialog_ = true;
            }
            ImGui::EndMenu();
        }

        // Open (for files only)
        if (!item.is_directory) {
            if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Open")) {
                if (on_double_click_) {
                    on_double_click_(item);
                }
            }
            // Create a configured Data Input node; Data Input Apply performs
            // the actual load/registration and exposes the authoritative
            // source preview. Selecting an already registered dataset still
            // shows the bounded Dataset Preview side pane.
            if (IsDatasetFile(item)) {
                if (ImGui::MenuItem(ICON_FA_DATABASE " Create Data Input")) {
                    CreateDataInputFromItem(item);
                }
            }
            // Quick Preview is raw file peeking for non-dataset assets only.
            if (!IsDatasetFile(item) && IsPreviewableFile(item)) {
                if (ImGui::MenuItem(ICON_FA_EYE " Quick Preview")) {
                    show_quick_preview_ = true;
                    quick_preview_path_ = item.absolute_path;
                }
            }
            // Open in CyxWiz Studio (for .cyxgraph files only)
            if (item.type == AssetType::Graph) {
                if (ImGui::MenuItem(ICON_FA_DIAGRAM_PROJECT " Open in CyxWiz Studio")) {
                    open_graph_in_studio();
                }
            }
        }

        ImGui::Separator();

        // Copy/Cut/Paste
        if (ImGui::MenuItem(ICON_FA_COPY " Copy", "Ctrl+C")) {
            CopySelectedAsset();
        }
        if (ImGui::MenuItem(ICON_FA_SCISSORS " Cut", "Ctrl+X")) {
            CutSelectedAsset();
        }
        if (ImGui::MenuItem(ICON_FA_PASTE " Paste", "Ctrl+V", false, !clipboard_path_.empty())) {
            PasteAsset();
        }

        ImGui::Separator();

        // Rename (F2)
        if (ImGui::MenuItem(ICON_FA_PENCIL " Rename", "F2")) {
            show_rename_dialog_ = true;
            std::strncpy(rename_buffer_, item.name.c_str(), sizeof(rename_buffer_) - 1);
        }

        // Delete (Delete)
        if (ImGui::MenuItem(ICON_FA_TRASH " Delete", "Del")) {
            show_delete_confirm_ = true;
        }

        ImGui::Separator();

        // Open in Explorer
        if (ImGui::MenuItem(ICON_FA_FOLDER " Show in Explorer")) {
            OpenInExplorer();
        }

        // Open in Terminal
        if (ImGui::MenuItem(ICON_FA_TERMINAL " Open in Terminal")) {
            OpenInTerminal();
        }

        ImGui::EndPopup();
    }
    ImGui::PopID();

    // Handle drag source - all items can be dragged
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
        ImGui::SetDragDropPayload("ASSET_PATH", item.absolute_path.c_str(), item.absolute_path.size() + 1);
        ImGui::Text("%s %s", KindIcon(assets::KindOfPath(item.absolute_path, item.is_directory), false), item.name.c_str());
        ImGui::EndDragDropSource();
    }

    // Handle drop target - directories accept dropped items
    if (item.is_directory && ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ASSET_PATH")) {
            std::string source_path(static_cast<const char*>(payload->Data));
            fs::path source(source_path);
            fs::path dest = fs::path(item.absolute_path) / source.filename();

            // Don't drop onto itself or its parent
            if (source_path != item.absolute_path && source.parent_path().string() != item.absolute_path) {
                try {
                    // Handle name conflicts
                    if (fs::exists(dest)) {
                        std::string stem = dest.stem().string();
                        std::string ext = dest.extension().string();
                        int counter = 1;
                        while (fs::exists(dest)) {
                            dest = fs::path(item.absolute_path) / (stem + "_" + std::to_string(counter++) + ext);
                        }
                    }
                    fs::rename(source, dest);
                    spdlog::info("Moved: {} -> {}", source.string(), dest.string());
                    // Defer refresh to next frame to avoid invalidating tree during iteration
                    needs_refresh_ = true;
                } catch (const std::exception& e) {
                    spdlog::error("Move failed: {}", e.what());
                }
            }
        }
        ImGui::EndDragDropTarget();
    }

    // Render descriptive content only after all last-item-dependent row
    // interactions (click, popup, and drag/drop) have been evaluated.
    if (item_hovered) {
        ImGui::BeginTooltip();
        ImGui::Text("%s", item.name.c_str());
        ImGui::Separator();
        ImGui::Text("Path: %s", item.relative_path.c_str());
        if (!item.is_directory) {
            ImGui::Text("Size: %s", FormatFileSize(item.file_size).c_str());
        }
        ImGui::Text("Modified: %s", item.modified_time.c_str());
        ImGui::EndTooltip();
    }

    // Recursively render children if node is open (and not a leaf)
    if (node_open && !is_leaf) {
        for (auto& child : item.children) {
            RenderAssetNode(*child, depth + 1);
        }
        ImGui::TreePop();
    }
}

void AssetBrowserPanel::RenderStatusBar() {
    int total_items = CountItems();
    int selected_count = static_cast<int>(selected_items_.size());
    auto& pm = ProjectManager::Instance();

    if (pm.HasActiveProject()) {
        if (selected_count > 0) {
            ImGui::Text("%d selected | %d items | %s", selected_count, total_items, pm.GetProjectName().c_str());
        } else {
            ImGui::Text("%d items | %s", total_items, pm.GetProjectName().c_str());
        }
    } else {
        ImGui::Text("%d items | No project", total_items);
    }
    if (asset_action_status_ != AssetActionStatus::None &&
        !asset_action_message_.empty()) {
        ImGui::SameLine();
        const ImVec4 color = asset_action_status_ == AssetActionStatus::Error
            ? ui::CurrentTokens().error
            : ui::CurrentTokens().success;
        ImGui::TextColored(color, "| %s", asset_action_message_.c_str());
    }
}

void AssetBrowserPanel::RenderNewFileDialog() {
    // Same dialog as File > New Script (TOFIX129 A2-4), opened in the folder
    // that was right-clicked, else the project's scripts folder (a script) or
    // the project folder (a file).
    if (new_file_request_) {
        std::string folder;
        if (context_menu_item_ && context_menu_item_->is_directory) folder = context_menu_item_->absolute_path;
        else if (context_menu_item_) folder = fs::path(context_menu_item_->absolute_path).parent_path().string();
        new_file_dialog_.Open(*new_file_request_, folder);
        new_file_request_.reset();
    }
    if (new_file_dialog_.Render() != NewFileDialog::Result::Created) return;
    Refresh();
    if (on_double_click_) {
        AssetItem new_item;
        const fs::path created(new_file_dialog_.CreatedPath());
        new_item.name = created.filename().string();
        new_item.absolute_path = created.string();
        new_item.relative_path = ProjectManager::Instance().MakeRelativePath(created.string());
        new_item.type = DetermineAssetType(created.string());
        new_item.is_directory = false;
        on_double_click_(new_item);
    }
}

void AssetBrowserPanel::RenderRenameDialog() {
    if (!show_rename_dialog_ || !context_menu_item_) return;

    ImGui::OpenPopup("Rename Asset");

    if (ImGui::BeginPopupModal("Rename Asset", &show_rename_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Rename: %s", context_menu_item_->name.c_str());
        ImGui::Separator();

        ImGui::InputText("##rename", rename_buffer_, sizeof(rename_buffer_));

        ImGui::Separator();

        if (ImGui::Button("Rename", ImVec2(120, 0))) {
            RenameSelectedAsset();
            show_rename_dialog_ = false;
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_rename_dialog_ = false;
        }

        ImGui::EndPopup();
    }
}

void AssetBrowserPanel::RenderDeleteConfirmDialog() {
    if (!show_delete_confirm_ || !context_menu_item_) return;

    ImGui::OpenPopup("Delete Asset");

    if (ImGui::BeginPopupModal("Delete Asset", &show_delete_confirm_, ImGuiWindowFlags_AlwaysAutoResize)) {
        int delete_count = selected_items_.empty() ? 1 : static_cast<int>(selected_items_.size());

        if (delete_count > 1) {
            ImGui::Text("Are you sure you want to delete %d items?", delete_count);
            ImGui::Separator();
            // Show first few items
            int shown = 0;
            for (auto* item : selected_items_) {
                if (shown >= 5) {
                    ImGui::Text("... and %d more", delete_count - 5);
                    break;
                }
                ImGui::BulletText("%s", item->name.c_str());
                shown++;
            }
        } else {
            ImGui::Text("Are you sure you want to delete:");
            ImGui::Text("%s", context_menu_item_ ? context_menu_item_->name.c_str() : "");
        }

        ImGui::Separator();
        ImGui::TextColored(ui::CurrentTokens().error, "This action cannot be undone");

        ImGui::Separator();

        if (ImGui::Button("Delete", ImVec2(120, 0))) {
            DeleteSelectedAsset();
            show_delete_confirm_ = false;
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_delete_confirm_ = false;
        }

        ImGui::EndPopup();
    }
}

void AssetBrowserPanel::RenderNewFolderDialog() {
    if (!show_new_folder_dialog_) return;

    auto& pm = ProjectManager::Instance();

    // Show warning if no project is open
    if (!pm.HasActiveProject()) {
        ImGui::OpenPopup("No Project Open##Folder");

        if (ImGui::BeginPopupModal("No Project Open##Folder", &show_new_folder_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextColored(ui::CurrentTokens().warning, ICON_FA_TRIANGLE_EXCLAMATION " Warning");
            ImGui::Separator();
            ImGui::Text("Cannot create a new folder without an open project.");
            ImGui::Text("Please create or open a project first.");
            ImGui::Separator();

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                show_new_folder_dialog_ = false;
            }
            ImGui::EndPopup();
        }
        return;
    }

    ImGui::OpenPopup("New Folder");

    if (ImGui::BeginPopupModal("New Folder", &show_new_folder_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Create a new folder");
        ImGui::Separator();

        ImGui::InputTextWithHint("##folder_name", "folder_name", new_folder_name_, sizeof(new_folder_name_));

        ImGui::Separator();

        if (ImGui::Button("Create", ImVec2(120, 0))) {
            CreateNewFolder();
            show_new_folder_dialog_ = false;
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_new_folder_dialog_ = false;
        }

        ImGui::EndPopup();
    }
}

void AssetBrowserPanel::BuildDirectoryTree() {
    directory_root_ = std::make_unique<AssetItem>();
    directory_root_->name = "Project";
    directory_root_->is_directory = true;
    directory_root_->is_expanded = true;

    auto& pm = ProjectManager::Instance();
    if (!pm.HasActiveProject()) return;

    fs::path project_root = pm.GetProjectRoot();
    directory_root_->absolute_path = project_root.string();
    directory_root_->relative_path = "";

    // Recursively build tree from a directory
    std::function<void(AssetItem&, const fs::path&)> build_tree;
    build_tree = [&](AssetItem& parent, const fs::path& dir_path) {
        if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) return;

        try {
            for (const auto& entry : fs::directory_iterator(dir_path)) {
                std::string filename = entry.path().filename().string();

                // Skip .cyxwiz project files - no need to show them in asset browser
                std::string ext = entry.path().extension().string();
                if (ext == ".cyxwiz") {
                    continue;
                }

                // Skip hidden files (starting with .) unless show_hidden_files_ is true
                if (!show_hidden_files_ && !filename.empty() && filename[0] == '.') {
                    continue;
                }

                auto item = std::make_unique<AssetItem>();
                item->name = entry.path().filename().string();
                item->absolute_path = entry.path().string();
                item->relative_path = pm.MakeRelativePath(entry.path().string());
                item->is_directory = entry.is_directory();

                if (entry.is_directory()) {
                    item->type = AssetType::Folder;
                    item->is_expanded = false;
                    // Recursively scan subdirectories
                    build_tree(*item, entry.path());
                } else {
                    item->type = DetermineAssetType(entry.path().string());
                    // Get file size
                    try {
                        item->file_size = fs::file_size(entry.path());
                    } catch (...) {
                        item->file_size = 0;
                    }
                }

                // Get last modified time
                try {
                    auto ftime = fs::last_write_time(entry.path());
                    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                    auto time = std::chrono::system_clock::to_time_t(sctp);
                    char buf[64];
                    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", std::localtime(&time));
                    item->modified_time = buf;
                } catch (...) {
                    item->modified_time = "Unknown";
                }

                parent.children.push_back(std::move(item));
            }

        } catch (const std::exception& e) {
            spdlog::warn("Error building directory tree for {}: {}", dir_path.string(), e.what());
        }
    };

    // Build tree from project root - scan ALL files and folders
    build_tree(*directory_root_, project_root);

    // Apply current sort mode
    SortAssets();
}

AssetBrowserPanel::AssetType AssetBrowserPanel::DetermineAssetType(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Scripts
    if (ext == ".py" || ext == ".pyw" || ext == ".cyx" || ext == ".ipynb") return AssetType::Script;

    // Models (note: .h5 can be Keras models OR HDF5 datasets - we prioritize datasets)
    if (ext == ".onnx" || ext == ".pt" || ext == ".safetensors" || ext == ".bin")
        return AssetType::Model;

    // Datasets: every format Data Input recognises, so each gets "Create Data Input"
    if (data_input::DetectFormat(path)) return AssetType::Dataset;

    // Checkpoints
    if (ext == ".ckpt" || ext == ".checkpoint")
        return AssetType::Checkpoint;

    // Exports
    if (ext == ".gguf" || ext == ".lora")
        return AssetType::Export;

    // Graphs
    if (ext == ".cyxgraph") return AssetType::Graph;

    // Plugins
    if (ext == ".dll" || ext == ".so" || ext == ".dylib")
        return AssetType::Plugin;

    return AssetType::Unknown;
}

int AssetBrowserPanel::CountItems() const {
    std::function<int(const AssetItem*)> count_recursive;
    count_recursive = [&](const AssetItem* item) {
        int count = item->is_directory ? 0 : 1;
        for (const auto& child : item->children) {
            count += count_recursive(child.get());
        }
        return count;
    };

    if (directory_root_) {
        return count_recursive(directory_root_.get());
    }

    return 0;
}

void AssetBrowserPanel::CreateNewFolder() {
    auto& pm = ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        spdlog::warn("Cannot create folder: no active project");
        return;
    }

    std::string folder_name = new_folder_name_;
    if (folder_name.empty()) {
        spdlog::warn("Folder name is empty");
        return;
    }

    // Determine target directory - use context_menu_item if it's a directory, otherwise use project root
    fs::path target_dir;
    if (context_menu_item_ && context_menu_item_->is_directory) {
        target_dir = context_menu_item_->absolute_path;
    } else if (context_menu_item_) {
        target_dir = fs::path(context_menu_item_->absolute_path).parent_path();
    } else {
        target_dir = pm.GetProjectRoot();
    }

    fs::path folder_path = target_dir / folder_name;

    // Create directory
    try {
        fs::create_directories(folder_path);
        spdlog::info("Created new folder: {}", folder_path.string());

        // Refresh to show new folder
        Refresh();

        // Clear input
        std::memset(new_folder_name_, 0, sizeof(new_folder_name_));

    } catch (const std::exception& e) {
        spdlog::error("Failed to create folder: {}", e.what());
    }
}

void AssetBrowserPanel::DeleteSelectedAsset() {
    // Delete all selected items (multi-select support)
    if (selected_items_.empty() && !context_menu_item_) return;

    // Build list of items to delete (selected items or context menu item)
    std::vector<AssetItem*> items_to_delete;
    if (!selected_items_.empty()) {
        items_to_delete.assign(selected_items_.begin(), selected_items_.end());
    } else if (context_menu_item_) {
        items_to_delete.push_back(context_menu_item_);
    }

    int deleted_count = 0;
    for (AssetItem* item : items_to_delete) {
        if (!item) continue;

        try {
            if (fs::is_directory(item->absolute_path)) {
                fs::remove_all(item->absolute_path);
            } else {
                fs::remove(item->absolute_path);
            }
            spdlog::info("Deleted asset: {}", item->absolute_path);
            deleted_count++;

            // Fire callback
            if (on_deleted_) {
                on_deleted_(*item);
            }
        } catch (const std::exception& e) {
            spdlog::error("Failed to delete {}: {}", item->absolute_path, e.what());
        }
    }

    // Clear selection
    ClearSelection();
    context_menu_item_ = nullptr;

    // Refresh to update view
    if (deleted_count > 0) {
        Refresh();
    }
}

void AssetBrowserPanel::RenameSelectedAsset() {
    if (!context_menu_item_) return;

    std::string new_name = rename_buffer_;
    if (new_name.empty() || new_name == context_menu_item_->name) {
        return;
    }

    try {
        fs::path old_path(context_menu_item_->absolute_path);
        fs::path new_path = old_path.parent_path() / new_name;

        fs::rename(old_path, new_path);
        spdlog::info("Renamed asset: {} -> {}", old_path.string(), new_path.string());

        // Refresh to update view
        Refresh();

    } catch (const std::exception& e) {
        spdlog::error("Failed to rename asset: {}", e.what());
    }
}

void AssetBrowserPanel::OpenInExplorer() {
    if (!context_menu_item_) return;

    const std::string target = context_menu_item_->absolute_path;
    if (LaunchAssetInFileBrowser(target, context_menu_item_->is_directory)) {
        spdlog::info("AssetBrowser: opened file browser for '{}'", target);
        asset_action_status_ = AssetActionStatus::Success;
        asset_action_message_ = "Opened in file browser";
    } else {
        spdlog::error("AssetBrowser: failed to open file browser for '{}'", target);
        asset_action_status_ = AssetActionStatus::Error;
        asset_action_message_ = "Could not open the selected asset in the file browser";
    }
}

void AssetBrowserPanel::OpenInTerminal() {
    if (!context_menu_item_) return;

    std::string dir = context_menu_item_->is_directory 
        ? context_menu_item_->absolute_path
        : fs::path(context_menu_item_->absolute_path).parent_path().string();

#ifdef _WIN32
    std::string command = "start cmd.exe /K cd /D "" + dir + """;
    system(command.c_str());
#elif defined(__APPLE__)
    std::string command = "open -a Terminal "" + dir + """;
    system(command.c_str());
#else
    std::string command = "gnome-terminal --working-directory="" + dir + """;
    system(command.c_str());
#endif
}

void AssetBrowserPanel::CopySelectedAsset() {
    if (!context_menu_item_) return;

    clipboard_path_ = context_menu_item_->absolute_path;
    clipboard_is_cut_ = false;
    spdlog::info("Copied: {}", clipboard_path_);
}

void AssetBrowserPanel::CutSelectedAsset() {
    if (!context_menu_item_) return;

    clipboard_path_ = context_menu_item_->absolute_path;
    clipboard_is_cut_ = true;
    spdlog::info("Cut: {}", clipboard_path_);
}

void AssetBrowserPanel::PasteAsset() {
    if (clipboard_path_.empty()) return;

    auto& pm = ProjectManager::Instance();
    if (!pm.HasActiveProject()) return;

    // Determine target directory
    fs::path target_dir;
    if (context_menu_item_ && context_menu_item_->is_directory) {
        target_dir = context_menu_item_->absolute_path;
    } else if (context_menu_item_) {
        target_dir = fs::path(context_menu_item_->absolute_path).parent_path();
    } else {
        target_dir = pm.GetProjectRoot();
    }

    fs::path source_path(clipboard_path_);
    fs::path dest_path = target_dir / source_path.filename();

    // Handle name conflicts
    if (fs::exists(dest_path)) {
        std::string stem = dest_path.stem().string();
        std::string ext = dest_path.extension().string();
        int counter = 1;
        while (fs::exists(dest_path)) {
            dest_path = target_dir / (stem + "_" + std::to_string(counter++) + ext);
        }
    }

    try {
        if (clipboard_is_cut_) {
            // Move file
            fs::rename(source_path, dest_path);
            spdlog::info("Moved: {} -> {}", source_path.string(), dest_path.string());
            clipboard_path_.clear();  // Clear clipboard after cut
        } else {
            // Copy file
            if (fs::is_directory(source_path)) {
                fs::copy(source_path, dest_path, fs::copy_options::recursive);
            } else {
                fs::copy_file(source_path, dest_path);
            }
            spdlog::info("Pasted: {} -> {}", source_path.string(), dest_path.string());
        }
        Refresh();
    } catch (const std::exception& e) {
        spdlog::error("Paste failed: {}", e.what());
    }
}

void AssetBrowserPanel::ExpandAll() {
    if (directory_root_) {
        SetExpandedRecursive(directory_root_.get(), true);
        force_tree_state_ = true;  // Force ImGui to apply the new state
    }
}

void AssetBrowserPanel::CollapseAll() {
    if (directory_root_) {
        SetExpandedRecursive(directory_root_.get(), false);
        force_tree_state_ = true;  // Force ImGui to apply the new state
    }
}

void AssetBrowserPanel::SetExpandedRecursive(AssetItem* item, bool expanded) {
    if (!item) return;
    if (item->is_directory) {
        item->is_expanded = expanded;
    }
    for (auto& child : item->children) {
        SetExpandedRecursive(child.get(), expanded);
    }
}

void AssetBrowserPanel::SortAssets() {
    if (directory_root_) {
        SortChildren(directory_root_.get());
    }
}

bool AssetBrowserPanel::IsSelected(AssetItem* item) const {
    return selected_items_.find(item) != selected_items_.end();
}

void AssetBrowserPanel::SelectItem(AssetItem* item, bool ctrl_held, bool shift_held) {
    if (!item) return;

    if (shift_held && last_clicked_item_) {
        // Shift+click: select range from last clicked to current
        SelectRange(last_clicked_item_, item);
    } else if (ctrl_held) {
        // Ctrl+click: toggle selection
        if (IsSelected(item)) {
            selected_items_.erase(item);
        } else {
            selected_items_.insert(item);
        }
        last_clicked_item_ = item;
    } else {
        // Normal click: clear selection and select only this item
        ClearSelection();
        selected_items_.insert(item);
        last_clicked_item_ = item;
    }
}

void AssetBrowserPanel::ClearSelection() {
    selected_items_.clear();
    last_clicked_item_ = nullptr;
}

void AssetBrowserPanel::GetFlatItemList(AssetItem* root, std::vector<AssetItem*>& out_list) {
    if (!root) return;
    for (auto& child : root->children) {
        out_list.push_back(child.get());
        if (child->is_directory && child->is_expanded) {
            GetFlatItemList(child.get(), out_list);
        }
    }
}

void AssetBrowserPanel::SelectRange(AssetItem* from, AssetItem* to) {
    if (!from || !to || !directory_root_) return;

    // Build flat list of visible items
    std::vector<AssetItem*> flat_list;
    GetFlatItemList(directory_root_.get(), flat_list);

    // Find indices
    int from_idx = -1, to_idx = -1;
    for (int i = 0; i < static_cast<int>(flat_list.size()); i++) {
        if (flat_list[i] == from) from_idx = i;
        if (flat_list[i] == to) to_idx = i;
    }

    if (from_idx == -1 || to_idx == -1) return;

    // Select range
    int start = std::min(from_idx, to_idx);
    int end = std::max(from_idx, to_idx);

    ClearSelection();
    for (int i = start; i <= end; i++) {
        selected_items_.insert(flat_list[i]);
    }
}

void AssetBrowserPanel::SortChildren(AssetItem* parent) {
    if (!parent) return;

    std::sort(parent->children.begin(), parent->children.end(),
        [this](const std::unique_ptr<AssetItem>& a, const std::unique_ptr<AssetItem>& b) {
            // Directories always come first
            if (a->is_directory != b->is_directory) {
                return a->is_directory;
            }

            switch (sort_mode_) {
                case SortMode::Name:
                    return a->name < b->name;
                case SortMode::Date:
                    return a->modified_time > b->modified_time;  // Newest first
                case SortMode::Size:
                    return a->file_size > b->file_size;  // Largest first
                case SortMode::Type:
                    if (a->type != b->type) {
                        return static_cast<int>(a->type) < static_cast<int>(b->type);
                    }
                    return a->name < b->name;
                default:
                    return a->name < b->name;
            }
        });

    // Recursively sort children
    for (auto& child : parent->children) {
        if (child->is_directory) {
            SortChildren(child.get());
        }
    }
}

std::string AssetBrowserPanel::FormatFileSize(std::uintmax_t size) const {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double display_size = static_cast<double>(size);

    while (display_size >= 1024.0 && unit_index < 4) {
        display_size /= 1024.0;
        unit_index++;
    }

    char buf[32];
    if (unit_index == 0) {
        std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(size));
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f %s", display_size, units[unit_index]);
    }
    return buf;
}

void AssetBrowserPanel::FilterAssets(const std::string& query) {
    // Store lowercase version of query for filtering
    current_search_query_ = query;
    std::transform(current_search_query_.begin(), current_search_query_.end(),
                   current_search_query_.begin(), ::tolower);
}

bool AssetBrowserPanel::MatchesSearch(const AssetItem& item, const std::string& query) const {
    if (query.empty()) return true;

    // Case-insensitive search in name
    std::string lower_name = item.name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    return lower_name.find(query) != std::string::npos;
}

bool AssetBrowserPanel::HasMatchingChildren(const AssetItem& item, const std::string& query) const {
    if (query.empty()) return true;

    for (const auto& child : item.children) {
        if (MatchesSearch(*child, query)) {
            return true;
        }
        if (child->is_directory && HasMatchingChildren(*child, query)) {
            return true;
        }
    }
    return false;
}

// Dataset integration methods

bool AssetBrowserPanel::IsDatasetFile(const AssetItem& item) const {
    return item.type == AssetType::Dataset;
}

bool AssetBrowserPanel::IsPreviewableFile(const AssetItem& item) const {
    if (item.is_directory) return false;

    auto content_type = ::gui::PreviewRenderer::DetectFileType(item.absolute_path);
    return content_type != ::gui::PreviewContentType::Unknown;
}

void AssetBrowserPanel::RenderQuickPreviewPopup() {
    if (!show_quick_preview_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);

    std::string title = "Quick Preview - " + fs::path(quick_preview_path_).filename().string() + "###QuickPreview";

    if (ImGui::Begin(title.c_str(), &show_quick_preview_, ImGuiWindowFlags_NoDocking)) {
        // File info bar
        auto content_type = ::gui::PreviewRenderer::DetectFileType(quick_preview_path_);
        ImGui::TextColored(ui::CurrentTokens().accent_text, "%s", quick_preview_path_.c_str());
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 80);

        // Type indicator
        const char* type_str = "Unknown";
        switch (content_type) {
            case ::gui::PreviewContentType::Image: type_str = "Image"; break;
            case ::gui::PreviewContentType::TableCSV: type_str = "CSV"; break;
            case ::gui::PreviewContentType::JSONTree: type_str = "JSON"; break;
            case ::gui::PreviewContentType::PlainText: type_str = "Text"; break;
            case ::gui::PreviewContentType::SyntaxText: type_str = "Code"; break;
            case ::gui::PreviewContentType::Markdown: type_str = "Markdown"; break;
            case ::gui::PreviewContentType::HexDump: type_str = "Binary"; break;
            default: break;
        }
        ImGui::TextDisabled("[%s]", type_str);

        ImGui::Separator();

        // Render preview content
        ImGui::BeginChild("##QuickPreviewContent", ImVec2(0, 0), false);
        quick_preview_renderer_.RenderFilePreview(quick_preview_path_);
        ImGui::EndChild();
    }
    ImGui::End();
}

void AssetBrowserPanel::CreateDataInputFromItem(const AssetItem& item) {
    if (IsDatasetFile(item) && on_create_data_input_) on_create_data_input_(item.absolute_path);
}
void AssetBrowserPanel::PreviewSplitter(bool beside) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 size = beside ? ImVec2(kPreviewHandle, std::max(avail.y - ImGui::GetFrameHeightWithSpacing(), 1.0f))
                               : ImVec2(std::max(avail.x, 1.0f), kPreviewHandle);
    ImGui::InvisibleButton("##preview_handle", size);
    const bool active = ImGui::IsItemActive();
    const bool hot = active || ImGui::IsItemHovered();
    if (hot) ImGui::SetMouseCursor(beside ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
    if (active) {
        if (beside) preview_side_width_ -= ImGui::GetIO().MouseDelta.x;
        else preview_below_height_ -= ImGui::GetIO().MouseDelta.y;
    }
    // A short grip in the surface tone, as in the analysis panels.
    const float grip = 28.0f;
    const ImU32 colour = ui::ToU32(hot ? t.border : t.border_soft);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    if (beside) {
        const ImVec2 a(pos.x + size.x * 0.5f - 1.0f, pos.y + (size.y - grip) * 0.5f);
        draw->AddRectFilled(a, ImVec2(a.x + 2.0f, a.y + grip), colour, 1.0f);
    } else {
        const ImVec2 a(pos.x + (size.x - grip) * 0.5f, pos.y + size.y * 0.5f - 1.0f);
        draw->AddRectFilled(a, ImVec2(a.x + grip, a.y + 2.0f), colour, 1.0f);
    }
}

void AssetBrowserPanel::RefreshFilePreview(const AssetItem& item) {
    preview_path_ = item.absolute_path;
    preview_modified_ = item.modified_time;
    preview_dataset_name_.clear();
    preview_column_types_.clear();
    preview_registry_name_.clear();
    file_peek_ = assets::FilePeek{};

    // A dataset a Data Input already loaded: its first rows through the data
    // service (real types, any size, nothing read again).
    if (IsDatasetFile(item)) {
        auto& registry = DataRegistry::Instance();
        if (auto name = registry.FindTabularDatasetBySourcePath(preview_path_)) {
            preview_registry_name_ = *name;
            DataPreviewRequest request;
            request.dataset_name = *name;
            request.row_limit = kPreviewRows;
            const DataPreviewPage page = DataPreviewService::PreviewRegisteredTabular(registry, request);
            if (page.ok) {
                preview_dataset_name_ = *name;
                file_peek_.ok = true;
                file_peek_.table = true;
                file_peek_.total_rows = page.total_rows;
                for (const auto& column : page.schema) {
                    file_peek_.columns.push_back(column.name);
                    file_peek_.numeric.push_back(IsNumericType(column.type));
                    preview_column_types_.push_back(column.type);
                }
                file_peek_.rows = page.rows;
                return;
            }
        }
    }
    // Otherwise a bounded read of the file itself.
    file_peek_ = assets::PeekFile(preview_path_);
    for (const bool numeric : file_peek_.numeric) preview_column_types_.push_back(numeric ? "num" : "text");
}

void AssetBrowserPanel::RenderFilePreview(const AssetItem& item, const ImVec2& size) {
    bool stale = preview_path_ != item.absolute_path || preview_modified_ != item.modified_time;
    // A Data Input loading (or dropping) this file changes what the preview shows.
    if (!stale && IsDatasetFile(item)) {
        const auto loaded = DataRegistry::Instance().FindTabularDatasetBySourcePath(item.absolute_path);
        stale = loaded.value_or(std::string()) != preview_registry_name_;
    }
    if (stale) RefreshFilePreview(item);
    const ui::Tokens& t = ui::CurrentTokens();
    const assets::FileKind kind = assets::KindOfPath(item.absolute_path, false);

    // The pane sits one step above the window, like the analysis settings
    // column; no outline (owner rule 2026-10-01).
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.bg_raised, 0.4f));
    ImGui::BeginChild("##file_preview", size, ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();

    // Header: the kind's icon on a tinted tile, the name and its summary.
    const float tile = ImGui::GetFrameHeight() * 1.3f;
    {
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImVec4 colour = KindColour(kind);
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(p, ImVec2(p.x + tile, p.y + tile), ui::ToU32(ui::WithAlpha(colour, 0.16f)), 6.0f);
        const char* icon = KindIcon(kind, false);
        const ImVec2 icon_size = ImGui::CalcTextSize(icon);
        draw->AddText(ImVec2(p.x + (tile - icon_size.x) * 0.5f, p.y + (tile - icon_size.y) * 0.5f), ui::ToU32(colour), icon);
        ImGui::Dummy(ImVec2(tile, tile));
    }
    ImGui::SameLine();
    const float hide_width = ImGui::CalcTextSize("Hide").x;
    const float text_right = ImGui::GetContentRegionMax().x - hide_width - ImGui::GetStyle().ItemSpacing.x;
    ImGui::BeginGroup();
    ImGui::PushTextWrapPos(text_right);
    {
        ui::FontScope font(ui::Font::Medium);
        ImGui::TextUnformatted(item.name.c_str());
    }
    std::string summary = assets::SummaryLine(assets::TypeLabel(item.absolute_path, false),
                                              item.file_size > 0 ? FormatFileSize(item.file_size) : std::string(),
                                              file_peek_);
    if (!item.modified_time.empty()) summary += " \xC2\xB7 modified " + item.modified_time;
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::TextUnformatted(summary.c_str());
    ImGui::PopStyleColor();
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
    ImGui::SameLine(ImGui::GetContentRegionMax().x - hide_width);
    if (ui::LinkButton("Hide")) show_dataset_preview_ = false;

    if (!preview_dataset_name_.empty()) {
        ImGui::TextColored(t.success, ICON_FA_CIRCLE_CHECK " Loaded as '%s'", preview_dataset_name_.c_str());
    }
    ImGui::Spacing();

    // Body: rows, lines, a picture, or why there is nothing to show.
    if (!file_peek_.ok) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.warning, "%s", file_peek_.error.empty() ? "Could not read the file." : file_peek_.error.c_str());
        ImGui::PopTextWrapPos();
    } else if (file_peek_.table) {
        RenderPreviewTable();
    } else if (!file_peek_.lines.empty()) {
        RenderPreviewLines();
    } else if (!file_peek_.note.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", file_peek_.note.c_str());
        ImGui::PopTextWrapPos();
    } else if (kind == assets::FileKind::Image) {
        const float height = std::max(120.0f, ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() * 2.0f);
        ImGui::BeginChild("##image_preview", ImVec2(0.0f, height));
        quick_preview_renderer_.RenderFilePreview(item.absolute_path);
        ImGui::EndChild();
    } else {
        ImGui::TextDisabled(assets::PeeksAsText(kind) ? "The file is empty." : "No preview for this kind of file.");
    }

    RenderPreviewActions(item);
    ImGui::EndChild();
    ImGui::PopStyleVar();
}

void AssetBrowserPanel::RenderPreviewTable() {
    const ui::Tokens& t = ui::CurrentTokens();
    const int columns = static_cast<int>(file_peek_.columns.size());
    const int rows = static_cast<int>(file_peek_.rows.size());
    if (columns == 0) {
        ImGui::TextDisabled("No columns.");
        return;
    }
    // Room for the note and the actions below the table.
    const float reserve = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetFrameHeightWithSpacing() * 1.5f;
    const float row_height = ImGui::GetTextLineHeightWithSpacing() + 2.0f;
    const float wanted = row_height * static_cast<float>(rows + 1) + ImGui::GetStyle().ScrollbarSize + 4.0f;
    const float height = std::clamp(wanted, row_height * 3.0f, std::max(row_height * 3.0f, ImGui::GetContentRegionAvail().y - reserve));

    ImGui::PushStyleColor(ImGuiCol_TableRowBg, t.bg_window);
    ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ui::Mix(t.bg_window, t.bg_raised, 0.2f));
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoBordersInBody;
    if (ImGui::BeginTable("##peek_table", columns + 1, flags, ImVec2(0.0f, height))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed);
        // Each column fits its header (name and type) and its cells; ImGui's
        // auto-fit leaves the header row out and clipped the type label.
        const float gap = ImGui::GetStyle().ItemSpacing.x;
        for (int c = 0; c < columns; ++c) {
            const auto column = static_cast<size_t>(c);
            float width = ImGui::CalcTextSize(file_peek_.columns[column].c_str()).x;
            if (column < preview_column_types_.size())
                width += gap + ImGui::CalcTextSize(preview_column_types_[column].c_str()).x;
            for (const auto& row : file_peek_.rows)
                if (column < row.size()) width = std::max(width, ImGui::CalcTextSize(row[column].c_str()).x);
            ImGui::TableSetupColumn(file_peek_.columns[column].c_str(), ImGuiTableColumnFlags_WidthFixed,
                                    std::min(width, 320.0f));
        }
        // Header: the name, then its type in the info tone.
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("#");
        for (int c = 0; c < columns; ++c) {
            ImGui::TableSetColumnIndex(c + 1);
            ImGui::TextUnformatted(file_peek_.columns[static_cast<size_t>(c)].c_str());
            if (static_cast<size_t>(c) < preview_column_types_.size()) {
                ImGui::SameLine();
                ImGui::TextColored(t.info, "%s", preview_column_types_[static_cast<size_t>(c)].c_str());
            }
        }
        for (int r = 0; r < rows; ++r) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextDisabled("%d", r + 1);
            const auto& row = file_peek_.rows[static_cast<size_t>(r)];
            for (int c = 0; c < columns && static_cast<size_t>(c) < row.size(); ++c) {
                ImGui::TableSetColumnIndex(c + 1);
                ImGui::TextUnformatted(row[static_cast<size_t>(c)].c_str());
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleColor(2);

    // Where the rows came from.
    std::string note = "First " + std::to_string(rows);
    if (file_peek_.total_rows >= 0)
        note += std::string(" of ") + (file_peek_.rows_estimated ? "about " : "") + std::to_string(file_peek_.total_rows);
    note += preview_dataset_name_.empty() ? " rows, read from the file. Nothing is loaded until a Data Input applies it."
                                          : " rows of the loaded dataset.";
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", note.c_str());
    ImGui::PopTextWrapPos();
}

void AssetBrowserPanel::RenderPreviewLines() {
    const ui::Tokens& t = ui::CurrentTokens();
    const float reserve = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetFrameHeightWithSpacing() * 1.5f;
    const float line_height = ImGui::GetTextLineHeightWithSpacing();
    const float wanted = line_height * static_cast<float>(file_peek_.lines.size()) + ImGui::GetStyle().WindowPadding.y * 2.0f +
                         ImGui::GetStyle().ScrollbarSize;
    const float height = std::clamp(wanted, line_height * 3.0f, std::max(line_height * 3.0f, ImGui::GetContentRegionAvail().y - reserve));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
    ImGui::BeginChild("##peek_lines", ImVec2(0.0f, height), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::PopStyleColor();
    ImFont* code = gui::GetCodeFont();
    if (code) ImGui::PushFont(code);
    for (const auto& line : file_peek_.lines) ImGui::TextUnformatted(line.c_str());
    if (code) ImGui::PopFont();
    ImGui::EndChild();
    if (file_peek_.more_lines)
        ImGui::TextDisabled("First %d %ss.", static_cast<int>(file_peek_.lines.size()), file_peek_.line_noun.c_str());
}

void AssetBrowserPanel::RenderPreviewActions(const AssetItem& item) {
    const assets::FileKind kind = assets::KindOfPath(item.absolute_path, false);
    ImGui::Spacing();
    ui::FlowRow row(ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX());
    const auto primary = [&row](const char* label) {
        row.Next(ui::ButtonWidth(label, ui::ButtonSize::Regular));
        return ui::PrimaryButton(label);
    };
    const auto secondary = [&row](const char* label) {
        row.Next(ui::ButtonWidth(label, ui::ButtonSize::Regular));
        return ui::SecondaryButton(label, true, nullptr, ui::ButtonSize::Regular);
    };

    // A Data Input is how data enters the Engine; a loaded dataset opens in
    // Data Studio, whose query and profile scale to any size.
    if (!preview_dataset_name_.empty()) {
        if (primary("Open in Data Studio") && on_open_in_data_studio_) on_open_in_data_studio_(preview_dataset_name_);
        if (secondary("Create Data Input")) CreateDataInputFromItem(item);
    } else if (IsDatasetFile(item)) {
        if (primary("Create Data Input")) CreateDataInputFromItem(item);
    }
    if (kind == assets::FileKind::Graph && primary("Open in CyxWiz Studio") && on_open_in_node_editor_) {
        on_open_in_node_editor_(item.absolute_path);
    }
    if (kind == assets::FileKind::Python) {
        if (secondary("Open in Script Editor") && on_double_click_) on_double_click_(item);
    } else if ((kind == assets::FileKind::Table || assets::PeeksAsText(kind)) && on_double_click_) {
        row.Next(ImGui::CalcTextSize("Open as text").x);
        if (ui::LinkButton("Open as text")) on_double_click_(item);
    }
}

} // namespace cyxwiz
