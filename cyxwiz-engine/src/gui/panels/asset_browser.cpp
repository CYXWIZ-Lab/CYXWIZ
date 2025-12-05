#include "asset_browser.h"
#include "../icons.h"
#include "../../core/project_manager.h"
#include <imgui.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace cyxwiz {

AssetBrowserPanel::AssetBrowserPanel()
    : Panel("Asset Browser", true)
    , last_clicked_item_(nullptr)
    , context_menu_item_(nullptr)
    , show_rename_dialog_(false)
    , show_delete_confirm_(false)
    , show_new_script_dialog_(false)
    , show_new_folder_dialog_(false)
{
    std::memset(search_buffer_, 0, sizeof(search_buffer_));
    std::memset(rename_buffer_, 0, sizeof(rename_buffer_));
    std::memset(new_script_name_, 0, sizeof(new_script_name_));
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
        std::lock_guard<std::mutex> lock(pending_tree_mutex_);
        if (pending_directory_root_) {
            directory_root_ = std::move(pending_directory_root_);
            SortAssets();
            ClearSelection();
        }
        scan_completed_.store(false);
        is_scanning_directory_.store(false);
    }

    ImGui::Begin(GetName(), &visible_);

    // Toolbar at top
    RenderToolbar();

    ImGui::Separator();

    // Keyboard shortcuts (when window is focused)
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
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
            show_new_script_dialog_ = true;
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

    // Main content area - directory view with optional preview pane
    float preview_width = 0.0f;

    // Check if any selected item is a dataset for preview pane
    bool has_dataset_selected = false;
    for (auto* item : selected_items_) {
        if (item && IsDatasetFile(*item)) {
            has_dataset_selected = true;
            break;
        }
    }

    if (show_dataset_preview_ && has_dataset_selected) {
        preview_width = 200.0f;
    }

    ImGui::BeginChild("AssetContentRegion", ImVec2(-preview_width, -ImGui::GetFrameHeightWithSpacing()), false);

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
                show_new_script_dialog_ = true;
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

    // Render dataset preview pane if needed
    if (show_dataset_preview_ && has_dataset_selected) {
        ImGui::SameLine();
        RenderDatasetPreview();
    }

    // Status bar at bottom
    RenderStatusBar();

    // Dialogs (these are modal popups, render at parent level)
    RenderNewScriptDialog();
    RenderNewFolderDialog();
    RenderRenameDialog();
    RenderDeleteConfirmDialog();

    ImGui::End();
}

void AssetBrowserPanel::SetProjectRoot(const std::string& root) {
    project_root_ = root;
    Refresh();
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

    spdlog::info("Refreshing asset browser async for project: {}", project_root_);

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

            // Count total entries first for progress reporting
            size_t total_entries = 0;
            size_t processed_entries = 0;

            try {
                for (auto it = fs::recursive_directory_iterator(project_root,
                         fs::directory_options::skip_permission_denied);
                     it != fs::recursive_directory_iterator(); ++it) {
                    total_entries++;
                }
            } catch (...) {
                // Ignore errors in counting
            }

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

                        // Update progress
                        processed_entries++;
                        if (total_entries > 0 && processed_entries % 50 == 0) {
                            float progress = static_cast<float>(processed_entries) / static_cast<float>(total_entries);
                            task.ReportProgress(progress * 0.9f, "Scanning files...");
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
    // Show loading indicator while scanning
    if (is_scanning_directory_.load()) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), ICON_FA_SPINNER " Scanning project files...");

        // Show existing tree while scanning (if any)
        if (!directory_root_) {
            return;
        }
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

    // Get icon for asset type
    const char* icon = GetAssetIcon(item.type);

    // For directories, show item count
    std::string label = item.name;
    if (item.is_directory && !item.children.empty()) {
        int count = static_cast<int>(item.children.size());
        label += " (" + std::to_string(count) + ")";
    }

    // Render tree node with icon
    void* node_id = &item;
    bool node_open = ImGui::TreeNodeEx(node_id, flags, "%s %s", icon, label.c_str());

    // Sync our is_expanded state with ImGui's actual state (for directories)
    if (item.is_directory && !is_leaf) {
        item.is_expanded = node_open;
    }

    // Show tooltip with file info on hover
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
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

    // Handle selection with Ctrl/Shift modifiers for multi-select
    if (ImGui::IsItemClicked(0)) {
        bool ctrl_held = ImGui::GetIO().KeyCtrl;
        bool shift_held = ImGui::GetIO().KeyShift;
        SelectItem(&item, ctrl_held, shift_held);
    }

    // Handle double-click for files (directories are handled by TreeNode arrow)
    if (!item.is_directory && ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
        // Special handling for graph files - open in Node Editor
        if (item.type == AssetType::Graph) {
            if (on_open_in_node_editor_) {
                on_open_in_node_editor_(item.absolute_path);
            }
        }
        // Special handling for dataset files - load into DataRegistry (async with loading indicator)
        else if (IsDatasetFile(item)) {
            LoadDatasetFromItemAsync(item);
        }
        // Fire callback for file double-click
        if (on_double_click_) {
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
                show_new_script_dialog_ = true;
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

            // View in Table (for tabular data files only)
            if (IsTableViewableFile(item)) {
                if (ImGui::MenuItem(ICON_FA_TABLE " View in Table")) {
                    if (on_view_in_table_) {
                        on_view_in_table_(item.absolute_path);
                    }
                }
            }

            // Load Dataset (for dataset files only - async with loading indicator)
            if (IsDatasetFile(item)) {
                if (ImGui::MenuItem(ICON_FA_DATABASE " Load Dataset")) {
                    LoadDatasetFromItemAsync(item);
                }
            }

            // Open in Node Editor (for .cyxgraph files only)
            if (item.type == AssetType::Graph) {
                if (ImGui::MenuItem(ICON_FA_DIAGRAM_PROJECT " Open in Node Editor")) {
                    if (on_open_in_node_editor_) {
                        on_open_in_node_editor_(item.absolute_path);
                    }
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
        ImGui::Text("%s %s", item.is_directory ? ICON_FA_FOLDER : GetAssetIcon(item.type), item.name.c_str());
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
}

void AssetBrowserPanel::RenderNewScriptDialog() {
    if (!show_new_script_dialog_) return;

    auto& pm = ProjectManager::Instance();

    // Show warning if no project is open
    if (!pm.HasActiveProject()) {
        ImGui::OpenPopup("No Project Open");

        if (ImGui::BeginPopupModal("No Project Open", &show_new_script_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), ICON_FA_TRIANGLE_EXCLAMATION " Warning");
            ImGui::Separator();
            ImGui::Text("Cannot create a new script without an open project.");
            ImGui::Text("Please create or open a project first.");
            ImGui::Separator();

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                show_new_script_dialog_ = false;
            }
            ImGui::EndPopup();
        }
        return;
    }

    ImGui::OpenPopup("New Script");

    if (ImGui::BeginPopupModal("New Script", &show_new_script_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Create a new CyxWiz script");
        ImGui::Separator();

        ImGui::InputTextWithHint("##script_name", "script_name.cyx", new_script_name_, sizeof(new_script_name_));

        ImGui::Separator();

        if (ImGui::Button("Create", ImVec2(120, 0))) {
            CreateNewScript();
            show_new_script_dialog_ = false;
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_new_script_dialog_ = false;
        }

        ImGui::EndPopup();
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
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "This action cannot be undone");

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
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), ICON_FA_TRIANGLE_EXCLAMATION " Warning");
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

    // Scripts
    if (ext == ".py" || ext == ".cyx") return AssetType::Script;

    // Models
    if (ext == ".h5" || ext == ".onnx" || ext == ".pt" || ext == ".safetensors" || ext == ".bin")
        return AssetType::Model;

    // Datasets
    if (ext == ".csv" || ext == ".json" || ext == ".parquet" || ext == ".arrow" || ext == ".txt")
        return AssetType::Dataset;

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

const char* AssetBrowserPanel::GetAssetIcon(AssetType type) const {
    switch (type) {
        case AssetType::Script:      return ICON_FA_FILE_CODE;
        case AssetType::Model:       return ICON_FA_BRAIN;
        case AssetType::Dataset:     return ICON_FA_DATABASE;
        case AssetType::Checkpoint:  return ICON_FA_FLOPPY_DISK;
        case AssetType::Export:      return ICON_FA_DOWNLOAD;
        case AssetType::Plugin:      return ICON_FA_PLUG;
        case AssetType::Graph:       return ICON_FA_DIAGRAM_PROJECT;
        case AssetType::Folder:      return ICON_FA_FOLDER;
        case AssetType::Unknown:     return ICON_FA_FILE;
        default:                     return ICON_FA_FILE;
    }
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

void AssetBrowserPanel::CreateNewScript() {
    auto& pm = ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        spdlog::warn("Cannot create script: no active project");
        return;
    }

    std::string script_name = new_script_name_;
    if (script_name.empty()) {
        spdlog::warn("Script name is empty");
        return;
    }

    // Ensure .cyx extension (default script extension)
    if (!script_name.ends_with(".py") && !script_name.ends_with(".cyx")) {
        script_name += ".cyx";
    }

    // Determine target directory - use context_menu_item if it's a directory, otherwise use scripts path
    fs::path target_dir;
    if (context_menu_item_ && context_menu_item_->is_directory) {
        target_dir = context_menu_item_->absolute_path;
    } else if (context_menu_item_) {
        target_dir = fs::path(context_menu_item_->absolute_path).parent_path();
    } else {
        target_dir = pm.GetScriptsPath();
    }

    std::string script_path = (target_dir / script_name).string();

    // Create file
    try {
        std::ofstream file(script_path);
        file << "# " << script_name << "\n";
        file << "# Created with CyxWiz Engine\n\n";
        file << "# CyxWiz Script\n";
        file << "# Use this script to define your ML training pipeline\n\n";
        file << "def main():\n";
        file << "    pass\n\n";
        file << "if __name__ == '__main__':\n";
        file << "    main()\n";
        file.close();

        spdlog::info("Created new script: {}", script_path);

        // Refresh to show new file
        Refresh();

        // Auto-open the new script in editor if callback is set
        if (on_double_click_) {
            AssetItem new_item;
            new_item.name = script_name;
            new_item.absolute_path = script_path;
            new_item.relative_path = pm.MakeRelativePath(script_path);
            new_item.type = AssetType::Script;
            new_item.is_directory = false;
            on_double_click_(new_item);
        }

        // Clear input
        std::memset(new_script_name_, 0, sizeof(new_script_name_));

    } catch (const std::exception& e) {
        spdlog::error("Failed to create script: {}", e.what());
    }
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

#ifdef _WIN32
    std::string command = "explorer.exe /select,"" + context_menu_item_->absolute_path + """;
    system(command.c_str());
#elif defined(__APPLE__)
    std::string command = "open -R "" + context_menu_item_->absolute_path + """;
    system(command.c_str());
#else
    // Linux - open parent directory
    std::string dir = fs::path(context_menu_item_->absolute_path).parent_path().string();
    std::string command = "xdg-open "" + dir + """;
    system(command.c_str());
#endif
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

bool AssetBrowserPanel::IsTableViewableFile(const AssetItem& item) const {
    if (item.is_directory) return false;

    std::string ext = fs::path(item.absolute_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Tabular data files that can be viewed in TableViewer
    return ext == ".csv" || ext == ".tsv" || ext == ".txt" ||
           ext == ".h5" || ext == ".hdf5" ||
           ext == ".xlsx" || ext == ".xls" ||
           ext == ".json" || ext == ".parquet";
}

void AssetBrowserPanel::LoadDatasetFromItem(const AssetItem& item) {
    // Use async loading by default for better UX
    LoadDatasetFromItemAsync(item);
}

void AssetBrowserPanel::LoadDatasetFromItemAsync(const AssetItem& item) {
    if (!IsDatasetFile(item)) return;

    if (is_loading_dataset_.load()) {
        spdlog::warn("Already loading a dataset, please wait...");
        return;
    }

    spdlog::info("Loading dataset async from Asset Browser: {}", item.absolute_path);

    is_loading_dataset_.store(true);
    loading_dataset_path_ = item.absolute_path;

    // Capture path and callback by value to avoid dangling references
    std::string path = item.absolute_path;
    DatasetCallback callback = on_dataset_loaded_;

    loading_task_id_ = AsyncTaskManager::Instance().RunAsync(
        "Loading: " + item.name,
        [this, path, callback](LambdaTask& task) {
            task.ReportProgress(0.1f, "Opening file...");

            auto& registry = DataRegistry::Instance();

            task.ReportProgress(0.3f, "Parsing data...");
            if (task.ShouldStop()) return;

            DatasetHandle handle = registry.LoadDataset(path);

            task.ReportProgress(0.9f, "Finalizing...");
            if (task.ShouldStop()) return;

            if (handle.IsValid()) {
                spdlog::info("Dataset loaded successfully: {} ({} samples)",
                    handle.GetName(), handle.GetInfo().num_samples);
                task.MarkCompleted();
            } else {
                task.MarkFailed("Failed to load dataset");
            }
        },
        nullptr,  // No progress callback needed - indicator is shown via is_loading_dataset_
        [this, path, callback](bool success, const std::string& error) {
            is_loading_dataset_.store(false);

            if (success && callback) {
                // Reload the handle on main thread and fire callback
                auto& registry = DataRegistry::Instance();
                auto handle = registry.GetDataset(path);
                if (handle.IsValid()) {
                    callback(path, handle);
                }
            } else if (!success) {
                spdlog::error("Async dataset load failed: {}", error);
            }
        }
    );
}

void AssetBrowserPanel::RenderDatasetPreview() {
    if (!show_dataset_preview_) return;

    // Get first selected item that is a dataset
    AssetItem* dataset_item = nullptr;
    for (auto* item : selected_items_) {
        if (item && IsDatasetFile(*item)) {
            dataset_item = item;
            break;
        }
    }

    if (!dataset_item) return;

    // Check if we need to update preview
    if (preview_path_ != dataset_item->absolute_path) {
        preview_path_ = dataset_item->absolute_path;
        auto& registry = DataRegistry::Instance();
        current_preview_ = registry.GetPreview(preview_path_, 5);
    }

    // Render preview in a side panel
    ImGui::BeginChild("DatasetPreviewPane", ImVec2(200, 0), true);

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), ICON_FA_DATABASE " Dataset Preview");
    ImGui::Separator();

    ImGui::Text("Type: %s", DataRegistry::TypeToString(current_preview_.type).c_str());
    ImGui::Text("Samples: %zu", current_preview_.num_samples);
    ImGui::Text("Classes: %zu", current_preview_.num_classes);

    if (!current_preview_.shape.empty()) {
        std::string shape_str = "[";
        for (size_t i = 0; i < current_preview_.shape.size(); ++i) {
            if (i > 0) shape_str += ", ";
            shape_str += std::to_string(current_preview_.shape[i]);
        }
        shape_str += "]";
        ImGui::Text("Shape: %s", shape_str.c_str());
    }

    ImGui::Text("Size: %s", FormatFileSize(current_preview_.file_size).c_str());

    ImGui::Separator();

    // Show column names for tabular data
    if (!current_preview_.columns.empty()) {
        ImGui::Text("Columns:");
        for (size_t i = 0; i < current_preview_.columns.size() && i < 5; ++i) {
            ImGui::BulletText("%s", current_preview_.columns[i].c_str());
        }
        if (current_preview_.columns.size() > 5) {
            ImGui::Text("  ... and %zu more", current_preview_.columns.size() - 5);
        }
    }

    ImGui::Separator();

    // Show loading indicator or load button
    if (is_loading_dataset_.load()) {
        // Animated loading spinner
        float time = static_cast<float>(ImGui::GetTime());
        const char* spinner_chars[] = {"|", "/", "-", "\\"};
        int spinner_idx = static_cast<int>(time * 8) % 4;

        ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "%s Loading...", spinner_chars[spinner_idx]);

        if (ImGui::Button("Cancel", ImVec2(-1, 0))) {
            if (loading_task_id_ != 0) {
                AsyncTaskManager::Instance().Cancel(loading_task_id_);
                is_loading_dataset_.store(false);
            }
        }
    } else {
        // Load button
        if (ImGui::Button("Load Dataset", ImVec2(-1, 0))) {
            LoadDatasetFromItemAsync(*dataset_item);
        }
    }

    ImGui::EndChild();
}

} // namespace cyxwiz

