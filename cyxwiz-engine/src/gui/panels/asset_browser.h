#pragma once

#include "../dialogs/new_script_dialog.h"
#include <chrono>
#include <future>
#include "../panel.h"
#include "../../core/data_registry.h"
#include "../../core/asset_preview.h"
#include <imgui.h>
#include "../../core/data_preview_service.h"
#include "../../core/async_task_manager.h"
#include "../preview/preview_renderer.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <set>
#include <atomic>

namespace cyxwiz {

/**
 * Asset Browser Panel - Directory-based filesystem browser
 * Shows actual filesystem structure of the project
 */
class AssetBrowserPanel : public Panel {
public:
    // Asset types for icon display
    enum class AssetType {
        Script,         // .py, .cyx
        Model,          // .h5, .onnx, .pt, .safetensors, .bin
        Dataset,        // .csv, .json, .parquet, .h5, .arrow, .txt
        Checkpoint,     // .ckpt, .pt, .checkpoint
        Export,         // .onnx, .gguf, .lora
        Plugin,         // .dll, .so, .dylib
        Graph,          // .cyxgraph - Node editor graphs
        Folder,         // Directory
        Unknown         // Unrecognized file
    };

    // Sort options
    enum class SortMode {
        Name,           // Alphabetical by name
        Date,           // By modified date (newest first)
        Size,           // By file size (largest first)
        Type            // By file type
    };

    // Asset item in tree
    struct AssetItem {
        std::string name;               // Display name
        std::string relative_path;      // Path relative to project root
        std::string absolute_path;      // Full filesystem path
        AssetType type;
        bool is_directory;
        bool is_expanded = false;
        std::vector<std::unique_ptr<AssetItem>> children;

        // File metadata
        std::uintmax_t file_size = 0;   // Size in bytes
        std::string modified_time;       // Last modified timestamp
    };

    // Callback types
    using AssetCallback = std::function<void(const AssetItem&)>;

    AssetBrowserPanel();
    ~AssetBrowserPanel() override = default;

    void Render() override;

    // Project integration
    void SetProjectRoot(const std::string& root);
    void Refresh();                 // Rescan filesystem
    void Clear();                   // Clear all assets

    // Callbacks for asset operations
    void SetOnAssetDoubleClick(AssetCallback callback) { on_double_click_ = std::move(callback); }
    void SetOnAssetDeleted(AssetCallback callback) { on_deleted_ = std::move(callback); }

    // Dataset callback - called when a dataset is double-clicked
    using DataInputSourceCallback = std::function<void(const std::string& path)>;
    void SetOnCreateDataInput(DataInputSourceCallback callback) { on_create_data_input_ = std::move(callback); }

    // Graph file callback - called when user opens a .cyxgraph file
    using GraphCallback = std::function<void(const std::string& path)>;
    void SetOnOpenInNodeEditor(GraphCallback callback) { on_open_in_node_editor_ = std::move(callback); }

    // Enable/disable the file preview pane
    void SetShowDatasetPreview(bool show) { show_dataset_preview_ = show; }

    // A loaded dataset's "Open in Data Studio" (its registry name).
    using DatasetNameCallback = std::function<void(const std::string& dataset_name)>;
    void SetOnOpenInDataStudio(DatasetNameCallback callback) { on_open_in_data_studio_ = std::move(callback); }

private:
    enum class AssetActionStatus { None, Success, Error };

    // UI Rendering
    void RenderToolbar();
    void RenderDirectoryView();
    void RenderAssetNode(AssetItem& item, int depth = 0);
    void RenderStatusBar();
    // The selected file's light preview (TOFIX140): beside the tree in a wide
    // panel, below it in a narrow one. A bounded peek only; data enters the
    // Engine through a Data Input.
    void RenderFilePreview(const AssetItem& item, const ImVec2& size);
    void RefreshFilePreview(const AssetItem& item);
    void RenderPreviewTable();
    void RenderPreviewLines();
    void RenderPreviewActions(const AssetItem& item);
    void PreviewSplitter(bool beside);

    // Dataset helpers
    bool IsDatasetFile(const AssetItem& item) const;
    void CreateDataInputFromItem(const AssetItem& item);

    // Dialogs
    void RenderNewScriptDialog();
    void RenderNewFolderDialog();
    void RenderRenameDialog();
    void RenderDeleteConfirmDialog();

    // Asset operations
    void BuildDirectoryTree();
    AssetType DetermineAssetType(const std::string& path);

    // Helper to count total items
    int CountItems() const;

    // Helper to format file size
    std::string FormatFileSize(std::uintmax_t size) const;

    // Expand/collapse all helpers
    void ExpandAll();
    void CollapseAll();
    void SetExpandedRecursive(AssetItem* item, bool expanded);

    // Sorting helpers
    void SortAssets();
    void SortChildren(AssetItem* parent);

    // Selection helpers
    bool IsSelected(AssetItem* item) const;
    void SelectItem(AssetItem* item, bool ctrl_held, bool shift_held);
    void ClearSelection();
    void SelectRange(AssetItem* from, AssetItem* to);
    void GetFlatItemList(AssetItem* root, std::vector<AssetItem*>& out_list);

    // Context menu actions
    void CreateNewFolder();
    void DeleteSelectedAsset();
    void RenameSelectedAsset();
    void OpenInExplorer();
    void OpenInTerminal();

    // Clipboard operations
    void CopySelectedAsset();
    void CutSelectedAsset();
    void PasteAsset();

    // Search/Filter
    void FilterAssets(const std::string& query);
    bool MatchesSearch(const AssetItem& item, const std::string& query) const;
    bool HasMatchingChildren(const AssetItem& item, const std::string& query) const;

    // State
    std::string project_root_;
    char search_buffer_[256];
    std::string current_search_query_;  // Cached lowercase search query

    // Asset tree
    std::unique_ptr<AssetItem> directory_root_;

    // Selection (multi-select support)
    std::set<AssetItem*> selected_items_;  // Set of selected items
    AssetItem* last_clicked_item_ = nullptr;  // For shift-click range selection
    AssetItem* context_menu_item_;

    // Dialogs
    bool show_rename_dialog_;
    bool show_delete_confirm_;
    bool show_new_script_dialog_;
    bool show_new_folder_dialog_;
    char rename_buffer_[256];
    NewScriptDialog new_script_dialog_;  // New > Script (shared dialog)
    char new_folder_name_[256];

    // Callbacks
    AssetCallback on_double_click_;
    AssetCallback on_deleted_;
    DataInputSourceCallback on_create_data_input_;

    GraphCallback on_open_in_node_editor_;

    DatasetNameCallback on_open_in_data_studio_;

    // File preview state: the peek (a loaded dataset's page is shown through
    // it too), its column type labels, and the pane's size per placement.
    bool show_dataset_preview_ = true;
    std::string preview_path_;
    std::string preview_modified_;
    std::string preview_dataset_name_;
    std::string preview_registry_name_;  // the registry's name for the file when last read
    assets::FilePeek file_peek_;
    std::vector<std::string> preview_column_types_;
    float preview_side_width_ = 400.0f;
    float preview_below_height_ = 320.0f;
    AssetActionStatus asset_action_status_ = AssetActionStatus::None;
    std::string asset_action_message_;

    // Clipboard state
    std::string clipboard_path_;      // Path of copied/cut file
    bool clipboard_is_cut_ = false;   // True if cut, false if copy

    // View options
    SortMode sort_mode_ = SortMode::Name;
    bool show_hidden_files_ = false;  // Show files starting with .

    // Deferred operations (to avoid invalidating iterators)
    bool needs_refresh_ = false;

    // Rescans keep the user's view: open folders, selection and the context
    // item are carried over by path when a new tree replaces the old one.
    void ApplyRescannedTree(std::unique_ptr<AssetItem> new_root);

    // Changes made outside the browser (exports, scripts, other tools) are
    // picked up by a cheap background fingerprint of the project every few
    // seconds; a rescan runs only when the fingerprint changes.
    void PollForExternalChanges();
    std::future<std::uint64_t> change_probe_;
    std::string change_probe_root_;
    std::uint64_t last_disk_signature_ = 0;
    bool have_disk_signature_ = false;
    std::chrono::steady_clock::time_point next_change_probe_{};

    // Force tree state update (for expand/collapse all)
    bool force_tree_state_ = false;

    // Async directory scanning state
    std::atomic<bool> is_scanning_directory_{false};
    uint64_t scanning_task_id_ = 0;
    std::unique_ptr<AssetItem> pending_directory_root_;  // Built in background thread
    std::mutex pending_tree_mutex_;  // Protects pending_directory_root_
    std::atomic<bool> scan_completed_{false};  // Signal that scan is done

    // Quick Preview popup state
    bool show_quick_preview_ = false;
    std::string quick_preview_path_;
    gui::PreviewRenderer quick_preview_renderer_;

    // Helper methods
    void RenderQuickPreviewPopup();
    bool IsPreviewableFile(const AssetItem& item) const;
};

} // namespace cyxwiz
