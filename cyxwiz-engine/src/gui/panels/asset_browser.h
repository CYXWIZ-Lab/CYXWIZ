#pragma once

#include "../panel.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <set>

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

private:
    // UI Rendering
    void RenderToolbar();
    void RenderDirectoryView();
    void RenderAssetNode(AssetItem& item, int depth = 0);
    void RenderStatusBar();

    // Dialogs
    void RenderNewScriptDialog();
    void RenderNewFolderDialog();
    void RenderRenameDialog();
    void RenderDeleteConfirmDialog();

    // Asset operations
    void BuildDirectoryTree();
    AssetType DetermineAssetType(const std::string& path);
    const char* GetAssetIcon(AssetType type) const;

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
    void CreateNewScript();
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
    char new_script_name_[256];
    char new_folder_name_[256];

    // Callbacks
    AssetCallback on_double_click_;
    AssetCallback on_deleted_;

    // Clipboard state
    std::string clipboard_path_;      // Path of copied/cut file
    bool clipboard_is_cut_ = false;   // True if cut, false if copy

    // View options
    SortMode sort_mode_ = SortMode::Name;
    bool show_hidden_files_ = false;  // Show files starting with .

    // Deferred operations (to avoid invalidating iterators)
    bool needs_refresh_ = false;

    // Force tree state update (for expand/collapse all)
    bool force_tree_state_ = false;
};

} // namespace cyxwiz
