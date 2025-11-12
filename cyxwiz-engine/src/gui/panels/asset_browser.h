#pragma once

#include "../panel.h"
#include <string>
#include <vector>
#include <memory>

namespace cyxwiz {

/**
 * Asset Browser Panel
 * Left-side tree view for browsing datasets, models, training runs, scripts, checkpoints, plugins
 */
class AssetBrowserPanel : public Panel {
public:
    AssetBrowserPanel();
    ~AssetBrowserPanel() override = default;

    void Render() override;

private:
    // Asset types
    enum class AssetType {
        Dataset,
        Model,
        TrainingRun,
        Script,
        Checkpoint,
        Plugin,
        Folder
    };

    // Asset item in tree
    struct AssetItem {
        std::string name;
        AssetType type;
        std::string path;
        std::vector<std::unique_ptr<AssetItem>> children;
        bool is_expanded = false;
    };

    // Render functions
    void RenderSearchBar();
    void RenderAssetTree();
    void RenderAssetNode(AssetItem* item);
    void RenderContextMenu(AssetItem* item);

    // Asset operations
    void LoadAssets();
    void FilterAssets(const std::string& filter);
    void OnAssetDoubleClick(AssetItem* item);
    void OnAssetDragStart(AssetItem* item);

    // Icon helpers
    const char* GetAssetIcon(AssetType type) const;

    // Data
    std::vector<std::unique_ptr<AssetItem>> root_items_;
    std::vector<AssetItem*> filtered_items_;

    // UI state
    char search_buffer_[256];
    AssetItem* selected_item_;
    AssetItem* context_menu_item_;
    bool show_context_menu_;
};

} // namespace cyxwiz
