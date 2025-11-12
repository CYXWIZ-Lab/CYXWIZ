#pragma once

#include "panel.h"
#include <vector>
#include <string>
#include <memory>

namespace gui {

/**
 * Asset Browser panel - hierarchical tree view of project assets.
 * Displays datasets, models, training runs, scripts, checkpoints, etc.
 */
class AssetBrowser : public Panel {
public:
    AssetBrowser();
    ~AssetBrowser() override;

    void Render() override;
    const char* GetName() const override { return "Asset Browser"; }

    // Asset management
    void RefreshAssets();
    void AddAsset(const std::string& category, const std::string& name);

private:
    struct AssetNode {
        std::string name;
        std::string path;
        bool is_folder;
        std::vector<std::unique_ptr<AssetNode>> children;

        AssetNode(const std::string& n, bool folder = false)
            : name(n), is_folder(folder) {}
    };

    void RenderNode(AssetNode* node);
    void RenderContextMenu();
    void InitializeDefaultStructure();

    std::unique_ptr<AssetNode> root_;
    AssetNode* selected_node_;
    char search_buffer_[256];
};

} // namespace gui
