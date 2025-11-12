#include "asset_browser.h"
#include <imgui.h>
#include <cstring>

namespace cyxwiz {

AssetBrowserPanel::AssetBrowserPanel()
    : Panel("Asset Browser", true)
    , selected_item_(nullptr)
    , context_menu_item_(nullptr)
    , show_context_menu_(false)
{
    std::memset(search_buffer_, 0, sizeof(search_buffer_));
    LoadAssets();
}

void AssetBrowserPanel::Render() {
    if (!visible_) return;

    ImGui::Begin(GetName(), &visible_);

    // Search bar at top
    RenderSearchBar();

    ImGui::Separator();

    // Asset tree
    RenderAssetTree();

    // Context menu
    if (show_context_menu_ && context_menu_item_) {
        RenderContextMenu(context_menu_item_);
    }

    ImGui::End();
}

void AssetBrowserPanel::RenderSearchBar() {
    ImGui::PushItemWidth(-1.0f);
    if (ImGui::InputTextWithHint("##search", "Search assets...", search_buffer_, sizeof(search_buffer_))) {
        FilterAssets(search_buffer_);
    }
    ImGui::PopItemWidth();
}

void AssetBrowserPanel::RenderAssetTree() {
    // Use child window for scrollable area
    ImGui::BeginChild("AssetTreeRegion", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

    // Render each root category
    for (auto& item : root_items_) {
        RenderAssetNode(item.get());
    }

    ImGui::EndChild();
}

void AssetBrowserPanel::RenderAssetNode(AssetItem* item) {
    if (!item) return;

    // Set up tree node flags
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;

    // Add leaf flag if no children
    if (item->children.empty()) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    // Add selected flag if this is the selected item
    if (item == selected_item_) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    // Get icon for asset type
    const char* icon = GetAssetIcon(item->type);

    // Render tree node with icon
    bool node_open = ImGui::TreeNodeEx(item, flags, "%s %s", icon, item->name.c_str());

    // Handle selection
    if (ImGui::IsItemClicked()) {
        selected_item_ = item;
    }

    // Handle double-click
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
        OnAssetDoubleClick(item);
    }

    // Handle right-click context menu
    if (ImGui::IsItemClicked(1)) {
        context_menu_item_ = item;
        show_context_menu_ = true;
        ImGui::OpenPopup("AssetContextMenu");
    }

    // Handle drag and drop
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
        // Set payload to carry asset path
        ImGui::SetDragDropPayload("ASSET_PATH", item->path.c_str(), item->path.size() + 1);
        ImGui::Text("Drag %s", item->name.c_str());
        ImGui::EndDragDropSource();
        OnAssetDragStart(item);
    }

    // Recursively render children if node is open
    // IMPORTANT: Only call TreePop() when TreeNodeEx actually pushed to the stack
    // NoTreePushOnOpen is set when children.empty(), so only pop when !children.empty()
    if (node_open && !item->children.empty()) {
        for (auto& child : item->children) {
            RenderAssetNode(child.get());
        }
        ImGui::TreePop();
    }
}

void AssetBrowserPanel::RenderContextMenu(AssetItem* item) {
    if (ImGui::BeginPopup("AssetContextMenu")) {
        ImGui::Text("%s", item->name.c_str());
        ImGui::Separator();

        if (ImGui::MenuItem("Open")) {
            OnAssetDoubleClick(item);
        }

        if (ImGui::MenuItem("Rename")) {
            // TODO: Implement rename dialog
        }

        if (ImGui::MenuItem("Delete")) {
            // TODO: Implement delete confirmation
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Show in Explorer")) {
            // TODO: Open OS file browser
        }

        if (ImGui::MenuItem("Copy Path")) {
            ImGui::SetClipboardText(item->path.c_str());
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Properties")) {
            // TODO: Show properties dialog
        }

        ImGui::EndPopup();
    } else {
        show_context_menu_ = false;
    }
}

void AssetBrowserPanel::LoadAssets() {
    // Create root categories
    auto datasets = std::make_unique<AssetItem>();
    datasets->name = "Datasets";
    datasets->type = AssetType::Folder;
    datasets->path = "assets/datasets";

    // Add sample dataset items
    auto dataset1 = std::make_unique<AssetItem>();
    dataset1->name = "MNIST";
    dataset1->type = AssetType::Dataset;
    dataset1->path = "assets/datasets/mnist";
    datasets->children.push_back(std::move(dataset1));

    auto dataset2 = std::make_unique<AssetItem>();
    dataset2->name = "CIFAR-10";
    dataset2->type = AssetType::Dataset;
    dataset2->path = "assets/datasets/cifar10";
    datasets->children.push_back(std::move(dataset2));

    root_items_.push_back(std::move(datasets));

    // Models category
    auto models = std::make_unique<AssetItem>();
    models->name = "Models";
    models->type = AssetType::Folder;
    models->path = "assets/models";

    auto model1 = std::make_unique<AssetItem>();
    model1->name = "ResNet50";
    model1->type = AssetType::Model;
    model1->path = "assets/models/resnet50.h5";
    models->children.push_back(std::move(model1));

    root_items_.push_back(std::move(models));

    // Training Runs category
    auto training_runs = std::make_unique<AssetItem>();
    training_runs->name = "Training Runs";
    training_runs->type = AssetType::Folder;
    training_runs->path = "assets/training";

    auto run1 = std::make_unique<AssetItem>();
    run1->name = "experiment_001";
    run1->type = AssetType::TrainingRun;
    run1->path = "assets/training/experiment_001";
    training_runs->children.push_back(std::move(run1));

    root_items_.push_back(std::move(training_runs));

    // Scripts category
    auto scripts = std::make_unique<AssetItem>();
    scripts->name = "Scripts";
    scripts->type = AssetType::Folder;
    scripts->path = "assets/scripts";

    auto script1 = std::make_unique<AssetItem>();
    script1->name = "preprocess.py";
    script1->type = AssetType::Script;
    script1->path = "assets/scripts/preprocess.py";
    scripts->children.push_back(std::move(script1));

    root_items_.push_back(std::move(scripts));

    // Checkpoints category
    auto checkpoints = std::make_unique<AssetItem>();
    checkpoints->name = "Checkpoints";
    checkpoints->type = AssetType::Folder;
    checkpoints->path = "assets/checkpoints";
    root_items_.push_back(std::move(checkpoints));

    // Plugins category
    auto plugins = std::make_unique<AssetItem>();
    plugins->name = "Plugins";
    plugins->type = AssetType::Folder;
    plugins->path = "assets/plugins";
    root_items_.push_back(std::move(plugins));
}

void AssetBrowserPanel::FilterAssets(const std::string& filter) {
    // TODO: Implement filtering logic
    // For now, just clear filtered items if search is empty
    if (filter.empty()) {
        filtered_items_.clear();
    }
}

void AssetBrowserPanel::OnAssetDoubleClick(AssetItem* item) {
    // TODO: Implement asset opening logic based on type
    // For now, just expand/collapse folders
    if (item->type == AssetType::Folder) {
        item->is_expanded = !item->is_expanded;
    }
}

void AssetBrowserPanel::OnAssetDragStart(AssetItem* item) {
    // TODO: Implement drag start logic
    // Could highlight drop zones, etc.
}

const char* AssetBrowserPanel::GetAssetIcon(AssetType type) const {
    switch (type) {
        case AssetType::Dataset:     return "\xef\x80\x87"; //
        case AssetType::Model:       return "\xef\x88\x9e"; //
        case AssetType::TrainingRun: return "\xef\x88\x91"; //
        case AssetType::Script:      return "\xef\x87\xab"; //
        case AssetType::Checkpoint:  return "\xef\x80\x87"; //
        case AssetType::Plugin:      return "\xef\x84\xb9"; //
        case AssetType::Folder:      return "\xef\x81\xbb"; //
        default:                     return "\xef\x85\xa8"; //
    }
}

} // namespace cyxwiz
