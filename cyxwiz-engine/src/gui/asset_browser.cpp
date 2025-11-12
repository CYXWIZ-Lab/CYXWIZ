#include "asset_browser.h"
#include <imgui.h>
#include <cstring>
#include <spdlog/spdlog.h>

namespace gui {

AssetBrowser::AssetBrowser()
    : selected_node_(nullptr) {
    memset(search_buffer_, 0, sizeof(search_buffer_));
    InitializeDefaultStructure();
}

AssetBrowser::~AssetBrowser() = default;

void AssetBrowser::Render() {
    if (!visible_) return;

    ImGui::Begin(GetName(), &visible_);

    // Search bar
    ImGui::PushItemWidth(-1);
    if (ImGui::InputTextWithHint("##search", "Search assets...", search_buffer_, IM_ARRAYSIZE(search_buffer_))) {
        // TODO: Implement search filtering
    }
    ImGui::PopItemWidth();

    ImGui::Separator();

    // Toolbar
    if (ImGui::Button("Refresh")) {
        RefreshAssets();
    }
    ImGui::SameLine();
    if (ImGui::Button("Import...")) {
        // TODO: Show import dialog
        spdlog::info("Import asset clicked");
    }
    ImGui::SameLine();
    if (ImGui::Button("New Folder")) {
        // TODO: Create new folder
        spdlog::info("New folder clicked");
    }

    ImGui::Separator();

    // Asset tree
    ImGui::BeginChild("AssetTree", ImVec2(0, 0), true);

    if (root_) {
        for (auto& child : root_->children) {
            RenderNode(child.get());
        }
    }

    // Handle right-click in empty space
    if (ImGui::BeginPopupContextWindow("AssetBrowserContext")) {
        RenderContextMenu();
        ImGui::EndPopup();
    }

    ImGui::EndChild();

    ImGui::End();
}

void AssetBrowser::RenderNode(AssetNode* node) {
    if (!node) return;

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

    if (node == selected_node_) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    if (!node->is_folder || node->children.empty()) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    // Icon based on type
    const char* icon = node->is_folder ? ICON_FOLDER : ICON_FILE;

    bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)node, flags, "%s %s", icon, node->name.c_str());

    // Handle selection
    if (ImGui::IsItemClicked()) {
        selected_node_ = node;
        spdlog::debug("Selected asset: {}", node->name);
    }

    // Context menu
    if (ImGui::BeginPopupContextItem()) {
        selected_node_ = node;
        RenderContextMenu();
        ImGui::EndPopup();
    }

    // Drag source
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
        ImGui::SetDragDropPayload("ASSET_ITEM", &node, sizeof(AssetNode*));
        ImGui::Text("Move %s", node->name.c_str());
        ImGui::EndDragDropSource();
    }

    // Render children
    // Only pop if TreeNodeEx actually pushed (i.e., NoTreePushOnOpen was not set)
    // NoTreePushOnOpen is set for: !node->is_folder || node->children.empty()
    // So we need to pop when: node_open && node->is_folder && !node->children.empty()
    if (node_open && node->is_folder && !node->children.empty()) {
        for (auto& child : node->children) {
            RenderNode(child.get());
        }
        ImGui::TreePop();
    }
}

void AssetBrowser::RenderContextMenu() {
    if (ImGui::MenuItem("Open")) {
        if (selected_node_) {
            spdlog::info("Open asset: {}", selected_node_->name);
            // TODO: Open asset in appropriate editor
        }
    }
    if (ImGui::MenuItem("Rename", "F2")) {
        if (selected_node_) {
            spdlog::info("Rename asset: {}", selected_node_->name);
            // TODO: Show rename dialog
        }
    }
    if (ImGui::MenuItem("Delete", "Del")) {
        if (selected_node_) {
            spdlog::info("Delete asset: {}", selected_node_->name);
            // TODO: Show confirmation dialog
        }
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Properties")) {
        if (selected_node_) {
            spdlog::info("Show properties for: {}", selected_node_->name);
            // TODO: Show properties panel
        }
    }
}

void AssetBrowser::RefreshAssets() {
    spdlog::info("Refreshing assets...");
    // TODO: Scan filesystem and rebuild tree
}

void AssetBrowser::AddAsset(const std::string& category, const std::string& name) {
    // TODO: Add asset to the appropriate category
    spdlog::info("Adding asset: {} to category: {}", name, category);
}

void AssetBrowser::InitializeDefaultStructure() {
    root_ = std::make_unique<AssetNode>("Root", true);

    // Create default folder structure
    auto datasets = std::make_unique<AssetNode>("Datasets", true);
    datasets->children.push_back(std::make_unique<AssetNode>("training_data.csv", false));
    datasets->children.push_back(std::make_unique<AssetNode>("validation_data.csv", false));
    datasets->children.push_back(std::make_unique<AssetNode>("test_data.csv", false));
    root_->children.push_back(std::move(datasets));

    auto models = std::make_unique<AssetNode>("Models", true);
    models->children.push_back(std::make_unique<AssetNode>("my_model.cyxwiz", false));
    models->children.push_back(std::make_unique<AssetNode>("pretrained_model.cyxwiz", false));
    root_->children.push_back(std::move(models));

    auto training_runs = std::make_unique<AssetNode>("Training Runs", true);
    training_runs->children.push_back(std::make_unique<AssetNode>("run_2025_01_15_001", false));
    training_runs->children.push_back(std::make_unique<AssetNode>("run_2025_01_14_003", false));
    root_->children.push_back(std::move(training_runs));

    auto scripts = std::make_unique<AssetNode>("Scripts", true);
    scripts->children.push_back(std::make_unique<AssetNode>("preprocessing.py", false));
    scripts->children.push_back(std::make_unique<AssetNode>("custom_loss.py", false));
    root_->children.push_back(std::move(scripts));

    auto checkpoints = std::make_unique<AssetNode>("Checkpoints", true);
    checkpoints->children.push_back(std::make_unique<AssetNode>("checkpoint_epoch_10.ckpt", false));
    checkpoints->children.push_back(std::make_unique<AssetNode>("checkpoint_epoch_20.ckpt", false));
    root_->children.push_back(std::move(checkpoints));

    spdlog::info("Asset browser initialized with default structure");
}

// Icon constants (use actual icon font if available, otherwise use text)
#ifndef ICON_FOLDER
#define ICON_FOLDER "\xef\x81\xbb"  // FontAwesome folder icon (or use "[+]")
#endif
#ifndef ICON_FILE
#define ICON_FILE "\xef\x85\x9b"    // FontAwesome file icon (or use "[-]")
#endif

} // namespace gui
