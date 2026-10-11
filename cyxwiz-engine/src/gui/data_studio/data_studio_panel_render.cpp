#include "data_studio_panel.h"

#include "../dataset_picker.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../separate_windows.h"

#include <imgui.h>

#include <algorithm>

namespace cyxwiz {

void DataStudioPanel::Render() {
    if (query_editor_) query_editor_->RenderWindows();
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(1200, 800), ImGuiCond_FirstUseEver);
    // Opened from elsewhere (a dashboard): bring the panel forward, also when
    // it is a hidden dock tab (its Begin returns false until focused).
    if (show_profile_ || show_query_) ImGui::SetNextWindowFocus();
    ::gui::NextWindowMayLeave("Data Studio");
    const bool expanded = ImGui::Begin("Data Studio", &visible_);
    ::gui::TabMenu("Data Studio", &visible_);
    if (expanded) {
        RenderToolbar();
        ImGui::Spacing();
        RenderTabBar();
        ImGui::Spacing();
        RenderStatusBar();
    }
    ImGui::End();
}

void DataStudioPanel::RenderToolbar() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Data Studio");
    ImGui::SameLine();
    RenderDatasetSelector();

    ImGui::SameLine();
    if (ui::SecondaryButton(ICON_FA_ROTATE " Refresh", !active_dataset_.empty(), "Pick a dataset first")) SetActiveDataset(active_dataset_);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Read the picked dataset again (after it was re-loaded)");

    // Cross-navigation: one click opens the Engine's Node Editor panel.
    ImGui::SameLine();
    if (open_node_editor_callback_) {
        if (ui::SecondaryButton("Open Node Editor")) open_node_editor_callback_();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Open the CyxWiz Studio (Node Editor) panel. "
                              "Switch to 'Data Pipeline' execution mode "
                              "there to build visual transformation graphs.");
        }
    } else {
        ImGui::TextColored(t.text_dim, "For visual pipelines, use CyxWiz Studio");
    }
}

void DataStudioPanel::RenderDatasetSelector() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "Dataset");
    ImGui::SameLine();
    // Every dataset the catalog knows: Data Input tables, node results, files.
    std::string picked = active_dataset_;
    ui::DatasetPickerOptions options;
    options.tabular_only = false;
    options.width = std::min(520.0f, ImGui::GetContentRegionAvail().x * 0.45f);
    if (ui::DatasetPicker("dataset_selector", picked, options)) SetActiveDataset(picked);
}

void DataStudioPanel::RenderTabBar() {
    // Unified Canvas Phase 5: Removed Pipeline tab (moved to Node Editor)
    if (ImGui::BeginTabBar("DataStudioTabs")) {
        if (ImGui::BeginTabItem("Query", nullptr, show_query_ ? ImGuiTabItemFlags_SetSelected : 0)) {
            show_query_ = false;
            if (query_editor_) query_editor_->Render();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Profile", nullptr, show_profile_ ? ImGuiTabItemFlags_SetSelected : 0)) {
            show_profile_ = false;
            if (profile_) profile_->Render();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Visualize")) {
            if (visualizer_) visualizer_->Render();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void DataStudioPanel::RenderStatusBar() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!active_dataset_.empty()) ImGui::TextColored(t.text_dim, "Dataset: %s", active_dataset_.c_str());
    else ImGui::TextColored(t.text_dim, "No dataset selected");
    ImGui::SameLine();
    ImGui::TextColored(t.text_faint, "\xC2\xB7");
    ImGui::SameLine();
    if (query_editor_ && query_editor_->IsRunning()) ImGui::TextColored(t.info, "Query running");
    else if (profile_ && profile_->IsRunning()) ImGui::TextColored(t.info, "Profiling");
    else ImGui::TextColored(t.text_dim, "Ready");
}

}  // namespace cyxwiz
