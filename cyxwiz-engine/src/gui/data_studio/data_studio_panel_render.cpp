#include "data_studio_panel.h"
#include "../../core/data_registry.h"
#include <imgui.h>

namespace cyxwiz {

void DataStudioPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(1200, 800), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Data Studio", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        RenderTabBar();

        ImGui::Separator();
        RenderStatusBar();
    }
    ImGui::End();
}

void DataStudioPanel::RenderToolbar() {
    ImGui::Text("Data Studio");
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    // Dataset selector
    RenderDatasetSelector();

    ImGui::SameLine();
    if (ImGui::Button("Refresh")) {
        // Refresh current dataset
        if (!active_dataset_.empty()) {
            SetActiveDataset(active_dataset_);
        }
    }

    // Cross-navigation: one click opens the Engine's Node Editor
    // panel. Replaces the old static text hint — the user no longer
    // has to hunt through the Engine toolbar to get there.
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    if (open_node_editor_callback_) {
        if (ImGui::Button("Open Node Editor")) {
            open_node_editor_callback_();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Open the CyxWiz Studio (Node Editor) panel. "
                              "Switch to 'Data Pipeline' execution mode "
                              "there to build visual transformation graphs.");
        }
    } else {
        ImGui::TextDisabled("For visual pipelines, use CyxWiz Studio");
    }
}

void DataStudioPanel::RenderDatasetSelector() {
    // Get available datasets from registry
    auto& registry = DataRegistry::Instance();
    auto datasets = registry.GetDatasetNames();

    ImGui::Text("Dataset:");
    ImGui::SameLine();

    if (ImGui::BeginCombo("##dataset_selector",
                         active_dataset_.empty() ? "Select dataset..." : active_dataset_.c_str())) {

        for (const auto& dataset_name : datasets) {
            bool is_selected = (dataset_name == active_dataset_);
            if (ImGui::Selectable(dataset_name.c_str(), is_selected)) {
                SetActiveDataset(dataset_name);
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }
}

void DataStudioPanel::RenderTabBar() {
    // Unified Canvas Phase 5: Removed Pipeline tab (moved to Node Editor)
    if (ImGui::BeginTabBar("DataStudioTabs")) {
        if (ImGui::BeginTabItem("Query")) {
            if (query_editor_) {
                query_editor_->Render();
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Analyze")) {
            if (analyzer_) {
                analyzer_->Render();
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Visualize")) {
            if (visualizer_) {
                visualizer_->Render();
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void DataStudioPanel::RenderStatusBar() {
    if (!active_dataset_.empty()) {
        ImGui::Text("Dataset: %s", active_dataset_.c_str());
    } else {
        ImGui::TextDisabled("No dataset selected");
    }

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    // Analysis status (Unified Canvas Phase 5: Removed pipeline status)
    if (analyzer_ && analyzer_->IsAnalysisRunning()) {
        ImGui::Text("Analyzing... %.0f%%", analyzer_->GetAnalysisProgress() * 100.0f);
    } else {
        ImGui::Text("Analysis: Ready");
    }
}

} // namespace cyxwiz