#include "data_studio_panel.h"
#include "../../core/data_registry.h"
#include <spdlog/spdlog.h>
#include <imgui.h>

namespace cyxwiz {

DataStudioPanel::DataStudioPanel()
    : selected_tab_(0)
    , visible_(true)
{
    // Initialize all components (Unified Canvas Phase 5: Removed pipeline_canvas_)
    query_editor_ = std::make_unique<QueryEditor>();
    analyzer_ = std::make_unique<Analyzer>();
    visualizer_ = std::make_unique<Visualizer>();

    spdlog::info("[Data Studio] Panel initialized (simplified - pipeline moved to Node Editor)");
}

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

    // Unified Canvas Phase 5: Add note about Node Editor for pipelines
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::TextDisabled("For visual pipelines, use CyxWiz Studio (Data Pipeline mode)");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Visual data transformation pipelines have moved to CyxWiz Studio.\n"
                         "Switch to 'Data Pipeline' execution mode in the Studio toolbar.");
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

void DataStudioPanel::SetActiveDataset(const std::string& dataset_name) {
    active_dataset_ = dataset_name;
    spdlog::info("[Data Studio] Set active dataset: {}", dataset_name);

    // Propagate to all components (Unified Canvas Phase 5: Removed pipeline_canvas_)
    if (query_editor_) {
        query_editor_->SetActiveDataset(dataset_name);
    }
    if (analyzer_) {
        analyzer_->SetActiveDataset(dataset_name);
    }
    if (visualizer_) {
        visualizer_->SetActiveDataset(dataset_name);
    }
}

} // namespace cyxwiz
