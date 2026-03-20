#include "data_studio_panel.h"
#include "../../core/data_registry.h"
#include "../main_window.h"
#include "../node_editor.h"
#include <spdlog/spdlog.h>
#include <imgui.h>

namespace cyxwiz {

DataStudioPanel::DataStudioPanel()
    : selected_tab_(0)
    , visible_(true)
{
    // Initialize all components
    pipeline_canvas_ = std::make_unique<PipelineCanvas>();
    query_editor_ = std::make_unique<QueryEditor>();
    analyzer_ = std::make_unique<Analyzer>();
    visualizer_ = std::make_unique<Visualizer>();

    spdlog::info("[Data Studio] Panel initialized");
}

void DataStudioPanel::Render() {
    if (!visible_) return;

    // Phase 5 Week 7: Check for deployment request
    if (pipeline_canvas_ && pipeline_canvas_->IsDeploymentRequested()) {
        OnDeployToNodeEditor();
        pipeline_canvas_->ClearDeploymentRequest();
    }

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

    ImGui::SameLine();
    ImGui::TextDisabled("|");

    ImGui::SameLine();
    if (ImGui::Button("Help")) {
        // TODO: Phase 1 Week 2 - Show help dialog
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
    if (ImGui::BeginTabBar("DataStudioTabs")) {
        if (ImGui::BeginTabItem("Pipeline")) {
            if (pipeline_canvas_) {
                pipeline_canvas_->Render();
            }
            ImGui::EndTabItem();
        }

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

    // Pipeline execution status
    if (pipeline_canvas_) {
        // TODO: Show pipeline execution status
        ImGui::Text("Pipeline: Ready");
    }

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    // Analysis status
    if (analyzer_ && analyzer_->IsAnalysisRunning()) {
        ImGui::Text("Analyzing... %.0f%%", analyzer_->GetAnalysisProgress() * 100.0f);
    } else {
        ImGui::Text("Analysis: Ready");
    }
}

void DataStudioPanel::SetActiveDataset(const std::string& dataset_name) {
    active_dataset_ = dataset_name;
    spdlog::info("[Data Studio] Set active dataset: {}", dataset_name);

    // Propagate to all components
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

// ============================================================================
// Phase 5 Week 7 - Node Editor Handoff
// ============================================================================

void DataStudioPanel::OnDeployToNodeEditor() {
    if (!pipeline_canvas_ || !pipeline_canvas_->IsDeploymentReady()) {
        spdlog::error("[Data Studio] Cannot deploy: no dataset ready for deployment");
        return;
    }

    std::string dataset_name = pipeline_canvas_->GetDeploymentDataset();
    if (dataset_name.empty()) {
        spdlog::error("[Data Studio] Cannot deploy: deployment dataset name is empty");
        return;
    }

    if (!main_window_) {
        spdlog::error("[Data Studio] Cannot deploy: MainWindow reference not set");
        return;
    }

    auto* node_editor = main_window_->GetNodeEditor();
    if (!node_editor) {
        spdlog::error("[Data Studio] Cannot deploy: NodeEditor not available");
        return;
    }

    spdlog::info("[Data Studio] Deploying dataset '{}' to Node Editor", dataset_name);

    // Handoff dataset to Node Editor
    node_editor->SetDatasetFromDataStudio(dataset_name);

    // Show Node Editor panel
    node_editor->Show();

    // Clear deployment status
    pipeline_canvas_->ClearDeploymentStatus();

    // Show success notification (using ImGui notification system if available)
    spdlog::info("[Data Studio] Successfully deployed dataset '{}' to Node Editor", dataset_name);
}

} // namespace cyxwiz
