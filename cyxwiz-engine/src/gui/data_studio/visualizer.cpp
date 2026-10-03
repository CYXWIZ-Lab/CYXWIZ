#include "visualizer.h"

#include "../../core/dataset_catalog.h"
#include "../dashboard/dashboard_window.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

Visualizer::Visualizer() = default;
Visualizer::~Visualizer() = default;

void Visualizer::SetActiveDataset(const std::string& dataset_name) {
    current_dataset_ = dataset_name;
    spdlog::info("[Data Studio] Visualize: {}", dataset_name);
}

void Visualizer::Render() {
    if (current_dataset_.empty()) {
        ImGui::TextDisabled("Pick a dataset above to plot it.");
        return;
    }
    auto& view = plots_[current_dataset_];
    if (!view) view = std::make_unique<dashboard::DashboardWindow>("visualize_" + current_dataset_, dashboard::DashboardWindow::Mode::Visualize);
    const auto entry = DatasetCatalog::Instance().Resolve(current_dataset_);
    if (entry) view->SetData(current_dataset_, entry->Shown());
    else view->ClearData("This dataset is no longer loaded: pick it again or load it in its Data Input.");
    view->RenderEmbedded();
}

}  // namespace cyxwiz
