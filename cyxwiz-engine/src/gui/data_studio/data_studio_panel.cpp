#include "data_studio_panel.h"
#include <spdlog/spdlog.h>

namespace cyxwiz {

DataStudioPanel::DataStudioPanel()
    : selected_tab_(0)
    , visible_(true)
{
    // Initialize all components (Unified Canvas Phase 5: Removed pipeline_canvas_)
    query_editor_ = std::make_unique<QueryEditor>();
    profile_ = std::make_unique<ProfileView>();
    visualizer_ = std::make_unique<Visualizer>();

    spdlog::info("[Data Studio] Panel initialized (simplified - pipeline moved to Node Editor)");
}

void DataStudioPanel::ShowQuery(const std::string& dataset_name, const std::string& sql) {
    SetActiveDataset(dataset_name);
    visible_ = true;
    show_query_ = true;
    if (query_editor_) {
        query_editor_->SetQuery(sql);
        query_editor_->ExecuteQuery();
    }
}

void DataStudioPanel::SetActiveDataset(const std::string& dataset_name) {
    active_dataset_ = dataset_name;
    spdlog::info("[Data Studio] Set active dataset: {}", dataset_name);

    // Propagate to all components (Unified Canvas Phase 5: Removed pipeline_canvas_)
    if (query_editor_) {
        query_editor_->SetActiveDataset(dataset_name);
    }
    if (profile_) {
        profile_->SetActiveDataset(dataset_name);
    }
    if (visualizer_) {
        visualizer_->SetActiveDataset(dataset_name);
    }
}

} // namespace cyxwiz
