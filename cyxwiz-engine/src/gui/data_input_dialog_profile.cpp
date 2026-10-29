// DataInputDialog profiling tab: Data Studio's profile view of the loaded
// dataset (TOFIX148 D2: one profile model and one view; roles are set in
// Data Studio).

#include "node_config_dialog.h"
#include "data_studio/image_gallery_view.h"
#include "data_studio/profile_view.h"
#include "ui_tokens.h"

#include <memory>

namespace gui {

DataInputDialog::~DataInputDialog() = default;

void DataInputDialog::RenderDataProfilingTab() {
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    ImGui::Spacing();
    // The data this dialog loaded, else what the node's last pipeline run produced.
    std::string dataset = data_load_state_ == DataLoadState::InMemory ? loaded_dataset_name_ : std::string();
    if (dataset.empty() && node_editor_ && node_) {
        const auto& facts = node_editor_->GetNodeRunFacts();
        if (const auto it = facts.find(node_->id); it != facts.end()) dataset = it->second.output_dataset;
    }
    if (dataset.empty()) {
        ImGui::TextColored(t.text_dim, "No data loaded in memory.");
        ImGui::TextColored(t.text_dim, "Click Apply (or run the pipeline) to load the data, then come back here for its profile.");
        return;
    }
    if (!profile_view_) {
        profile_view_ = std::make_unique<cyxwiz::ProfileView>();
        profile_view_->SetRolesEditable(false);
    }
    if (profile_dataset_ != dataset) {
        profile_dataset_ = dataset;
        profile_view_->SetActiveDataset(profile_dataset_);
    }
    if (node_editor_) profile_view_->SetLiveCompile(node_editor_->GetLiveCompile());
    profile_view_->Render();
}

}  // namespace gui
