#include "node_config_dialog.h"
#include "loaders/data_loader.h"
#include "../core/data_registry.h"
#include "../core/project_manager.h"

#include <spdlog/spdlog.h>

#include <filesystem>

namespace gui {

// HDF5 Apply (TOFIX189): the inspector's selection is read, audited and staged
// on a worker; the node and the registry change only here, on the UI thread,
// once the result is back and the node still has the settings it started with.
void DataInputDialog::BeginHdf5Apply() {
    hdf5_loaded_source_.Cancel();
    hdf5_apply_.Cancel();
    hdf5_apply_parameters_.clear();
    data_load_state_ = DataLoadState::NotLoaded;
    loaded_rows_ = loaded_cols_ = 0;
    loaded_memory_bytes_ = 0;
    loaded_backend_ = 0;
    loaded_memory_is_estimate_ = false;
    node_->parameters["data_loaded"] = "false";
    SyncHdf5InspectorSource();

    const auto fail = [this](const std::string& message) {
        apply_status_message_ = message;
        apply_success_ = false;
        apply_in_progress_ = false;
        has_changes_ = false;
    };
    const auto selection = hdf5_inspector_.Selection();
    if (selection.data_path.empty()) {
        fail("Choose the dataset to load in the HDF5 hierarchy");
        return;
    }
    // The inspector chooses the paths; a saved numeric policy and memory cap are kept.
    const auto saved = cyxwiz::ReadHdf5InputSettings(node_->parameters);
    cyxwiz::Hdf5ApplyPreparationRequest request;
    request.source.path = hdf5_source_resolved_path_;
    request.source.settings = saved.ok ? saved.settings : cyxwiz::Hdf5InputSettings{};
    request.source.settings.selection = selection;
    const auto previous = node_->parameters.find("dataset_name");
    hdf5_previous_dataset_name_ = previous == node_->parameters.end() ? "" : previous->second;
    request.dataset_name = GenerateDatasetName();
    request.publication = cyxwiz::DataRegistry::Instance().CaptureTabularPublication(request.dataset_name);
    request.parameters = node_->parameters;
    hdf5_apply_parameters_ = node_->parameters;
    loaded_dataset_name_ = request.dataset_name;

    hdf5_apply_.Start(std::move(request), cyxwiz::ProjectManager::Instance().GetSessionToken());
    if (!hdf5_apply_.Busy()) {
        fail(hdf5_apply_.Result().error);
        return;
    }
    apply_status_message_ = "Reading " + std::filesystem::path(hdf5_source_resolved_path_).filename().string() + "...";
}

void DataInputDialog::PollHdf5Apply() {
    if (!hdf5_apply_.Busy()) return;
    // Check the project owner before touching the node during delivery.
    if (!hdf5_apply_.OwnerAlive()) {
        hdf5_apply_.Cancel();
        apply_in_progress_ = false;
        return;
    }
    if (!node_ || node_->parameters != hdf5_apply_parameters_) {
        hdf5_apply_.Cancel();
        apply_in_progress_ = false;
        apply_status_message_ = "HDF5 load discarded: the node's settings changed while it read";
        return;
    }
    if (!hdf5_apply_.Poll()) return;
    apply_in_progress_ = false;
    has_changes_ = false;
    auto& result = hdf5_apply_.Result();
    if (result.status != cyxwiz::Hdf5TableStatus::Ok || !result.publication) {
        apply_success_ = false;
        apply_status_message_ = result.error.empty() ? "HDF5 load failed" : result.error;
        spdlog::warn("DataInputDialog: HDF5 Apply failed - {}", apply_status_message_);
        return;
    }
    const auto chosen = hdf5_inspector_.Selection();
    const auto staged = cyxwiz::ReadHdf5InputSettings(result.parameters).settings.selection;
    if (chosen.data_path != staged.data_path || chosen.label_path != staged.label_path) {
        apply_success_ = false;
        apply_status_message_ = "HDF5 load discarded: a different dataset or label set is selected now";
        return;
    }
    std::string error;
    if (!cyxwiz::DataRegistry::Instance().TryPublishPreparedArrowTable(*result.publication, error)) {
        apply_success_ = false;
        apply_status_message_ = "HDF5 load could not be registered: " + error;
        return;
    }
    node_->parameters.swap(result.parameters);
    result.publication->Notify();

    // A renamed replacement owns the node now; the previous registration goes.
    if (!hdf5_previous_dataset_name_.empty() && hdf5_previous_dataset_name_ != loaded_dataset_name_) {
        if (auto* previous_loader = cyxwiz::loaders::GetByRegisteredDataset(hdf5_previous_dataset_name_))
            previous_loader->Unregister(hdf5_previous_dataset_name_);
    }
    hdf5_previous_dataset_name_.clear();
    loaded_rows_ = std::stoll(node_->parameters["loaded_rows"]);
    loaded_cols_ = std::stoll(node_->parameters["loaded_cols"]);
    loaded_memory_bytes_ = std::stoull(node_->parameters["memory_bytes"]);
    loaded_backend_ = 1;
    loaded_memory_is_estimate_ = false;
    data_load_state_ = DataLoadState::InMemory;
    apply_success_ = true;
    apply_status_message_ = "Loaded " + std::to_string(loaded_rows_) + " rows x " +
        std::to_string(loaded_cols_) + " columns from " + result.source->canonical_path;
    spdlog::info("DataInputDialog: HDF5 Apply loaded '{}' ({} rows, {} columns)", loaded_dataset_name_,
                 loaded_rows_, loaded_cols_);
}
} // namespace gui
