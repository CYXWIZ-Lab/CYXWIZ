#include "node_config_dialog.h"
#include "../core/data_registry.h"
#include "../core/project_manager.h"

namespace gui {

bool DataInputDialog::BuildHdf5RestoreRequest(cyxwiz::Hdf5RegisteredSourceRequest& request,
                                            std::string& error) const {
    if (!node_ || !IsHdf5Source()) {
        error = "HDF5 source is no longer selected";
        return false;
    }
    const auto saved = cyxwiz::ReadHdf5InputSettings(node_->parameters);
    if (!saved.ok) {
        error = saved.error;
        return false;
    }
    const auto selection = hdf5_inspector_.Selection();
    if (selection.data_path != saved.settings.selection.data_path ||
        selection.label_path != saved.settings.selection.label_path) {
        error = "HDF5 selection differs from the saved dataset selection";
        return false;
    }
    const auto name = node_->parameters.find("dataset_name");
    if (name == node_->parameters.end() || name->second.empty()) {
        error = "HDF5 source has no saved dataset registration";
        return false;
    }
    request.dataset_name = name->second;
    request.resolved_path = hdf5_source_resolved_path_;
    request.settings = saved.settings;
    auto& registry = cyxwiz::DataRegistry::Instance();
    request.dataset = registry.GetArrowDataset(request.dataset_name);
    request.registered_source_path = registry.GetTabularSourcePath(request.dataset_name).value_or("");
    return true;
}

void DataInputDialog::BeginHdf5LoadedSourceVerification() {
    hdf5_loaded_source_.Cancel();
    hdf5_restore_parameters_.clear();
    if (!node_ || !IsHdf5Source()) return;
    data_load_state_ = DataLoadState::NotLoaded;
    loaded_rows_ = loaded_cols_ = 0;
    loaded_memory_bytes_ = 0;
    loaded_backend_ = 0;
    loaded_dataset_name_.clear();
    loaded_memory_is_estimate_ = false;
    apply_success_ = false;
    node_->parameters["data_loaded"] = "false";
    hdf5_restore_parameters_ = node_->parameters;
    SyncHdf5InspectorSource();
    cyxwiz::Hdf5RegisteredSourceRequest request;
    if (!BuildHdf5RestoreRequest(request, apply_status_message_)) return;
    loaded_dataset_name_ = request.dataset_name;
    if (!request.dataset) {
        apply_status_message_ = "Saved HDF5 dataset has no compatible in-memory registration";
        return;
    }
    hdf5_loaded_source_.Start(std::move(request), cyxwiz::ProjectManager::Instance().GetSessionToken());
    apply_status_message_ = hdf5_loaded_source_.Busy()
        ? "Verifying registered HDF5 source..." : hdf5_loaded_source_.Result().error;
}

void DataInputDialog::PollHdf5LoadedSourceVerification() {
    if (!hdf5_loaded_source_.Busy()) return;
    // Check the project owner before dereferencing the node during delivery.
    if (!hdf5_loaded_source_.OwnerAlive()) {
        hdf5_loaded_source_.Cancel();
        return;
    }
    if (!node_ || node_->parameters != hdf5_restore_parameters_) {
        hdf5_loaded_source_.Cancel();
        apply_status_message_ = "HDF5 loaded-state verification discarded: node settings changed";
        return;
    }
    cyxwiz::Hdf5RegisteredSourceRequest current;
    if (!BuildHdf5RestoreRequest(current, apply_status_message_)) {
        hdf5_loaded_source_.Cancel();
        return;
    }
    if (!hdf5_loaded_source_.Poll(current)) return;
    const auto& result = hdf5_loaded_source_.Result();
    if (result.status != cyxwiz::Hdf5TableStatus::Ok) {
        apply_status_message_ = result.error;
        return;
    }
    auto parameters = node_->parameters;
    parameters["data_loaded"] = "true";
    parameters["loaded_rows"] = std::to_string(result.rows);
    parameters["loaded_cols"] = std::to_string(result.columns);
    parameters["memory_bytes"] = std::to_string(result.bytes);
    node_->parameters.swap(parameters);
    loaded_rows_ = result.rows;
    loaded_cols_ = result.columns;
    loaded_memory_bytes_ = result.bytes;
    loaded_backend_ = 1;
    loaded_memory_is_estimate_ = false;
    data_load_state_ = DataLoadState::InMemory;
    apply_success_ = true;
    apply_status_message_ = "Verified HDF5 registration: " + loaded_dataset_name_;
}
} // namespace gui
