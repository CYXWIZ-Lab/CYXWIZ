#include "hdf5_source_inspector.h"
#include "../core/async_task_manager.h"
#include "../core/hdf5_input_settings.h"

#include <cstring>
#include <utility>

namespace gui {

Hdf5SourceInspector::~Hdf5SourceInspector() { Cancel(); }

void Hdf5SourceInspector::Cancel() {
    if (task_) task_->RequestCancel();
    task_.reset();
    // Dropping this request's state is the generation boundary. A late worker
    // retains its own state, but can no longer publish into this inspector.
    state_.reset();
}

void Hdf5SourceInspector::Reset() {
    Cancel();
    source_path_.clear();
    source_stamp_.reset();
    std::strcpy(group_path_, "/");
    std::strcpy(data_path_, "/data");
    label_path_[0] = '\0';
    selection_role_ = 0;
    rows_per_page_ = 20;
    row_offset_ = column_offset_ = 0;
    hierarchy_ = {};
    page_ = {};
    data_ = {};
    view_ = {};
    error_.clear();
}

void Hdf5SourceInspector::SetSource(const std::string& path) {
    if (source_path_ == path) return;
    Reset();
    source_path_ = path;
}

void Hdf5SourceInspector::InvalidateSelection() {
    Cancel();
    page_ = {};
    data_ = {};
    view_ = {};
    row_offset_ = column_offset_ = 0;
    error_.clear();
}

void Hdf5SourceInspector::SetSelection(const cyxwiz::Hdf5TableSelection& selection) {
    InvalidateSelection();
    if (selection.data_path.size() >= sizeof(data_path_) ||
        selection.label_path.size() >= sizeof(label_path_) ||
        selection.data_path.find('\0') != std::string::npos ||
        selection.label_path.find('\0') != std::string::npos) {
        error_ = "Dataset paths must be at most 4096 bytes without null characters";
        return;
    }
    std::strcpy(data_path_, selection.data_path.c_str());
    std::strcpy(label_path_, selection.label_path.c_str());
}

void Hdf5SourceInspector::RestoreSettings(
    const std::map<std::string, std::string>& parameters) {
    const auto restored = cyxwiz::ReadHdf5InputSettings(parameters);
    if (!restored.ok) {
        InvalidateSelection();
        data_path_[0] = '\0';
        label_path_[0] = '\0';
        error_ = restored.error;
        return;
    }
    SetSelection(restored.settings.selection);
}

void Hdf5SourceInspector::Start(cyxwiz::Hdf5InspectionRequest request) {
    Cancel();
    error_.clear();
    if (source_path_.empty()) {
        error_ = "No HDF5 source selected";
        return;
    }
    request.path = source_path_;
    request.expected_source = source_stamp_;
    request_ = std::move(request);
    try {
        state_ = std::make_shared<cyxwiz::Hdf5InspectionTaskResult>();
        task_ = cyxwiz::MakeHdf5InspectionTask(request_, state_);
        cyxwiz::AsyncTaskManager::Instance().Submit(task_);
    } catch (const std::exception& error) {
        Cancel();
        error_ = std::string("Cannot start HDF5 inspection: ") + error.what();
    }
}

void Hdf5SourceInspector::Browse(const std::string& group, uint64_t offset, bool refresh) {
    if (group.size() >= sizeof(group_path_) || group.find('\0') != std::string::npos) {
        Cancel();
        hierarchy_ = {};
        error_ = "Group path must be at most 4096 bytes without null characters";
        return;
    }
    // group may refer to group_path_ through a temporary string at the callsite.
    std::strcpy(group_path_, group.c_str());
    hierarchy_ = {};
    if (refresh) {
        source_stamp_.reset();
        InvalidateSelection();
    }
    cyxwiz::Hdf5InspectionRequest request;
    request.kind = cyxwiz::Hdf5InspectionKind::Hierarchy;
    request.browse.group_path = group;
    request.browse.offset = offset;
    request.browse.limit = 64;
    Start(std::move(request));
}

void Hdf5SourceInspector::Preview(uint64_t row_offset, uint64_t column_offset) {
    row_offset_ = row_offset;
    column_offset_ = column_offset;
    page_ = {};
    view_ = {};
    cyxwiz::Hdf5InspectionRequest request;
    request.kind = cyxwiz::Hdf5InspectionKind::Preview;
    request.selection = Selection();
    request.window = {row_offset, static_cast<uint64_t>(rows_per_page_), column_offset, 32};
    Start(std::move(request));
}

void Hdf5SourceInspector::Poll() {
    if (!state_) return;
    if (!state_->done.load()) {
        // A queued task can be retired without running its lambda.
        const auto status = task_->GetState();
        if (status == cyxwiz::TaskState::Cancelled) {
            error_ = "HDF5 inspection cancelled";
            Cancel();
        }
        return;
    }
    auto result = std::move(state_->result);
    task_.reset();
    state_.reset();
    if (result.source_changed) {
        source_stamp_.reset();
        hierarchy_ = {};
        page_ = {};
        data_ = {};
        view_ = {};
    }
    if (result.status != cyxwiz::Hdf5TableStatus::Ok) {
        error_ = result.error;
        return;
    }
    source_stamp_ = std::move(result.source);
    if (request_.kind == cyxwiz::Hdf5InspectionKind::Hierarchy)
        hierarchy_ = std::move(result.hierarchy);
    else {
        page_ = std::move(result.preview);
        data_ = std::move(result.data);
    }
}
} // namespace gui
