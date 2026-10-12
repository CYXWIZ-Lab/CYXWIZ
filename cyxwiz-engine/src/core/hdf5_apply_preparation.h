#pragma once

#include "data_registry.h"
#include "hdf5_source_load_task.h"

#include <map>

namespace cyxwiz {

struct Hdf5ApplyPreparationRequest {
    Hdf5SourceLoadRequest source;
    std::string dataset_name;
    DataRegistry::TabularPublicationToken publication;
    // The dialog owns common source/category settings and project-relative paths.
    // This private copy receives canonical HDF5 settings and loaded-state fields.
    std::map<std::string, std::string> parameters;
};

struct Hdf5ApplyPreparationResult {
    Hdf5TableStatus status = Hdf5TableStatus::ReadFailed;
    std::string error;
    bool source_changed = false;
    std::shared_ptr<const DatasetAuditResult> audit;
    std::optional<Hdf5SourceStamp> source;
    std::map<std::string, std::string> parameters;
    std::unique_ptr<DataRegistry::PreparedArrowPublication> publication;
};

struct Hdf5ApplyPreparationState {
    std::atomic<bool> done{false};
    Hdf5ApplyPreparationResult result;
};

// Worker-only read/audit/staging. Never mutates a node or registry. The owner
// retains state until delivery; task history owns neither state nor its table.
// Before committing, the UI must recheck owner, node/settings and cancellation,
// then publish, swap these parameters without throwing, and notify observers.
// Source identity is a final worker-side snapshot, not a file lock/content hash.
std::shared_ptr<AsyncTask> MakeHdf5ApplyPreparationTask(
    Hdf5ApplyPreparationRequest request, const std::shared_ptr<Hdf5ApplyPreparationState>& state);

} // namespace cyxwiz
