#pragma once

#include "hdf5_input_settings.h"
#include "hdf5_source_identity.h"

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace cyxwiz {
class AsyncTask;
struct DatasetAuditResult;

struct Hdf5SourceLoadRequest {
    std::string path;
    Hdf5InputSettings settings;
    std::optional<Hdf5SourceStamp> expected_source;
    std::function<bool()> cancel_requested;
};

struct Hdf5SourceLoadResult {
    Hdf5TableReadResult read;
    std::optional<Hdf5SourceStamp> source;
    bool source_changed = false;
    // Audit refusal retains diagnostics only; cancellation/source changes clear them.
    std::shared_ptr<const DatasetAuditResult> audit;
};

struct Hdf5SourceLoadTaskResult {
    std::atomic<bool> done{false};
    Hdf5SourceLoadResult result;
};

// Synchronous worker boundary shared by standalone preparation and Apply staging.
// Progress callbacks run on the calling worker; failures return no table/source.
Hdf5SourceLoadResult PrepareHdf5Source(
    const Hdf5SourceLoadRequest& request,
    const std::function<void(float, const std::string&)>& progress = {});

// Prepares a private, validated and audited Arrow table; never registers a dataset.
// Audit scratch and diagnostics are outside the materialization byte cap. The
// worker publishes terminal state before done (release). Queued cancellation
// may prevent Execute entirely; callers must also observe the task state.
std::shared_ptr<AsyncTask> MakeHdf5SourceLoadTask(
    Hdf5SourceLoadRequest request, std::shared_ptr<Hdf5SourceLoadTaskResult> state);

} // namespace cyxwiz
