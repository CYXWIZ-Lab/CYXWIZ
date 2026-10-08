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
};

struct Hdf5SourceLoadTaskResult {
    std::atomic<bool> done{false};
    Hdf5SourceLoadResult result;
};

// Prepares a private, validated Arrow table; never registers a dataset. The
// worker publishes terminal state before done (release). Queued cancellation
// may prevent Execute entirely; callers must also observe the task state.
std::shared_ptr<AsyncTask> MakeHdf5SourceLoadTask(
    Hdf5SourceLoadRequest request, std::shared_ptr<Hdf5SourceLoadTaskResult> state);

} // namespace cyxwiz
