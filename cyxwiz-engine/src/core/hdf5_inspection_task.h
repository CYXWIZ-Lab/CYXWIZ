#pragma once

#include "data_preview_service.h"
#include "hdf5_table_adapter.h"
#include "hdf5_source_identity.h"
#include "../utils/hdf5_browser.h"

#include <atomic>
#include <memory>
#include <optional>

namespace cyxwiz {
class AsyncTask;

enum class Hdf5InspectionKind { Hierarchy, Preview };

struct Hdf5InspectionRequest {
    std::string path;
    Hdf5InspectionKind kind = Hdf5InspectionKind::Hierarchy;
    Hdf5BrowseRequest browse;
    Hdf5TableSelection selection;
    Hdf5TablePreviewRequest window;
    std::optional<Hdf5SourceStamp> expected_source;
};

struct Hdf5InspectionResult {
    Hdf5TableStatus status = Hdf5TableStatus::ReadFailed;
    std::string error;
    bool source_changed = false;
    std::optional<Hdf5SourceStamp> source;
    Hdf5BrowsePage hierarchy;
    DataPreviewPage preview;
    Hdf5TableDatasetInfo data;
    std::optional<Hdf5TableDatasetInfo> labels;
};

struct Hdf5InspectionTaskResult {
    // Worker writes result, then publishes done. No dialog or node is captured.
    std::atomic<bool> done{false};
    Hdf5InspectionResult result;
};

// Fixed 16 MiB sampled-read policy; Preserve numeric types. File identity is
// checked before/after I/O on the worker, including expected_source when set.
std::shared_ptr<AsyncTask> MakeHdf5InspectionTask(
    Hdf5InspectionRequest request, std::shared_ptr<Hdf5InspectionTaskResult> state);
} // namespace cyxwiz
