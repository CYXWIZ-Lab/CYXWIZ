#pragma once

#include "hdf5_input_settings.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace cyxwiz {
class ArrowDataset;

struct Hdf5RegisteredSourceRequest {
    std::string dataset_name;
    std::string resolved_path;
    std::string registered_source_path;
    Hdf5InputSettings settings;
    std::shared_ptr<ArrowDataset> dataset;
};

struct Hdf5RegisteredSourceResult {
    Hdf5TableStatus status = Hdf5TableStatus::ReadFailed;
    std::string error;
    int64_t rows = 0;
    int64_t columns = 0;
    size_t bytes = 0;
};

// Worker-only filesystem snapshot/metadata verification. Does not read HDF5
// payloads, audit/scan Arrow values, or touch the registry. The registered
// dataset must remain immutable. This is provenance matching, not a content hash.
Hdf5RegisteredSourceResult VerifyHdf5RegisteredSource(
    const Hdf5RegisteredSourceRequest& request,
    const std::function<bool()>& cancelled = {});

} // namespace cyxwiz
