#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace arrow { class Table; }

namespace cyxwiz {

enum class Hdf5TableStatus {
    Ok, DependencyUnavailable, InvalidSelection, InvalidFile, MissingDataset,
    UnsupportedLayout, UnsupportedRank, UnsupportedType, EmptyDataset,
    LabelRowMismatch, ResourceLimit, Cancelled, ReadFailed
};

enum class Hdf5NumericPolicy { Preserve, Float64 };

struct Hdf5TableSelection {
    std::string data_path = "/data";
    std::string label_path;
};

struct Hdf5TableReadOptions {
    Hdf5TableSelection selection;
    Hdf5NumericPolicy numeric_policy = Hdf5NumericPolicy::Preserve;
    // Conservative Arrow payload + validity + scratch estimate, excluding
    // HDF5's internal caches and allocator overhead. Zero permits no load.
    uint64_t max_materialized_bytes = 256ULL * 1024 * 1024;
    std::function<bool()> cancel_requested;
};

struct Hdf5TableDatasetInfo {
    std::string path;
    std::vector<uint64_t> shape;
    std::string source_type;
};

struct Hdf5TableReadResult {
    Hdf5TableStatus status = Hdf5TableStatus::ReadFailed;
    std::string error;
    std::shared_ptr<arrow::Table> table;
    Hdf5TableDatasetInfo data;
    std::optional<Hdf5TableDatasetInfo> labels;
    uint64_t estimated_materialized_bytes = 0;
};

bool Hdf5TableSupportAvailable();

// Host file ingress, not Tensor computation. Explicit hard-linked numeric
// rank-1/2 data only; optional rank-1 labels append as column "label".
// Preserve retains primitive widths/sign; Float64 explicitly permits precision
// loss for compatibility with DataConvert. Reads are cancellable between slabs.
// At most 4096 output columns and 65536 values per scratch slab.
Hdf5TableReadResult ReadHdf5Table(
    const std::string& path, const Hdf5TableReadOptions& options);

} // namespace cyxwiz
