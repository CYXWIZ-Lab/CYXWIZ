#pragma once

#include <arrow/result.h>
#include <arrow/status.h>

#include <memory>
#include <string>

namespace cyxwiz {

class SparseFeatureDataset;

/** Versioned Arrow IPC persistence for immutable sparse feature datasets. */
class SparseFeatureDatasetCache {
public:
    static constexpr int kFormatVersion = 1;

    static arrow::Status SaveAtomically(
        const SparseFeatureDataset& dataset,
        const std::string& path);

    static arrow::Result<std::shared_ptr<SparseFeatureDataset>> Load(
        const std::string& path);
};

} // namespace cyxwiz
