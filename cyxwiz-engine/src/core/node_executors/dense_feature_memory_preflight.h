#pragma once

#include "materialization_memory_preflight.h"

#include <arrow/api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

// Guards the common native row-major feature-matrix boundary before
// ReadFeatureMatrix copies Arrow columns into float temporaries and a dense
// double matrix. workspace_columns accounts conservatively for labels,
// predictions, and operator-specific model state that scales with row count.
inline arrow::Result<MaterializationMemoryEstimate>
EmitDenseFeatureMemoryPreflight(
    const std::shared_ptr<arrow::Table>& input,
    const std::vector<std::string>& resolved_features,
    uint64_t workspace_columns,
    const std::string& operation,
    const std::string& suggestion,
    const MaterializationMemoryContext& memory_context,
    const PipelineOperatorProgressCallback& callback,
    float progress = 0.15f) {
    const uint64_t planned_rows = static_cast<uint64_t>(
        std::max<int64_t>(0, input ? input->num_rows() : 0));
    if (planned_rows == 0) {
        return arrow::Status::Invalid(operation + ": input table has no rows");
    }
    if (resolved_features.empty()) {
        return arrow::Status::Invalid(
            operation + ": no numeric feature columns resolved");
    }

    uint64_t planned_columns = 0;
    const bool columns_valid = CheckedAddU64(
        static_cast<uint64_t>(resolved_features.size()),
        workspace_columns,
        planned_columns);
    const auto estimate = EstimateDenseMaterializationMemory(
        planned_rows,
        planned_columns,
        static_cast<uint64_t>(sizeof(double)));
    auto checked_estimate = estimate;
    if (!columns_valid) {
        checked_estimate.overflow = true;
        checked_estimate.estimated_peak_bytes =
            (std::numeric_limits<uint64_t>::max)();
    }

    return EmitMaterializationMemoryPreflight(
        checked_estimate,
        operation,
        "planned_dense_columns",
        suggestion,
        memory_context,
        callback,
        SaturatingMaterializationItemCount(planned_rows, planned_columns),
        progress);
}

} // namespace cyxwiz
