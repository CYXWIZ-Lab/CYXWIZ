#pragma once

#include "../materialization_memory_guard.h"
#include "pipeline_operator.h"

#include <arrow/result.h>
#include <arrow/status.h>

#include <cstdint>
#include <limits>
#include <string>

namespace cyxwiz {

// Publishes one canonical operator preflight event and stops before the
// operator's first materializing allocation when the shared policy blocks it.
inline arrow::Result<MaterializationMemoryEstimate>
EmitMaterializationMemoryPreflight(
    MaterializationMemoryEstimate estimate,
    const std::string& operation,
    const std::string& dimension_name,
    const std::string& suggestion,
    const MaterializationMemoryContext& memory_context,
    const PipelineOperatorProgressCallback& callback,
    uint64_t total_items,
    float progress) {
    const auto decision = EvaluateMaterializationMemory(estimate, memory_context);
    const std::string message = BuildMaterializationMemoryPreflightMessage(
        operation, dimension_name, estimate, decision, suggestion);

    if (callback) {
        PipelineOperatorProgress event;
        event.stage = operation + " memory preflight";
        event.message = message;
        event.status = MaterializationMemoryRiskToProgressStatus(decision.risk);
        event.progress = progress;
        event.estimated_memory_bytes = estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(decision.risk);
        event.processed_items = 0;
        event.total_items = total_items;
        callback(event);
    }

    if (decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + message);
    }
    return estimate;
}

inline uint64_t SaturatingMaterializationItemCount(
    uint64_t rows,
    uint64_t dimensions) {
    uint64_t items = 0;
    if (!CheckedMulU64(rows, dimensions, items)) {
        return (std::numeric_limits<uint64_t>::max)();
    }
    return items;
}

} // namespace cyxwiz
