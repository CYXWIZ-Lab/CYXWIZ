#pragma once

#include "dataset_batcher.h"
#include "debug_trace_record.h"
#include "graph_compiler.h"

#include <cstddef>
#include <string>

namespace cyxwiz {

inline constexpr size_t kDebugBatchInspectorMaxRows = 32;
inline constexpr size_t kDebugBatchInspectorMaxClasses = 64;

void AttachDebugBatchInspection(
    DebugTraceRecord& trace,
    const Batch& batch,
    const TrainingConfiguration& config,
    const std::string& dataset_name,
    const std::string& batcher_source,
    size_t requested_batch_size);

} // namespace cyxwiz
