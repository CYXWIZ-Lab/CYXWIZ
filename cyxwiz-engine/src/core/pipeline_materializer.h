#pragma once

#include "../gui/node_editor.h"
#include <string>
#include <vector>

namespace cyxwiz {

class DataRegistry;

/**
 * MaterializeResult — outcome of a PipelineMaterializer::Materialize pass.
 *
 * - effective_dataset_name: the name to dispatch to the training batcher.
 *   Either equals source_dataset_name (when no Cat-1 operators fired or the
 *   source is not in-memory Arrow), or the registry name of a newly
 *   registered "<source>__materialized" Arrow dataset that holds the
 *   transformed table.
 * - operators_applied: how many Cat-1 IPipelineOperator instances ran.
 *   Zero means pass-through; the dispatcher should observe identical
 *   behavior to the pre-materializer code path.
 * - success: false only if a Configure() or Apply() call reported a hard
 *   error. A pass-through (no ops registered for any node) is success=true.
 * - error_message: populated only when success=false.
 */
struct MaterializeResult {
    std::string effective_dataset_name;
    int operators_applied = 0;
    bool success = true;
    std::string error_message;
};

/**
 * PipelineMaterializer — Stage 1 integration point for Cat-1 IPipelineOperator.
 *
 * Walks the training graph forward from the DataInput node and, for each
 * visited node that has a creator registered with PipelineOperatorFactory,
 * runs Configure() then Apply() on the current Arrow table. Threads the
 * resulting table through to the next op. When at least one operator fires,
 * the final table is registered in DataRegistry under
 * "<source_dataset_name>__materialized" and that name is returned as the
 * effective dataset for training launch.
 *
 * **In-memory Arrow only.** When the source dataset is not registered as
 * an Arrow dataset (Parquet-backed, image, audio, text, ...), the
 * materializer returns success with effective_dataset_name = source and
 * zero operators applied. This is intentional for v1: Phase 4 targets the
 * Arrow tabular path. Larger-than-RAM Parquet sources stay on the
 * disk-backed code path until a streaming Cat-1 path is designed.
 *
 * **No model layers.** The walk visits the entire forward closure from
 * DataInput, but model layers / loss / optimizer / output never have a
 * registered IPipelineOperator, so they're naturally no-ops. Adding a
 * Cat-1 operator downstream of model layers is user error and will be
 * surfaced by the compile gate in a later pass — not here.
 *
 * **Stateless.** Materializer is a free helper: no instance state, single
 * static entry point. Call from MainWindow::StartTrainingFromGraph after
 * BuildCompileResult succeeds and before the IsArrowDataset / IsParquetBackedDataset
 * dispatch.
 */
class PipelineMaterializer {
public:
    static MaterializeResult Materialize(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        DataRegistry& registry,
        const std::string& source_dataset_name);

    // Suffix used to register the transformed Arrow table when ops fire.
    // Exposed for the DataRegistry cleanup cascade and tests.
    static constexpr const char* kMaterializedSuffix = "__materialized";
};

} // namespace cyxwiz
