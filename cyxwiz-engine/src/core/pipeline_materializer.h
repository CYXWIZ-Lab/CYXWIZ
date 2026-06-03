#pragma once

#include "../gui/node_editor.h"

#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

class DataRegistry;

/**
 * MaterializeResult - outcome of a PipelineMaterializer::Materialize pass.
 *
 * - effective_dataset_name: the name to dispatch to the training batcher.
 *   Either equals source_dataset_name when no Cat-1 operators fired or the
 *   source is not in-memory Arrow, or the registry name of a newly
 *   registered "<source>__materialized" Arrow dataset.
 * - operators_applied: how many Cat-1 IPipelineOperator instances ran.
 *   Zero means pass-through.
 * - success: false only if a Configure(), Apply(), or registry operation
 *   reported a hard error.
 * - error_message: populated only when success=false.
 */
struct MaterializeResult {
    std::string effective_dataset_name;
    int operators_applied = 0;
    bool success = true;
    std::string error_message;
};

/**
 * MaterializeTableResult - table-level outcome before registry storage.
 *
 * This is intentionally separate from MaterializeResult so tests and future
 * non-registry callers can validate the graph walk without pulling in every
 * DataRegistry loader dependency.
 */
struct MaterializeTableResult {
    std::shared_ptr<arrow::Table> table;
    int operators_applied = 0;
    bool success = true;
    std::string error_message;
};

/**
 * PipelineMaterializer - Cat-1 IPipelineOperator integration point.
 *
 * Walks the graph forward from DataInput and applies registered
 * IPipelineOperator instances to the current Arrow table. MaterializeTable()
 * performs only the table transformation. Materialize() wraps that result in
 * DataRegistry lookup/storage under "<source_dataset_name>__materialized".
 *
 * In-memory Arrow only: non-Arrow sources pass through unchanged. This keeps
 * Parquet, image, audio, and legacy text paths stable while Arrow operators
 * are migrated in focused slices.
 */
class PipelineMaterializer {
public:
    static MaterializeTableResult MaterializeTable(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        const std::shared_ptr<arrow::Table>& source_table,
        const std::string& source_dataset_name = {});

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
