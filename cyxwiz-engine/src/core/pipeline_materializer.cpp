#include "pipeline_materializer.h"

#include "arrow_dataset.h"
#include "data_registry.h"

#include <arrow/table.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

MaterializeResult PipelineMaterializer::Materialize(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    DataRegistry& registry,
    const std::string& source_dataset_name) {

    MaterializeResult result;
    result.effective_dataset_name = source_dataset_name;

    if (source_dataset_name.empty()) {
        result.success = false;
        result.error_message = "PipelineMaterializer: source_dataset_name is empty";
        return result;
    }

    // v1: only the in-memory Arrow path is materialized. Parquet-backed,
    // image, audio, and legacy text sources fall through unchanged.
    if (!registry.IsArrowDataset(source_dataset_name)) {
        spdlog::debug("PipelineMaterializer: source '{}' is not an Arrow dataset, "
                      "skipping materialization", source_dataset_name);
        return result;
    }

    auto source_dataset = registry.GetArrowDataset(source_dataset_name);
    if (!source_dataset) {
        result.success = false;
        result.error_message = "PipelineMaterializer: failed to fetch Arrow dataset '" +
                               source_dataset_name + "' from registry";
        return result;
    }

    auto source_table = source_dataset->GetArrowTable();
    if (!source_table) {
        result.success = false;
        result.error_message = "PipelineMaterializer: Arrow dataset '" +
                               source_dataset_name + "' has a null table";
        return result;
    }

    auto table_result = MaterializeTable(
        nodes, links, source_table, source_dataset_name);
    if (!table_result.success) {
        result.success = false;
        result.error_message = table_result.error_message;
        return result;
    }

    result.operators_applied = table_result.operators_applied;
    if (table_result.operators_applied == 0) {
        return result;
    }

    const std::string materialized_name =
        source_dataset_name + kMaterializedSuffix;
    auto registered =
        registry.RegisterArrowTable(table_result.table, materialized_name);
    if (!registered) {
        result.success = false;
        result.error_message = "PipelineMaterializer: RegisterArrowTable failed for '" +
                               materialized_name + "'";
        return result;
    }

    result.effective_dataset_name = materialized_name;
    spdlog::info("PipelineMaterializer: materialized '{}' -> '{}' ({} operators applied)",
                 source_dataset_name, materialized_name, result.operators_applied);
    return result;
}

} // namespace cyxwiz
