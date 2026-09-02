#pragma once

#include "materialization_cache.h"
#include "node_executors/pipeline_operator.h"
#include "../gui/node_editor.h"

#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

class DataRegistry;
class SparseFeatureDataset;

enum class PipelineMaterializerSourceKind {
    Unknown,
    ArrowTable,
    ParquetBacked,
    ImageDataset,
    AudioDataset,
    TextDataset,
    SparseFeatureDataset,
};

enum class MaterializationFailureKind {
    None,
    Cancelled,
    Capacity,
    Error,
};

const char* MaterializationFailureKindName(MaterializationFailureKind kind);

const char* PipelineMaterializerSourceKindName(
    PipelineMaterializerSourceKind kind);

PipelineMaterializerSourceKind ResolvePipelineMaterializerSourceKind(
    const DataRegistry& registry,
    const std::string& source_dataset_name);

/**
 * MaterializeResult - outcome of a PipelineMaterializer::Materialize pass.
 *
 * - effective_dataset_name: the name to dispatch to the training batcher.
 *   Either equals source_dataset_name when no Cat-1 operators fired or the
 *   source is not in-memory Arrow, or the registry name of a newly
 *   registered "<source>__materialized" Arrow dataset.
 * - operators_applied: how many Cat-1 IPipelineOperator instances ran.
 *   Zero means pass-through.
 * - source_kind / skipped_unsupported_source / unsupported_source_reason:
 *   records the storage scope that caused a pass-through when the source is
 *   not an in-memory Arrow table.
 * - cache_*: narrow persistent materialization cache diagnostics. Cache is
 *   disabled by default unless the caller supplies a MaterializationCacheConfig.
 * - success: false only if a Configure(), Apply(), registry operation, or
 *   explicit require-hit cache policy reported a hard error.
 * - error_message: populated only when success=false.
 */
struct MaterializeResult {
    std::string effective_dataset_name;
    int operators_applied = 0;
    PipelineMaterializerSourceKind source_kind =
        PipelineMaterializerSourceKind::Unknown;
    // Typed identity of the effective registry artifact. Today this matches
    // source_kind for pass-through and is ArrowTable after dense operator
    // materialization. Sparse vectorizer emission will set it explicitly.
    PipelineMaterializerSourceKind effective_kind =
        PipelineMaterializerSourceKind::Unknown;
    bool skipped_unsupported_source = false;
    std::string unsupported_source_reason;
    std::string diagnostic_message;
    MaterializationCacheStatus cache_status =
        MaterializationCacheStatus::Disabled;
    std::string cache_key;
    std::string cache_artifact_path;
    std::string cache_manifest_path;
    int64_t cache_row_count = 0;
    int64_t cache_column_count = 0;
    std::string cache_message;
    bool loaded_from_cache = false;
    bool saved_to_cache = false;
    bool success = true;
    std::string error_message;
    MaterializationFailureKind failure_kind = MaterializationFailureKind::None;
    int failed_node_id = -1;
    std::string failed_node_name;
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
    std::shared_ptr<SparseFeatureDataset> sparse_dataset;
    int operators_applied = 0;
    bool memory_preflight_observed = false;
    PipelineOperatorProgress memory_preflight;
    bool success = true;
    std::string error_message;
    MaterializationFailureKind failure_kind = MaterializationFailureKind::None;
    int failed_node_id = -1;
    std::string failed_node_name;
};

struct MaterializationCacheability {
    bool cacheable = true;
    bool valid = true;
    std::string reason;
    std::vector<MaterializationCacheDependencyIdentity> dependencies;
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
    static MaterializationCacheability EvaluateCacheability(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        const std::string& source_dataset_name = {});

    // Runs the same operator-owned estimator used by materialization and stops
    // at its first memory decision, before that operator starts materializing.
    // Later data-dependent operators remain unknown until their inputs exist.
    static MaterializeTableResult PreflightTable(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        const std::shared_ptr<arrow::Table>& source_table,
        const std::string& source_dataset_name = {},
        MaterializationMemoryContext memory_context = {});

    static MaterializeTableResult MaterializeTable(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        const std::shared_ptr<arrow::Table>& source_table,
        const std::string& source_dataset_name = {},
        PipelineOperatorProgressCallback progress_callback = {},
        PipelineOperatorExecutionContext execution_context = {});

    static MaterializeResult Materialize(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        DataRegistry& registry,
        const std::string& source_dataset_name,
        PipelineOperatorProgressCallback progress_callback = {},
        PipelineOperatorExecutionContext execution_context = {});

    static MaterializeResult Materialize(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        DataRegistry& registry,
        const std::string& source_dataset_name,
        const MaterializationCacheConfig& cache_config,
        PipelineOperatorProgressCallback progress_callback = {},
        PipelineOperatorExecutionContext execution_context = {});

    // Suffix used to register the transformed Arrow table when ops fire.
    // Exposed for the DataRegistry cleanup cascade and tests.
    static constexpr const char* kMaterializedSuffix = "__materialized";
};

} // namespace cyxwiz
