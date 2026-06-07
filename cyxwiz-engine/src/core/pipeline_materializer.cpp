#include "pipeline_materializer.h"

#include "arrow_dataset.h"
#include "data_registry.h"
#include "pipeline_runtime_capabilities.h"

#include <arrow/table.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

namespace {

PipelineStorageBackend ToStorageBackend(PipelineMaterializerSourceKind kind) {
    switch (kind) {
    case PipelineMaterializerSourceKind::ArrowTable:
        return PipelineStorageBackend::ArrowTable;
    case PipelineMaterializerSourceKind::ParquetBacked:
        return PipelineStorageBackend::ParquetBacked;
    case PipelineMaterializerSourceKind::ImageDataset:
        return PipelineStorageBackend::ImageDataset;
    case PipelineMaterializerSourceKind::AudioDataset:
        return PipelineStorageBackend::AudioDataset;
    case PipelineMaterializerSourceKind::TextDataset:
        return PipelineStorageBackend::TextDataset;
    case PipelineMaterializerSourceKind::Unknown:
        return PipelineStorageBackend::Unknown;
    }
    return PipelineStorageBackend::Unknown;
}

} // namespace

const char* PipelineMaterializerSourceKindName(
    PipelineMaterializerSourceKind kind) {
    switch (kind) {
    case PipelineMaterializerSourceKind::ArrowTable:
        return "ArrowTable";
    case PipelineMaterializerSourceKind::ParquetBacked:
        return "ParquetBacked";
    case PipelineMaterializerSourceKind::ImageDataset:
        return "ImageDataset";
    case PipelineMaterializerSourceKind::AudioDataset:
        return "AudioDataset";
    case PipelineMaterializerSourceKind::TextDataset:
        return "TextDataset";
    case PipelineMaterializerSourceKind::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

PipelineMaterializerSourceKind ResolvePipelineMaterializerSourceKind(
    const DataRegistry& registry,
    const std::string& source_dataset_name) {
    if (registry.IsArrowDataset(source_dataset_name)) {
        return PipelineMaterializerSourceKind::ArrowTable;
    }
    if (registry.IsParquetBackedDataset(source_dataset_name)) {
        return PipelineMaterializerSourceKind::ParquetBacked;
    }
    if (registry.IsImageDataset(source_dataset_name)) {
        return PipelineMaterializerSourceKind::ImageDataset;
    }
    if (registry.IsAudioDataset(source_dataset_name)) {
        return PipelineMaterializerSourceKind::AudioDataset;
    }
    if (registry.IsTextDataset(source_dataset_name)) {
        return PipelineMaterializerSourceKind::TextDataset;
    }
    return PipelineMaterializerSourceKind::Unknown;
}

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

    result.source_kind = ResolvePipelineMaterializerSourceKind(
        registry, source_dataset_name);

    const auto backend_support =
        ResolvePipelineMaterializerStorageBackendSupport(
            ToStorageBackend(result.source_kind));

    if (!backend_support.materializer_supported) {
        result.skipped_unsupported_source = true;
        result.unsupported_source_reason =
            backend_support.reason ? backend_support.reason
                                   : "storage backend is unsupported";
        spdlog::debug("PipelineMaterializer: source '{}' has kind '{}', "
                      "skipping materialization: {}",
                      source_dataset_name,
                      PipelineMaterializerSourceKindName(result.source_kind),
                      result.unsupported_source_reason);
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
