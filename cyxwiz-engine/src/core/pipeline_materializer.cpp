#include "pipeline_materializer.h"

#include "arrow_dataset.h"
#include "data_registry.h"
#include "materialization_memory_guard.h"
#include "pipeline_runtime_capabilities.h"

#include <arrow/table.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <filesystem>
#include <new>
#include <system_error>

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

bool CacheEnabled(const MaterializationCacheConfig& config) {
    return config.mode != MaterializationCacheMode::Disabled &&
           !config.cache_root.empty();
}

std::string FindNodeParameter(const gui::MLNode& node,
                              const std::vector<const char*>& names) {
    for (const char* name : names) {
        auto it = node.parameters.find(name);
        if (it != node.parameters.end() && !it->second.empty()) {
            return it->second;
        }
    }
    return {};
}

std::string FindSourceFilePath(const std::vector<gui::MLNode>& nodes,
                               const std::string& source_dataset_name) {
    const gui::MLNode* fallback_data_input = nullptr;
    for (const auto& node : nodes) {
        if (node.type != gui::NodeType::DataInput &&
            node.type != gui::NodeType::DatasetInput) {
            continue;
        }
        if (!fallback_data_input) {
            fallback_data_input = &node;
        }

        const std::string node_dataset = FindNodeParameter(
            node, {"dataset_name", "dataset"});
        if (!source_dataset_name.empty() &&
            node_dataset != source_dataset_name) {
            continue;
        }

        const std::string path = FindNodeParameter(
            node, {"file_path", "source_path", "raw_source_path",
                   "data_path", "path"});
        if (!path.empty()) {
            return path;
        }
    }

    if (fallback_data_input) {
        return FindNodeParameter(
            *fallback_data_input,
            {"file_path", "source_path", "raw_source_path",
             "data_path", "path"});
    }
    return {};
}

uint64_t FileTimeStampSeconds(const std::filesystem::file_time_type& time) {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
        time.time_since_epoch()).count();
    return seconds > 0 ? static_cast<uint64_t>(seconds) : 0;
}

void PopulateSourceFileIdentity(MaterializationCacheKeyInput& input,
                                const std::string& source_path) {
    if (source_path.empty()) {
        return;
    }

    std::error_code ec;
    std::filesystem::path path(source_path);
    auto absolute = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        ec.clear();
        absolute = std::filesystem::absolute(path, ec);
    }
    input.source_file_path = ec ? source_path : absolute.string();

    ec.clear();
    if (std::filesystem::is_regular_file(path, ec)) {
        ec.clear();
        input.source_file_size =
            static_cast<uint64_t>(std::filesystem::file_size(path, ec));
        if (ec) {
            input.source_file_size = 0;
        }
        ec.clear();
        input.source_file_mtime = FileTimeStampSeconds(
            std::filesystem::last_write_time(path, ec));
        if (ec) {
            input.source_file_mtime = 0;
        }
    }
}

MaterializationCacheKeyInput BuildCacheKeyInput(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::string& source_dataset_name,
    const std::string& source_schema_fingerprint,
    const std::vector<MaterializationCacheDependencyIdentity>& dependencies) {
    MaterializationCacheKeyInput input;
    input.source_dataset_name = source_dataset_name;
    input.source_identity = "arrow:" + source_dataset_name;
    input.source_schema_fingerprint = source_schema_fingerprint;
    input.dependencies = dependencies;
    input.nodes = nodes;
    input.links = links;
    PopulateSourceFileIdentity(
        input, FindSourceFilePath(nodes, source_dataset_name));
    return input;
}

std::shared_ptr<ArrowDataset> LoadCachedArrowDataset(
    const MaterializationCacheManifest& manifest,
    const std::string& materialized_name) {
    if (manifest.artifact_format == "feather") {
        return ArrowDataset::FromFeather(manifest.artifact_path, materialized_name);
    }
    return ArrowDataset::FromParquet(manifest.artifact_path, materialized_name);
}

bool ReplacePath(const std::filesystem::path& source,
                 const std::filesystem::path& target,
                 std::string* error) {
    std::error_code ec;
    std::filesystem::rename(source, target, ec);
    if (!ec) {
        return true;
    }

    std::filesystem::remove(target, ec);
    ec.clear();
    std::filesystem::rename(source, target, ec);
    if (ec) {
        if (error) {
            *error = ec.message();
        }
        return false;
    }
    return true;
}

bool ExportCachedArtifact(const std::shared_ptr<arrow::Table>& table,
                          const std::string& materialized_name,
                          const MaterializationCacheConfig& config,
                          const std::string& cache_key,
                          std::filesystem::path& artifact_path,
                          std::string* error) {
    artifact_path = MaterializationCacheArtifactPath(config, cache_key);
    std::error_code ec;
    std::filesystem::create_directories(artifact_path.parent_path(), ec);
    if (ec) {
        if (error) *error = ec.message();
        return false;
    }

    const auto temp_path = artifact_path.string() + ".tmp";
    std::filesystem::remove(temp_path, ec);

    ArrowDataset dataset(table, materialized_name);
    const bool exported = config.artifact_format == "feather"
        ? dataset.ExportFeather(temp_path)
        : dataset.ExportParquet(temp_path);
    if (!exported) {
        if (error) *error = "failed to export cached materialization artifact";
        std::filesystem::remove(temp_path, ec);
        return false;
    }

    return ReplacePath(temp_path, artifact_path, error);
}

void SaveCacheManifestLastUsed(const MaterializationCacheManifest& manifest,
                               const std::filesystem::path& manifest_path) {
    auto updated = manifest;
    updated.cache_status = MaterializationCacheStatus::Hit;
    updated.last_used_at.clear();
    std::string ignored;
    if (!WriteMaterializationCacheManifest(updated, manifest_path, &ignored)) {
        spdlog::warn("PipelineMaterializer: failed to update cache manifest '{}': {}",
                     manifest_path.string(), ignored);
    }
}

void SetMaterializationFailure(
    MaterializeResult& result,
    MaterializationFailureKind kind,
    std::string message,
    int node_id = -1,
    std::string node_name = {}) {
    result.success = false;
    result.failure_kind = kind;
    result.error_message = std::move(message);
    result.failed_node_id = node_id;
    result.failed_node_name = std::move(node_name);
}

bool StopIfMaterializationCancelled(
    const PipelineOperatorExecutionContext& context,
    MaterializeResult& result) {
    if (!context.IsCancellationRequested()) return false;
    SetMaterializationFailure(
        result,
        MaterializationFailureKind::Cancelled,
        "PipelineMaterializer: cancelled before publishing output");
    return true;
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

const char* MaterializationFailureKindName(MaterializationFailureKind kind) {
    switch (kind) {
    case MaterializationFailureKind::None:
        return "none";
    case MaterializationFailureKind::Cancelled:
        return "cancelled";
    case MaterializationFailureKind::Capacity:
        return "capacity";
    case MaterializationFailureKind::Error:
        return "error";
    }
    return "error";
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
    const std::string& source_dataset_name,
    PipelineOperatorProgressCallback progress_callback,
    PipelineOperatorExecutionContext execution_context) {
    MaterializationCacheConfig cache_config;
    return Materialize(nodes, links, registry, source_dataset_name,
                       cache_config, std::move(progress_callback),
                       std::move(execution_context));
}

MaterializeResult PipelineMaterializer::Materialize(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    DataRegistry& registry,
    const std::string& source_dataset_name,
    const MaterializationCacheConfig& cache_config,
    PipelineOperatorProgressCallback progress_callback,
    PipelineOperatorExecutionContext execution_context) try {

    MaterializeResult result;
    result.effective_dataset_name = source_dataset_name;
    result.cache_status = cache_config.mode == MaterializationCacheMode::Disabled
        ? MaterializationCacheStatus::Disabled
        : MaterializationCacheStatus::Miss;

    if (StopIfMaterializationCancelled(execution_context, result)) {
        return result;
    }

    if (source_dataset_name.empty()) {
        SetMaterializationFailure(
            result,
            MaterializationFailureKind::Error,
            "PipelineMaterializer: source_dataset_name is empty");
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
        result.diagnostic_message =
            "PipelineMaterializer skipped dataset '" + source_dataset_name +
            "' (" + PipelineMaterializerSourceKindName(result.source_kind) +
            "): " + result.unsupported_source_reason;
        if (cache_config.mode != MaterializationCacheMode::Disabled) {
            result.cache_status = MaterializationCacheStatus::Unsupported;
            result.cache_message = result.unsupported_source_reason;
        }
        spdlog::debug("{}", result.diagnostic_message);
        return result;
    }

    auto source_dataset = registry.GetArrowDataset(source_dataset_name);
    if (!source_dataset) {
        SetMaterializationFailure(
            result,
            MaterializationFailureKind::Error,
            "PipelineMaterializer: failed to fetch Arrow dataset '" +
                source_dataset_name + "' from registry");
        return result;
    }

    auto source_table = source_dataset->GetArrowTable();
    if (!source_table) {
        SetMaterializationFailure(
            result,
            MaterializationFailureKind::Error,
            "PipelineMaterializer: Arrow dataset '" +
                source_dataset_name + "' has a null table");
        return result;
    }

    const std::string materialized_name =
        source_dataset_name + kMaterializedSuffix;
    const std::string source_schema_fingerprint =
        ComputeSchemaFingerprint(source_table->schema());
    MaterializationCacheability cacheability;
    if (cache_config.mode != MaterializationCacheMode::Disabled) {
        cacheability = EvaluateCacheability(
            nodes, links, source_dataset_name);
    }
    if (!cacheability.valid) {
        result.cache_status = MaterializationCacheStatus::Unsupported;
        result.cache_message = "Materialization cache input is invalid: " +
                               cacheability.reason + ".";
        SetMaterializationFailure(
            result, MaterializationFailureKind::Error,
            "PipelineMaterializer: " + cacheability.reason);
        return result;
    }
    const bool cache_enabled =
        CacheEnabled(cache_config) && cacheability.cacheable;

    if (cache_config.mode != MaterializationCacheMode::Disabled &&
        !cacheability.cacheable) {
        result.cache_status = MaterializationCacheStatus::Unsupported;
        result.cache_message = "Materialization cache bypassed: " +
                               cacheability.reason + ".";
        if (cache_config.mode == MaterializationCacheMode::RequireHit) {
            SetMaterializationFailure(
                result, MaterializationFailureKind::Error,
                "PipelineMaterializer: require-hit cache policy cannot be "
                "used because " + cacheability.reason);
            return result;
        }
    }

    if (cache_config.mode != MaterializationCacheMode::Disabled &&
        cache_config.cache_root.empty() && cacheability.cacheable) {
        result.cache_status = MaterializationCacheStatus::Unsupported;
        result.cache_message = "materialization cache root is empty";
    }

    if (cache_enabled) {
        result.cache_key = ComputeMaterializationCacheKey(BuildCacheKeyInput(
            nodes, links, source_dataset_name, source_schema_fingerprint,
            cacheability.dependencies));
        const auto manifest_path =
            MaterializationCacheManifestPath(cache_config, result.cache_key);
        result.cache_manifest_path = manifest_path.string();
        result.cache_artifact_path =
            MaterializationCacheArtifactPath(cache_config, result.cache_key).string();

        if (cache_config.mode != MaterializationCacheMode::Rebuild) {
            MaterializationCacheManifest manifest;
            std::string read_error;
            if (ReadMaterializationCacheManifest(manifest_path, manifest, &read_error)) {
                auto validation = ValidateMaterializationCacheManifest(
                    manifest, result.cache_key, source_schema_fingerprint);
                result.cache_status = validation.status;
                result.cache_message = validation.message;
                if (!validation.manifest.artifact_path.empty()) {
                    result.cache_artifact_path = validation.manifest.artifact_path;
                }
                if (validation.usable) {
                    result.cache_row_count = validation.manifest.row_count;
                    result.cache_column_count = validation.manifest.column_count;
                }

                if (validation.usable) {
                    if (StopIfMaterializationCancelled(
                            execution_context, result)) {
                        return result;
                    }
                    const auto cache_load_started =
                        std::chrono::steady_clock::now();
                    auto cached = LoadCachedArrowDataset(
                        validation.manifest, materialized_name);
                    if (cached && cached->GetArrowTable()) {
                        if (StopIfMaterializationCancelled(
                                execution_context, result)) {
                            return result;
                        }
                        auto registered = registry.RegisterArrowTable(
                            cached->GetArrowTable(), materialized_name);
                        if (!registered) {
                            SetMaterializationFailure(
                                result,
                                MaterializationFailureKind::Error,
                                "PipelineMaterializer: RegisterArrowTable failed for cached '" +
                                    materialized_name + "'");
                            return result;
                        }
                        result.effective_dataset_name = materialized_name;
                        result.operators_applied =
                            validation.manifest.operators_applied;
                        result.cache_status = MaterializationCacheStatus::Hit;
                        result.loaded_from_cache = true;
                        const auto cache_load_elapsed =
                            std::chrono::steady_clock::now() -
                            cache_load_started;
                        const auto cache_load_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                cache_load_elapsed).count();
                        std::error_code artifact_size_error;
                        const uint64_t artifact_bytes =
                            static_cast<uint64_t>(std::filesystem::file_size(
                                result.cache_artifact_path,
                                artifact_size_error));
                        const uint64_t safe_artifact_bytes =
                            artifact_size_error ? 0 : artifact_bytes;
                        const uint64_t expanded_bytes =
                            static_cast<uint64_t>(cached->GetMemoryUsage());

                        std::ostringstream cache_message;
                        cache_message
                            << "Preprocessing skipped: reused cached "
                               "materialization ("
                            << result.cache_row_count << " rows x "
                            << result.cache_column_count << " columns";
                        if (safe_artifact_bytes > 0) {
                            cache_message << ", "
                                          << FormatMaterializationBytes(
                                                 safe_artifact_bytes)
                                          << " on disk";
                        }
                        if (expanded_bytes > 0) {
                            cache_message << " -> "
                                          << FormatMaterializationBytes(
                                                 expanded_bytes)
                                          << " in memory";
                        }
                        cache_message << ", loaded in "
                                      << std::fixed << std::setprecision(1)
                                      << static_cast<double>(cache_load_ms) /
                                             1000.0
                                      << " s).";
                        result.cache_message = cache_message.str();
                        SaveCacheManifestLastUsed(validation.manifest,
                                                  manifest_path);
                        spdlog::info(
                            "PipelineMaterializer: cache hit; skipped {} "
                            "preprocessing operator(s) by reusing '{}' -> '{}' "
                            "(rows={}, columns={}, artifact_bytes={}, "
                            "expanded_bytes={}, load_ms={}, path='{}')",
                            result.operators_applied,
                            source_dataset_name,
                            materialized_name,
                            result.cache_row_count,
                            result.cache_column_count,
                            safe_artifact_bytes,
                            expanded_bytes,
                            cache_load_ms,
                            result.cache_artifact_path);
                        return result;
                    }
                    result.cache_status = MaterializationCacheStatus::Corrupt;
                    result.cache_message =
                        "cached materialization artifact could not be loaded";
                }
            } else if (std::filesystem::exists(manifest_path)) {
                result.cache_status = MaterializationCacheStatus::Corrupt;
                result.cache_message = "cache manifest is corrupt: " + read_error;
            } else {
                result.cache_status = MaterializationCacheStatus::Miss;
                result.cache_message =
                    "Materialization cache miss; rebuilding prepared dataset.";
            }
        } else {
            result.cache_status = MaterializationCacheStatus::Stale;
            result.cache_message = "materialization cache rebuild requested";
        }

        if (cache_config.mode == MaterializationCacheMode::RequireHit) {
            std::string message =
                "PipelineMaterializer: materialization cache require-hit failed";
            if (!result.cache_message.empty()) {
                message += ": " + result.cache_message;
            }
            SetMaterializationFailure(
                result, MaterializationFailureKind::Error, std::move(message));
            return result;
        }
    }

    auto table_result = MaterializeTable(
        nodes, links, source_table, source_dataset_name, progress_callback,
        execution_context);
    if (!table_result.success) {
        SetMaterializationFailure(
            result,
            table_result.failure_kind == MaterializationFailureKind::None
                ? MaterializationFailureKind::Error
                : table_result.failure_kind,
            table_result.error_message,
            table_result.failed_node_id,
            table_result.failed_node_name);
        return result;
    }

    result.operators_applied = table_result.operators_applied;
    if (table_result.operators_applied == 0) {
        if (cache_enabled && result.cache_message.empty()) {
            result.cache_message =
                "no materializer operators applied; cache artifact not written";
        }
        result.cache_artifact_path.clear();
        result.cache_manifest_path.clear();
        return result;
    }

    if (StopIfMaterializationCancelled(execution_context, result)) {
        return result;
    }

    result.effective_dataset_name = materialized_name;

    if (cache_enabled) {
        std::filesystem::path artifact_path;
        std::string cache_error;
        if (ExportCachedArtifact(table_result.table, materialized_name,
                                 cache_config, result.cache_key,
                                 artifact_path, &cache_error)) {
            MaterializationCacheManifest manifest;
            manifest.cache_key = result.cache_key;
            manifest.source_dataset_name = source_dataset_name;
            manifest.effective_dataset_name = materialized_name;
            manifest.artifact_path = artifact_path.string();
            manifest.artifact_format = cache_config.artifact_format;
            manifest.row_count = table_result.table ? table_result.table->num_rows() : 0;
            manifest.column_count = table_result.table ? table_result.table->num_columns() : 0;
            result.cache_row_count = manifest.row_count;
            result.cache_column_count = manifest.column_count;
            manifest.schema_fingerprint = source_schema_fingerprint;
            manifest.dependencies = cacheability.dependencies;
            manifest.operators_applied = result.operators_applied;
            manifest.cache_status = MaterializationCacheStatus::Saved;

            const auto manifest_path =
                MaterializationCacheManifestPath(cache_config, result.cache_key);
            result.cache_manifest_path = manifest_path.string();
            if (WriteMaterializationCacheManifest(manifest, manifest_path,
                                                  &cache_error)) {
                result.cache_status = MaterializationCacheStatus::Saved;
                result.cache_artifact_path = artifact_path.string();
                result.cache_message = "Materialization completed and saved.";
                result.saved_to_cache = true;
            } else {
                result.cache_status = MaterializationCacheStatus::SaveFailed;
                result.cache_artifact_path = artifact_path.string();
                result.cache_message =
                    "materialization cache manifest save failed: " + cache_error;
            }
        } else {
            result.cache_status = MaterializationCacheStatus::SaveFailed;
            result.cache_message =
                "materialization cache artifact save failed: " + cache_error;
        }
    }

    if (StopIfMaterializationCancelled(execution_context, result)) {
        return result;
    }
    auto registered =
        registry.RegisterArrowTable(table_result.table, materialized_name);
    if (!registered) {
        SetMaterializationFailure(
            result,
            MaterializationFailureKind::Error,
            "PipelineMaterializer: RegisterArrowTable failed for '" +
                materialized_name + "'");
        return result;
    }

    spdlog::info("PipelineMaterializer: materialized '{}' -> '{}' ({} operators applied)",
                 source_dataset_name, materialized_name, result.operators_applied);
    return result;
} catch (const std::bad_alloc&) {
    MaterializeResult result;
    result.effective_dataset_name = source_dataset_name;
    SetMaterializationFailure(
        result,
        MaterializationFailureKind::Capacity,
        "PipelineMaterializer: allocation failed while preparing dataset '" +
            source_dataset_name +
            "'. Reduce input rows or output dimensions and retry.");
    return result;
}

} // namespace cyxwiz
