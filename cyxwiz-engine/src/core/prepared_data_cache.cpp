#include "prepared_data_cache.h"

#include "arrow_dataset.h"
#include "data_registry.h"
#include "node_executors/pipeline_operator_factory.h"
#include "parquet_backed_dataset.h"
#include "pipeline_runtime_capabilities.h"
#include "project_data_path.h"
#include "training_input_preparation.h"
#include "sparse_feature_dataset.h"
#include "sparse_feature_dataset_cache.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <set>
#include <system_error>

namespace cyxwiz {

namespace {

const gui::MLNode* FindNode(const std::vector<gui::MLNode>& nodes, int id) {
    for (const auto& node : nodes)
        if (node.id == id) return &node;
    return nullptr;
}

std::string Param(const gui::MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}

uint64_t FileTimeStampSeconds(const std::filesystem::file_time_type& time) {
    const auto seconds =
        std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch()).count();
    return seconds > 0 ? static_cast<uint64_t>(seconds) : 0;
}

bool ReplacePath(const std::filesystem::path& source, const std::filesystem::path& target,
                 std::string* error) {
    std::error_code ec;
    std::filesystem::rename(source, target, ec);
    if (!ec) return true;
    std::filesystem::remove(target, ec);
    ec.clear();
    std::filesystem::rename(source, target, ec);
    if (ec) {
        if (error) *error = ec.message();
        return false;
    }
    return true;
}

// A source file's path, size and time in the key.
void PopulateSourceFileIdentity(MaterializationCacheKeyInput& input, const std::string& source_path) {
    if (source_path.empty()) return;
    std::error_code ec;
    const std::filesystem::path path(source_path);
    auto absolute = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        ec.clear();
        absolute = std::filesystem::absolute(path, ec);
    }
    input.source_file_path = ec ? source_path : absolute.string();
    ec.clear();
    if (std::filesystem::is_regular_file(path, ec)) {
        ec.clear();
        input.source_file_size = static_cast<uint64_t>(std::filesystem::file_size(path, ec));
        if (ec) input.source_file_size = 0;
        ec.clear();
        input.source_file_mtime = FileTimeStampSeconds(std::filesystem::last_write_time(path, ec));
        if (ec) input.source_file_mtime = 0;
    }
}

// The artifact file written atomically, and read back as a dataset.
bool ExportPreparedArtifact(const std::shared_ptr<arrow::Table>& table,
                            const std::string& name,
                            const MaterializationCacheConfig& config,
                            const std::string& cache_key,
                            std::filesystem::path& artifact_path,
                            std::string* error);
std::shared_ptr<ArrowDataset> LoadPreparedArtifact(const MaterializationCacheManifest& manifest,
                                                   const std::string& name);

}  // namespace

PreparedDataKey ComputePreparedDataKey(const std::vector<gui::MLNode>& nodes,
                                       const std::vector<gui::NodeLink>& links,
                                       int feeder,
                                       DataRegistry& registry,
                                       const std::string& project_root) {
    PreparedDataKey result;
    // Not a table path: nothing more to learn.
    const auto refuse = [&result](std::string reason) {
        result.table_path = false;
        result.cacheable = false;
        result.key.clear();
        result.reason = std::move(reason);
        return result;
    };
    // A table path that is not kept on disk: the walk goes on.
    std::string keep_out;
    const auto not_kept = [&keep_out](std::string reason) {
        if (keep_out.empty()) keep_out = std::move(reason);
    };
    if (!FindNode(nodes, feeder)) return refuse("no step feeds the Data Split");
    {
        std::string format = Param(*FindNode(nodes, feeder), "output_format");
        std::transform(format.begin(), format.end(), format.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        result.sparse = format == "sparse";
    }

    // The feeder and every node before it.
    std::set<int> slice;
    std::vector<int> pending{feeder};
    while (!pending.empty()) {
        const int id = pending.back();
        pending.pop_back();
        if (!slice.insert(id).second) continue;
        for (const auto& link : links)
            if (link.to_node == id) pending.push_back(link.from_node);
    }

    MaterializationCacheKeyInput input;
    const gui::MLNode* source = nullptr;
    auto& factory = PipelineOperatorFactory::Instance();
    for (const int id : slice) {
        const gui::MLNode* node = FindNode(nodes, id);
        if (!node) return refuse("a step before the Data Split is missing");
        input.nodes.push_back(*node);
        if (node->type == gui::NodeType::DataInput || node->type == gui::NodeType::DatasetInput) {
            if (source) return refuse("more than one Data Input feeds the Data Split");
            source = node;
            continue;
        }
        ++result.steps;
        const auto support = ResolvePipelineRuntimeSupport(node->type);
        if (support.mode == PipelineRuntimeSupportMode::OperatorBacked &&
            support.materializer_arrow_table_supported) {
            auto op = factory.Create(node->type);
            std::string error;
            if (!op) {
                not_kept("'" + node->name + "' cannot be inspected");
            } else if (!op->Configure(node->parameters, error)) {
                not_kept("'" + node->name + "' settings are invalid: " + error);
            } else if (!op->IsCacheable()) {
                not_kept("'" + node->name + "' keeps fitted state");
            } else {
                std::vector<PipelineOperatorCacheDependency> dependencies;
                if (!op->CollectCacheDependencies(dependencies, error)) not_kept("'" + node->name + "': " + error);
                for (const auto& dependency : dependencies) {
                    MaterializationCacheDependencyIdentity identity;
                    if (ResolveMaterializationCacheDependencyIdentity(
                            node->id, dependency.role, ResolveProjectDataPath(dependency.path, project_root),
                            identity, &error))
                        result.dependencies.push_back(std::move(identity));
                    else
                        not_kept("'" + node->name + "': " + error);
                }
            }
        } else if (node->type == gui::NodeType::SQLQuery) {
            not_kept("'" + node->name + "' runs SQL, which can read any file");
        } else if (IsPipelineRunTrainingStep(node->type)) {
            // A setting naming a file (a lookup table, a Data Convert target)
            // makes the step depend on that file's content.
            for (const auto& [key, value] : node->parameters) {
                if (value.empty() || value.size() > 1024) continue;
                const std::string path = ResolveProjectDataPath(value, project_root);
                std::error_code ec;
                if (!std::filesystem::is_regular_file(path, ec)) continue;
                MaterializationCacheDependencyIdentity identity;
                std::string error;
                if (ResolveMaterializationCacheDependencyIdentity(node->id, "param:" + key, path, identity, &error))
                    result.dependencies.push_back(std::move(identity));
            }
        } else {
            return refuse("'" + node->name + "' is not a table step");
        }
    }
    if (!source) return refuse("no Data Input feeds the Data Split");
    for (const auto& link : links)
        if (slice.count(link.from_node) && slice.count(link.to_node)) input.links.push_back(link);

    result.source_dataset_name = Param(*source, "dataset_name");
    std::shared_ptr<arrow::Table> source_table;
    if (const auto memory = registry.GetArrowDataset(result.source_dataset_name)) {
        source_table = memory->GetArrowTable();
        if (source_table) result.source_schema_fingerprint = ComputeSchemaFingerprint(source_table->schema());
    } else if (const auto disk = registry.GetParquetBackedDataset(result.source_dataset_name)) {
        result.source_schema_fingerprint = ComputeSchemaFingerprint(disk->GetSchema());
    } else {
        // Still a table path: a run loads the data itself.
        result.table_path = true;
        result.reason = "the Data Input's data is not loaded";
        return result;
    }
    result.table_path = true;
    if (!keep_out.empty()) {
        result.reason = keep_out;
        return result;
    }

    // "executor:" keeps these entries apart from the materializer's.
    input.source_dataset_name = result.source_dataset_name;
    input.source_identity = "executor:" + result.source_dataset_name;
    // The steps learn from the Train rows of the graph's split (TOFIX178 S0-g2).
    const auto policy = TrainingSplitPolicyOf(nodes, links);
    if (policy) input.source_identity += "|split:" + TrainingSplitPolicyJson(*policy).dump();
    // Supplied Validation / Test rows are in the entry (TOFIX178 S0-g4): their
    // Data Inputs' settings and files are in the key.
    if (policy && policy->Supplies()) {
        for (const auto& [id, role] : {std::pair{policy->validation_input, "Validation"},
                                       std::pair{policy->test_input, "Test"}}) {
            const gui::MLNode* supplied = id >= 0 ? FindNode(nodes, id) : nullptr;
            if (!supplied) continue;
            input.nodes.push_back(*supplied);
            std::string path;
            for (const char* key : {"file_path", "source_path", "raw_source_path", "data_path", "path"}) {
                path = Param(*supplied, key);
                if (!path.empty()) break;
            }
            MaterializationCacheDependencyIdentity identity;
            std::string error;
            if (!path.empty() &&
                ResolveMaterializationCacheDependencyIdentity(supplied->id, std::string("supplied:") + role,
                                                              ResolveProjectDataPath(path, project_root), identity,
                                                              &error))
                result.dependencies.push_back(std::move(identity));
            else
                not_kept(std::string("the ") + role + " input has no file to identify it");
        }
        if (!keep_out.empty()) {
            result.reason = keep_out;
            return result;
        }
    }
    input.source_schema_fingerprint = result.source_schema_fingerprint;
    input.dependencies = result.dependencies;
    std::string source_path;
    for (const char* key : {"file_path", "source_path", "raw_source_path", "data_path", "path"}) {
        source_path = Param(*source, key);
        if (!source_path.empty()) break;
    }
    PopulateSourceFileIdentity(input, ResolveProjectDataPath(source_path, project_root));
    if (input.source_file_size == 0 && input.source_file_mtime == 0) {
        if (!source_table) {
            result.reason = "the Data Input has no file and its rows are kept on disk";
            return result;
        }
        input.source_content_fingerprint = ComputeTableContentFingerprint(source_table);
    }
    result.cacheable = true;
    result.key = ComputeMaterializationCacheKey(input);
    return result;
}

std::filesystem::path SparseCacheArtifactPath(const MaterializationCacheConfig& config,
                                              const std::string& cache_key) {
    return MaterializationCacheEntryDirectory(config, cache_key) / "data.csr.arrow";
}

bool ProbePreparedData(const MaterializationCacheConfig& config,
                       const PreparedDataKey& key,
                       MaterializationCacheManifest* manifest) {
    if (!key.cacheable || config.mode == MaterializationCacheMode::Disabled ||
        config.mode == MaterializationCacheMode::Rebuild || config.cache_root.empty())
        return false;
    MaterializationCacheManifest read;
    if (!ReadMaterializationCacheManifest(MaterializationCacheManifestPath(config, key.key), read))
        return false;
    const auto validation =
        ValidateMaterializationCacheManifest(read, key.key, key.source_schema_fingerprint);
    const std::string format = key.sparse ? kSparseCacheArtifactFormat : config.artifact_format;
    if (!validation.usable || validation.manifest.artifact_format != format) return false;
    if (manifest) *manifest = validation.manifest;
    return true;
}

std::optional<std::string> LoadPreparedData(const MaterializationCacheConfig& config,
                                            const PreparedDataKey& key,
                                            DataRegistry& registry,
                                            const std::string& name,
                                            std::string* error) {
    MaterializationCacheManifest manifest;
    if (!ProbePreparedData(config, key, &manifest)) {
        if (error) *error = "no usable prepared data on disk";
        return std::nullopt;
    }
    std::string registered = name;
    if (key.sparse) {
        // Sparse features keep the name they were saved under.
        auto loaded = SparseFeatureDatasetCache::Load(manifest.artifact_path);
        if (!loaded.ok()) {
            if (error) *error = "the prepared sparse features could not be read: " + loaded.status().ToString();
            return std::nullopt;
        }
        registered = loaded.ValueOrDie()->GetName();
        if (!registry.RegisterSparseFeatureDataset(loaded.ValueOrDie())) {
            if (error) *error = "the prepared sparse features could not be registered";
            return std::nullopt;
        }
    } else {
        const auto dataset = LoadPreparedArtifact(manifest, name);
        if (!dataset || !dataset->GetArrowTable()) {
            if (error) *error = "the prepared data file could not be read: " + manifest.artifact_path;
            return std::nullopt;
        }
        if (!registry.RegisterArrowTable(dataset->GetArrowTable(), name)) {
            if (error) *error = "the prepared data could not be registered";
            return std::nullopt;
        }
    }
    auto used = manifest;
    used.cache_status = MaterializationCacheStatus::Hit;
    used.last_used_at.clear();
    WriteMaterializationCacheManifest(used, MaterializationCacheManifestPath(config, key.key));
    return registered;
}

namespace {

// The manifest after an artifact is written, then the size policy.
bool FinishPreparedEntry(const MaterializationCacheConfig& config,
                         const PreparedDataKey& key,
                         const std::string& name,
                         const std::filesystem::path& artifact_path,
                         const std::string& format,
                         int64_t rows,
                         int64_t columns,
                         std::string* error) {
    MaterializationCacheManifest manifest;
    manifest.cache_key = key.key;
    manifest.source_dataset_name = key.source_dataset_name;
    manifest.effective_dataset_name = name;
    manifest.artifact_path = artifact_path.string();
    manifest.artifact_format = format;
    manifest.row_count = rows;
    manifest.column_count = columns;
    manifest.schema_fingerprint = key.source_schema_fingerprint;
    manifest.dependencies = key.dependencies;
    manifest.operators_applied = key.steps;
    manifest.cache_status = MaterializationCacheStatus::Saved;
    if (!WriteMaterializationCacheManifest(manifest, MaterializationCacheManifestPath(config, key.key), error))
        return false;
    if (config.max_total_bytes > 0 || config.max_entries > 0) {
        const auto pruned = PruneMaterializationCache(config, key.key);
        if (pruned.removed_entries > 0)
            spdlog::info("Prepared data cache: the size policy removed {} older entr{} ({} bytes)",
                         pruned.removed_entries, pruned.removed_entries == 1 ? "y" : "ies", pruned.freed_bytes);
        if (!pruned.error.empty()) spdlog::warn("Prepared data cache cleanup: {}", pruned.error);
    }
    return true;
}

bool CanSave(const MaterializationCacheConfig& config, const PreparedDataKey& key, std::string* error) {
    if (key.cacheable && config.mode != MaterializationCacheMode::Disabled && !config.cache_root.empty())
        return true;
    if (error) *error = key.cacheable ? "the cache is off" : key.reason;
    return false;
}

}  // namespace

bool SavePreparedData(const MaterializationCacheConfig& config,
                      const PreparedDataKey& key,
                      const std::shared_ptr<arrow::Table>& table,
                      const std::string& name,
                      std::string* error) {
    if (!table || !CanSave(config, key, error)) return false;
    std::filesystem::path artifact_path;
    if (!ExportPreparedArtifact(table, name, config, key.key, artifact_path, error)) return false;
    return FinishPreparedEntry(config, key, name, artifact_path, config.artifact_format, table->num_rows(),
                               table->num_columns(), error);
}

bool SavePreparedSparseData(const MaterializationCacheConfig& config,
                            const PreparedDataKey& key,
                            const SparseFeatureDataset& dataset,
                            std::string* error) {
    if (!CanSave(config, key, error)) return false;
    const auto artifact_path = SparseCacheArtifactPath(config, key.key);
    std::error_code ec;
    std::filesystem::create_directories(artifact_path.parent_path(), ec);
    const auto status = SparseFeatureDatasetCache::SaveAtomically(dataset, artifact_path.string());
    if (!status.ok()) {
        if (error) *error = status.ToString();
        return false;
    }
    return FinishPreparedEntry(config, key, dataset.GetName(), artifact_path, kSparseCacheArtifactFormat,
                               dataset.GetNumRows(), dataset.GetNumFeatures() + (dataset.GetLabels() ? 1 : 0), error);
}

namespace {

bool ExportPreparedArtifact(const std::shared_ptr<arrow::Table>& table,
                            const std::string& name,
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
    ArrowDataset dataset(table, name);
    const bool exported = config.artifact_format == "feather" ? dataset.ExportFeather(temp_path)
                                                              : dataset.ExportParquet(temp_path);
    if (!exported) {
        if (error) *error = "failed to export the prepared data file";
        std::filesystem::remove(temp_path, ec);
        return false;
    }
    return ReplacePath(temp_path, artifact_path, error);
}

std::shared_ptr<ArrowDataset> LoadPreparedArtifact(const MaterializationCacheManifest& manifest,
                                                   const std::string& name) {
    if (manifest.artifact_format == "feather") return ArrowDataset::FromFeather(manifest.artifact_path, name);
    return ArrowDataset::FromParquet(manifest.artifact_path, name);
}

}  // namespace

}  // namespace cyxwiz
