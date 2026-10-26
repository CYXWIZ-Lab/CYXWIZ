#pragma once

// The table training reads, kept on disk between sessions (TOFIX178 S0-e2):
// the result of the node wired into the Data Split's Training Dataset input,
// keyed by that node and every step before it. A Train after reopening the
// project loads it instead of running the data steps again.

#include "graph_model.h"
#include "materialization_cache.h"

#include <arrow/api.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

class ArrowDataset;
class DataRegistry;
class SparseFeatureDataset;

struct PreparedDataKey {
    // Every step before the Split is a table step on a loaded table: the
    // pipeline run prepares it (TOFIX178 S0-e2), cached or not.
    bool table_path = false;
    bool cacheable = false;
    std::string key;     // empty when not cacheable
    std::string reason;  // why not, when not
    std::string source_dataset_name;
    std::string source_schema_fingerprint;
    std::vector<MaterializationCacheDependencyIdentity> dependencies;
    int steps = 0;  // data steps before the feeder, the feeder included
    bool sparse = false;  // the feeder makes sparse features (TOFIX178 S0-f)
};

// The artifact format and file of a sparse entry.
inline constexpr const char* kSparseCacheArtifactFormat = "sparse_csr_arrow_ipc_v1";
std::filesystem::path SparseCacheArtifactPath(const MaterializationCacheConfig& config,
                                              const std::string& cache_key);

// The key of the table `feeder` prepares: the Data Input (its file's size
// and time, or its rows when it has no file), and each step from it to
// `feeder` with its settings and the files it reads. Not cacheable when a
// step keeps fitted state, runs SQL (a query can read any file) or is not a
// table step. The Data Input's data must be loaded.
PreparedDataKey ComputePreparedDataKey(const std::vector<gui::MLNode>& nodes,
                                       const std::vector<gui::NodeLink>& links,
                                       int feeder,
                                       DataRegistry& registry,
                                       const std::string& project_root);

// The manifest of a usable entry for `key`; false when none.
bool ProbePreparedData(const MaterializationCacheConfig& config,
                       const PreparedDataKey& key,
                       MaterializationCacheManifest* manifest = nullptr);

// Registers the entry: a table as `name`, sparse features under the name
// they were saved with. Returns the registered name; nothing (with the
// reason) when it cannot be read.
std::optional<std::string> LoadPreparedData(const MaterializationCacheConfig& config,
                                            const PreparedDataKey& key,
                                            DataRegistry& registry,
                                            const std::string& name,
                                            std::string* error = nullptr);

// Writes `table` as the entry for `key` (artifact, then manifest), then
// applies the cache's size policy.
bool SavePreparedData(const MaterializationCacheConfig& config,
                      const PreparedDataKey& key,
                      const std::shared_ptr<arrow::Table>& table,
                      const std::string& name,
                      std::string* error = nullptr);

// Writes sparse features as the entry for `key`.
bool SavePreparedSparseData(const MaterializationCacheConfig& config,
                            const PreparedDataKey& key,
                            const SparseFeatureDataset& dataset,
                            std::string* error = nullptr);

// Shared with the materializer: a source file's path, size and time in the
// key, the artifact file written atomically, and read back as a dataset.
void PopulateSourceFileIdentity(MaterializationCacheKeyInput& input, const std::string& source_path);
bool ExportPreparedArtifact(const std::shared_ptr<arrow::Table>& table,
                            const std::string& name,
                            const MaterializationCacheConfig& config,
                            const std::string& cache_key,
                            std::filesystem::path& artifact_path,
                            std::string* error);
std::shared_ptr<ArrowDataset> LoadPreparedArtifact(const MaterializationCacheManifest& manifest,
                                                   const std::string& name);

}  // namespace cyxwiz
