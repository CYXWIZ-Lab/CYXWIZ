#pragma once

#include "../gui/node_editor.h"

#include <arrow/api.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

inline constexpr int kMaterializationCacheSchemaVersion = 2;

enum class MaterializationCacheMode {
    Disabled,
    Auto,
    Rebuild,
    RequireHit
};

enum class MaterializationCacheStatus {
    Disabled,
    Miss,
    Hit,
    Stale,
    Saved,
    SaveFailed,
    Corrupt,
    Unsupported
};

struct MaterializationCacheConfig {
    MaterializationCacheMode mode = MaterializationCacheMode::Disabled;
    std::filesystem::path cache_root;
    std::string artifact_format = "parquet";
};

struct MaterializationCacheDependencyIdentity {
    int node_id = -1;
    std::string role;
    std::string path;
    uint64_t byte_size = 0;
    std::string content_sha256;
};

struct MaterializationCacheKeyInput {
    std::string source_dataset_name;
    std::string source_identity;
    std::string source_file_path;
    uint64_t source_file_size = 0;
    uint64_t source_file_mtime = 0;
    std::string source_schema_fingerprint;
    std::vector<MaterializationCacheDependencyIdentity> dependencies;
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
};

struct MaterializationCacheManifest {
    std::string cache_key;
    std::string source_dataset_name;
    std::string effective_dataset_name;
    std::string artifact_path;
    std::string artifact_format = "parquet";
    int64_t row_count = 0;
    int64_t column_count = 0;
    std::string schema_fingerprint;
    std::vector<MaterializationCacheDependencyIdentity> dependencies;
    int operators_applied = 0;
    std::string engine_version;
    int materializer_cache_schema_version =
        kMaterializationCacheSchemaVersion;
    std::string created_at;
    std::string last_used_at;
    MaterializationCacheStatus cache_status = MaterializationCacheStatus::Miss;
    std::string stale_reason;
};

struct MaterializationCacheValidationResult {
    MaterializationCacheStatus status = MaterializationCacheStatus::Miss;
    bool usable = false;
    std::string message;
    MaterializationCacheManifest manifest;
};

const char* MaterializationCacheModeName(MaterializationCacheMode mode);
const char* MaterializationCacheStatusName(MaterializationCacheStatus status);

std::string ComputeSchemaFingerprint(
    const std::shared_ptr<arrow::Schema>& schema);
std::string ComputeMaterializationCacheKey(
    const MaterializationCacheKeyInput& input);
bool ResolveMaterializationCacheDependencyIdentity(
    int node_id,
    const std::string& role,
    const std::string& path,
    MaterializationCacheDependencyIdentity& identity,
    std::string* error = nullptr);

std::filesystem::path MaterializationCacheEntryDirectory(
    const MaterializationCacheConfig& config,
    const std::string& cache_key);
std::filesystem::path MaterializationCacheManifestPath(
    const MaterializationCacheConfig& config,
    const std::string& cache_key);
std::filesystem::path MaterializationCacheArtifactPath(
    const MaterializationCacheConfig& config,
    const std::string& cache_key);

bool WriteMaterializationCacheManifest(
    const MaterializationCacheManifest& manifest,
    const std::filesystem::path& manifest_path,
    std::string* error = nullptr);
bool ReadMaterializationCacheManifest(
    const std::filesystem::path& manifest_path,
    MaterializationCacheManifest& manifest,
    std::string* error = nullptr);

MaterializationCacheValidationResult ValidateMaterializationCacheManifest(
    const MaterializationCacheManifest& manifest,
    const std::string& expected_cache_key,
    const std::string& expected_schema_fingerprint);

} // namespace cyxwiz
