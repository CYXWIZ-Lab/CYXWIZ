#include "materialization_cache.h"

#include <cyxwiz/utilities.h>
#include <nlohmann/json.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cyxwiz {
namespace {

constexpr const char* kMaterializerOperatorVersion = "materializer-v1";

std::string Hex(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

uint64_t Fnv1a64(const std::string& text) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char ch : text) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string StableFingerprint(const std::string& text) {
    return Hex(Fnv1a64(text));
}

// 128-bit key: two FNV-1a 64 passes with independent offset bases.
std::string StableKey(const std::string& text) {
    uint64_t second = 0x6c62272e07bb0142ull;
    for (unsigned char ch : text) {
        second ^= static_cast<uint64_t>(ch);
        second *= 1099511628211ull;
    }
    return Hex(Fnv1a64(text)) + Hex(second);
}

// Word-at-a-time FNV-style mix for large buffers.
void MixBytes(uint64_t& hash, const uint8_t* data, int64_t size) {
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        uint64_t word = 0;
        std::memcpy(&word, data + i, sizeof(word));
        hash ^= word;
        hash *= 1099511628211ull;
        hash ^= hash >> 29;
    }
    for (; i < size; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 1099511628211ull;
    }
}

void MixValue(uint64_t& hash, int64_t value) {
    MixBytes(hash, reinterpret_cast<const uint8_t*>(&value), sizeof(value));
}

void MixArrayData(uint64_t& hash, const std::shared_ptr<arrow::ArrayData>& data) {
    if (!data) {
        MixValue(hash, -1);
        return;
    }
    MixValue(hash, data->length);
    MixValue(hash, data->offset);
    MixValue(hash, data->null_count.load());
    for (const auto& buffer : data->buffers) {
        if (buffer && buffer->is_cpu()) {
            MixValue(hash, buffer->size());
            MixBytes(hash, buffer->data(), buffer->size());
        } else {
            MixValue(hash, 0);
        }
    }
    for (const auto& child : data->child_data) {
        MixArrayData(hash, child);
    }
    if (data->dictionary) {
        MixArrayData(hash, data->dictionary);
    }
}

std::string NowIsoLikeUtc() {
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto count = seconds.time_since_epoch().count();
    return std::to_string(count);
}

std::string EscapePart(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '|':
            out << "\\|";
            break;
        case '=':
            out << "\\=";
            break;
        default:
            out << ch;
            break;
        }
    }
    return out.str();
}

std::string CanonicalKeyInput(const MaterializationCacheKeyInput& input) {
    std::ostringstream out;
    out << "schema_version=" << kMaterializationCacheSchemaVersion << "\n";
    out << "operator_version=" << kMaterializerOperatorVersion << "\n";
    out << "source_dataset_name=" << EscapePart(input.source_dataset_name) << "\n";
    out << "source_identity=" << EscapePart(input.source_identity) << "\n";
    out << "source_file_path=" << EscapePart(input.source_file_path) << "\n";
    out << "source_file_size=" << input.source_file_size << "\n";
    out << "source_file_mtime=" << input.source_file_mtime << "\n";
    out << "source_schema=" << EscapePart(input.source_schema_fingerprint) << "\n";
    if (!input.source_content_fingerprint.empty()) {
        out << "source_content=" << EscapePart(input.source_content_fingerprint) << "\n";
    }

    auto dependencies = input.dependencies;
    std::sort(
        dependencies.begin(), dependencies.end(),
        [](const MaterializationCacheDependencyIdentity& a,
           const MaterializationCacheDependencyIdentity& b) {
            if (a.node_id != b.node_id) return a.node_id < b.node_id;
            if (a.role != b.role) return a.role < b.role;
            if (a.path != b.path) return a.path < b.path;
            return a.content_sha256 < b.content_sha256;
        });
    for (const auto& dependency : dependencies) {
        out << "dependency=" << dependency.node_id << "|"
            << EscapePart(dependency.role) << "|"
            << EscapePart(dependency.path) << "|"
            << dependency.byte_size << "|"
            << EscapePart(dependency.content_sha256) << "\n";
    }

    auto nodes = input.nodes;
    std::sort(nodes.begin(), nodes.end(), [](const gui::MLNode& a,
                                             const gui::MLNode& b) {
        return a.id < b.id;
    });
    for (const auto& node : nodes) {
        // Display names do not change the prepared data.
        out << "node=" << node.id << "|"
            << static_cast<int>(node.type) << "\n";
        for (const auto& [key, value] : node.parameters) {
            out << "param=" << node.id << "|"
                << EscapePart(key) << "=" << EscapePart(value) << "\n";
        }
    }

    auto links = input.links;
    std::sort(links.begin(), links.end(), [](const gui::NodeLink& a,
                                             const gui::NodeLink& b) {
        if (a.from_node != b.from_node) return a.from_node < b.from_node;
        if (a.from_pin != b.from_pin) return a.from_pin < b.from_pin;
        if (a.to_node != b.to_node) return a.to_node < b.to_node;
        if (a.to_pin != b.to_pin) return a.to_pin < b.to_pin;
        return a.id < b.id;
    });
    for (const auto& link : links) {
        out << "link=" << link.from_node << "|"
            << link.from_pin << "|"
            << link.to_node << "|"
            << link.to_pin << "|"
            << static_cast<int>(link.type) << "\n";
    }

    return out.str();
}

MaterializationCacheStatus StatusFromName(const std::string& name) {
    if (name == "disabled") return MaterializationCacheStatus::Disabled;
    if (name == "miss") return MaterializationCacheStatus::Miss;
    if (name == "hit") return MaterializationCacheStatus::Hit;
    if (name == "stale") return MaterializationCacheStatus::Stale;
    if (name == "saved") return MaterializationCacheStatus::Saved;
    if (name == "save_failed") return MaterializationCacheStatus::SaveFailed;
    if (name == "corrupt") return MaterializationCacheStatus::Corrupt;
    if (name == "unsupported") return MaterializationCacheStatus::Unsupported;
    return MaterializationCacheStatus::Corrupt;
}

nlohmann::json ManifestToJson(const MaterializationCacheManifest& manifest) {
    nlohmann::json dependencies = nlohmann::json::array();
    for (const auto& dependency : manifest.dependencies) {
        dependencies.push_back({
            {"node_id", dependency.node_id},
            {"role", dependency.role},
            {"path", dependency.path},
            {"byte_size", dependency.byte_size},
            {"content_sha256", dependency.content_sha256},
        });
    }
    return {
        {"cache_key", manifest.cache_key},
        {"source_dataset_name", manifest.source_dataset_name},
        {"effective_dataset_name", manifest.effective_dataset_name},
        {"artifact_path", manifest.artifact_path},
        {"artifact_format", manifest.artifact_format},
        {"row_count", manifest.row_count},
        {"column_count", manifest.column_count},
        {"schema_fingerprint", manifest.schema_fingerprint},
        {"dependencies", std::move(dependencies)},
        {"operators_applied", manifest.operators_applied},
        {"engine_version", manifest.engine_version},
        {"materializer_cache_schema_version",
         manifest.materializer_cache_schema_version},
        {"created_at", manifest.created_at},
        {"last_used_at", manifest.last_used_at},
        {"cache_status", MaterializationCacheStatusName(manifest.cache_status)},
        {"stale_reason", manifest.stale_reason},
    };
}

bool JsonToManifest(const nlohmann::json& j,
                    MaterializationCacheManifest& manifest,
                    std::string* error) {
    try {
        manifest.cache_key = j.at("cache_key").get<std::string>();
        manifest.source_dataset_name =
            j.value("source_dataset_name", std::string{});
        manifest.effective_dataset_name =
            j.value("effective_dataset_name", std::string{});
        manifest.artifact_path = j.value("artifact_path", std::string{});
        manifest.artifact_format = j.value("artifact_format", "parquet");
        manifest.row_count = j.value("row_count", int64_t{0});
        manifest.column_count = j.value("column_count", int64_t{0});
        manifest.schema_fingerprint =
            j.value("schema_fingerprint", std::string{});
        manifest.dependencies.clear();
        if (const auto dependencies = j.find("dependencies");
            dependencies != j.end()) {
            if (!dependencies->is_array()) {
                throw std::runtime_error("dependencies must be an array");
            }
            for (const auto& value : *dependencies) {
                MaterializationCacheDependencyIdentity dependency;
                dependency.node_id = value.value("node_id", -1);
                dependency.role = value.value("role", std::string{});
                dependency.path = value.value("path", std::string{});
                dependency.byte_size = value.value("byte_size", uint64_t{0});
                dependency.content_sha256 =
                    value.value("content_sha256", std::string{});
                manifest.dependencies.push_back(std::move(dependency));
            }
        }
        manifest.operators_applied = j.value("operators_applied", 0);
        manifest.engine_version = j.value("engine_version", std::string{});
        manifest.materializer_cache_schema_version =
            j.value("materializer_cache_schema_version", 0);
        manifest.created_at = j.value("created_at", std::string{});
        manifest.last_used_at = j.value("last_used_at", std::string{});
        manifest.cache_status =
            StatusFromName(j.value("cache_status", std::string{"corrupt"}));
        manifest.stale_reason = j.value("stale_reason", std::string{});
        return true;
    } catch (const std::exception& ex) {
        if (error) {
            *error = ex.what();
        }
        return false;
    }
}

} // namespace

const char* MaterializationCacheModeName(MaterializationCacheMode mode) {
    switch (mode) {
    case MaterializationCacheMode::Disabled:
        return "disabled";
    case MaterializationCacheMode::Auto:
        return "auto";
    case MaterializationCacheMode::Rebuild:
        return "rebuild";
    case MaterializationCacheMode::RequireHit:
        return "require_hit";
    }
    return "unknown";
}

const char* MaterializationCacheStatusName(MaterializationCacheStatus status) {
    switch (status) {
    case MaterializationCacheStatus::Disabled:
        return "disabled";
    case MaterializationCacheStatus::Miss:
        return "miss";
    case MaterializationCacheStatus::Hit:
        return "hit";
    case MaterializationCacheStatus::Stale:
        return "stale";
    case MaterializationCacheStatus::Saved:
        return "saved";
    case MaterializationCacheStatus::SaveFailed:
        return "save_failed";
    case MaterializationCacheStatus::Corrupt:
        return "corrupt";
    case MaterializationCacheStatus::Unsupported:
        return "unsupported";
    }
    return "unknown";
}

std::string ComputeSchemaFingerprint(
    const std::shared_ptr<arrow::Schema>& schema) {
    if (!schema) {
        return StableFingerprint("null_schema");
    }
    return StableFingerprint(schema->ToString(/*show_metadata=*/true));
}

std::string ComputeTableContentFingerprint(
    const std::shared_ptr<arrow::Table>& table) {
    if (!table) {
        return StableFingerprint("null_table");
    }
    uint64_t hash = 14695981039346656037ull;
    MixValue(hash, table->num_rows());
    MixValue(hash, table->num_columns());
    for (const auto& column : table->columns()) {
        if (!column) {
            MixValue(hash, -1);
            continue;
        }
        MixValue(hash, column->num_chunks());
        for (const auto& chunk : column->chunks()) {
            MixArrayData(hash, chunk ? chunk->data() : nullptr);
        }
    }
    return Hex(hash);
}

std::string ComputeMaterializationCacheKey(
    const MaterializationCacheKeyInput& input) {
    return StableKey(CanonicalKeyInput(input));
}

bool ResolveMaterializationCacheDependencyIdentity(
    int node_id,
    const std::string& role,
    const std::string& path_text,
    MaterializationCacheDependencyIdentity& identity,
    std::string* error) {
    if (role.empty()) {
        if (error) *error = "cache dependency role is empty";
        return false;
    }
    if (path_text.empty()) {
        if (error) *error = "cache dependency path is empty";
        return false;
    }

    std::error_code ec;
    std::filesystem::path path(path_text);
    auto normalized = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        ec.clear();
        normalized = std::filesystem::absolute(path, ec).lexically_normal();
    }
    if (ec || !std::filesystem::is_regular_file(normalized, ec) || ec) {
        if (error) {
            *error = "cache dependency is not a readable file at '" +
                     normalized.string() + "'";
        }
        return false;
    }

    const auto byte_size = std::filesystem::file_size(normalized, ec);
    if (ec) {
        if (error) {
            *error = "could not read cache dependency size at '" +
                     normalized.string() + "': " + ec.message();
        }
        return false;
    }

    const auto hash = Utilities::HashFile(normalized.string(), "sha256");
    const bool valid_sha256 = hash.success &&
        hash.sha256_hash.size() == 64 &&
        std::all_of(
            hash.sha256_hash.begin(), hash.sha256_hash.end(),
            [](unsigned char ch) { return std::isxdigit(ch) != 0; });
    if (!valid_sha256) {
        if (error) {
            *error = hash.error_message.empty()
                ? "could not compute SHA-256 for cache dependency '" +
                      normalized.string() + "'"
                : hash.error_message;
        }
        return false;
    }

    identity = {};
    identity.node_id = node_id;
    identity.role = role;
    identity.path = normalized.string();
    identity.byte_size = byte_size;
    identity.content_sha256 = hash.sha256_hash;
    std::transform(
        identity.content_sha256.begin(), identity.content_sha256.end(),
        identity.content_sha256.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return true;
}

std::filesystem::path MaterializationCacheDirectory(
    const MaterializationCacheConfig& config) {
    return config.cache_root / "cache" / "materialized";
}

std::filesystem::path MaterializationCacheEntryDirectory(
    const MaterializationCacheConfig& config,
    const std::string& cache_key) {
    return MaterializationCacheDirectory(config) / cache_key;
}

std::filesystem::path MaterializationCacheManifestPath(
    const MaterializationCacheConfig& config,
    const std::string& cache_key) {
    return MaterializationCacheEntryDirectory(config, cache_key) /
           "manifest.json";
}

std::filesystem::path MaterializationCacheArtifactPath(
    const MaterializationCacheConfig& config,
    const std::string& cache_key) {
    const std::string extension =
        config.artifact_format == "feather" ? ".feather" : ".parquet";
    return MaterializationCacheEntryDirectory(config, cache_key) /
           ("data" + extension);
}

bool WriteMaterializationCacheManifest(
    const MaterializationCacheManifest& manifest,
    const std::filesystem::path& manifest_path,
    std::string* error) {
    std::error_code ec;
    std::filesystem::create_directories(manifest_path.parent_path(), ec);
    if (ec) {
        if (error) *error = ec.message();
        return false;
    }

    auto to_write = manifest;
    const auto now = NowIsoLikeUtc();
    if (to_write.created_at.empty()) {
        to_write.created_at = now;
    }
    if (to_write.last_used_at.empty() ||
        to_write.cache_status == MaterializationCacheStatus::Hit) {
        to_write.last_used_at = now;
    }
    if (to_write.materializer_cache_schema_version == 0) {
        to_write.materializer_cache_schema_version =
            kMaterializationCacheSchemaVersion;
    }

    const auto temp_path = manifest_path.string() + ".tmp";
    {
        std::ofstream out(temp_path, std::ios::binary);
        if (!out) {
            if (error) *error = "failed to open temporary manifest";
            return false;
        }
        out << ManifestToJson(to_write).dump(2);
        if (!out.good()) {
            if (error) *error = "failed to write temporary manifest";
            return false;
        }
    }

    std::filesystem::rename(temp_path, manifest_path, ec);
    if (ec) {
        std::filesystem::remove(manifest_path, ec);
        ec.clear();
        std::filesystem::rename(temp_path, manifest_path, ec);
        if (ec) {
            if (error) *error = ec.message();
            return false;
        }
    }
    return true;
}

bool ReadMaterializationCacheManifest(
    const std::filesystem::path& manifest_path,
    MaterializationCacheManifest& manifest,
    std::string* error) {
    std::ifstream in(manifest_path, std::ios::binary);
    if (!in) {
        if (error) *error = "manifest not found";
        return false;
    }

    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& ex) {
        if (error) *error = ex.what();
        return false;
    }
    return JsonToManifest(j, manifest, error);
}

MaterializationCacheValidationResult ValidateMaterializationCacheManifest(
    const MaterializationCacheManifest& manifest,
    const std::string& expected_cache_key,
    const std::string& expected_schema_fingerprint) {
    MaterializationCacheValidationResult result;
    result.manifest = manifest;

    if (manifest.materializer_cache_schema_version !=
        kMaterializationCacheSchemaVersion) {
        result.status = MaterializationCacheStatus::Stale;
        result.message = "materializer cache schema version changed";
        return result;
    }
    if (manifest.cache_key.empty() ||
        manifest.cache_key != expected_cache_key) {
        result.status = MaterializationCacheStatus::Stale;
        result.message = "cache key does not match requested graph";
        return result;
    }
    if (manifest.schema_fingerprint != expected_schema_fingerprint) {
        result.status = MaterializationCacheStatus::Stale;
        result.message = "source schema fingerprint changed";
        return result;
    }
    std::error_code artifact_error;
    if (manifest.artifact_path.empty() ||
        !std::filesystem::is_regular_file(manifest.artifact_path, artifact_error) ||
        artifact_error) {
        result.status = MaterializationCacheStatus::Stale;
        result.message = "cached materialization artifact is missing or unreadable: '" +
                         manifest.artifact_path + "'";
        return result;
    }

    result.status = MaterializationCacheStatus::Hit;
    result.usable = true;
    result.message = "cached prepared dataset is valid";
    return result;
}

std::string MaterializationArtifactIdentity(
    const MaterializationCacheManifest& manifest) {
    std::error_code ec;
    const auto path = std::filesystem::weakly_canonical(manifest.artifact_path, ec);
    if (ec || !std::filesystem::is_regular_file(path, ec) || ec) return {};
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) return {};
    const auto modified = std::filesystem::last_write_time(path, ec);
    if (ec) return {};
    return manifest.cache_key + "\n" + manifest.artifact_format + "\n" +
           path.string() + "\n" + std::to_string(size) + "\n" +
           fmt::format("{}", modified.time_since_epoch().count()) + "\n" +
           std::to_string(manifest.row_count) + ":" +
           std::to_string(manifest.column_count) + ":" +
           std::to_string(manifest.operators_applied);
}

namespace {

struct CacheEntryOnDisk {
    std::filesystem::path directory;
    std::string key;
    uint64_t bytes = 0;
    uint64_t last_used = 0;
    bool outdated = false;  // older cache schema: can never be reused
};

bool LooksLikeCacheKey(const std::string& name) {
    return !name.empty() && name.size() <= 64 &&
           std::all_of(name.begin(), name.end(), [](unsigned char ch) {
               return std::isxdigit(ch) != 0;
           });
}

// Only hex-named entry directories are considered, so nothing else under the
// project's cache folder is ever touched.
std::vector<CacheEntryOnDisk> ListCacheEntries(
    const MaterializationCacheConfig& config) {
    std::vector<CacheEntryOnDisk> entries;
    if (config.cache_root.empty()) {
        return entries;
    }
    std::error_code ec;
    const auto root = MaterializationCacheDirectory(config);
    if (!std::filesystem::is_directory(root, ec)) {
        return entries;
    }
    for (std::filesystem::directory_iterator it(root, ec), end;
         !ec && it != end; it.increment(ec)) {
        std::error_code entry_ec;
        if (!it->is_directory(entry_ec)) {
            continue;
        }
        CacheEntryOnDisk entry;
        entry.directory = it->path();
        entry.key = it->path().filename().string();
        if (!LooksLikeCacheKey(entry.key)) {
            continue;
        }
        for (std::filesystem::recursive_directory_iterator
                 file(entry.directory, entry_ec), file_end;
             !entry_ec && file != file_end; file.increment(entry_ec)) {
            std::error_code size_ec;
            if (file->is_regular_file(size_ec)) {
                const auto size = file->file_size(size_ec);
                if (!size_ec) entry.bytes += size;
            }
        }
        MaterializationCacheManifest manifest;
        if (ReadMaterializationCacheManifest(entry.directory / "manifest.json",
                                             manifest)) {
            try {
                entry.last_used = manifest.last_used_at.empty()
                    ? 0 : std::stoull(manifest.last_used_at);
            } catch (...) {
                entry.last_used = 0;
            }
            entry.outdated = manifest.materializer_cache_schema_version != 0 &&
                             manifest.materializer_cache_schema_version !=
                                 kMaterializationCacheSchemaVersion;
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

bool RemoveCacheEntry(const CacheEntryOnDisk& entry, std::string& error) {
    std::error_code ec;
    std::filesystem::remove_all(entry.directory, ec);
    if (ec) {
        error = "could not remove '" + entry.directory.string() + "': " +
                ec.message();
        return false;
    }
    return true;
}

} // namespace

MaterializationCacheUsage MeasureMaterializationCache(
    const MaterializationCacheConfig& config) {
    MaterializationCacheUsage usage;
    for (const auto& entry : ListCacheEntries(config)) {
        ++usage.entries;
        usage.total_bytes += entry.bytes;
    }
    return usage;
}

MaterializationCachePruneResult PruneMaterializationCache(
    const MaterializationCacheConfig& config,
    const std::string& keep_key) {
    MaterializationCachePruneResult result;
    auto entries = ListCacheEntries(config);
    uint64_t total = 0;
    for (const auto& entry : entries) total += entry.bytes;
    int count = static_cast<int>(entries.size());

    // Least recently used first; entries never used sort first.
    std::sort(entries.begin(), entries.end(),
              [](const CacheEntryOnDisk& a, const CacheEntryOnDisk& b) {
                  return a.last_used < b.last_used;
              });
    const auto over_limit = [&]() {
        return (config.max_total_bytes > 0 && total > config.max_total_bytes) ||
               (config.max_entries > 0 && count > config.max_entries);
    };
    std::vector<bool> removed(entries.size(), false);
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        if (!entry.outdated || entry.key == keep_key) continue;
        std::string error;
        if (RemoveCacheEntry(entry, error)) {
            removed[i] = true;
            ++result.removed_entries;
            result.freed_bytes += entry.bytes;
            total -= std::min(total, entry.bytes);
            --count;
        } else if (result.error.empty()) {
            result.error = error;
        }
    }
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        if (!over_limit()) break;
        if (removed[i] || entry.key == keep_key) continue;
        std::string error;
        if (RemoveCacheEntry(entry, error)) {
            ++result.removed_entries;
            result.freed_bytes += entry.bytes;
            total -= std::min(total, entry.bytes);
            --count;
        } else if (result.error.empty()) {
            result.error = error;
        }
    }
    result.remaining_entries = count;
    result.remaining_bytes = total;
    return result;
}

MaterializationCachePruneResult ClearMaterializationCache(
    const MaterializationCacheConfig& config) {
    MaterializationCachePruneResult result;
    for (const auto& entry : ListCacheEntries(config)) {
        std::string error;
        if (RemoveCacheEntry(entry, error)) {
            ++result.removed_entries;
            result.freed_bytes += entry.bytes;
        } else {
            ++result.remaining_entries;
            result.remaining_bytes += entry.bytes;
            if (result.error.empty()) result.error = error;
        }
    }
    return result;
}

} // namespace cyxwiz
