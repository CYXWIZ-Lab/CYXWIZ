#include "materialization_cache.h"

#include <cyxwiz/utilities.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
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
        out << "node=" << node.id << "|"
            << static_cast<int>(node.type) << "|"
            << EscapePart(node.name) << "\n";
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

std::string ComputeMaterializationCacheKey(
    const MaterializationCacheKeyInput& input) {
    return StableFingerprint(CanonicalKeyInput(input));
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

std::filesystem::path MaterializationCacheEntryDirectory(
    const MaterializationCacheConfig& config,
    const std::string& cache_key) {
    return config.cache_root / "cache" / "materialized" / cache_key;
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
    if (manifest.artifact_path.empty() ||
        !std::filesystem::exists(manifest.artifact_path)) {
        result.status = MaterializationCacheStatus::Stale;
        result.message = "cached materialization artifact is missing";
        return result;
    }

    result.status = MaterializationCacheStatus::Hit;
    result.usable = true;
    result.message = "cached prepared dataset is valid";
    return result;
}

} // namespace cyxwiz
