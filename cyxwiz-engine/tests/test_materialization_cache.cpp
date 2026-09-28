#include "../src/core/materialization_cache.h"

#include <arrow/api.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

namespace fs = std::filesystem;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

gui::MLNode MakeNode(int id,
                     gui::NodeType type,
                     std::string name,
                     std::map<std::string, std::string> parameters = {}) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = std::move(name);
    node.parameters = std::move(parameters);
    return node;
}

cyxwiz::MaterializationCacheKeyInput MakeKeyInput() {
    cyxwiz::MaterializationCacheKeyInput input;
    input.source_dataset_name = "sentiment";
    input.source_identity = "arrow:sentiment";
    input.source_file_path = "D:/datasets/sentiment.csv";
    input.source_file_size = 1024;
    input.source_file_mtime = 2048;
    input.source_schema_fingerprint = "schema_a";
    input.dependencies = {{
        2,
        "fitted_state",
        "D:/artifacts/tfidf.cyxstate.json",
        512,
        std::string(64, 'a'),
    }};
    input.nodes = {
        MakeNode(2, gui::NodeType::TextTokenizer, "Tokenizer",
                 {{"text_col", "statement"}, {"max_length", "128"}}),
        MakeNode(1, gui::NodeType::DataInput, "Input",
                 {{"dataset_name", "sentiment"}}),
    };
    input.links = {
        {10, 1, 100, 2, 200, gui::LinkType::TensorFlow},
    };
    return input;
}

} // namespace

int main() {
    Check(std::string(cyxwiz::MaterializationCacheModeName(
              cyxwiz::MaterializationCacheMode::RequireHit)) == "require_hit",
          "cache mode names should expose require-hit policy");
    Check(std::string(cyxwiz::MaterializationCacheStatusName(
              cyxwiz::MaterializationCacheStatus::SaveFailed)) == "save_failed",
          "cache status names should expose save failures");

    const auto schema = arrow::schema({
        arrow::field("tok_0", arrow::int32()),
        arrow::field("y", arrow::int32()),
    });
    const auto same_schema = arrow::schema({
        arrow::field("tok_0", arrow::int32()),
        arrow::field("y", arrow::int32()),
    });
    const auto changed_schema = arrow::schema({
        arrow::field("tok_0", arrow::int32()),
        arrow::field("tok_1", arrow::int32()),
        arrow::field("y", arrow::int32()),
    });
    Check(cyxwiz::ComputeSchemaFingerprint(schema) ==
              cyxwiz::ComputeSchemaFingerprint(same_schema),
          "same Arrow schemas should produce the same fingerprint");
    Check(cyxwiz::ComputeSchemaFingerprint(schema) !=
              cyxwiz::ComputeSchemaFingerprint(changed_schema),
          "schema changes should affect the fingerprint");

    auto input = MakeKeyInput();
    const std::string key = cyxwiz::ComputeMaterializationCacheKey(input);
    auto reordered = input;
    std::swap(reordered.nodes[0], reordered.nodes[1]);
    Check(cyxwiz::ComputeMaterializationCacheKey(reordered) == key,
          "node ordering should not affect the materialization cache key");

    auto changed_param = input;
    changed_param.nodes[0].parameters["max_length"] = "256";
    Check(cyxwiz::ComputeMaterializationCacheKey(changed_param) != key,
          "materializer parameter changes should invalidate the cache key");

    auto changed_link = input;
    changed_link.links[0].to_pin = 201;
    Check(cyxwiz::ComputeMaterializationCacheKey(changed_link) != key,
          "materializer link changes should invalidate the cache key");

    auto changed_source = input;
    changed_source.source_file_size = 4096;
    Check(cyxwiz::ComputeMaterializationCacheKey(changed_source) != key,
          "source file size changes should invalidate the cache key");

    Check(key.size() == 32,
          "materialization cache keys should be 128-bit (32 hex characters)");

    auto renamed = input;
    renamed.nodes[0].name = "Renamed tokenizer";
    Check(cyxwiz::ComputeMaterializationCacheKey(renamed) == key,
          "renaming a node should not invalidate the prepared data");

    auto with_content = input;
    with_content.source_content_fingerprint = "rows_a";
    auto changed_content = input;
    changed_content.source_content_fingerprint = "rows_b";
    Check(cyxwiz::ComputeMaterializationCacheKey(with_content) != key &&
              cyxwiz::ComputeMaterializationCacheKey(with_content) !=
                  cyxwiz::ComputeMaterializationCacheKey(changed_content),
          "row content fingerprints should key sources without a file identity");

    {
        arrow::Int64Builder builder_a;
        Check(builder_a.AppendValues({1, 2, 3}).ok(), "append rows a");
        arrow::Int64Builder builder_b;
        Check(builder_b.AppendValues({1, 2, 4}).ok(), "append rows b");
        auto field_schema = arrow::schema({arrow::field("x", arrow::int64())});
        auto table_a = arrow::Table::Make(field_schema, {builder_a.Finish().ValueOrDie()});
        auto table_a2 = arrow::Table::Make(field_schema, {table_a->column(0)->chunk(0)});
        auto table_b = arrow::Table::Make(field_schema, {builder_b.Finish().ValueOrDie()});
        Check(cyxwiz::ComputeTableContentFingerprint(table_a) ==
                  cyxwiz::ComputeTableContentFingerprint(table_a2),
              "the same rows should give the same content fingerprint");
        Check(cyxwiz::ComputeTableContentFingerprint(table_a) !=
                  cyxwiz::ComputeTableContentFingerprint(table_b),
              "changed rows should change the content fingerprint");
    }

    auto changed_dependency = input;
    changed_dependency.dependencies[0].content_sha256 = std::string(64, 'b');
    Check(cyxwiz::ComputeMaterializationCacheKey(changed_dependency) != key,
          "fitted-state content changes should invalidate the cache key");

    const fs::path root =
        fs::temp_directory_path() / "cyxwiz_materialization_cache_test";
    fs::remove_all(root);
    cyxwiz::MaterializationCacheConfig config;
    config.mode = cyxwiz::MaterializationCacheMode::Auto;
    config.cache_root = root / ".cyxwiz";

    const auto dependency_path = root / "fitted_state.cyxstate.json";
    fs::create_directories(root);
    {
        std::ofstream state(dependency_path, std::ios::binary);
        state << "state-a";
    }
    cyxwiz::MaterializationCacheDependencyIdentity dependency_a;
    std::string error;
    Check(cyxwiz::ResolveMaterializationCacheDependencyIdentity(
              2, "fitted_state", dependency_path.string(), dependency_a,
              &error),
          "fitted-state identity should resolve: " + error);
    {
        std::ofstream state(dependency_path,
                            std::ios::binary | std::ios::trunc);
        state << "state-b";
    }
    cyxwiz::MaterializationCacheDependencyIdentity dependency_b;
    Check(cyxwiz::ResolveMaterializationCacheDependencyIdentity(
              2, "fitted_state", dependency_path.string(), dependency_b,
              &error),
          "changed fitted-state identity should resolve: " + error);
    Check(dependency_a.path == dependency_b.path &&
              dependency_a.content_sha256 != dependency_b.content_sha256,
          "same-path fitted-state mutation should change SHA-256 identity");
    const auto entry_dir =
        cyxwiz::MaterializationCacheEntryDirectory(config, key);
    const auto manifest_path =
        cyxwiz::MaterializationCacheManifestPath(config, key);
    const auto artifact_path =
        cyxwiz::MaterializationCacheArtifactPath(config, key);
    Check(entry_dir.filename().string() == key,
          "cache entry directory should end with the cache key");
    Check(manifest_path.filename().string() == "manifest.json",
          "cache manifest path should use manifest.json");
    Check(artifact_path.filename().string() == "data.parquet",
          "default cache artifact should be parquet");

    fs::create_directories(artifact_path.parent_path());
    {
        std::ofstream artifact(artifact_path, std::ios::binary);
        artifact << "parquet fixture placeholder";
    }

    cyxwiz::MaterializationCacheManifest manifest;
    manifest.cache_key = key;
    manifest.source_dataset_name = "sentiment";
    manifest.effective_dataset_name = "sentiment__materialized";
    manifest.artifact_path = artifact_path.string();
    manifest.artifact_format = "parquet";
    manifest.row_count = 3;
    manifest.column_count = 2;
    manifest.schema_fingerprint = cyxwiz::ComputeSchemaFingerprint(schema);
    manifest.dependencies = input.dependencies;
    manifest.operators_applied = 1;
    manifest.engine_version = "test";
    manifest.materializer_cache_schema_version =
        cyxwiz::kMaterializationCacheSchemaVersion;
    manifest.cache_status = cyxwiz::MaterializationCacheStatus::Saved;

    Check(cyxwiz::WriteMaterializationCacheManifest(
              manifest, manifest_path, &error),
          "manifest write should succeed: " + error);

    cyxwiz::MaterializationCacheManifest loaded;
    Check(cyxwiz::ReadMaterializationCacheManifest(
              manifest_path, loaded, &error),
          "manifest read should succeed: " + error);
    Check(loaded.cache_key == manifest.cache_key,
          "cache key should round-trip through manifest JSON");
    Check(loaded.cache_status == cyxwiz::MaterializationCacheStatus::Saved,
          "cache status should round-trip through manifest JSON");
    Check(loaded.dependencies.size() == 1 &&
              loaded.dependencies[0].content_sha256 ==
                  input.dependencies[0].content_sha256,
          "cache dependency identity should round-trip through manifest JSON");

    auto validation = cyxwiz::ValidateMaterializationCacheManifest(
        loaded, key, manifest.schema_fingerprint);
    Check(validation.usable,
          "matching manifest with existing artifact should be usable");
    Check(validation.status == cyxwiz::MaterializationCacheStatus::Hit,
          "matching manifest should validate as cache hit");

    auto stale_schema = cyxwiz::ValidateMaterializationCacheManifest(
        loaded, key, cyxwiz::ComputeSchemaFingerprint(changed_schema));
    Check(!stale_schema.usable &&
              stale_schema.status == cyxwiz::MaterializationCacheStatus::Stale,
          "schema drift should make the manifest stale");

    fs::remove(artifact_path);
    auto missing_artifact = cyxwiz::ValidateMaterializationCacheManifest(
        loaded, key, manifest.schema_fingerprint);
    Check(!missing_artifact.usable &&
              missing_artifact.status ==
                  cyxwiz::MaterializationCacheStatus::Stale,
          "missing artifact should make the manifest stale");

    const auto corrupt_path = entry_dir / "corrupt_manifest.json";
    {
        std::ofstream corrupt(corrupt_path, std::ios::binary);
        corrupt << "{not valid json";
    }
    cyxwiz::MaterializationCacheManifest corrupt_manifest;
    Check(!cyxwiz::ReadMaterializationCacheManifest(
              corrupt_path, corrupt_manifest, &error),
          "corrupt manifest should fail closed");

    // Size policy: least recently used entries go first; keep_key stays.
    {
        const auto make_entry = [&](const std::string& entry_key, int bytes,
                                    const std::string& last_used,
                                    int schema_version =
                                        cyxwiz::kMaterializationCacheSchemaVersion) {
            const auto dir = cyxwiz::MaterializationCacheEntryDirectory(config, entry_key);
            fs::create_directories(dir);
            {
                std::ofstream data(dir / "data.parquet", std::ios::binary);
                data << std::string(static_cast<size_t>(bytes), 'x');
            }
            cyxwiz::MaterializationCacheManifest entry_manifest;
            entry_manifest.cache_key = entry_key;
            entry_manifest.artifact_path = (dir / "data.parquet").string();
            entry_manifest.last_used_at = last_used;
            entry_manifest.created_at = last_used;
            entry_manifest.cache_status = cyxwiz::MaterializationCacheStatus::Saved;
            entry_manifest.materializer_cache_schema_version = schema_version;
            std::string write_error;
            Check(cyxwiz::WriteMaterializationCacheManifest(
                      entry_manifest, dir / "manifest.json", &write_error),
                  "write test manifest: " + write_error);
        };
        fs::remove_all(cyxwiz::MaterializationCacheDirectory(config));
        make_entry("aaaa", 1000, "100");  // least recently used
        make_entry("bbbb", 1000, "200");
        make_entry("cccc", 1000, "300");  // most recently used
        fs::create_directories(cyxwiz::MaterializationCacheDirectory(config) / "not-a-key");

        const auto usage = cyxwiz::MeasureMaterializationCache(config);
        Check(usage.entries == 3 && usage.total_bytes > 3000,
              "usage should count only cache entries");

        auto limited = config;
        limited.max_entries = 2;
        auto pruned = cyxwiz::PruneMaterializationCache(limited, "aaaa");
        Check(pruned.removed_entries == 1 && pruned.remaining_entries == 2,
              "entry limit should remove one entry");
        Check(fs::exists(cyxwiz::MaterializationCacheEntryDirectory(config, "aaaa")) &&
                  !fs::exists(cyxwiz::MaterializationCacheEntryDirectory(config, "bbbb")) &&
                  fs::exists(cyxwiz::MaterializationCacheEntryDirectory(config, "cccc")),
              "prune should keep keep_key and remove the least recently used other entry");

        auto size_limited = config;
        size_limited.max_total_bytes = 2500;  // one entry (data + manifest) fits
        auto size_pruned = cyxwiz::PruneMaterializationCache(size_limited);
        Check(size_pruned.removed_entries == 1 && size_pruned.remaining_entries == 1 &&
                  fs::exists(cyxwiz::MaterializationCacheEntryDirectory(config, "cccc")),
              "size limit should remove least recently used entries until it fits");

        make_entry("dddd", 10, "900", cyxwiz::kMaterializationCacheSchemaVersion - 1);
        auto outdated_pruned = cyxwiz::PruneMaterializationCache(config);
        Check(outdated_pruned.removed_entries == 1 &&
                  !fs::exists(cyxwiz::MaterializationCacheEntryDirectory(config, "dddd")) &&
                  fs::exists(cyxwiz::MaterializationCacheEntryDirectory(config, "cccc")),
              "entries from an older cache schema should be removed even under the limit");

        auto cleared = cyxwiz::ClearMaterializationCache(config);
        Check(cleared.removed_entries == 1 &&
                  cyxwiz::MeasureMaterializationCache(config).entries == 0,
              "clear should remove every cache entry");
        Check(fs::exists(cyxwiz::MaterializationCacheDirectory(config) / "not-a-key"),
              "clear should not touch folders that are not cache entries");
    }

    fs::remove_all(root);
    std::cout << "Materialization cache tests passed\n";
    return 0;
}
