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

    fs::remove_all(root);
    std::cout << "Materialization cache tests passed\n";
    return 0;
}
