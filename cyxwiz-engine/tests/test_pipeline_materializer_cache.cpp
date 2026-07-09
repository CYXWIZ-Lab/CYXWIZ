#include "../src/core/arrow_dataset.h"
#include "../src/core/data_registry.h"
#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"
#include "../src/core/pipeline_materializer.h"

#include <arrow/api.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeTextTable() {
    auto text = FinishStringArray({
        "Small happy text sample",
        "Another happy sample",
        "Sad text pipeline example",
        "Another sad example",
    });
    auto label = FinishStringArray({"positive", "positive", "negative", "negative"});
    auto schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
        arrow::field("label", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {text, label}, 4);
}

gui::MLNode MakeDataInputNode(const std::string& dataset_name,
                              std::string source_file_path = {}) {
    gui::MLNode node;
    node.id = 1;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Data Input";
    node.parameters = {
        {"dataset_name", dataset_name},
        {"data_loaded", "true"},
        {"file_category", "text"},
        {"label_column", "label"},
    };
    if (!source_file_path.empty()) {
        node.parameters["file_path"] = std::move(source_file_path);
    }
    return node;
}

gui::MLNode MakeTokenizerNode(std::string max_length = "4") {
    gui::MLNode node;
    node.id = 2;
    node.type = gui::NodeType::TextTokenizer;
    node.category = gui::NodeCategory::TextProcessing;
    node.name = "Text Tokenizer";
    node.parameters = {
        {"text_col", "text"},
        {"label_col", "label"},
        {"tokenizer_type", "1"},
        {"max_length", std::move(max_length)},
        {"lowercase", "true"},
        {"min_word_freq", "1"},
        {"max_vocab_size", "100"},
    };
    return node;
}

} // namespace

namespace cyxwiz {

PipelineOperatorFactory& PipelineOperatorFactory::Instance() {
    static PipelineOperatorFactory instance;
    return instance;
}

PipelineOperatorFactory::PipelineOperatorFactory() = default;

std::unique_ptr<IPipelineOperator> PipelineOperatorFactory::Create(
    gui::NodeType type) const {
    if (type == gui::NodeType::TextTokenizer) {
        return std::make_unique<TextTokenizerOperator>();
    }
    return nullptr;
}

bool PipelineOperatorFactory::HasOperator(gui::NodeType type) const {
    return type == gui::NodeType::TextTokenizer;
}

void PipelineOperatorFactory::RegisterCreator(gui::NodeType, Creator) {}

std::vector<gui::NodeType> PipelineOperatorFactory::GetSupportedTypes() const {
    return {gui::NodeType::TextTokenizer};
}

} // namespace cyxwiz

int main() {
    constexpr const char* kDatasetName = "materializer_cache_source";
    const std::string materialized_name =
        std::string(kDatasetName) + cyxwiz::PipelineMaterializer::kMaterializedSuffix;
    const auto cache_root =
        std::filesystem::temp_directory_path() / "cyxwiz_pipeline_materializer_cache_test";
    std::filesystem::remove_all(cache_root);

    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnregisterTabularDataset(kDatasetName);
    registry.UnregisterTabularDataset(materialized_name);
    Check(registry.RegisterArrowTable(MakeTextTable(), kDatasetName) != nullptr,
          "source Arrow dataset should register");

    std::vector<gui::MLNode> nodes = {
        MakeDataInputNode(kDatasetName),
        MakeTokenizerNode(),
    };
    std::vector<gui::NodeLink> links = {
        {1, 1, 0, 2, 0, gui::LinkType::TensorFlow},
    };

    auto disabled = cyxwiz::PipelineMaterializer::Materialize(
        nodes, links, registry, kDatasetName);
    Check(disabled.success, disabled.error_message);
    Check(disabled.cache_status == cyxwiz::MaterializationCacheStatus::Disabled,
          "default materializer call should keep cache disabled");
    registry.UnregisterTabularDataset(materialized_name);

    cyxwiz::MaterializationCacheConfig cache_config;
    cache_config.mode = cyxwiz::MaterializationCacheMode::Auto;
    cache_config.cache_root = cache_root;

    auto saved = cyxwiz::PipelineMaterializer::Materialize(
        nodes, links, registry, kDatasetName, cache_config);
    Check(saved.success, saved.error_message);
    Check(saved.operators_applied == 1,
          "cache miss should materialize tokenizer output");
    Check(saved.cache_status == cyxwiz::MaterializationCacheStatus::Saved,
          "cache miss should save prepared dataset");
    Check(saved.saved_to_cache, "cache miss should report saved_to_cache");
    Check(!saved.cache_key.empty(), "cache save should report cache key");
    Check(std::filesystem::exists(saved.cache_artifact_path),
          "cache save should write artifact");
    Check(std::filesystem::exists(saved.cache_manifest_path),
          "cache save should report written manifest path");
    auto saved_dataset = registry.GetArrowDataset(materialized_name);
    Check(saved_dataset && saved_dataset->GetArrowTable() &&
              saved_dataset->GetArrowTable()->GetColumnByName("tok_0") != nullptr,
          "saved materialized dataset should expose token columns");
    Check(saved.cache_row_count == saved_dataset->GetArrowTable()->num_rows(),
          "cache save should report prepared row count");
    Check(saved.cache_column_count == saved_dataset->GetArrowTable()->num_columns(),
          "cache save should report prepared column count");

    registry.UnregisterTabularDataset(materialized_name);
    auto hit = cyxwiz::PipelineMaterializer::Materialize(
        nodes, links, registry, kDatasetName, cache_config);
    Check(hit.success, hit.error_message);
    Check(hit.cache_status == cyxwiz::MaterializationCacheStatus::Hit,
          "second matching materialization should hit cache");
    Check(hit.loaded_from_cache, "cache hit should report loaded_from_cache");
    Check(hit.operators_applied == 1,
          "cache hit should restore operators_applied from manifest");
    Check(hit.cache_key == saved.cache_key,
          "cache hit should use original cache key");
    Check(hit.cache_manifest_path == saved.cache_manifest_path,
          "cache hit should report original manifest path");
    Check(hit.cache_row_count == saved.cache_row_count,
          "cache hit should report manifest row count");
    Check(hit.cache_column_count == saved.cache_column_count,
          "cache hit should report manifest column count");
    Check(registry.GetArrowDataset(materialized_name) != nullptr,
          "cache hit should register prepared dataset");

    auto changed_nodes = nodes;
    changed_nodes[1] = MakeTokenizerNode("5");
    registry.UnregisterTabularDataset(materialized_name);
    auto changed = cyxwiz::PipelineMaterializer::Materialize(
        changed_nodes, links, registry, kDatasetName, cache_config);
    Check(changed.success, changed.error_message);
    Check(changed.cache_status == cyxwiz::MaterializationCacheStatus::Saved,
          "changed graph should rebuild and save a new prepared dataset");
    Check(changed.cache_key != saved.cache_key,
          "changed node parameters should change materialization cache key");

    const auto source_path = cache_root / "source.csv";
    {
        std::ofstream out(source_path, std::ios::binary);
        out << "text,label\nsmall,positive\n";
    }
    auto file_nodes = nodes;
    file_nodes[0] = MakeDataInputNode(kDatasetName, source_path.string());
    registry.UnregisterTabularDataset(materialized_name);
    auto file_saved = cyxwiz::PipelineMaterializer::Materialize(
        file_nodes, links, registry, kDatasetName, cache_config);
    Check(file_saved.success, file_saved.error_message);
    Check(file_saved.cache_status == cyxwiz::MaterializationCacheStatus::Saved,
          "source file backed cache should save prepared dataset");
    {
        std::ofstream out(source_path, std::ios::binary | std::ios::app);
        out << "changed,negative\n";
    }
    registry.UnregisterTabularDataset(materialized_name);
    auto file_changed = cyxwiz::PipelineMaterializer::Materialize(
        file_nodes, links, registry, kDatasetName, cache_config);
    Check(file_changed.success, file_changed.error_message);
    Check(file_changed.cache_status == cyxwiz::MaterializationCacheStatus::Saved,
          "changed source file should rebuild and save prepared dataset");
    Check(file_changed.cache_key != file_saved.cache_key,
          "source file stat changes should change materialization cache key");

    cyxwiz::MaterializationCacheConfig require_hit_config;
    require_hit_config.mode = cyxwiz::MaterializationCacheMode::RequireHit;
    require_hit_config.cache_root = cache_root / "empty_require_hit";
    registry.UnregisterTabularDataset(materialized_name);
    auto required = cyxwiz::PipelineMaterializer::Materialize(
        nodes, links, registry, kDatasetName, require_hit_config);
    Check(!required.success,
          "require-hit cache policy should fail instead of rebuilding on miss");
    Check(required.cache_status == cyxwiz::MaterializationCacheStatus::Miss,
          "require-hit miss should report miss status");

    registry.UnregisterTabularDataset(kDatasetName);
    registry.UnregisterTabularDataset(materialized_name);
    std::filesystem::remove_all(cache_root);

    std::cout << "Pipeline materializer cache tests passed\n";
    return 0;
}