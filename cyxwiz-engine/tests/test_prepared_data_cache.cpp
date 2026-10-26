#include <algorithm>
#include "../src/core/arrow_dataset.h"
#include "../src/core/data_registry.h"
#include "../src/core/node_executors/count_vectorizer_operator.h"
#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"
#include "../src/core/prepared_data_cache.h"

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

gui::MLNode MakeSparseCountNode() {
    gui::MLNode node;
    node.id = 5;
    node.type = gui::NodeType::CountVectorizer;
    node.category = gui::NodeCategory::TextProcessing;
    node.name = "Sparse Count Vectorizer";
    node.parameters = {
        {"text_col", "text"},
        {"label_col", "label"},
        {"max_features", "8"},
        {"norm", "none"},
        {"stop_words", "none"},
        {"output_format", "sparse"},
    };
    return node;
}

gui::MLNode MakeStatefulNode() {
    gui::MLNode node;
    node.id = 3;
    node.type = gui::NodeType::StandardScaler;
    node.category = gui::NodeCategory::Preprocessing;
    node.name = "Stateful Transform";
    return node;
}

gui::MLNode MakeCacheDependentNode(const std::filesystem::path& state_path) {
    gui::MLNode node;
    node.id = 4;
    node.type = gui::NodeType::MinMaxScaler;
    node.category = gui::NodeCategory::Preprocessing;
    node.name = "Cache-dependent Transform";
    node.parameters["state_path"] = state_path.string();
    return node;
}

} // namespace

namespace cyxwiz {

class NonCacheablePassThroughOperator final : public IPipelineOperator {
public:
    std::string GetName() const override { return "StandardScaler"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    bool IsCacheable() const override { return false; }
    bool Configure(const std::map<std::string, std::string>&, std::string&) override { return true; }
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override {
        return input;
    }
};

class CacheDependentPassThroughOperator final : public IPipelineOperator {
public:
    std::string GetName() const override { return "MinMaxScaler"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    bool Configure(const std::map<std::string, std::string>& parameters,
                   std::string& error) override {
        const auto path = parameters.find("state_path");
        if (path == parameters.end() || path->second.empty()) {
            error = "state_path is required";
            return false;
        }
        state_path_ = path->second;
        return true;
    }
    bool CollectCacheDependencies(
        std::vector<PipelineOperatorCacheDependency>& dependencies,
        std::string&) const override {
        dependencies.push_back({"fitted_state", state_path_});
        return true;
    }
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override {
        return input;
    }

private:
    std::string state_path_;
};

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
    if (type == gui::NodeType::CountVectorizer) {
        return std::make_unique<CountVectorizerOperator>();
    }
    if (type == gui::NodeType::StandardScaler) {
        return std::make_unique<NonCacheablePassThroughOperator>();
    }
    if (type == gui::NodeType::MinMaxScaler) {
        return std::make_unique<CacheDependentPassThroughOperator>();
    }
    return nullptr;
}

bool PipelineOperatorFactory::HasOperator(gui::NodeType type) const {
    return type == gui::NodeType::TextTokenizer ||
           type == gui::NodeType::CountVectorizer ||
           type == gui::NodeType::StandardScaler ||
           type == gui::NodeType::MinMaxScaler;
}

void PipelineOperatorFactory::RegisterCreator(gui::NodeType, Creator) {}

std::vector<gui::NodeType> PipelineOperatorFactory::GetSupportedTypes() const {
    return {gui::NodeType::TextTokenizer, gui::NodeType::CountVectorizer,
            gui::NodeType::StandardScaler,
            gui::NodeType::MinMaxScaler};
}

} // namespace cyxwiz

namespace {

constexpr const char* kDatasetName = "prepared_cache_tokenizer_source";

std::filesystem::path CacheRoot() {
    return std::filesystem::temp_directory_path() / "cyxwiz_prepared_data_cache_test";
}

cyxwiz::MaterializationCacheConfig AutoConfig(const std::filesystem::path& root) {
    cyxwiz::MaterializationCacheConfig config;
    config.mode = cyxwiz::MaterializationCacheMode::Auto;
    config.cache_root = root;
    return config;
}

std::vector<gui::NodeLink> TokenizerLinks() {
    return {{1, 1, 0, 2, 0, gui::LinkType::TensorFlow}};
}

std::vector<gui::MLNode> TokenizerGraph() {
    return {MakeDataInputNode(kDatasetName), MakeTokenizerNode()};
}

// What the pipeline run's Text Tokenizer step makes from the source table.
std::shared_ptr<arrow::Table> ApplyOperator(const gui::MLNode& node,
                                            const std::shared_ptr<arrow::Table>& input) {
    auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(node.type);
    Check(op != nullptr, node.name + " has an operator");
    std::string error;
    Check(op->Configure(node.parameters, error), node.name + " configures: " + error);
    auto result = op->Apply(input);
    Check(result.ok(), node.name + " applies: " + result.status().ToString());
    return result.ValueOrDie();
}

// The entry for a prepared table: probe, save, probe, load, mark used, rebuild
// (was the materializer's Disabled / Saved / Hit / Rebuild statuses).
void TestPreparedTableRoundTrip(cyxwiz::DataRegistry& registry) {
    const auto root = CacheRoot() / "round_trip";
    std::filesystem::remove_all(root);
    const auto nodes = TokenizerGraph();
    const auto links = TokenizerLinks();
    const auto key = cyxwiz::ComputePreparedDataKey(nodes, links, 2, registry, "");
    Check(key.table_path && key.cacheable && !key.key.empty() && key.steps == 1 && !key.sparse,
          "a tokenizer path on a loaded table is cacheable: " + key.reason);

    cyxwiz::MaterializationCacheConfig disabled;
    disabled.cache_root = root;
    std::string error;
    Check(!cyxwiz::ProbePreparedData(disabled, key), "a disabled cache finds nothing");
    const auto prepared =
        ApplyOperator(nodes[1], registry.GetArrowDataset(kDatasetName)->GetArrowTable());
    Check(!cyxwiz::SavePreparedData(disabled, key, prepared, "prepared_disabled", &error) &&
              error == "the cache is off",
          "a disabled cache saves nothing: " + error);

    const auto config = AutoConfig(root);
    Check(!cyxwiz::ProbePreparedData(config, key), "nothing is on disk before a save (miss)");
    error.clear();
    Check(cyxwiz::SavePreparedData(config, key, prepared, "ds_operator_TextTokenizer_2", &error),
          "the tokenized table saves: " + error);
    cyxwiz::MaterializationCacheManifest manifest;
    Check(cyxwiz::ProbePreparedData(config, key, &manifest), "the saved entry is found without loading it");
    Check(manifest.cache_key == key.key && manifest.row_count == prepared->num_rows() &&
              manifest.column_count == prepared->num_columns() && manifest.operators_applied == 1 &&
              manifest.cache_status == cyxwiz::MaterializationCacheStatus::Saved &&
              manifest.source_dataset_name == kDatasetName,
          "the manifest keeps the key, the rows, the columns and the steps");
    Check(std::filesystem::exists(manifest.artifact_path) &&
              std::filesystem::exists(cyxwiz::MaterializationCacheManifestPath(config, key.key)),
          "the artifact and its manifest are written");

    const std::string loaded_name = "prepared_round_trip";
    registry.UnregisterTabularDataset(loaded_name);
    const auto loaded = cyxwiz::LoadPreparedData(config, key, registry, loaded_name, &error);
    Check(loaded && *loaded == loaded_name, "the entry loads under the asked name: " + error);
    const auto loaded_table = registry.GetArrowDataset(loaded_name)->GetArrowTable();
    Check(loaded_table->GetColumnByName("tok_0") != nullptr && loaded_table->GetColumnByName("text") == nullptr &&
              loaded_table->num_rows() == prepared->num_rows(),
          "the loaded table is the tokenized one");
    Check(cyxwiz::ProbePreparedData(config, key, &manifest) &&
              manifest.cache_status == cyxwiz::MaterializationCacheStatus::Hit,
          "a load marks the entry used (hit)");

    auto rebuild = config;
    rebuild.mode = cyxwiz::MaterializationCacheMode::Rebuild;
    Check(!cyxwiz::ProbePreparedData(rebuild, key), "a requested rebuild ignores the entry");
    Check(cyxwiz::SavePreparedData(rebuild, key, prepared, "ds_operator_TextTokenizer_2", &error),
          "a rebuild writes the entry again: " + error);
    Check(cyxwiz::ProbePreparedData(config, key, &manifest) &&
              manifest.cache_status == cyxwiz::MaterializationCacheStatus::Saved,
          "after the rebuild the entry is a fresh save");

    // The artifact goes away: the entry is not usable and the load says so.
    const std::string held = manifest.artifact_path + ".held";
    std::filesystem::rename(manifest.artifact_path, held);
    Check(!cyxwiz::ProbePreparedData(config, key), "a missing artifact is not a hit");
    Check(!cyxwiz::LoadPreparedData(config, key, registry, "prepared_missing", &error) && !error.empty(),
          "a load without the artifact fails with the reason");
    std::filesystem::rename(held, manifest.artifact_path);

    registry.UnregisterTabularDataset(loaded_name);
    std::filesystem::remove_all(root);
}

// The key covers the Data Input and the steps to the feeder, not the model,
// the Split or names (was the materializer's ProbeCache key checks).
void TestKeyCoversTheSteps(cyxwiz::DataRegistry& registry) {
    const auto nodes = TokenizerGraph();
    const auto links = TokenizerLinks();
    const auto base = cyxwiz::ComputePreparedDataKey(nodes, links, 2, registry, "");
    Check(base.cacheable, "base key: " + base.reason);

    auto changed = nodes;
    changed[1] = MakeTokenizerNode("5");
    Check(cyxwiz::ComputePreparedDataKey(changed, links, 2, registry, "").key != base.key,
          "a tokenizer setting changes the key");

    // A sparse vectorizer that leads nowhere is not on the path.
    auto disconnected = nodes;
    auto sparse = MakeSparseCountNode();
    sparse.id = 50;
    disconnected.push_back(sparse);
    const auto disconnected_key = cyxwiz::ComputePreparedDataKey(disconnected, links, 2, registry, "");
    Check(disconnected_key.key == base.key && !disconnected_key.sparse,
          "a disconnected sparse vectorizer changes neither the key nor the representation");

    const auto node = [](int id, gui::NodeType type, std::map<std::string, std::string> parameters) {
        gui::MLNode n;
        n.id = id;
        n.type = type;
        n.name = "n" + std::to_string(id);
        n.parameters = std::move(parameters);
        return n;
    };
    auto graph = nodes;
    graph.push_back(node(3, gui::NodeType::DataSplit, {{"train_ratio", "0.8"}, {"val_ratio", "0.1"}}));
    graph.push_back(node(4, gui::NodeType::Dense, {{"units", "64"}}));
    auto graph_links = links;
    graph_links.push_back({20, 2, 0, 3, 0, gui::LinkType::TensorFlow});
    graph_links.push_back({21, 3, 0, 4, 0, gui::LinkType::TensorFlow});
    const auto key_of = [&](const std::vector<gui::MLNode>& g) {
        return cyxwiz::ComputePreparedDataKey(g, graph_links, 2, registry, "").key;
    };
    const std::string with_model = key_of(graph);
    Check(with_model == base.key, "the model and the Split after the feeder are not in the key");
    auto model_edit = graph;
    model_edit[3].parameters["units"] = "128";
    Check(key_of(model_edit) == with_model, "a model layer edit keeps the prepared data's key");
    auto split_edit = graph;
    split_edit[2].parameters["train_ratio"] = "0.7";
    Check(key_of(split_edit) == with_model, "a Data Split edit keeps the prepared data's key");
    auto renamed = graph;
    renamed[1].name = "Tokenizer renamed";
    Check(key_of(renamed) == with_model, "a renamed step keeps the key");
    auto operator_edit = graph;
    operator_edit[1] = MakeTokenizerNode("7");
    Check(key_of(operator_edit) != with_model, "an operator edit changes the key");

    // The source file's size and time are in the key.
    const auto root = CacheRoot() / "source_file";
    std::filesystem::create_directories(root);
    const auto source_path = root / "source.csv";
    {
        std::ofstream out(source_path, std::ios::binary);
        out << "text,label\nsmall,positive\n";
    }
    auto file_nodes = nodes;
    file_nodes[0] = MakeDataInputNode(kDatasetName, source_path.string());
    const auto file_key = cyxwiz::ComputePreparedDataKey(file_nodes, links, 2, registry, "");
    Check(file_key.cacheable && file_key.key != base.key, "a Data Input file is part of the key");
    {
        std::ofstream out(source_path, std::ios::binary | std::ios::app);
        out << "changed,negative\n";
    }
    Check(cyxwiz::ComputePreparedDataKey(file_nodes, links, 2, registry, "").key != file_key.key,
          "an edited source file changes the key");
    std::filesystem::remove_all(root);
}

// A step that keeps fitted state is run but never kept on disk; a step's
// cache dependency (a fitted-state file) is in the key.
void TestFittedStateAndDependencies(cyxwiz::DataRegistry& registry) {
    const std::vector<gui::MLNode> stateful_nodes = {MakeDataInputNode(kDatasetName), MakeStatefulNode()};
    const std::vector<gui::NodeLink> stateful_links = {{2, 1, 0, 3, 0, gui::LinkType::TensorFlow}};
    const auto stateful = cyxwiz::ComputePreparedDataKey(stateful_nodes, stateful_links, 3, registry, "");
    Check(stateful.table_path && !stateful.cacheable && stateful.key.empty() &&
              stateful.reason.find("keeps fitted state") != std::string::npos,
          "a fitted-state step stays a table path that is not kept: " + stateful.reason);
    const auto root = CacheRoot() / "fitted_state";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto config = AutoConfig(root);
    std::string error;
    Check(!cyxwiz::ProbePreparedData(config, stateful) &&
              !cyxwiz::SavePreparedData(config, stateful, MakeTextTable(), "never", &error) &&
              error == stateful.reason,
          "a fitted-state path neither loads nor saves, and says why: " + error);

    const auto state_path = root / "fitted_state.cyxstate.json";
    {
        std::ofstream state(state_path, std::ios::binary);
        state << "state-a";
    }
    const std::vector<gui::MLNode> dependent_nodes = {MakeDataInputNode(kDatasetName),
                                                      MakeCacheDependentNode(state_path)};
    const std::vector<gui::NodeLink> dependent_links = {{3, 1, 0, 4, 0, gui::LinkType::TensorFlow}};
    const auto saved = cyxwiz::ComputePreparedDataKey(dependent_nodes, dependent_links, 4, registry, "");
    Check(saved.cacheable && saved.dependencies.size() == 1, "a cache dependency is resolved: " + saved.reason);
    Check(cyxwiz::SavePreparedData(config, saved, MakeTextTable(), "dependent", &error),
          "the dependent step's output saves: " + error);
    const auto again = cyxwiz::ComputePreparedDataKey(dependent_nodes, dependent_links, 4, registry, "");
    Check(again.key == saved.key && cyxwiz::ProbePreparedData(config, again),
          "unchanged fitted state permits a hit");
    {
        std::ofstream state(state_path, std::ios::binary | std::ios::trunc);
        state << "state-b";
    }
    const auto changed = cyxwiz::ComputePreparedDataKey(dependent_nodes, dependent_links, 4, registry, "");
    Check(changed.cacheable && changed.key != saved.key && !cyxwiz::ProbePreparedData(config, changed),
          "same-path fitted-state mutation changes the key (a miss)");
    std::filesystem::remove(state_path);
    const auto missing = cyxwiz::ComputePreparedDataKey(dependent_nodes, dependent_links, 4, registry, "");
    Check(missing.table_path && !missing.cacheable &&
              missing.reason.find("Cache-dependent Transform") != std::string::npos,
          "a missing fitted-state file is not cacheable and names the step: " + missing.reason);
    std::filesystem::remove_all(root);
}

// Sparse features are kept as their versioned CSR artifact and come back as
// the typed sparse entry, never a dense table.
void TestSparseEntry(cyxwiz::DataRegistry& registry) {
    const auto root = CacheRoot() / "sparse";
    std::filesystem::remove_all(root);
    const std::vector<gui::MLNode> nodes = {MakeDataInputNode(kDatasetName), MakeSparseCountNode()};
    const std::vector<gui::NodeLink> links = {{1, 1, 0, 5, 0, gui::LinkType::TensorFlow}};
    const auto key = cyxwiz::ComputePreparedDataKey(nodes, links, 5, registry, "");
    Check(key.cacheable && key.sparse, "a sparse vectorizer path is a sparse entry: " + key.reason);

    auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(gui::NodeType::CountVectorizer);
    std::string error;
    Check(op && op->Configure(nodes[1].parameters, error), "sparse vectorizer configures: " + error);
    const std::string name = "ds_operator_CountVectorizer_5";
    auto sparse = op->ApplySparse(registry.GetArrowDataset(kDatasetName)->GetArrowTable(), name);
    Check(sparse.ok(), "sparse vectorizer applies: " + sparse.status().ToString());

    const auto config = AutoConfig(root);
    Check(cyxwiz::SavePreparedSparseData(config, key, *sparse.ValueOrDie(), &error), "sparse save: " + error);
    cyxwiz::MaterializationCacheManifest manifest;
    Check(cyxwiz::ProbePreparedData(config, key, &manifest) &&
              manifest.artifact_format == cyxwiz::kSparseCacheArtifactFormat &&
              manifest.artifact_path == cyxwiz::SparseCacheArtifactPath(config, key.key).string() &&
              manifest.artifact_path.ends_with(".csr.arrow"),
          "sparse features use their versioned CSR cache artifact");
    auto dense_key = key;
    dense_key.sparse = false;
    Check(!cyxwiz::ProbePreparedData(config, dense_key), "a CSR entry is never read as a table");

    registry.UnregisterTabularDataset(name);
    const auto loaded = cyxwiz::LoadPreparedData(config, key, registry, "unused_name", &error);
    Check(loaded && *loaded == name, "sparse features load under the name they were saved with: " + error);
    const auto info = registry.InspectSparseFeatureDataset(name);
    Check(registry.IsSparseFeatureDataset(name) && !registry.IsArrowDataset(name) && info &&
              info->num_rows == 4 && info->num_features > 0 && info->has_labels,
          "the loaded CSR entry keeps rows, features and labels");
    registry.UnregisterTabularDataset(name);
    std::filesystem::remove_all(root);
}

}  // namespace

int main() {
    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnregisterTabularDataset(kDatasetName);
    Check(registry.RegisterArrowTable(MakeTextTable(), kDatasetName) != nullptr,
          "source Arrow dataset should register");
    const auto cache_root = CacheRoot();
    std::filesystem::remove_all(cache_root);

    TestPreparedTableRoundTrip(registry);
    TestKeyCoversTheSteps(registry);
    TestFittedStateAndDependencies(registry);
    TestSparseEntry(registry);
    registry.UnregisterTabularDataset(kDatasetName);
    // The table training reads, kept on disk between sessions (TOFIX178
    // S0-e2): Data Input (a file) -> Filter Rows -> Text Tokenizer -> Split.
    {
        const auto work = cache_root / "prepared";
        std::filesystem::create_directories(work);
        const auto csv = work / "text.csv";
        {
            std::ofstream out(csv, std::ios::binary);
            out << "text,label\nSmall happy text,positive\nAnother happy one,positive\nSad text here,negative\n";
        }
        const std::string source = "prepared_cache_source";
        registry.RegisterArrowTable(MakeTextTable(), source);
        gui::MLNode filter;
        filter.id = 7;
        filter.type = gui::NodeType::FilterRows;
        filter.name = "Keep rows";
        filter.parameters = {{"condition", "label != ''"}};
        gui::MLNode split;
        split.id = 3;
        split.type = gui::NodeType::DataSplit;
        split.name = "Split";
        gui::MLNode dense;
        dense.id = 4;
        dense.type = gui::NodeType::Dense;
        dense.name = "Dense";
        dense.parameters = {{"units", "16"}};
        std::vector<gui::MLNode> graph = {MakeDataInputNode(source, csv.string()), filter, MakeTokenizerNode(),
                                          split, dense};
        std::vector<gui::NodeLink> graph_links = {
            {1, 1, 0, 7, 0, gui::LinkType::TensorFlow},
            {2, 7, 0, 2, 0, gui::LinkType::TensorFlow},
            {3, 2, 0, 3, 0, gui::LinkType::TensorFlow},
            {4, 3, 0, 4, 0, gui::LinkType::TensorFlow},
        };
        const auto key_of = [&](const std::vector<gui::MLNode>& g) {
            return cyxwiz::ComputePreparedDataKey(g, graph_links, 2, registry, work.string());
        };
        const auto base = key_of(graph);
        Check(base.cacheable && !base.key.empty(), "a Filter Rows + tokenizer path is cacheable: " + base.reason);
        Check(base.steps == 2, "the key covers the two steps before the Split");
        Check(key_of(graph).key == base.key, "the same graph gives the same key");
        auto model_edit = graph;
        model_edit[4].parameters["units"] = "64";
        Check(key_of(model_edit).key == base.key, "a model edit keeps the key");
        auto filter_edit = graph;
        filter_edit[1].parameters["condition"] = "label == 'positive'";
        Check(key_of(filter_edit).key != base.key, "a Filter Rows edit changes the key");
        auto sql = graph;
        sql[1].type = gui::NodeType::SQLQuery;
        sql[1].parameters = {{"query", "SELECT * FROM input"}};
        const auto sql_key = key_of(sql);
        Check(!sql_key.cacheable && sql_key.reason.find("SQL") != std::string::npos,
              "a SQL step is never cached: " + sql_key.reason);
        Check(sql_key.table_path && sql_key.steps == 2, "a SQL path still runs in the pipeline run");
        Check(base.table_path, "a cacheable path is a table path");
        auto image = graph;
        image[1].type = gui::NodeType::QualityAnalyzer;
        Check(!key_of(image).table_path, "an image step is not a table path");

        cyxwiz::MaterializationCacheConfig config;
        config.mode = cyxwiz::MaterializationCacheMode::Auto;
        config.cache_root = work;
        Check(!cyxwiz::ProbePreparedData(config, base), "nothing is on disk before a save");
        const auto table = MakeTextTable();
        std::string error;
        Check(cyxwiz::SavePreparedData(config, base, table, "prepared_out", &error), "the save works: " + error);
        Check(cyxwiz::ProbePreparedData(config, base), "the saved entry is found");
        Check(!cyxwiz::ProbePreparedData(config, key_of(filter_edit)), "another key finds nothing");
        Check(cyxwiz::LoadPreparedData(config, base, registry, "prepared_loaded", &error) == "prepared_loaded",
              "the load works: " + error);
        const auto loaded = registry.GetArrowDataset("prepared_loaded");
        Check(loaded && loaded->GetArrowTable()->Equals(*table), "the loaded table equals the saved one");
        auto rebuild = config;
        rebuild.mode = cyxwiz::MaterializationCacheMode::Rebuild;
        Check(!cyxwiz::ProbePreparedData(rebuild, base), "a requested rebuild ignores the entry");

        // The source file changes: its size and time change the key.
        {
            std::ofstream out(csv, std::ios::binary | std::ios::app);
            out << "More text,negative\n";
        }
        Check(key_of(graph).key != base.key, "an edited source file changes the key");
        registry.UnregisterTabularDataset(source);
        registry.UnregisterTabularDataset("prepared_loaded");
        std::filesystem::remove_all(cache_root);
    }

    std::cout << "Prepared data cache tests passed\n";
    return 0;
}
