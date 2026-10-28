// The text path training reads (TOFIX178 S0-e/S0-f2): a Text Tokenizer step
// runs in the pipeline run (the canvas's engine), its result is the table the
// batcher and the model read, and PrepareStepResult keeps it on disk so the
// next preparation loads it instead of running the step again.

#include "../src/core/arrow_dataset.h"
#include "../src/core/data_registry.h"
#include "../src/core/dataset_batcher.h"
#include "../src/core/model_builder.h"
#include "../src/core/pipeline_executor.h"
#include "../src/core/prepared_data_cache.h"
#include "../src/core/training_input_preparation.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckFinite(float value, const std::string& message) {
    Check(std::isfinite(value), message);
}

fs::path WorkDir() {
    return fs::temp_directory_path() / "cyxwiz_text_arrow_materializer";
}

fs::path WriteTextCsv() {
    const auto path = WorkDir() / "text.csv";
    std::ofstream out(path, std::ios::binary);
    out << "text,label\n"
           "Small text sample,positive\n"
           "Another small sample,positive\n"
           "Text pipelines should tokenize,negative\n";
    return path;
}

// One input and one output pin, as the canvas's nodes have.
gui::MLNode WithPins(gui::MLNode node) {
    node.inputs = {{0, gui::PinType::Tensor, "in", true}};
    node.outputs = {{0, gui::PinType::Tensor, "out", false}};
    return node;
}

gui::MLNode MakeDataInputNode(const fs::path& csv) {
    gui::MLNode node;
    node.id = 1;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Data Input";
    node.parameters = {{"source_type", "file"}, {"file_path", csv.string()}, {"type", "csv"},
                       {"has_header", "true"}, {"dataset_name", "text_arrow_source"}};
    return WithPins(node);
}

gui::MLNode MakeTokenizerNode(int id = 2, std::string max_length = "4") {
    gui::MLNode node;
    node.id = id;
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
    return WithPins(node);
}

gui::MLNode MakeNode(int id, gui::NodeType type, const std::string& name) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    return WithPins(node);
}

std::string RunJson(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links) {
    nlohmann::json pipeline;
    std::string error;
    Check(cyxwiz::PipelineJsonFromGraph(nodes, links, pipeline, error), "pipeline JSON: " + error);
    return pipeline.dump();
}

std::shared_ptr<arrow::Table> TableOf(const std::string& name) {
    const auto dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset(name);
    Check(dataset && dataset->GetArrowTable(), "'" + name + "' is a registered table");
    return dataset->GetArrowTable();
}

// The pipeline run's Text Tokenizer step: token columns and y, no raw text.
std::shared_ptr<arrow::Table> TestTokenizerStep(const fs::path& csv) {
    const std::vector<gui::MLNode> nodes = {MakeDataInputNode(csv), MakeTokenizerNode()};
    const std::vector<gui::NodeLink> links = {{1, 1, 0, 2, 0, gui::LinkType::TensorFlow}};
    cyxwiz::PipelineExecutor executor;
    Check(executor.ExecutePipeline(RunJson(nodes, links)), "the tokenizer step runs: " + executor.GetLastError());
    const auto result = executor.NodeResults().find(2);
    Check(result != executor.NodeResults().end() && result->second == "ds_operator_TextTokenizer_2",
          "the step's result is its node result");
    const auto table = TableOf(result->second);
    Check(table->num_rows() == 3, "expected 3 tokenized rows");
    Check(table->num_columns() == 5, "expected 4 token columns plus y");
    Check(table->GetColumnByName("tok_0") && table->GetColumnByName("tok_3"), "token columns tok_0..tok_3");
    Check(table->GetColumnByName("y") != nullptr, "the y label column");
    Check(table->GetColumnByName("text") == nullptr, "raw text column should not remain after tokenization");
    executor.ReleaseAll({result->second});
    return table;
}

// Graph shape: only the branch into training is prepared; a cycle stops the run.
void TestGraphShape(const fs::path& csv) {
    auto& registry = cyxwiz::DataRegistry::Instance();
    // A side branch (a second tokenizer that leads nowhere, like a Word
    // Embeddings node saving its vectors) is not part of the training table.
    const auto side = MakeTokenizerNode(5);
    const std::vector<gui::MLNode> nodes = {MakeDataInputNode(csv), MakeTokenizerNode(), side,
                                            MakeNode(6, gui::NodeType::DataSplit, "Split"),
                                            MakeNode(7, gui::NodeType::DataLoader, "Loader")};
    const std::vector<gui::NodeLink> links = {
        {1, 1, 0, 2, 0, gui::LinkType::TensorFlow},
        {2, 1, 0, 5, 0, gui::LinkType::TensorFlow},
        {3, 2, 0, 6, 0, gui::LinkType::TensorFlow},
        {4, 6, 0, 7, 0, gui::LinkType::TensorFlow},
    };
    Check(cyxwiz::TrainingInputFeeder(nodes, links) == 2, "the Split's input is the training tokenizer");
    registry.UnregisterTabularDataset("ds_operator_TextTokenizer_5");
    const auto prepared = cyxwiz::PrepareStepResult(nodes, links, 2, registry, {}, {});
    Check(prepared.ok && prepared.ran && prepared.dataset == "ds_operator_TextTokenizer_2",
          "a side branch does not block preparing the training branch: " + prepared.error);
    Check(registry.GetArrowDataset("ds_operator_TextTokenizer_5") == nullptr, "the side branch is not run");
    registry.UnregisterTabularDataset(prepared.dataset);

    // Two steps feeding each other.
    const std::vector<gui::NodeLink> cyclic = {
        {1, 1, 0, 2, 0, gui::LinkType::TensorFlow},
        {2, 2, 0, 5, 0, gui::LinkType::TensorFlow},
        {3, 5, 0, 2, 0, gui::LinkType::TensorFlow},
    };
    cyxwiz::PipelineExecutor executor;
    Check(!executor.ExecutePipeline(RunJson({MakeDataInputNode(csv), MakeTokenizerNode(), side}, cyclic)),
          "a cyclic graph does not run");
    Check(!executor.GetLastError().empty(), "the cycle is reported: " + executor.GetLastError());
    executor.ReleaseAll();
}

// PrepareStepResult: a miss runs the step and keeps it; a hit loads it; a
// rebuild runs again; a changed setting misses; a cancel gives no result.
void TestPreparedStepCache(const fs::path& csv) {
    auto& registry = cyxwiz::DataRegistry::Instance();
    const auto source = MakeDataInputNode(csv);
    // The Data Input's data is loaded (as after Apply): the key reads it.
    Check(registry.LoadTabularCSV(csv.string(), "text_arrow_source") ==
              cyxwiz::DataRegistry::TabularLoadBackend::InMemory,
          "the source table loads");
    const std::vector<gui::MLNode> nodes = {source, MakeTokenizerNode(), MakeNode(3, gui::NodeType::DataSplit, "Split")};
    const std::vector<gui::NodeLink> links = {{1, 1, 0, 2, 0, gui::LinkType::TensorFlow},
                                              {2, 2, 0, 3, 0, gui::LinkType::TensorFlow}};
    const auto key = cyxwiz::ComputePreparedDataKey(nodes, links, 2, registry, "");
    Check(key.cacheable, "the tokenizer path is cacheable: " + key.reason);

    const auto root = WorkDir() / "prepared_cache";
    fs::remove_all(root);
    cyxwiz::MaterializationCacheConfig config;
    config.mode = cyxwiz::MaterializationCacheMode::Auto;
    config.cache_root = root;

    const auto miss = cyxwiz::PrepareStepResult(nodes, links, 2, registry, {}, config);
    Check(miss.ok && miss.ran && !miss.from_disk && miss.dataset == "ds_operator_TextTokenizer_2",
          "a miss runs the step: " + miss.error);
    Check(cyxwiz::ProbePreparedData(config, key), "the run keeps its result on disk");
    const auto ran_table = TableOf(miss.dataset);
    registry.UnregisterTabularDataset(miss.dataset);

    std::vector<std::string> messages;
    const auto hit = cyxwiz::PrepareStepResult(nodes, links, 2, registry, {}, config, {},
                                               [&messages](float, const std::string& m) { messages.push_back(m); });
    Check(hit.ok && hit.from_disk && !hit.ran && hit.dataset.rfind("prepared_", 0) == 0,
          "a hit loads the result instead of running the step: " + hit.error);
    Check(!messages.empty() && messages.front().find("from disk") != std::string::npos,
          "a hit says it loads from disk");
    Check(TableOf(hit.dataset)->Equals(*ran_table), "the loaded table equals the one the run made");
    registry.UnregisterTabularDataset(hit.dataset);

    auto rebuild = config;
    rebuild.mode = cyxwiz::MaterializationCacheMode::Rebuild;
    const auto rebuilt = cyxwiz::PrepareStepResult(nodes, links, 2, registry, {}, rebuild);
    Check(rebuilt.ok && rebuilt.ran && !rebuilt.from_disk, "a rebuild runs the step again: " + rebuilt.error);
    registry.UnregisterTabularDataset(rebuilt.dataset);

    auto changed = nodes;
    changed[1] = MakeTokenizerNode(2, "5");
    const auto changed_run = cyxwiz::PrepareStepResult(changed, links, 2, registry, {}, config);
    // Five token columns and the label; the Split's rows also carry their
    // partition (TOFIX178 S0-g3).
    Check(changed_run.ok && changed_run.ran && !changed_run.from_disk &&
              TableOf(changed_run.dataset)->num_columns() == 7 &&
              TableOf(changed_run.dataset)->schema()->GetFieldIndex("__partition__") >= 0,
          "a changed setting misses and runs: " + changed_run.error);
    registry.UnregisterTabularDataset(changed_run.dataset);

    const auto cancelled = cyxwiz::PrepareStepResult(changed, links, 2, registry, {}, rebuild, [] { return true; });
    Check(!cancelled.ok && cancelled.cancelled && cancelled.dataset.empty(),
          "a cancelled preparation is a cancel, not a result");
    Check(registry.GetArrowDataset("ds_operator_TextTokenizer_2") == nullptr, "a cancel publishes nothing");

    registry.UnregisterTabularDataset("text_arrow_source");
    fs::remove_all(root);
}

}  // namespace

int main() {
    fs::remove_all(WorkDir());
    fs::create_directories(WorkDir());
    const auto csv = WriteTextCsv();

    const auto table = TestTokenizerStep(csv);
    TestGraphShape(csv);
    TestPreparedStepCache(csv);

    const auto parquet_path = WorkDir() / "cyxwiz_text_arrow_materializer.parquet";
    cyxwiz::ArrowDataset tokenized_dataset(table, "tokenized_text");
    Check(tokenized_dataset.ExportParquet(parquet_path.string()), "tokenized table should export to Parquet");

    auto reloaded = cyxwiz::ArrowDataset::FromParquet(parquet_path.string(), "tokenized_text_reloaded");
    Check(reloaded != nullptr, "exported Parquet should reload");
    auto reloaded_table = reloaded->GetArrowTable();
    Check(reloaded_table != nullptr, "reloaded table should not be null");
    Check(reloaded_table->num_rows() == table->num_rows(), "reloaded row count should match");
    Check(reloaded_table->num_columns() == table->num_columns(), "reloaded column count should match");
    Check(reloaded_table->GetColumnByName("tok_0") != nullptr, "reloaded table missing tok_0");
    Check(reloaded_table->GetColumnByName("y") != nullptr, "reloaded table missing y");

    auto batcher_dataset = std::make_shared<cyxwiz::ArrowDataset>(table, "tokenized_text_train");
    cyxwiz::ArrowDatasetBatcher batcher(batcher_dataset, "y", /*batch_size=*/2, /*shuffle=*/false,
                                        /*train_split=*/1.0f, /*is_training=*/true);
    batcher.SetOneHotEncoding(2);

    auto batch = batcher.GetNextBatch();
    Check(batch.IsValid(), "training batch should be valid");
    Check(batch.size == 2, "training batch should contain 2 samples");
    Check(batch.data.Shape().size() == 2, "feature tensor should be 2D");
    Check(batch.data.Shape()[0] == 2, "feature batch dimension should be 2");
    Check(batch.data.Shape()[1] == 4, "feature width should equal max_length");
    Check(batch.labels.Shape().size() == 2, "label tensor should be 2D");
    Check(batch.labels.Shape()[0] == 2, "label batch dimension should be 2");
    Check(batch.labels.Shape()[1] == 2, "label width should equal num_classes");

    const float* label_data = batch.labels.Data<float>();
    for (size_t r = 0; r < batch.size; ++r) {
        const float row_sum = label_data[r * 2] + label_data[r * 2 + 1];
        Check(row_sum == 1.0f, "one-hot label row should sum to 1");
    }

    cyxwiz::TrainingConfiguration train_config;
    train_config.input_size = 4;
    train_config.input_shape = {4};
    train_config.output_size = 2;
    train_config.preprocessing_domain = cyxwiz::PreprocessingDomain::Text;
    train_config.loss_type = gui::NodeType::CrossEntropyLoss;
    train_config.optimizer_type = gui::NodeType::Adam;
    train_config.learning_rate = 0.001f;

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    train_config.layers.push_back(dense);

    auto built = cyxwiz::BuildSequentialFromConfig(train_config);
    Check(built.ok(), "training smoke should build model/loss/optimizer");
    Check(built.model != nullptr, "training smoke missing model");
    Check(built.loss != nullptr, "training smoke missing loss");
    Check(built.optimizer != nullptr, "training smoke missing optimizer");

    auto predictions = built.model->Forward(batch.data);
    Check(predictions.Shape().size() == 2, "training smoke predictions should be 2D");
    Check(predictions.Shape()[0] == batch.size, "training smoke prediction batch should match input batch");
    Check(predictions.Shape()[1] == 2, "training smoke prediction width should equal output_size");

    auto loss = built.loss->Forward(predictions, batch.labels);
    Check(loss.NumElements() == 1, "training smoke loss should be scalar");
    CheckFinite(loss.Data<float>()[0], "training smoke loss should be finite");

    auto grad = built.loss->Backward(predictions, batch.labels);
    built.model->Backward(grad);
    built.model->UpdateParameters(built.optimizer.get());

    cyxwiz::DataRegistry::Instance().UnregisterTabularDataset("ds_operator_TextTokenizer_2");
    std::error_code error;
    fs::remove_all(WorkDir(), error);

    std::cout << "Text Arrow materializer path passed\n";
    return 0;
}
