#include "../src/core/arrow_dataset.h"
#include "../src/core/dataset_batcher.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/model_builder.h"
#include "../src/core/pipeline_materializer.h"
#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"

#include <arrow/api.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckFinite(float value, const std::string& message) {
    Check(std::isfinite(value), message);
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
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

gui::MLNode MakeDataInputNode() {
    gui::MLNode node;
    node.id = 1;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Data Input";
    return node;
}

gui::MLNode MakeTokenizerNode() {
    gui::MLNode node;
    node.id = 2;
    node.type = gui::NodeType::TextTokenizer;
    node.category = gui::NodeCategory::TextProcessing;
    node.name = "Text Tokenizer";
    node.parameters = {
        {"text_col", "text"},
        {"label_col", "label"},
        {"tokenizer_type", "1"},
        {"max_length", "4"},
        {"lowercase", "true"},
        {"min_word_freq", "1"},
        {"max_vocab_size", "100"},
    };
    return node;
}

cyxwiz::TrainingConfiguration MakeTrainingConfig(
    const std::filesystem::path& checkpoint_dir) {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "runtime_text";
    config.input_size = 4;
    config.input_shape = {4};
    config.output_size = 2;
    config.preprocessing_domain = cyxwiz::PreprocessingDomain::Text;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::Adam;
    config.learning_rate = 0.001f;
    config.train_ratio = 0.75f;
    config.shuffle = false;
    config.num_workers = 0;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
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
    const auto work_dir =
        std::filesystem::temp_directory_path() /
        "cyxwiz_text_arrow_training_launch";
    std::filesystem::create_directories(work_dir);
    std::filesystem::current_path(work_dir);

    std::vector<gui::MLNode> nodes = {
        MakeDataInputNode(),
        MakeTokenizerNode(),
    };
    std::vector<gui::NodeLink> links = {
        {1, 1, 0, 2, 0, gui::LinkType::TensorFlow},
    };

    auto materialized = cyxwiz::PipelineMaterializer::MaterializeTable(
        nodes, links, MakeTextTable());
    Check(materialized.success, materialized.error_message);
    Check(materialized.operators_applied == 1, "expected one tokenizer operator");
    Check(materialized.table != nullptr, "materialized table should not be null");
    Check(materialized.table->GetColumnByName("y") != nullptr,
          "materialized table should expose y label column");

    auto dataset = std::make_shared<cyxwiz::ArrowDataset>(
        materialized.table, "runtime_text_materialized");

    std::atomic<bool> started{false};
    std::atomic<bool> completed{false};
    std::atomic<bool> success{false};

    const auto checkpoint_dir = work_dir / "checkpoints";
    auto config = MakeTrainingConfig(checkpoint_dir);
    std::thread worker([&]() {
        started.store(true);

        cyxwiz::ArrowDatasetBatcher batcher(
            dataset,
            "y",
            /*batch_size=*/2,
            /*shuffle=*/false,
            config.train_ratio,
            /*is_training=*/true);
        batcher.SetOneHotEncoding(2);

        auto batch = batcher.GetNextBatch();
        Check(batch.IsValid(), "worker training batch should be valid");
        Check(batch.data.Shape().size() == 2,
              "worker feature tensor should be 2D");
        Check(batch.data.Shape()[1] == 4,
              "worker feature width should equal max_length");

        auto built = cyxwiz::BuildSequentialFromConfig(config);
        Check(built.ok(), "worker should build model/loss/optimizer");
        Check(built.model != nullptr, "worker missing model");
        Check(built.loss != nullptr, "worker missing loss");
        Check(built.optimizer != nullptr, "worker missing optimizer");

        auto predictions = built.model->Forward(batch.data);
        auto loss = built.loss->Forward(predictions, batch.labels);
        Check(loss.NumElements() == 1, "worker loss should be scalar");
        CheckFinite(loss.Data<float>()[0], "worker loss should be finite");

        auto grad = built.loss->Backward(predictions, batch.labels);
        built.model->Backward(grad);
        built.model->UpdateParameters(built.optimizer.get());

        success.store(true);
        completed.store(true);
    });
    worker.join();

    Check(started.load(), "worker training launch should start");
    Check(completed.load(), "worker training launch should complete");
    Check(success.load(), "worker training launch should succeed");

    std::cout << "Text Arrow runtime training launch passed\n";
    return 0;
}
