#include "../src/core/arrow_dataset.h"
#include "../src/core/data_registry.h"
#include "../src/core/dataset_batcher.h"
#include "../src/core/model_builder.h"
#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"
#include "../src/gui/graph_training_launcher.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr const char* kDatasetName = "gui_text_runtime";
constexpr const char* kMaterializedDatasetName = "gui_text_runtime__materialized";
constexpr const char* kUnusedDatasetName = "unused_gui_text_runtime";

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
    node.parameters = {
        {"dataset_name", kDatasetName},
        {"data_loaded", "true"},
        {"file_category", "text"},
        {"label_column", "label"},
    };
    return node;
}

gui::MLNode MakeUnusedDataInputNode() {
    gui::MLNode node;
    node.id = 99;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Unused Data Input";
    node.parameters = {
        {"dataset_name", kUnusedDatasetName},
        {"data_loaded", "true"},
        {"file_category", "text"},
        {"label_column", "wrong_label"},
    };
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
    config.is_valid = true;
    config.dataset_name = kDatasetName;
    config.input_size = 4;
    config.input_shape = {4};
    config.output_size = 2;
    config.preprocessing_domain = cyxwiz::PreprocessingDomain::Text;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::Adam;
    config.learning_rate = 0.001f;
    config.train_ratio = 0.75f;
    config.shuffle = false;
    config.epochs = 1;
    config.batch_size = 2;
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
        "cyxwiz_text_gui_training_launch";
    std::filesystem::create_directories(work_dir);
    std::filesystem::current_path(work_dir);

    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnregisterTabularDataset(kDatasetName);
    registry.UnregisterTabularDataset(kMaterializedDatasetName);
    registry.UnregisterTabularDataset(kUnusedDatasetName);
    Check(registry.RegisterArrowTable(MakeTextTable(), kDatasetName) != nullptr,
          "raw text Arrow dataset should register");
    Check(registry.RegisterArrowTable(MakeTextTable(), kUnusedDatasetName) != nullptr,
          "unused raw text Arrow dataset should register");

    std::vector<gui::MLNode> nodes = {
        MakeUnusedDataInputNode(),
        MakeDataInputNode(),
        MakeTokenizerNode(),
    };
    std::vector<gui::NodeLink> links = {
        {1, 1, 0, 2, 0, gui::LinkType::TensorFlow},
    };

    bool dispatch_called = false;
    bool callback_started = false;
    bool callback_finished = false;

    auto config = MakeTrainingConfig(work_dir / "checkpoints");
    auto dispatch = [&](cyxwiz::TrainingConfiguration dispatch_config,
                        const std::string& dataset_name,
                        const std::string& label_column,
                        int epochs,
                        int batch_size,
                        std::weak_ptr<cyxwiz::TrainingPlotPanel>,
                        std::function<void(bool)> callback) {
        dispatch_called = true;
        Check(dataset_name == kMaterializedDatasetName,
              "dispatch should receive materialized dataset");
        Check(dispatch_config.dataset_name == kMaterializedDatasetName,
              "config dataset name should match materialized dataset");
        Check(label_column == "y", "dispatch should receive runtime y label");
        Check(epochs == 1, "epochs should come from compiled config");
        Check(batch_size == 2, "batch size should come from compiled config");

        if (callback) {
            callback(true);
            callback_started = true;
        }

        auto dataset = registry.GetArrowDataset(dataset_name);
        Check(dataset != nullptr, "materialized Arrow dataset should exist");
        auto table = dataset->GetArrowTable();
        Check(table != nullptr, "materialized table should exist");
        Check(table->GetColumnByName("tok_0") != nullptr,
              "materialized table should expose token columns");
        Check(table->GetColumnByName("y") != nullptr,
              "materialized table should expose y label");

        cyxwiz::ArrowDatasetBatcher batcher(
            dataset,
            label_column,
            batch_size,
            /*shuffle=*/false,
            dispatch_config.train_ratio,
            /*is_training=*/true);
        batcher.SetOneHotEncoding(2);

        auto batch = batcher.GetNextBatch();
        Check(batch.IsValid(), "GUI launch batch should be valid");
        Check(batch.data.Shape().size() == 2, "GUI launch features should be 2D");
        Check(batch.data.Shape()[1] == 4,
              "GUI launch feature width should equal tokenizer max_length");

        auto built = cyxwiz::BuildSequentialFromConfig(dispatch_config);
        Check(built.ok(), "GUI launch should build model/loss/optimizer");
        auto predictions = built.model->Forward(batch.data);
        auto loss = built.loss->Forward(predictions, batch.labels);
        Check(loss.NumElements() == 1, "GUI launch loss should be scalar");
        CheckFinite(loss.Data<float>()[0], "GUI launch loss should be finite");

        auto grad = built.loss->Backward(predictions, batch.labels);
        built.model->Backward(grad);
        built.model->UpdateParameters(built.optimizer.get());

        if (callback) {
            callback(false);
            callback_finished = true;
        }
        return true;
    };

    auto result = gui::StartGraphTrainingFromCompiledConfig(
        nodes,
        links,
        std::move(config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [](bool) {},
        dispatch);

    Check(result.started, result.error_message);
    Check(dispatch_called, "dispatch should be called");
    Check(callback_started, "training start callback should fire");
    Check(callback_finished, "training finish callback should fire");
    Check(result.effective_dataset_name == kMaterializedDatasetName,
          "result should report materialized dataset");
    Check(result.label_column == "y", "result should report resolved y label");
    Check(result.operators_applied == 1, "expected one tokenizer operator");
    Check(result.epochs == 1, "result epochs should match config");
    Check(result.batch_size == 2, "result batch size should match config");

    registry.UnregisterTabularDataset(kDatasetName);
    registry.UnregisterTabularDataset(kMaterializedDatasetName);
    registry.UnregisterTabularDataset(kUnusedDatasetName);

    std::cout << "Text GUI training launch helper passed\n";
    return 0;
}
