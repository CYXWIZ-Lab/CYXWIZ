#include "../src/core/arrow_dataset.h"
#include "../src/core/data_registry.h"
#include "../src/core/dataset_batcher.h"
#include "../src/core/model_builder.h"
#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"
#include "../src/core/pipeline_materializer.h"
#include "../src/core/pipeline_runtime_capabilities.h"
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
constexpr const char* kScopeArrowDatasetName = "gui_text_runtime_scope_arrow";
constexpr const char* kScopeTextDatasetName = "gui_text_runtime_scope_legacy_text";

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

gui::MLNode MakeDataInputNode(
    const std::string& dataset_name = kDatasetName) {
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

gui::MLNode MakeOptimizerNode(
    int id,
    const std::string& name,
    const std::string& epochs,
    const std::string& batch_size) {

    gui::MLNode node;
    node.id = id;
    node.type = gui::NodeType::Adam;
    node.category = gui::NodeCategory::Training;
    node.name = name;
    node.parameters = {
        {"epochs", epochs},
        {"batch_size", batch_size},
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
    config.data_source_node_id = 1;
    config.optimizer_node_id = 4;
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
        MakeOptimizerNode(3, "Stale Adam", "99", "99"),
        MakeDataInputNode(),
        MakeTokenizerNode(),
        MakeOptimizerNode(4, "Selected Adam", "", ""),
    };
    std::vector<gui::NodeLink> links = {
        {1, 1, 0, 2, 0, gui::LinkType::TensorFlow},
    };

    registry.UnregisterTabularDataset(kScopeArrowDatasetName);
    registry.UnregisterTabularDataset(
        std::string(kScopeArrowDatasetName) +
        cyxwiz::PipelineMaterializer::kMaterializedSuffix);
    Check(registry.RegisterArrowTable(MakeTextTable(), kScopeArrowDatasetName) != nullptr,
          "Arrow source should register for materializer scope test");
    std::vector<gui::MLNode> scope_nodes = {
        MakeDataInputNode(kScopeArrowDatasetName),
        MakeTokenizerNode(),
    };
    auto arrow_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeArrowDatasetName);
    Check(arrow_scope.success, arrow_scope.error_message);
    Check(arrow_scope.source_kind ==
              cyxwiz::PipelineMaterializerSourceKind::ArrowTable,
          "Arrow source should report ArrowTable source kind");
    Check(!arrow_scope.skipped_unsupported_source,
          "Arrow source should not report unsupported-source skip");
    Check(arrow_scope.unsupported_source_reason.empty(),
          "Arrow source should not report unsupported-source reason");
    Check(arrow_scope.operators_applied == 1,
          "Arrow source should apply tokenizer through registry materializer");
    registry.UnregisterTabularDataset(kScopeArrowDatasetName);
    registry.UnregisterTabularDataset(
        std::string(kScopeArrowDatasetName) +
        cyxwiz::PipelineMaterializer::kMaterializedSuffix);

    cyxwiz::DataRegistry::TextDatasetEntry text_entry;
    text_entry.source_path = "legacy_text.csv";
    text_entry.text_column = "text";
    text_entry.label_column = "label";
    text_entry.num_samples = 3;
    registry.RegisterTextDataset(kScopeTextDatasetName, text_entry);
    auto text_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeTextDatasetName);
    Check(text_scope.success, text_scope.error_message);
    Check(text_scope.effective_dataset_name == kScopeTextDatasetName,
          "legacy text source should pass through unchanged");
    Check(text_scope.operators_applied == 0,
          "legacy text source should not apply Arrow operators");
    Check(text_scope.source_kind ==
              cyxwiz::PipelineMaterializerSourceKind::TextDataset,
          "legacy text source should report TextDataset source kind");
    Check(text_scope.skipped_unsupported_source,
          "legacy text source should report unsupported-source skip");
    const auto text_backend_support =
        cyxwiz::ResolvePipelineMaterializerStorageBackendSupport(
            cyxwiz::PipelineStorageBackend::TextDataset);
    Check(text_backend_support.reason != nullptr,
          "text materializer backend reason should be registered");
    Check(text_scope.unsupported_source_reason == text_backend_support.reason,
          "legacy text source should expose central materializer skip reason");
    registry.UnregisterTextDataset(kScopeTextDatasetName);

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
    Check(result.materializer_source_kind ==
              cyxwiz::PipelineMaterializerSourceKind::ArrowTable,
          "result should report ArrowTable materializer source kind");
    Check(!result.materializer_skipped_unsupported_source,
          "result should not report unsupported materializer skip for Arrow source");
    Check(result.materializer_unsupported_source_reason.empty(),
          "result should not report unsupported materializer reason for Arrow source");
    Check(result.epochs == 1, "result epochs should match config");
    Check(result.batch_size == 2, "result batch size should match config");

    registry.UnregisterTabularDataset(kDatasetName);
    registry.UnregisterTabularDataset(kMaterializedDatasetName);
    registry.UnregisterTabularDataset(kUnusedDatasetName);

    cyxwiz::DataRegistry::TextDatasetEntry launch_text_entry = text_entry;
    launch_text_entry.num_samples = 3;
    registry.RegisterTextDataset(kScopeTextDatasetName, launch_text_entry);

    auto legacy_text_config = MakeTrainingConfig(work_dir / "legacy_text_checkpoints");
    legacy_text_config.dataset_name = kScopeTextDatasetName;
    bool legacy_text_dispatch_called = false;
    auto legacy_text_dispatch = [&](cyxwiz::TrainingConfiguration dispatch_config,
                                    const std::string& dataset_name,
                                    const std::string& label_column,
                                    int epochs,
                                    int batch_size,
                                    std::weak_ptr<cyxwiz::TrainingPlotPanel>,
                                    std::function<void(bool)> callback) {
        legacy_text_dispatch_called = true;
        Check(dispatch_config.dataset_name == kScopeTextDatasetName,
              "legacy text config should keep original dataset");
        Check(dataset_name == kScopeTextDatasetName,
              "legacy text dispatch should receive original dataset");
        Check(label_column == "label",
              "legacy text dispatch should keep configured label column");
        Check(epochs == 1, "legacy text epochs should match config");
        Check(batch_size == 2, "legacy text batch size should match config");
        if (callback) {
            callback(true);
        }
        return true;
    };

    std::vector<gui::MLNode> legacy_text_nodes = {
        MakeDataInputNode(kScopeTextDatasetName),
        MakeTokenizerNode(),
        MakeOptimizerNode(4, "Text Adam", "", ""),
    };
    auto legacy_text_result = gui::StartGraphTrainingFromCompiledConfig(
        legacy_text_nodes,
        links,
        std::move(legacy_text_config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [](bool) {},
        legacy_text_dispatch);

    Check(legacy_text_result.started, legacy_text_result.error_message);
    Check(legacy_text_dispatch_called,
          "legacy text dispatch should be called");
    Check(legacy_text_result.effective_dataset_name == kScopeTextDatasetName,
          "legacy text result should keep original dataset");
    Check(legacy_text_result.operators_applied == 0,
          "legacy text result should not apply Arrow materializer operators");
    Check(legacy_text_result.materializer_source_kind ==
              cyxwiz::PipelineMaterializerSourceKind::TextDataset,
          "legacy text result should report TextDataset source kind");
    Check(legacy_text_result.materializer_skipped_unsupported_source,
          "legacy text result should report unsupported materializer skip");
    Check(legacy_text_result.materializer_unsupported_source_reason ==
              text_backend_support.reason,
          "legacy text result should expose central skip reason");
    registry.UnregisterTextDataset(kScopeTextDatasetName);

    std::cout << "Text GUI training launch helper passed\n";
    return 0;
}
