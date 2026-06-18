#include "../src/core/arrow_dataset.h"
#include "../src/core/data_registry.h"
#include "../src/core/dataset_batcher.h"
#include "../src/core/model_builder.h"
#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/core/pipeline_materializer.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/core/training_run_comparison.h"
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
constexpr const char* kScopeParquetDatasetName = "gui_text_runtime_scope_parquet";
constexpr const char* kScopeImageDatasetName = "gui_text_runtime_scope_image";
constexpr const char* kScopeAudioDatasetName = "gui_text_runtime_scope_audio";
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

std::shared_ptr<arrow::Table> MakeSequenceTable() {
    auto tokens = FinishStringArray({
        "John lives in Berlin",
        "Mary works in Paris",
    });
    auto ner_tags = FinishStringArray({
        "B-PER O O B-LOC",
        "B-PER O O B-LOC",
    });

    auto schema = arrow::schema({
        arrow::field("tokens", arrow::utf8()),
        arrow::field("ner_tags", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {tokens, ner_tags}, 2);
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

void CheckUnsupportedMaterializerSource(
    const cyxwiz::MaterializeResult& result,
    const std::string& dataset_name,
    cyxwiz::PipelineMaterializerSourceKind source_kind,
    cyxwiz::PipelineStorageBackend backend,
    const std::string& label) {

    Check(result.success, result.error_message);
    Check(result.effective_dataset_name == dataset_name,
          label + " source should pass through unchanged");
    Check(result.operators_applied == 0,
          label + " source should not apply Arrow operators");
    Check(result.source_kind == source_kind,
          label + " source should report its source kind");
    Check(result.skipped_unsupported_source,
          label + " source should report unsupported-source skip");
    const auto backend_support =
        cyxwiz::ResolvePipelineMaterializerStorageBackendSupport(backend);
    Check(backend_support.reason != nullptr,
          label + " materializer backend reason should be registered");
    Check(result.unsupported_source_reason == backend_support.reason,
          label + " source should expose central materializer skip reason");
}

void TestTrainingRunComparisonRecord() {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "sentiment_v1";
    config.epochs = 8;
    config.batch_size = 32;
    config.learning_rate = 0.0002f;
    config.save_best_checkpoint = true;
    config.early_stopping_patience = 3;
    config.checkpoint_dir = "runs/sentiment";

    cyxwiz::CompiledLayer gru;
    gru.type = gui::NodeType::GRU;
    gru.parameters["hidden_size"] = "96";
    gru.parameters["num_layers"] = "2";
    gru.parameters["bidirectional"] = "true";
    config.layers.push_back(gru);

    cyxwiz::TrainingMetrics metrics;
    metrics.train_loss = 0.25f;
    metrics.train_accuracy = 0.91f;
    metrics.val_loss_history = {0.9f, 0.7f, 0.8f};
    metrics.val_accuracy_history = {0.65f, 0.72f, 0.70f};
    metrics.test_loss = 0.68f;
    metrics.test_accuracy = 0.71f;
    metrics.checkpoint_used = "runs/sentiment/best_checkpoint.cyxckpt";

    const auto record = cyxwiz::MakeTrainingRunComparisonRecord(
        "run-001", config, metrics, 12.5f,
        metrics.checkpoint_used,
        "complete");

    Check(record.run_id == "run-001", "run comparison should keep run id");
    Check(record.run_status == "complete",
          "run comparison should keep run status");
    Check(record.dataset_name == "sentiment_v1",
          "run comparison should keep dataset");
    Check(record.primary_layer_type == "GRU",
          "run comparison should keep primary layer type");
    Check(record.architecture_summary == "GRU",
          "run comparison should summarize architecture");
    Check(record.model_layer_count == 1,
          "run comparison should count model layers");
    Check(record.model_family == "GRU",
          "run comparison should detect GRU family");
    Check(record.bidirectional,
          "run comparison should preserve bidirectional flag");
    Check(record.hidden_size == 96,
          "run comparison should preserve hidden size");
    Check(record.num_layers == 2,
          "run comparison should preserve recurrent layer count");
    Check(record.best_val_loss == 0.7f,
          "run comparison should compute best validation loss");
    Check(record.best_val_accuracy == 0.72f,
          "run comparison should compute best validation accuracy");
    Check(record.has_validation_metrics,
          "run comparison should mark validation metrics present");
    Check(record.has_test_metrics,
          "run comparison should mark test metrics present");
    Check(record.checkpoint_used == "runs/sentiment/best_checkpoint.cyxckpt",
          "run comparison should keep checkpoint used");
    Check(record.final_test_accuracy == 0.71f,
          "run comparison should keep final test accuracy");

    const std::string csv = cyxwiz::TrainingRunComparisonTableSummary({record});
    Check(csv.find("run_id,run_status,dataset_name,model_family") == 0,
          "run comparison CSV should include stable header");
    Check(csv.find("run-001,complete,sentiment_v1,GRU") != std::string::npos,
          "run comparison CSV should include record row");

    const auto output_path =
        std::filesystem::temp_directory_path() /
        "cyxwiz_training_run_comparison" /
        "runs.csv";
    std::string error;
    Check(cyxwiz::WriteTrainingRunComparisonCsv(output_path, {record}, &error),
          "run comparison CSV export should succeed: " + error);
    Check(std::filesystem::exists(output_path),
          "run comparison CSV export should create output file");

    auto weaker = record;
    weaker.run_id = "run-002";
    weaker.final_test_accuracy = 0.60f;
    auto sorted = cyxwiz::SortTrainingRunComparisonsByBestMetric({weaker, record});
    Check(sorted.size() == 2,
          "run comparison sort should keep all records");
    Check(sorted.front().run_id == "run-001",
          "run comparison sort should prefer higher test accuracy");

    config.checkpoint_dir.clear();
    const auto default_checkpoint_record =
        cyxwiz::MakeTrainingRunComparisonRecord(
            "run-default-checkpoint", config, metrics, 1.0f);
    Check(default_checkpoint_record.checkpoint_used ==
              "default .cyxwiz/checkpoints run folder",
          "run comparison should make default checkpoint root explicit");
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
    TestTrainingRunComparisonRecord();

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
    registry.UnregisterTabularDataset(kScopeParquetDatasetName);
    registry.UnregisterImageDataset(kScopeImageDatasetName);
    registry.UnregisterAudioDataset(kScopeAudioDatasetName);
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

    const auto parquet_path = work_dir / "materializer_scope.parquet";
    std::remove(parquet_path.string().c_str());
    cyxwiz::ArrowDataset parquet_fixture(
        MakeTextTable(), kScopeParquetDatasetName);
    Check(parquet_fixture.ExportParquet(parquet_path.string()),
          "materializer scope Parquet fixture should export");
    auto parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        parquet_path.string(), kScopeParquetDatasetName);
    Check(parquet_dataset != nullptr,
          "materializer scope Parquet fixture should open");
    registry.RegisterParquetBacked(kScopeParquetDatasetName, parquet_dataset);
    auto parquet_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeParquetDatasetName);
    CheckUnsupportedMaterializerSource(
        parquet_scope,
        kScopeParquetDatasetName,
        cyxwiz::PipelineMaterializerSourceKind::ParquetBacked,
        cyxwiz::PipelineStorageBackend::ParquetBacked,
        "Parquet-backed");
    registry.UnregisterTabularDataset(kScopeParquetDatasetName);
    parquet_dataset.reset();
    std::remove(parquet_path.string().c_str());

    cyxwiz::DataRegistry::ImageDatasetEntry image_entry;
    image_entry.folder_path = "images";
    image_entry.num_images = 4;
    image_entry.num_classes = 2;
    registry.RegisterImageDataset(kScopeImageDatasetName, image_entry);
    auto image_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeImageDatasetName);
    CheckUnsupportedMaterializerSource(
        image_scope,
        kScopeImageDatasetName,
        cyxwiz::PipelineMaterializerSourceKind::ImageDataset,
        cyxwiz::PipelineStorageBackend::ImageDataset,
        "Image");
    registry.UnregisterImageDataset(kScopeImageDatasetName);

    cyxwiz::DataRegistry::AudioDatasetEntry audio_entry;
    audio_entry.folder_path = "audio";
    audio_entry.num_samples = 4;
    audio_entry.num_classes = 2;
    registry.RegisterAudioDataset(kScopeAudioDatasetName, audio_entry);
    auto audio_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeAudioDatasetName);
    CheckUnsupportedMaterializerSource(
        audio_scope,
        kScopeAudioDatasetName,
        cyxwiz::PipelineMaterializerSourceKind::AudioDataset,
        cyxwiz::PipelineStorageBackend::AudioDataset,
        "Audio");
    registry.UnregisterAudioDataset(kScopeAudioDatasetName);

    cyxwiz::DataRegistry::TextDatasetEntry text_entry;
    text_entry.source_path = "legacy_text.csv";
    text_entry.text_column = "text";
    text_entry.label_column = "label";
    text_entry.num_samples = 3;
    registry.RegisterTextDataset(kScopeTextDatasetName, text_entry);
    auto text_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeTextDatasetName);
    CheckUnsupportedMaterializerSource(
        text_scope,
        kScopeTextDatasetName,
        cyxwiz::PipelineMaterializerSourceKind::TextDataset,
        cyxwiz::PipelineStorageBackend::TextDataset,
        "Legacy text");
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
        Check(!dispatch_config.save_best_checkpoint,
              "save_best_checkpoint should come from compiled config");
        Check(dispatch_config.early_stopping_patience == 0,
              "early stopping patience should come from compiled config");
        Check(dispatch_config.checkpoint_dir ==
                  (work_dir / "checkpoints").string(),
              "checkpoint directory should come from compiled config");

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

    auto sequence_config =
        MakeTrainingConfig(work_dir / "sequence_launch_checkpoints");
    sequence_config.sequence_batch.enabled = true;
    sequence_config.sequence_batch.token_column = "tokens";
    sequence_config.sequence_batch.tag_column = "ner_tags";
    bool sequence_dispatch_called = false;
    auto sequence_dispatch = [&](
        cyxwiz::TrainingConfiguration dispatch_config,
        const std::string&,
        const std::string&,
        int,
        int,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>,
        std::function<void(bool)>) {
        sequence_dispatch_called = true;
        Check(dispatch_config.sequence_batch.enabled,
              "sequence launch should preserve sequence_batch.enabled");
        Check(dispatch_config.sequence_batch.token_column == "tokens",
              "sequence launch should preserve token column");
        Check(dispatch_config.sequence_batch.tag_column == "ner_tags",
              "sequence launch should preserve tag column");
        return true;
    };
    auto sequence_result = gui::StartGraphTrainingFromCompiledConfig(
        nodes,
        links,
        std::move(sequence_config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [](bool) {},
        sequence_dispatch);

    Check(sequence_result.started,
          sequence_result.error_message);
    Check(sequence_dispatch_called,
          "sequence batch launch should call dispatch");

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
    const auto legacy_text_backend_support =
        cyxwiz::ResolvePipelineMaterializerStorageBackendSupport(
            cyxwiz::PipelineStorageBackend::TextDataset);
    Check(legacy_text_result.materializer_unsupported_source_reason ==
              legacy_text_backend_support.reason,
          "legacy text result should expose central skip reason");
    registry.UnregisterTextDataset(kScopeTextDatasetName);

    std::cout << "Text GUI training launch helper passed\n";
    return 0;
}
