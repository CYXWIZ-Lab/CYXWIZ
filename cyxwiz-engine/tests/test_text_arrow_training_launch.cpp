#include "../src/core/arrow_dataset.h"
#include "../src/core/dataset_batcher.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/model_builder.h"
#include "../src/core/data_registry.h"
#include "../src/core/training_input_preparation.h"

#include <arrow/api.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

std::filesystem::path WriteTextCsv(const std::filesystem::path& dir) {
    const auto path = dir / "runtime_text.csv";
    std::ofstream out(path, std::ios::binary);
    out << "text,label\n"
           "Small happy text sample,positive\n"
           "Another happy sample,positive\n"
           "Sad text pipeline example,negative\n"
           "Another sad example,negative\n";
    return path;
}

gui::MLNode WithPins(gui::MLNode node) {
    node.inputs = {{0, gui::PinType::Tensor, "in", true}};
    node.outputs = {{0, gui::PinType::Tensor, "out", false}};
    return node;
}

gui::MLNode MakeDataInputNode(const std::filesystem::path& csv) {
    gui::MLNode node;
    node.id = 1;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Data Input";
    node.parameters = {{"source_type", "file"}, {"file_path", csv.string()},
                       {"type", "csv"}, {"has_header", "true"}};
    return WithPins(node);
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
    return WithPins(node);
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

int main() {
    const auto work_dir =
        std::filesystem::temp_directory_path() /
        "cyxwiz_text_arrow_training_launch";
    std::filesystem::create_directories(work_dir);
    std::filesystem::current_path(work_dir);

    // The step before training runs in the pipeline run (TOFIX178 S0-e),
    // the same engine as the canvas; training reads its result.
    std::vector<gui::MLNode> nodes = {
        MakeDataInputNode(WriteTextCsv(work_dir)),
        MakeTokenizerNode(),
    };
    std::vector<gui::NodeLink> links = {
        {1, 1, 0, 2, 0, gui::LinkType::TensorFlow},
    };
    auto& registry = cyxwiz::DataRegistry::Instance();
    const auto prepared = cyxwiz::PrepareStepResult(nodes, links, 2, registry, {}, {});
    Check(prepared.ok && prepared.ran, "the tokenizer step runs: " + prepared.error);
    auto dataset = registry.GetArrowDataset(prepared.dataset);
    Check(dataset != nullptr && dataset->GetArrowTable() != nullptr, "the prepared table should be registered");
    Check(dataset->GetArrowTable()->GetColumnByName("y") != nullptr,
          "the prepared table should expose the y label column");

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

    registry.UnregisterTabularDataset(prepared.dataset);
    std::cout << "Text Arrow runtime training launch passed\n";
    return 0;
}
