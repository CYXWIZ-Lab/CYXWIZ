#include "../src/core/arrow_dataset.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/core/training_executor.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishFloatArray(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeTrainingTable() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::float32()),
    });

    return arrow::Table::Make(
        schema,
        {
            FinishFloatArray({0.0f, 0.1f, 0.9f, 1.0f, 0.2f, 0.8f}),
            FinishFloatArray({0.0f, 0.2f, 0.8f, 1.0f, 0.1f, 0.9f}),
            FinishFloatArray({0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f}),
        },
        6);
}

void WriteParquetWithRowGroupSize(const cyxwiz::ArrowDataset& dataset,
                                  const std::string& path,
                                  int64_t row_group_size) {
    auto table = dataset.GetArrowTable();
    Check(table != nullptr, "source table should exist for row-group parquet write");
    auto output = arrow::io::FileOutputStream::Open(path);
    Check(output.ok(), output.status().ToString());
    auto status = parquet::arrow::WriteTable(*table,
                                             arrow::default_memory_pool(),
                                             *output,
                                             row_group_size);
    Check(status.ok(), status.ToString());
}

cyxwiz::TrainingConfiguration MakeConfig(const std::filesystem::path& checkpoint_dir) {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "training_executor_parity";
    config.input_size = 2;
    config.input_shape = {2};
    config.output_size = 2;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.train_ratio = 0.67f;
    config.shuffle = false;
    config.num_workers = 0;
    config.batch_size = 2;
    config.epochs = 1;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
}

void RunExecutor(cyxwiz::TrainingExecutor& executor,
                 const std::string& label) {
    bool saw_epoch = false;
    bool completed = false;
    cyxwiz::TrainingMetrics final_metrics;

    executor.Train(
        1,
        2,
        nullptr,
        [&](int epoch,
            float train_loss,
            float,
            float val_loss,
            float,
            float) {
            Check(epoch == 1, label + " epoch callback should report epoch 1");
            Check(std::isfinite(train_loss), label + " train loss should be finite");
            Check(std::isfinite(val_loss), label + " val loss should be finite");
            saw_epoch = true;
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
            completed = true;
        });

    Check(saw_epoch, label + " should run an epoch callback");
    Check(completed, label + " should run completion callback");
    Check(final_metrics.is_complete, label + " should mark training complete");
    Check(!final_metrics.is_training, label + " should clear training state");
    Check(final_metrics.current_epoch == 1, label + " should finish epoch 1");
    Check(final_metrics.total_epochs == 1, label + " should keep total epochs");
    Check(final_metrics.total_batches == 2, label + " should train two batches");
    Check(final_metrics.loss_history.size() == 1,
          label + " should store one train loss history entry");
    Check(final_metrics.val_loss_history.size() == 1,
          label + " should store one validation loss history entry");
    Check(std::isfinite(final_metrics.train_loss),
          label + " final train loss should be finite");
    Check(std::isfinite(final_metrics.val_loss),
          label + " final validation loss should be finite");
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path work_dir =
        fs::temp_directory_path() / "cyxwiz_training_executor_arrow_parquet";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);

    const auto dataset = std::make_shared<cyxwiz::ArrowDataset>(
        MakeTrainingTable(), "training_executor_arrow");
    const auto config = MakeConfig(work_dir / "checkpoints");

    {
        auto sequence_config = config;
        sequence_config.sequence_batch.enabled = true;
        sequence_config.sequence_batch.token_column = "tokens";
        sequence_config.sequence_batch.tag_column = "ner_tags";
        cyxwiz::TrainingExecutor sequence_executor(
            sequence_config, dataset, "label");
        bool saw_batch = false;
        bool saw_epoch = false;
        bool completed = false;
        sequence_executor.Train(
            1,
            sequence_config.batch_size,
            [&](int, int, int, float, float) { saw_batch = true; },
            [&](int, float, float, float, float, float) { saw_epoch = true; },
            [&](const cyxwiz::TrainingMetrics&) { completed = true; });
        Check(!saw_batch,
              "sequence batch guard should reject before any training batch");
        Check(!saw_epoch,
              "sequence batch guard should reject before epoch callback");
        Check(!completed,
              "sequence batch guard should reject before completion callback");
        Check(!sequence_executor.IsTraining(),
              "sequence batch guard should clear executor training state");
    }

    {
        cyxwiz::TrainingExecutor arrow_executor(config, dataset, "label");
        RunExecutor(arrow_executor, "Arrow TrainingExecutor");
    }

    const fs::path parquet_path = work_dir / "training_executor.parquet";
    WriteParquetWithRowGroupSize(*dataset, parquet_path.string(), 2);
    auto parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        parquet_path.string(), "training_executor_parquet");
    Check(parquet_dataset != nullptr, "Parquet fixture should open");

    {
        cyxwiz::TrainingExecutor parquet_executor(config, parquet_dataset, "label");
        RunExecutor(parquet_executor, "Parquet TrainingExecutor");
    }

    parquet_dataset.reset();
    fs::remove_all(work_dir);

    std::cout << "TrainingExecutor Arrow/Parquet parity passed\n";
    return 0;
}
