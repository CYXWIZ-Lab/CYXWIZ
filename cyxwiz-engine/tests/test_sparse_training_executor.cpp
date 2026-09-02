#include "core/sparse_feature_dataset.h"
#include "core/debug_run_paths.h"
#include "core/training_batcher_setup.h"
#include "core/training_executor.h"
#include "core/training_trace_collector.h"
#include "route_qualification_test_fixture.h"

#include <arrow/api.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<cyxwiz::SparseFeatureDataset> MakeDataset() {
    arrow::Int32Builder label_builder;
    Check(label_builder.AppendValues({0, 1, 0, 1}).ok(),
          "failed to append sparse training labels");
    auto labels = label_builder.Finish();
    Check(labels.ok(), labels.status().ToString());

    cyxwiz::SparseFeatureDataset::Contents contents;
    contents.name = "sparse_training_executor";
    contents.num_rows = 4;
    contents.num_features = 4;
    contents.row_offsets = {0, 2, 4, 6, 8};
    contents.column_indices = {0, 2, 1, 3, 0, 3, 1, 2};
    contents.values = {1.0f, 0.5f, 1.0f, 0.5f,
                       0.5f, 1.0f, 0.5f, 1.0f};
    contents.feature_names = {"alpha", "beta", "delta", "gamma"};
    contents.labels = std::make_shared<arrow::ChunkedArray>(
        labels.ValueOrDie());
    contents.label_name = "label";
    auto dataset = cyxwiz::SparseFeatureDataset::Create(std::move(contents));
    Check(dataset.ok(), dataset.status().ToString());
    return dataset.ValueOrDie();
}

cyxwiz::TrainingConfiguration MakeConfig(
    const std::filesystem::path& checkpoint_dir) {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "sparse_training_executor";
    config.input_size = 4;
    config.input_shape = {4};
    config.output_size = 2;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.05f;
    config.batch_size = 2;
    config.epochs = 1;
    config.shuffle = false;
    config.has_data_split = true;
    config.train_ratio = 0.5f;
    config.val_ratio = 0.25f;
    config.test_ratio = 0.25f;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.log_interval = 0;
    config.forbid_native_cpu_fallback = true;
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path work_dir =
        fs::temp_directory_path() / "cyxwiz_sparse_training_executor";
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(
        work_dir / "debug_runs");
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    auto config = MakeConfig(work_dir / "checkpoints");
    auto batchers = cyxwiz::BuildSparseTrainingBatchers(
        config, MakeDataset(), config.batch_size);
    auto resolved = cyxwiz::TakeResolvedExternalBatchers(
        std::move(batchers));
    Check(resolved.train != nullptr && resolved.dev != nullptr &&
              resolved.test != nullptr,
          "sparse Train/Dev/Test lifecycle must resolve every batcher");

    cyxwiz::TrainingExecutor executor(config, std::move(resolved));
    cyxwiz::TrainingMetrics final_metrics;
    bool completed = false;
    int batch_callbacks = 0;
    executor.Train(
        1,
        2,
        [&](int, int, int, float, float) { ++batch_callbacks; },
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
            completed = true;
        });

    Check(completed && final_metrics.terminal_status == "completed",
          "sparse training lifecycle must complete");
    Check(batch_callbacks == 1 && final_metrics.optimizer_step_count == 1,
          "sparse training lifecycle must execute and update one Train batch");
    Check(final_metrics.has_validation_metrics &&
              final_metrics.has_test_metrics,
          "sparse training lifecycle must evaluate Dev and Test roles");

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    bool saw_sparse_forward = false;
    bool saw_sparse_backward = false;
    for (const auto& event : trace.recent_events) {
        saw_sparse_forward = saw_sparse_forward ||
                             event.stage == "SparseFirstProjection.Forward";
        saw_sparse_backward = saw_sparse_backward ||
                              event.stage == "SparseFirstProjection.Backward";
    }
    if (!saw_sparse_forward || !saw_sparse_backward) {
        std::cerr << "Sparse trace stages:";
        for (const auto& event : trace.recent_events) {
            std::cerr << ' ' << event.stage;
        }
        std::cerr << '\n';
    }
    Check(saw_sparse_forward && saw_sparse_backward,
          "training trace must prove sparse first-layer forward/backward use");
    Check(trace.native_cpu_fallback_count == 0,
          "strict sparse training must not use native CPU fallback");

    std::cout << "Sparse TrainingExecutor lifecycle tests passed\n";
    return 0;
}
