// Training loop benchmark (TOFIX178 0.6).
//
// cyxwiz-bench-training-data times the batcher and the model alone; this
// times an epoch of the Engine's own TrainingExecutor (validation, metrics,
// checkpoints, trace) on TF-IDF-shaped sparse rows, so the two together show
// what the loop adds around the steps. The model is the sentiment example's:
// features-128-ReLU-Dropout-64-ReLU-Dropout-classes, AdamW, cross entropy.
//
//   cyxwiz-bench-training-loop [--rows N] [--features F] [--nnz K] [--classes C]
//                              [--batch B] [--epochs E] [--checkpoint 0|1]
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device. Prints one
// JSON line per epoch.

#include "core/debug_run_paths.h"
#include "core/sparse_feature_dataset.h"
#include "core/training_batcher_setup.h"
#include "core/training_executor.h"
#include "core/training_trace_collector.h"
#include "route_qualification_test_fixture.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

std::string Arg(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (name == argv[i]) return argv[i + 1];
    return fallback;
}

std::shared_ptr<cyxwiz::SparseFeatureDataset> MakeDataset(int64_t rows, int features, int nnz, int classes) {
    cyxwiz::SparseFeatureDataset::Contents contents;
    contents.name = "bench_training_loop";
    contents.num_rows = rows;
    contents.num_features = features;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> column(0, features - 1);
    std::uniform_real_distribution<float> value(0.01f, 0.5f);
    arrow::Int32Builder labels;
    contents.row_offsets.push_back(0);
    for (int64_t r = 0; r < rows; ++r) {
        std::vector<int32_t> cols;
        for (int i = 0; i < nnz; ++i) cols.push_back(column(rng));
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        for (const int32_t c : cols) {
            contents.column_indices.push_back(c);
            contents.values.push_back(value(rng));
        }
        contents.row_offsets.push_back(static_cast<int32_t>(contents.column_indices.size()));
        (void)labels.Append(static_cast<int32_t>(r % classes));
    }
    std::shared_ptr<arrow::Array> label_array;
    (void)labels.Finish(&label_array);
    contents.labels = std::make_shared<arrow::ChunkedArray>(label_array);
    contents.label_name = "label";
    auto created = cyxwiz::SparseFeatureDataset::Create(std::move(contents));
    if (!created.ok()) {
        std::cerr << created.status().ToString() << "\n";
        std::exit(1);
    }
    return created.ValueOrDie();
}

cyxwiz::CompiledLayer Layer(gui::NodeType type, int units = 0, float dropout = 0.0f) {
    cyxwiz::CompiledLayer layer;
    layer.type = type;
    layer.units = units;
    layer.dropout_rate = dropout;
    return layer;
}

}  // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const int64_t rows = std::stoll(Arg(argc, argv, "--rows", "52681"));
    const int features = std::stoi(Arg(argc, argv, "--features", "8000"));
    const int nnz = std::stoi(Arg(argc, argv, "--nnz", "40"));
    const int classes = std::stoi(Arg(argc, argv, "--classes", "7"));
    const int batch = std::stoi(Arg(argc, argv, "--batch", "128"));
    const int epochs = std::stoi(Arg(argc, argv, "--epochs", "2"));
    const bool checkpoint = Arg(argc, argv, "--checkpoint", "1") != "0";

    const fs::path work_dir = fs::temp_directory_path() / "cyxwiz_bench_training_loop";
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(work_dir / "debug_runs");
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "bench_training_loop";
    config.input_size = static_cast<size_t>(features);
    config.input_shape = {static_cast<size_t>(features)};
    config.output_size = static_cast<size_t>(classes);
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::AdamW;
    config.learning_rate = 0.001f;
    config.batch_size = batch;
    config.epochs = epochs;
    config.shuffle = true;
    config.has_data_split = true;
    config.train_ratio = 0.8f;
    config.val_ratio = 0.1f;
    config.test_ratio = 0.1f;
    config.save_best_checkpoint = checkpoint;
    config.early_stopping_patience = 0;
    config.log_interval = 10;
    config.checkpoint_dir = (work_dir / "checkpoints").string();
    config.layers = {Layer(gui::NodeType::Dense, 128), Layer(gui::NodeType::ReLU),
                     Layer(gui::NodeType::Dropout, 0, 0.3f), Layer(gui::NodeType::Dense, 64),
                     Layer(gui::NodeType::ReLU), Layer(gui::NodeType::Dropout, 0, 0.3f),
                     Layer(gui::NodeType::Dense, classes)};

    auto batchers = cyxwiz::BuildSparseTrainingBatchers(config, MakeDataset(rows, features, nnz, classes), batch);
    auto resolved = cyxwiz::TakeResolvedExternalBatchers(std::move(batchers));
    if (!resolved.train) {
        std::cerr << "the sparse batchers could not be built\n";
        return 1;
    }
    cyxwiz::TrainingExecutor executor(config, std::move(resolved));
    int train_batches = 0;
    executor.Train(
        epochs, batch, [&](int, int, int, float, float) { ++train_batches; },
        [&](int epoch, float, float, float, float val_acc, float epoch_time) {
            std::cout << nlohmann::json{{"epoch", epoch}, {"seconds", epoch_time}, {"train_batches", train_batches},
                                        {"val_acc", val_acc}, {"rows", rows}, {"features", features},
                                        {"nnz", nnz}, {"checkpoint", checkpoint}}
                             .dump()
                      << std::endl;
            train_batches = 0;
        },
        nullptr);
    // Where the time went, and the host syncs (with
    // CYXWIZ_PROFILE_STAGE_SYNC=1 the stages are device time).
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    for (const auto& timing : trace.stage_timings)
        std::cout << nlohmann::json{{"stage", timing.stage}, {"count", timing.count}, {"total_ms", timing.total_ms},
                                    {"max_ms", timing.max_ms}}
                         .dump()
                  << std::endl;
    for (const auto& group : trace.arrayfire_host_sync_groups)
        std::cout << nlohmann::json{{"host_sync", group.category}, {"reason", group.reason},
                                    {"operation", group.operation}, {"events", group.event_count},
                                    {"bytes", group.bytes}}
                         .dump()
                  << std::endl;
    std::cout << nlohmann::json{{"host_syncs", trace.arrayfire_host_sync_count},
                                {"host_sync_bytes", trace.arrayfire_host_sync_bytes},
                                {"transfers", trace.transfer_event_count},
                                {"transfer_bytes", trace.transfer_known_bytes}}
                     .dump()
              << std::endl;
    return 0;
}
