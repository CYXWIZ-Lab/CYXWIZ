#include "core/arrow_dataset.h"
#include "core/debug_run_paths.h"
#include "core/execution_device_preferences.h"
#include "core/parquet_backed_dataset.h"
#include "core/test_executor.h"
#include "core/training_executor.h"
#include "core/training_trace_collector.h"
#include "route_qualification_test_fixture.h"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <parquet/arrow/writer.h>
#include <thread>

namespace {
using namespace cyxwiz;
void Require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}

std::shared_ptr<arrow::Table> MakeTable(size_t features) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  for (size_t col = 0; col <= features; ++col) {
    arrow::FloatBuilder builder;
    for (int row = 0; row < 10; ++row)
      Require(builder
                  .Append(col == features
                              ? static_cast<float>(row % 2)
                              : .1f * static_cast<float>(row + col + 1))
                  .ok(),
              "append fixture");
    auto result = builder.Finish();
    Require(result.ok(), "finish fixture column");
    arrays.push_back(*result);
    fields.push_back(
        arrow::field(col == features ? "label" : "x" + std::to_string(col),
                     arrow::float32()));
  }
  return arrow::Table::Make(arrow::schema(fields), arrays);
}

TrainingConfiguration Configuration(gui::NodeType type, int mode, int prefetch,
                                    const std::filesystem::path &dir) {
  TrainingConfiguration config;
  config.dataset_name = "spatial_ingress";
  config.input_shape = type == gui::NodeType::Upsample
                           ? std::vector<size_t>{1, 2, 1}
                           : std::vector<size_t>{1, 1, 4};
  config.input_size = type == gui::NodeType::Upsample ? 2 : 4;
  config.output_size = 2;
  config.loss_type = gui::NodeType::CrossEntropyLoss;
  config.optimizer_type = gui::NodeType::SGD;
  config.learning_rate = .01f;
  config.model_seed = 100;
  config.shuffle = false;
  config.stratified = false;
  config.has_data_split = true;
  config.train_ratio = .7f;
  config.val_ratio = .1f;
  config.test_ratio = .2f;
  config.num_workers = 0;
  config.prefetch_factor = prefetch;
  config.log_interval = 1;
  config.batch_size = 3;
  config.epochs = 2;
  config.save_best_checkpoint = false;
  config.early_stopping_patience = 0;
  config.checkpoint_dir = dir.string();
  config.forbid_native_cpu_fallback = true;
  for (auto node_type : {type, gui::NodeType::ReLU, gui::NodeType::Flatten,
                         gui::NodeType::Dense}) {
    CompiledLayer layer;
    layer.type = node_type;
    layer.node_id = static_cast<int>(100 + config.layers.size());
    layer.units = 2;
    config.layers.push_back(layer);
  }
  config.layers.front().upsample_mode = mode;
  return config;
}

std::vector<float> Run(const TrainingConfiguration &config,
                       const std::shared_ptr<ArrowDataset> &arrow,
                       const std::shared_ptr<ParquetBackedDataset> &parquet,
                       DeviceType backend, int device_id,
                       const std::string &backend_name) {
  std::vector<float> weights;
  std::exception_ptr failure;
  SetPendingExecutionDeviceSelection(backend, device_id);
  std::thread worker([&] {
    try {
      const ScopedDebugRunRootOverrideForTesting root(
          std::filesystem::path(config.checkpoint_dir) / "trace");
      auto executor =
          arrow ? std::make_unique<TrainingExecutor>(config, arrow, "label")
                : std::make_unique<TrainingExecutor>(config, parquet, "label");
      int batches = 0, epochs = 0, completed = 0;
      TrainingMetrics final;
      executor->Train(
          2, 3,
          [&](int, int, int total, float loss, float) {
            Require(total == 3 && std::isfinite(loss),
                    "three finite-loss batches per epoch");
            ++batches;
          },
          [&](int, float, float, float validation_loss, float, float) {
            Require(std::isfinite(validation_loss), "finite validation loss");
            ++epochs;
          },
          [&](const TrainingMetrics &metrics) {
            final = metrics;
            ++completed;
          });
      Require(final.terminal_status == "completed",
              "training terminal: " + final.status_message);
      Require(batches == 6 && epochs == 2 && completed == 1,
              "batch/epoch/terminal callback cadence");
      Require(final.optimizer_step_count == 6,
              "six optimizer updates including partial batches");
      Require(final.has_validation_metrics && final.has_test_metrics,
              "validation and held-out test executed");
      const auto trace = TrainingTraceCollector::Instance().Snapshot();
      Require(trace.native_cpu_fallback_count == 0,
              "strict spatial training native fallback");
      Require(trace.effective_backend == backend_name &&
                  trace.effective_device_id == device_id,
              "exact effective backend/device: " + trace.effective_backend);
      Require(trace.requested_backend == trace.effective_backend,
              "no startup backend substitution");
      uint64_t sync_events = 0, sync_bytes = 0;
      for (const auto &group : trace.arrayfire_host_sync_groups) {
        Require(group.category == "loss_scalar_readback" ||
                    group.category == "metric_scalar_readback",
                "only named scalar reporting may read training tensors: " +
                    group.category);
        sync_events += group.event_count;
        sync_bytes += group.bytes;
      }
      Require(sync_events == trace.arrayfire_host_sync_count &&
                  sync_bytes == trace.arrayfire_host_sync_bytes,
              "host-sync groups reconcile with run totals");
      auto model = std::shared_ptr<SequentialModel>(executor->ReleaseModel());
      Require(model && model->Size() == 4, "trained spatial head retained");
      for (const auto &[name, tensor] : model->GetParameters()) {
        const auto *data =
            tensor.ReadData<float>(); // Explicit post-training test inspection.
        weights.insert(weights.end(), data, data + tensor.NumElements());
      }
      auto tester = arrow ? std::make_unique<TestExecutor>(
                                config, arrow, "label",
                                TestDatasetScope::EntireProvidedDataset)
                          : std::make_unique<TestExecutor>(
                                config, parquet, "label",
                                TestDatasetScope::EntireProvidedDataset);
      tester->SetModel(model);
      TestingMetrics tested;
      int test_batches = 0;
      tester->Test(
          3,
          [&](int, int total, float) {
            Require(total == 4, "four full-source test batches");
            ++test_batches;
          },
          [&](const TestingMetrics &metrics) { tested = metrics; });
      Require(tested.is_complete && tested.total_samples == 10 &&
                  test_batches == 4,
              "TestExecutor consumes every sample including partial batch: " +
                  tested.status_message);
      Require(std::isfinite(tested.test_loss),
              "finite held-out evaluation loss");
      std::cout << "Spatial ingress passed: " << backend_name << " "
                << (arrow ? "Arrow" : "Parquet")
                << " prefetch=" << config.prefetch_factor
                << " train_batches=6 test_batches=4 host_sync_events="
                << trace.arrayfire_host_sync_count
                << " host_sync_bytes=" << trace.arrayfire_host_sync_bytes
                << '\n';
    } catch (...) {
      failure = std::current_exception();
    }
  });
  worker.join();
  if (failure)
    std::rethrow_exception(failure);
  return weights;
}
} // namespace

int RunSpatialIngressTests(const std::string &backend) {
  try {
    const auto type = backend == "cpu"      ? cyxwiz::DeviceType::CPU
                      : backend == "cuda"   ? cyxwiz::DeviceType::CUDA
                      : backend == "opencl" ? cyxwiz::DeviceType::OPENCL
                                            : cyxwiz::DeviceType::ONEAPI;
    Require(backend == "cpu" || backend == "cuda" || backend == "opencl" ||
                backend == "oneapi",
            "unknown backend");
    if (backend == "oneapi" && !cyxwiz::IsUncertifiedOneAPITrainingEnabled()) {
      std::cout << "SKIP: oneAPI uncertified-training policy\n";
      return 77;
    }
    const auto inventory = cyxwiz::Device::GetAvailableDevices();
    int device_id = -1;
    for (const auto &device : inventory)
      if (device.type == type && device.device_selectable) {
        device_id = device.device_id;
        break;
      }
    if (device_id < 0) {
      Require(backend != "cpu", "ArrayFire CPU is required for this matrix");
      std::cout << "SKIP: requested backend unavailable\n";
      return 77;
    }
    cyxwiz::test::InstallQualifiedRouteSnapshot(inventory);
    const auto root =
        std::filesystem::temp_directory_path() /
        ("cyxwiz_spatial_ingress_" + backend + "_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    for (auto node : {gui::NodeType::Upsample, gui::NodeType::PixelShuffle}) {
      for (int mode = 0; mode < (node == gui::NodeType::Upsample ? 2 : 1);
           ++mode) {
        const auto name =
            std::to_string(static_cast<int>(node)) + "_" + std::to_string(mode);
        auto config = Configuration(node, mode, 0, root / name);
        const auto table = MakeTable(config.input_size);
        const auto arrow = std::make_shared<cyxwiz::ArrowDataset>(table, name);
        const auto path = root / (name + ".parquet");
        auto file = arrow::io::FileOutputStream::Open(path.string());
        Require(file.ok(), "open parquet fixture");
        // Parquet partitions whole row groups. Single-row groups align its
        // split with Arrow and make each full batch cross group boundaries.
        Require(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                           *file, 1)
                    .ok(),
                "write parquet fixture");
        Require((*file)->Close().ok(), "close parquet fixture");
        const auto parquet =
            cyxwiz::ParquetBackedDataset::Open(path.string(), name);
        Require(parquet != nullptr, "open parquet dataset");
        std::vector<float> baseline;
        for (bool disk : {false, true})
          for (int prefetch : {0, 2}) {
            config.prefetch_factor = prefetch;
            config.checkpoint_dir =
                (root / (name + (disk ? "_parquet" : "_arrow") +
                         std::to_string(prefetch)))
                    .string();
            const auto values =
                Run(config, disk ? nullptr : arrow, disk ? parquet : nullptr,
                    type, device_id, "arrayfire_" + backend);
            if (baseline.empty())
              baseline = values;
            Require(values.size() == baseline.size(), "parameter count parity");
            for (size_t i = 0; i < values.size(); ++i)
              Require(std::isfinite(values[i]) &&
                          std::abs(values[i] - baseline[i]) < 1e-5f,
                      "Arrow/Parquet/prefetch final weight parity");
          }
      }
    }
    std::cout << "Spatial ingress matrix passed: 12 training/evaluation runs; "
                 "evidence "
              << root.string() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
