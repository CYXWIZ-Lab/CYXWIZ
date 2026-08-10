#include "../../src/core/arrow_dataset.h"
#include "../../src/core/debug_run_paths.h"
#include "../../src/core/execution_device_preferences.h"
#include "../../src/core/training_executor.h"
#include "../../src/core/training_trace_collector.h"

#include <arrow/api.h>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kFeatureCount = 170;

struct BenchmarkOptions {
    int batch_size = 254;
    int warmup_batches = 2;
    int measured_batches = 8;
    std::string device = "current";
    std::optional<int> device_id;
    std::filesystem::path trace_root =
        cyxwiz::GetDebugRunRoot() / "benchmarks" / "aps_dense";
};

int ParsePositiveInt(std::string_view text, const char* option_name) {
    int value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() ||
        value <= 0) {
        throw std::invalid_argument(
            std::string(option_name) + " requires a positive integer");
    }
    return value;
}

int ParseNonNegativeInt(std::string_view text, const char* option_name) {
    int value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() ||
        value < 0) {
        throw std::invalid_argument(
            std::string(option_name) +
            " requires a non-negative integer");
    }
    return value;
}

void PrintUsage() {
    std::cout
        << "Usage: test_dense_training_benchmark [options]\n"
        << "  --batch-size N       Batch size (default 254)\n"
        << "  --warmup-batches N   Unmeasured warmup batches (default 2)\n"
        << "  --measured-batches N Timed batches (default 8)\n"
        << "  --device NAME        current|cpu|cuda|opencl|oneapi\n"
        << "  --device-id N        ArrayFire device id for --device\n"
        << "  --trace-root PATH    Benchmark trace output directory\n";
}

BenchmarkOptions ParseOptions(int argc, char** argv) {
    BenchmarkOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (option == "--help") {
            PrintUsage();
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument(
                std::string(option) + " requires a value");
        }
        const std::string_view value = argv[++index];
        if (option == "--batch-size") {
            options.batch_size = ParsePositiveInt(value, "--batch-size");
        } else if (option == "--warmup-batches") {
            options.warmup_batches =
                ParsePositiveInt(value, "--warmup-batches");
        } else if (option == "--measured-batches") {
            options.measured_batches =
                ParsePositiveInt(value, "--measured-batches");
        } else if (option == "--device") {
            options.device = value;
        } else if (option == "--device-id") {
            options.device_id =
                ParseNonNegativeInt(value, "--device-id");
        } else if (option == "--trace-root") {
            options.trace_root = std::filesystem::path(std::string(value));
        } else {
            throw std::invalid_argument(
                "Unknown benchmark option: " + std::string(option));
        }
    }
    return options;
}

std::optional<cyxwiz::DeviceType> ParseDeviceType(
    const std::string& name) {
    if (name == "current") return std::nullopt;
    if (name == "cpu") return cyxwiz::DeviceType::CPU;
    if (name == "cuda") return cyxwiz::DeviceType::CUDA;
    if (name == "opencl") return cyxwiz::DeviceType::OPENCL;
    if (name == "oneapi") return cyxwiz::DeviceType::ONEAPI;
    throw std::invalid_argument(
        "--device must be current, cpu, cuda, opencl, or oneapi");
}

int ResolveDeviceId(cyxwiz::DeviceType type,
                    const std::optional<int>& requested_id) {
    if (requested_id.has_value()) {
        return *requested_id;
    }
    for (const auto& device : cyxwiz::Device::GetAvailableDevices()) {
        if (device.type == type) {
            return device.device_id;
        }
    }
    throw std::runtime_error(
        "Requested ArrayFire backend has no discovered device");
}

std::shared_ptr<arrow::Array> FinishFloatArray(
    const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    const auto append_status = builder.AppendValues(values);
    if (!append_status.ok()) {
        throw std::runtime_error(append_status.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    const auto finish_status = builder.Finish(&array);
    if (!finish_status.ok()) {
        throw std::runtime_error(finish_status.ToString());
    }
    return array;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeDataset(size_t row_count) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    fields.reserve(kFeatureCount + 1);
    columns.reserve(kFeatureCount + 1);

    std::vector<float> values(row_count);
    for (int feature = 0; feature < kFeatureCount; ++feature) {
        for (size_t row = 0; row < row_count; ++row) {
            const size_t pattern =
                (row * 17 + static_cast<size_t>(feature) * 13) % 257;
            values[row] = static_cast<float>(pattern) / 256.0f;
        }
        fields.push_back(
            arrow::field("x" + std::to_string(feature), arrow::float32()));
        columns.push_back(FinishFloatArray(values));
    }

    for (size_t row = 0; row < row_count; ++row) {
        values[row] = static_cast<float>(row % 2);
    }
    fields.push_back(arrow::field("label", arrow::float32()));
    columns.push_back(FinishFloatArray(values));

    auto table = arrow::Table::Make(
        arrow::schema(std::move(fields)),
        std::move(columns),
        static_cast<int64_t>(row_count));
    return std::make_shared<cyxwiz::ArrowDataset>(
        std::move(table), "ticket85_aps_dense_benchmark");
}

cyxwiz::CompiledLayer MakeLayer(gui::NodeType type,
                                int node_id,
                                std::string name) {
    cyxwiz::CompiledLayer layer;
    layer.type = type;
    layer.node_id = node_id;
    layer.name = std::move(name);
    return layer;
}

cyxwiz::TrainingConfiguration MakeConfiguration(
    const BenchmarkOptions& options) {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "ticket85_aps_dense_benchmark";
    config.input_size = kFeatureCount;
    config.input_shape = {kFeatureCount};
    config.output_size = 1;
    config.loss_type = gui::NodeType::BCEWithLogits;
    config.optimizer_type = gui::NodeType::Adam;
    config.learning_rate = 0.001f;
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.has_data_split = true;
    config.shuffle = false;
    config.num_workers = 0;
    config.batch_size = options.batch_size;
    config.epochs = 1;
    config.log_interval = 0;
    config.validation_freq = 1;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.checkpoint_dir = (options.trace_root / "checkpoints").string();
    config.forbid_native_cpu_fallback = true;
    config.target.required_by_objective = true;
    config.target.origin = cyxwiz::TargetOrigin::DatasetColumn;
    config.target.value_kind = cyxwiz::TargetValueKind::Categorical;
    config.target.primary_column = "label";
    config.target.width = 1;

    auto dense_128 = MakeLayer(gui::NodeType::Dense, 1, "Dense128");
    dense_128.units = 128;
    config.layers.push_back(std::move(dense_128));
    config.layers.push_back(MakeLayer(gui::NodeType::ReLU, 2, "ReLU128"));
    auto dropout_128 = MakeLayer(gui::NodeType::Dropout, 3, "Dropout128");
    dropout_128.dropout_rate = 0.2f;
    config.layers.push_back(std::move(dropout_128));
    auto dense_64 = MakeLayer(gui::NodeType::Dense, 4, "Dense64");
    dense_64.units = 64;
    config.layers.push_back(std::move(dense_64));
    config.layers.push_back(MakeLayer(gui::NodeType::ReLU, 5, "ReLU64"));
    auto dropout_64 = MakeLayer(gui::NodeType::Dropout, 6, "Dropout64");
    dropout_64.dropout_rate = 0.2f;
    config.layers.push_back(std::move(dropout_64));
    auto dense_1 = MakeLayer(gui::NodeType::Dense, 7, "Dense1");
    dense_1.units = 1;
    config.layers.push_back(std::move(dense_1));
    return config;
}

struct Measurement {
    bool started = false;
    bool finished = false;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    cyxwiz::TrainingTraceSummary before;
    cyxwiz::TrainingTraceSummary after;
};

int RunBenchmark(const BenchmarkOptions& options) {
    const int total_batches =
        options.warmup_batches + options.measured_batches;
    const size_t row_count =
        static_cast<size_t>(total_batches) *
        static_cast<size_t>(options.batch_size);

    const cyxwiz::ScopedDebugRunRootOverrideForTesting trace_root(
        options.trace_root);
    std::filesystem::create_directories(options.trace_root);
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::ClearNextRunExecutionPolicy();
    if (const auto device_type = ParseDeviceType(options.device)) {
        cyxwiz::SetPendingExecutionDeviceSelection(
            *device_type,
            ResolveDeviceId(*device_type, options.device_id));
    }
    cyxwiz::SetNextRunExecutionPolicy(
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);

    auto dataset = MakeDataset(row_count);
    auto config = MakeConfiguration(options);
    cyxwiz::TrainingExecutor executor(config, std::move(dataset), "label");
    Measurement measurement;
    bool completed = false;
    executor.Train(
        1,
        options.batch_size,
        [&](int, int batch, int callback_total_batches, float, float) {
            if (callback_total_batches != total_batches) {
                throw std::runtime_error(
                    "Benchmark dataset produced an unexpected batch count");
            }
            if (batch == options.warmup_batches) {
                measurement.before =
                    cyxwiz::TrainingTraceCollector::Instance().Snapshot();
                measurement.start = std::chrono::steady_clock::now();
                measurement.started = true;
            }
            if (batch == total_batches) {
                measurement.end = std::chrono::steady_clock::now();
                measurement.after =
                    cyxwiz::TrainingTraceCollector::Instance().Snapshot();
                measurement.finished = true;
            }
        },
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            completed = metrics.is_complete;
        });
    cyxwiz::ClearNextRunExecutionPolicy();

    if (!completed || !measurement.started || !measurement.finished) {
        throw std::runtime_error(
            "Dense training benchmark did not complete its measured window");
    }
    const auto final_trace =
        cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    if (final_trace.native_cpu_fallback_count != 0) {
        throw std::runtime_error(
            "Dense training benchmark observed native CPU fallback");
    }
    uint64_t loss_scalar_readbacks = 0;
    uint64_t metric_scalar_readbacks = 0;
    for (const auto& group : final_trace.arrayfire_host_sync_groups) {
        if (group.category == "loss_scalar_readback") {
            loss_scalar_readbacks += group.event_count;
        } else if (group.category == "metric_scalar_readback") {
            metric_scalar_readbacks += group.event_count;
        }
    }
    if (loss_scalar_readbacks != 2 || metric_scalar_readbacks != 2) {
        throw std::runtime_error(
            "First/final reporting cadence must read exactly two loss and two metric scalars");
    }

    const double elapsed_seconds =
        std::chrono::duration<double>(measurement.end - measurement.start)
            .count();
    if (elapsed_seconds <= 0.0) {
        throw std::runtime_error(
            "Dense training benchmark measured a non-positive duration");
    }
    const uint64_t sync_count =
        measurement.after.arrayfire_host_sync_count -
        measurement.before.arrayfire_host_sync_count;
    const uint64_t sync_bytes =
        measurement.after.arrayfire_host_sync_bytes -
        measurement.before.arrayfire_host_sync_bytes;
    const double measured_batches =
        static_cast<double>(options.measured_batches);

    std::cout << "Ticket 85 APS-style dense training benchmark\n";
#ifdef NDEBUG
    std::cout << "build_type=release\n";
#else
    std::cout << "build_type=debug\n";
#endif
    std::cout << "run_id=" << final_trace.run_id << '\n';
    std::cout << "effective_backend=" << final_trace.effective_backend << '\n';
    std::cout << "effective_device_id=" << final_trace.effective_device_id
              << '\n';
    std::cout << "effective_device_name="
              << final_trace.effective_device_name << '\n';
    std::cout << "fallback_policy=" << final_trace.fallback_policy << '\n';
    std::cout << "native_cpu_fallback_count="
              << final_trace.native_cpu_fallback_count << '\n';
    std::cout << "shape=170->128->64->1\n";
    std::cout << "batch_size=" << options.batch_size << '\n';
    std::cout << "warmup_batches=" << options.warmup_batches << '\n';
    std::cout << "measured_batches=" << options.measured_batches << '\n';
    std::cout << "metric_report_interval=" << config.log_interval << '\n';
    std::cout << "metric_report_samples=2\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "elapsed_seconds=" << elapsed_seconds << '\n';
    std::cout << "batches_per_second="
              << measured_batches / elapsed_seconds << '\n';
    std::cout << "host_sync_count=" << sync_count << '\n';
    std::cout << "host_syncs_per_second="
              << static_cast<double>(sync_count) / elapsed_seconds << '\n';
    std::cout << "host_syncs_per_batch="
              << static_cast<double>(sync_count) / measured_batches << '\n';
    std::cout << "host_sync_bytes=" << sync_bytes << '\n';
    std::cout << "host_sync_bytes_per_second="
              << static_cast<double>(sync_bytes) / elapsed_seconds << '\n';
    std::cout << "host_sync_bytes_per_batch="
              << static_cast<double>(sync_bytes) / measured_batches << '\n';
    std::cout << "trace_path="
              << (options.trace_root / "current_training_trace.json").string()
              << '\n';
    std::cout << "host_sync_summary="
              << final_trace.arrayfire_host_sync_summary << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return RunBenchmark(ParseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "Dense training benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
