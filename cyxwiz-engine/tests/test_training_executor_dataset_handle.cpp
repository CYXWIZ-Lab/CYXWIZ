#include "../src/core/data_registry.h"
#include "../src/core/dataset_base.h"
#include "../src/core/debug_run_paths.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/training_executor.h"
#include "../src/core/training_trace_collector.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/device.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

class LifecycleDataset final : public cyxwiz::Dataset {
public:
    LifecycleDataset() {
        cyxwiz::SplitConfig split;
        split.train_ratio = 1.0f;
        split.val_ratio = 0.0f;
        split.test_ratio = 0.0f;
        split.shuffle = false;
        SetSplit(split);
    }

    size_t Size() const override {
        return samples_.size();
    }

    std::pair<std::vector<float>, int> GetItem(
        size_t index) const override {
        return {samples_.at(index), labels_.at(index)};
    }

    cyxwiz::DatasetInfo GetInfo() const override {
        cyxwiz::DatasetInfo info;
        info.name = "legacy_lifecycle_dataset";
        info.shape = {2};
        info.num_samples = samples_.size();
        info.num_classes = 2;
        info.train_count = samples_.size();
        info.is_loaded = true;
        return info;
    }

private:
    const std::vector<std::vector<float>> samples_ = {
        {0.0f, 0.0f},
        {0.1f, 0.2f},
        {0.2f, 0.1f},
        {0.8f, 0.9f},
        {0.9f, 0.8f},
        {1.0f, 1.0f},
    };
    const std::vector<int> labels_ = {0, 0, 0, 1, 1, 1};
};

cyxwiz::TrainingConfiguration MakeConfig(
    const std::filesystem::path& checkpoint_dir) {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "legacy_lifecycle_dataset";
    config.input_size = 2;
    config.input_shape = {2};
    config.output_size = 2;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.has_data_split = true;
    config.shuffle = false;
    config.num_workers = 0;
    config.batch_size = 2;
    config.epochs = 1;
    config.log_interval = 0;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.forbid_native_cpu_fallback = true;
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
}

enum class PausedAction {
    Resume,
    Cancel,
};

void CommitCpuPreferencesSelection() {
    const auto result = cyxwiz::CommitExecutionDeviceSelection(
        {cyxwiz::DeviceType::CPU, 0, {}});
    Check(result.committed &&
              result.stage ==
                  cyxwiz::DeviceSelectionTransactionStage::Complete &&
              result.status ==
                  cyxwiz::DeviceSelectionTransactionStatus::Committed &&
              result.activation.success &&
              result.activation.execution_validated &&
              result.activation.effective_type == cyxwiz::DeviceType::CPU &&
              result.activation.effective_device_id == 0,
          "Preferences must verify and commit exact ArrayFire CPU device 0");
}

void CheckPausedLifecycle(
    const cyxwiz::DatasetHandle& dataset,
    const std::filesystem::path& work_dir,
    PausedAction action) {
    const bool cancel = action == PausedAction::Cancel;
    const std::string action_name = cancel ? "cancel" : "resume";
    CommitCpuPreferencesSelection();
    cyxwiz::TrainingExecutor executor(
        MakeConfig(work_dir / action_name), dataset);
    std::atomic<int> batch_callback_count{0};
    std::atomic<bool> pause_requested{false};
    cyxwiz::TrainingMetrics final_metrics;
    int completion_callback_count = 0;
    std::exception_ptr training_error;

    std::thread training_thread([&]() {
        try {
            executor.Train(
                1,
                2,
                [&](int, int batch, int, float, float) {
                    ++batch_callback_count;
                    if (batch == 1) {
                        executor.Pause();
                        pause_requested.store(true);
                    }
                },
                nullptr,
                [&](const cyxwiz::TrainingMetrics& metrics) {
                    ++completion_callback_count;
                    final_metrics = metrics;
                });
        } catch (...) {
            training_error = std::current_exception();
        }
    });

    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(10);
    while (!pause_requested.load() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!pause_requested.load() || !executor.IsPaused()) {
        executor.Stop();
        training_thread.join();
        Check(false, "legacy pause/" + action_name +
                         " should reach the paused state");
    }

    const auto paused_metrics = executor.GetMetrics();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const auto stable_metrics = executor.GetMetrics();
    Check(paused_metrics.current_batch == 1 &&
              paused_metrics.optimizer_step_count == 1 &&
              stable_metrics.current_batch == 1 &&
              stable_metrics.optimizer_step_count == 1 &&
              batch_callback_count.load() == 1,
          "legacy pause/" + action_name +
              " must not advance or duplicate work while paused");

    if (cancel) {
        executor.Stop();
    } else {
        executor.Resume();
    }
    training_thread.join();

    Check(training_error == nullptr,
          "legacy pause/" + action_name + " should not throw");
    Check(completion_callback_count == 1,
          "legacy pause/" + action_name +
              " should invoke completion exactly once");
    Check(!final_metrics.is_training && !final_metrics.is_paused &&
              final_metrics.is_complete,
          "legacy pause/" + action_name +
              " should close lifecycle state");

    if (cancel) {
        Check(batch_callback_count.load() == 1 &&
                  final_metrics.current_batch == 1 &&
                  final_metrics.optimizer_step_count == 1,
              "legacy cancellation must not consume another batch");
        Check(final_metrics.current_epoch == 1 &&
                  final_metrics.last_executed_epoch == 0 &&
                  final_metrics.loss_history.empty() &&
                  final_metrics.terminal_status == "cancelled" &&
                  final_metrics.terminal_reason == "user_cancelled",
              "legacy cancellation must preserve partial-epoch truth");
    } else {
        Check(batch_callback_count.load() == 3 &&
                  final_metrics.current_batch == 3 &&
                  final_metrics.optimizer_step_count == 3,
              "legacy resume must consume all three batches exactly once");
        Check(final_metrics.current_epoch == 1 &&
                  final_metrics.last_executed_epoch == 1 &&
                  final_metrics.loss_history.size() == 1 &&
                  final_metrics.terminal_status == "completed" &&
                  final_metrics.terminal_reason == "completed_all_epochs",
              "legacy resume must preserve completed terminal truth");
    }

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.execution_platform == "arrayfire" &&
              trace.requested_backend == "arrayfire_cpu" &&
              trace.requested_device_id == 0 &&
              trace.effective_backend == "arrayfire_cpu" &&
              trace.effective_device_id == 0,
          "legacy lifecycle must retain the exact requested/effective CPU "
          "route identity");
    Check(trace.requested_qualification_evidence_available &&
              trace.requested_route_qualified &&
              trace.effective_qualification_evidence_available &&
              trace.effective_route_qualified &&
              trace.activation_succeeded && trace.execution_validated &&
              !trace.selection_fallback_applied &&
              trace.preflight_stage == "complete",
          "legacy lifecycle must retain certified exact-route preflight "
          "truth without selection fallback");
    Check(trace.native_cpu_fallback_count == 0 &&
              trace.fallback_policy == "forbid_native_cpu_fallback",
          "legacy lifecycle must retain strict zero-fallback truth");
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    const fs::path work_dir = fs::temp_directory_path() /
        ("cyxwiz_legacy_dataset_lifecycle_" + std::to_string(nonce));
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(
        work_dir / "debug_runs");

    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    Check(activation.success && activation.execution_validated,
          "legacy lifecycle harness requires exact ArrayFire CPU activation");
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    auto underlying = std::make_shared<LifecycleDataset>();
    cyxwiz::DatasetHandle dataset(
        underlying, "legacy_lifecycle_dataset");
    Check(dataset.IsValid() && dataset.Size(cyxwiz::DatasetSplit::Train) == 6,
          "legacy lifecycle fixture must expose six training samples");

    CheckPausedLifecycle(dataset, work_dir, PausedAction::Resume);
    CheckPausedLifecycle(dataset, work_dir, PausedAction::Cancel);

    fs::remove_all(work_dir);
    std::cout << "Legacy DatasetHandle lifecycle contracts passed\n";
    return 0;
}
