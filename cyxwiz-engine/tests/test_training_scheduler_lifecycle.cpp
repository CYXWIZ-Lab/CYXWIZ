#include "../src/core/arrow_dataset.h"
#include "../src/core/checkpoint_payload_io.h"
#include "../src/core/debug_run_paths.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/training_executor.h"
#include "../src/core/training_trace_collector.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/device.h>
#include <cyxwiz/optimizers/sgd.h>
#include <cyxwiz/scheduler.h>

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void CheckNear(double actual,
               double expected,
               double tolerance,
               const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message << " actual=" << actual
                  << " expected=" << expected << '\n';
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishFloatArray(
    const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) {
        const auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    const auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeDataset() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::float32()),
    });
    auto table = arrow::Table::Make(
        schema,
        {
            FinishFloatArray({0.0f, 0.1f, 0.9f, 1.0f, 0.2f, 0.8f}),
            FinishFloatArray({0.0f, 0.2f, 0.8f, 1.0f, 0.1f, 0.9f}),
            FinishFloatArray({0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f}),
        },
        6);
    return std::make_shared<cyxwiz::ArrowDataset>(
        std::move(table), "scheduler_lifecycle");
}

cyxwiz::TrainingConfiguration MakeConfig(
    const std::filesystem::path& checkpoint_dir) {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "scheduler_lifecycle";
    config.input_size = 2;
    config.input_shape = {2};
    config.output_size = 2;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.08f;
    config.train_ratio = 0.67f;
    config.shuffle = false;
    config.num_workers = 0;
    config.batch_size = 2;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.log_interval = 0;
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
}

void SelectStrictArrayFireCpu() {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = std::find_if(
        devices.begin(), devices.end(), [](const cyxwiz::DeviceInfo& device) {
            return device.type == cyxwiz::DeviceType::CPU;
        });
    Check(cpu != devices.end(),
          "scheduler lifecycle requires an ArrayFire CPU route");
    cyxwiz::test::InstallQualifiedRouteSnapshot(devices);
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(
        cyxwiz::DeviceType::CPU, cpu->device_id);
    cyxwiz::ClearNextRunExecutionPolicy();
    cyxwiz::SetNextRunExecutionPolicy(
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
}

cyxwiz::TrainingMetrics Run(
    cyxwiz::TrainingExecutor& executor,
    int epochs) {
    SelectStrictArrayFireCpu();
    cyxwiz::TrainingMetrics final_metrics;
    int completion_count = 0;
    executor.Train(
        epochs,
        2,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
            ++completion_count;
        });
    Check(completion_count == 1,
          "scheduler lifecycle run should complete exactly once");
    Check(final_metrics.terminal_status == "completed" &&
              final_metrics.terminal_reason == "completed_all_epochs",
          "scheduler lifecycle run should complete all epochs");
    return final_metrics;
}

void TestEpochSchedulerFamilyBindings() {
    struct Case {
        std::string name;
        cyxwiz::TrainingSchedulerSpec specification;
        double expected_initial_lr;
        double expected_epoch_one_lr;
    };
    const std::vector<Case> cases = {
        {"ExponentialLR",
         cyxwiz::ExponentialLRSchedulerSpec{0.5},
         0.08,
         0.04},
        {"CosineAnnealingLR",
         cyxwiz::CosineAnnealingLRSchedulerSpec{2, 0.0},
         0.08,
         0.04},
        {"LinearWarmupLR",
         cyxwiz::LinearWarmupLRSchedulerSpec{2, 0.08, 0.0},
         0.0,
         0.04},
    };

    for (const auto& test_case : cases) {
        cyxwiz::SGDOptimizer optimizer(0.08);
        cyxwiz::TrainingSchedulerController controller(
            test_case.specification);
        std::string error;
        Check(controller.Attach(optimizer, std::nullopt, error),
              test_case.name + " should attach: " + error);
        Check(controller.GetCadence() ==
                  cyxwiz::TrainingSchedulerCadence::CompletedEpoch,
              test_case.name + " should use completed-epoch cadence");
        CheckNear(optimizer.GetLearningRate(),
                  test_case.expected_initial_lr,
                  1.0e-12,
                  test_case.name + " initial learning rate");

        Check(!controller.OnOptimizerStep().stepped &&
                  !controller.OnOptimizerStep().stepped,
              test_case.name +
                  " should not advance at optimizer-update boundaries");
        const auto epoch_advance =
            controller.OnEpochCompleted(std::nullopt);
        Check(epoch_advance.ok && epoch_advance.stepped,
              test_case.name + " should advance after a completed epoch");
        CheckNear(epoch_advance.learning_rate,
                  test_case.expected_epoch_one_lr,
                  1.0e-12,
                  test_case.name + " epoch-one learning rate");

        cyxwiz::TrainingSchedulerResumeState state;
        Check(controller.ExportResumeState(state, error),
              test_case.name + " should export lifecycle state: " + error);
        Check(state.completed_epochs == 1 &&
                  state.completed_optimizer_steps == 2 &&
                  state.scheduler_state.last_step == 1,
              test_case.name + " should retain exact lifecycle cursors");
    }
}

void TestExplicitResumeRequiresSchedulerState() {
    auto config = MakeConfig(
        std::filesystem::temp_directory_path() /
        "cyxwiz_training_scheduler_invalid_resume");
    cyxwiz::TrainingExecutor executor(config, MakeDataset(), "label");
    cyxwiz::TrainingSchedulerResumeState incomplete_resume;
    incomplete_resume.completed_epochs = 2;
    incomplete_resume.completed_optimizer_steps = 4;
    std::string error;
    Check(!executor.ConfigureScheduler(
              cyxwiz::StepLRSchedulerSpec{1, 0.5},
              incomplete_resume,
              error),
          "explicit scheduler resume should reject a missing backend state");
    Check(error.find("scheduler state") != std::string::npos,
          "incomplete scheduler resume should explain the missing state");
}

void TestEpochCadencePersistedResume(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "source-checkpoints");
    cyxwiz::TrainingExecutor source(config, dataset, "label");
    std::string error;
    Check(source.ConfigureScheduler(
              cyxwiz::StepLRSchedulerSpec{1, 0.5}, error),
          "StepLR configuration should be accepted: " + error);

    const auto source_metrics = Run(source, 2);
    Check(source_metrics.optimizer_step_count == 4,
          "two epochs should perform four optimizer updates");
    Check(source_metrics.scheduler_step_count == 2,
          "StepLR should advance once per completed epoch");
    Check(source_metrics.learning_rate_history.size() == 2,
          "StepLR should record one LR value per scheduler advance");
    CheckNear(source_metrics.learning_rate_history[0], 0.04, 1.0e-8,
              "StepLR epoch 1 learning rate");
    CheckNear(source_metrics.learning_rate_history[1], 0.02, 1.0e-8,
              "StepLR epoch 2 learning rate");

    const auto source_state = source.ExportSchedulerResumeState(error);
    Check(source_state.has_value(),
          "StepLR lifecycle state should export: " + error);
    Check(source_state->completed_epochs == 2 &&
              source_state->completed_optimizer_steps == 4 &&
              source_state->scheduler_state.last_step == 2,
          "StepLR lifecycle state should retain absolute cursors");

    const auto payload_root = work_dir / "scheduler-payload";
    cyxwiz::CheckpointPayloadDescriptor descriptor;
    const auto* source_scheduler = source.GetScheduler();
    Check(source_scheduler != nullptr,
          "configured executor should expose its scheduler state owner");
    Check(cyxwiz::SaveSchedulerPayloadV2(
              payload_root,
              "scheduler/state.bin",
              *source_scheduler,
              descriptor,
              error),
          "StepLR scheduler payload should save: " + error);

    cyxwiz::SGDOptimizer payload_optimizer(
        static_cast<double>(config.learning_rate));
    cyxwiz::StepLR payload_scheduler(&payload_optimizer, 1, 0.5);
    Check(cyxwiz::LoadSchedulerPayloadV2(
              payload_root, descriptor, payload_scheduler, error),
          "StepLR scheduler payload should load: " + error);
    cyxwiz::SchedulerState loaded_scheduler_state;
    Check(payload_scheduler.ExportState(loaded_scheduler_state, error),
          "loaded StepLR state should export: " + error);

    cyxwiz::TrainingSchedulerResumeState resume_state;
    resume_state.scheduler_state = std::move(loaded_scheduler_state);
    resume_state.completed_epochs = source_state->completed_epochs;
    resume_state.completed_optimizer_steps =
        source_state->completed_optimizer_steps;

    auto resumed_config = MakeConfig(work_dir / "resumed-checkpoints");
    cyxwiz::TrainingExecutor resumed(resumed_config, dataset, "label");
    Check(resumed.ConfigureScheduler(
              cyxwiz::StepLRSchedulerSpec{1, 0.5},
              resume_state,
              error),
          "persisted StepLR resume should be accepted: " + error);

    const auto resumed_metrics = Run(resumed, 1);
    Check(resumed_metrics.optimizer_step_count == 2 &&
              resumed_metrics.scheduler_step_count == 1,
          "resumed run should report only newly executed update/scheduler work");
    Check(resumed_metrics.learning_rate_history.size() == 1,
          "resumed StepLR run should record one new LR value");
    CheckNear(resumed_metrics.learning_rate_history.front(), 0.01, 1.0e-8,
              "resumed StepLR should continue at absolute epoch 3");

    const auto resumed_state = resumed.ExportSchedulerResumeState(error);
    Check(resumed_state.has_value(),
          "resumed StepLR state should export: " + error);
    Check(resumed_state->completed_epochs == 3 &&
              resumed_state->completed_optimizer_steps == 6 &&
              resumed_state->scheduler_state.last_step == 3,
          "resumed StepLR should advance absolute lifecycle cursors");
}

void TestPlateauUsesValidationCadenceOnly(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "plateau-checkpoints");
    config.validation_freq = 2;
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    std::string error;
    Check(executor.ConfigureScheduler(
              cyxwiz::ReduceLROnPlateauSchedulerSpec{
                  "min", 0.5, 10, 1.0e-4, 0.0},
              error),
          "ReduceLROnPlateau configuration should be accepted: " + error);

    const auto metrics = Run(executor, 3);
    Check(metrics.optimizer_step_count == 6,
          "plateau run should perform six optimizer updates");
    Check(metrics.scheduler_step_count == 2 &&
              metrics.learning_rate_history.size() == 2,
          "plateau scheduler should advance only on validation epochs 2 and 3");

    const auto state = executor.ExportSchedulerResumeState(error);
    Check(state.has_value(),
          "plateau lifecycle state should export: " + error);
    Check(state->completed_epochs == 3 &&
              state->completed_optimizer_steps == 6 &&
              state->scheduler_state.last_step == 3,
          "plateau state should retain the last validation epoch and run cursors");
}

void TestPlateauSkipsEmptyValidationPartition(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "empty-validation-checkpoints");
    config.train_ratio = 1.0f;
    config.validation_freq = 1;
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    std::string error;
    Check(executor.ConfigureScheduler(
              cyxwiz::ReduceLROnPlateauSchedulerSpec{
                  "min", 0.5, 10, 1.0e-4, 0.0},
              error),
          "empty-validation plateau configuration should be accepted: " +
              error);

    const auto metrics = Run(executor, 2);
    Check(metrics.val_sample_count == 0,
          "empty-validation fixture should expose zero validation samples");
    Check(metrics.scheduler_step_count == 0 &&
              metrics.learning_rate_history.empty(),
          "plateau scheduler must not advance on a fabricated empty metric");

    const auto state = executor.ExportSchedulerResumeState(error);
    Check(state.has_value(),
          "empty-validation plateau state should export: " + error);
    Check(state->completed_epochs == 2 &&
              state->completed_optimizer_steps == 6 &&
              state->scheduler_state.last_step == 0,
          "empty-validation plateau state should retain run cursors without "
          "a scheduler step");
}

void TestOneCycleUsesRealOptimizerUpdates(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "one-cycle-checkpoints");
    config.grad_accum_steps = 3;
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    std::string error;
    Check(executor.ConfigureScheduler(
              cyxwiz::OneCycleLRSchedulerSpec{
                  0.1, 3, 0.3, 25.0, 10000.0},
              error),
          "OneCycleLR configuration should be accepted: " + error);

    const auto metrics = Run(executor, 3);
    Check(metrics.optimizer_step_count == 3,
          "forced partial accumulation should produce one update per epoch");
    Check(metrics.scheduler_step_count == metrics.optimizer_step_count &&
              metrics.learning_rate_history.size() == 3,
          "OneCycleLR should advance exactly after each real optimizer update");

    const auto state = executor.ExportSchedulerResumeState(error);
    Check(state.has_value(),
          "OneCycleLR lifecycle state should export: " + error);
    Check(state->completed_epochs == 3 &&
              state->completed_optimizer_steps == 3 &&
              state->scheduler_state.last_step == 3,
          "OneCycleLR state should match completed epoch/update truth");

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    int scheduler_events = 0;
    for (const auto& event : trace.recent_events) {
        if (event.stage == "TrainingScheduler.Advance") {
            ++scheduler_events;
        }
    }
    Check(scheduler_events == metrics.scheduler_step_count,
          "trace scheduler events should reconcile with TrainingMetrics");
    Check(trace.native_cpu_fallback_count == 0,
          "strict scheduler lifecycle run should use no native CPU fallback");
    Check(trace.requested_backend == "arrayfire_cpu" &&
              trace.effective_backend == "arrayfire_cpu",
          "scheduler lifecycle trace should preserve exact CPU route identity");
}

void TestBestCheckpointRestoresSchedulerCursor(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "best-checkpoints");
    config.save_best_checkpoint = true;
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    std::string error;
    Check(executor.ConfigureScheduler(
              cyxwiz::StepLRSchedulerSpec{1, 0.0}, error),
          "best-checkpoint StepLR configuration should be accepted: " +
              error);

    const auto metrics = Run(executor, 3);
    Check(metrics.last_executed_epoch == 3 &&
              metrics.restored_checkpoint_epoch == 1,
          "run history should retain epoch 3 while the active model restores "
          "the frozen epoch-1 checkpoint");
    Check(metrics.scheduler_step_count == 3 &&
              metrics.learning_rate_history.size() == 3,
          "scheduler metrics should retain all executed scheduler work");

    const auto state = executor.ExportSchedulerResumeState(error);
    Check(state.has_value(),
          "restored best scheduler state should export: " + error);
    Check(state->completed_epochs == 1 &&
              state->completed_optimizer_steps == 2 &&
              state->scheduler_state.last_step == 1,
          "active scheduler cursor should align with the restored best model");
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path work_dir =
        fs::temp_directory_path() / "cyxwiz_training_scheduler_lifecycle";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(
        work_dir / "debug_runs");

    const auto dataset = MakeDataset();
    TestEpochSchedulerFamilyBindings();
    TestExplicitResumeRequiresSchedulerState();
    TestEpochCadencePersistedResume(dataset, work_dir);
    TestPlateauUsesValidationCadenceOnly(dataset, work_dir);
    TestPlateauSkipsEmptyValidationPartition(dataset, work_dir);
    TestOneCycleUsesRealOptimizerUpdates(dataset, work_dir);
    TestBestCheckpointRestoresSchedulerCursor(dataset, work_dir);

    fs::remove_all(work_dir);
    std::cout << "Training scheduler lifecycle passed\n";
    return 0;
}
