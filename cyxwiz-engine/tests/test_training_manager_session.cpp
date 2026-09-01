#include "../src/core/arrow_dataset.h"
#include "../src/core/async_task_manager.h"
#include "../src/core/crash_run_recorder.h"
#include "../src/core/debug_run_paths.h"
#include "../src/core/training_manager.h"
#include "../src/core/training_trace_collector.h"
#include "../src/gui/panels/training_plot_panel.h"
#include "route_qualification_test_fixture.h"

#include <arrow/api.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishFloatArray(
    const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) {
        Check(builder.Append(value).ok(), "failed to append float fixture");
    }
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "failed to finish float fixture");
    return array;
}

std::shared_ptr<arrow::Array> FinishIntArray(
    const std::vector<int32_t>& values) {
    arrow::Int32Builder builder;
    for (int32_t value : values) {
        Check(builder.Append(value).ok(), "failed to append label fixture");
    }
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "failed to finish label fixture");
    return array;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeDataset() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::int32()),
    });
    auto table = arrow::Table::Make(schema, {
        FinishFloatArray({0.0f, 0.1f, 0.2f, 0.8f, 0.9f, 1.0f}),
        FinishFloatArray({0.0f, 0.2f, 0.1f, 0.9f, 0.8f, 1.0f}),
        FinishIntArray({0, 0, 0, 1, 1, 1}),
    }, 6);
    return std::make_shared<cyxwiz::ArrowDataset>(
        table, "training_manager_session");
}

cyxwiz::TrainingConfiguration MakeConfig(
    const std::filesystem::path& checkpoint_dir) {
    cyxwiz::TrainingConfiguration config;
    config.is_valid = true;
    config.dataset_name = "training_manager_session";
    config.input_size = 2;
    config.input_shape = {2};
    config.output_size = 2;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.train_ratio = 2.0f / 3.0f;
    config.val_ratio = 1.0f / 3.0f;
    config.test_ratio = 0.0f;
    config.has_data_split = true;
    config.shuffle = false;
    config.num_workers = 0;
    config.prefetch_factor = 0;
    config.batch_size = 2;
    config.epochs = 1;
    config.log_interval = 1;
    config.validation_freq = 1;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.checkpoint_dir = checkpoint_dir.string();
    config.target.primary_column = "label";
    config.dataset_roles.train.label_column = "label";

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
}

bool WaitFor(const std::function<bool()>& predicate,
             std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
    return predicate();
}

std::shared_ptr<cyxwiz::AsyncTask> WaitForTerminalTask(
    cyxwiz::TrainingManager& manager,
    cyxwiz::AsyncTaskManager& tasks,
    uint64_t task_id,
    const std::string& session_name) {
    Check(WaitFor([&] { return !manager.IsTrainingActive(); },
                  std::chrono::seconds(60)),
          session_name + " should finish within the timeout");
    manager.WaitForTrainingStop();
    Check(WaitFor([&] {
              const auto task = tasks.GetTask(task_id);
              return task &&
                  (task->GetState() == cyxwiz::TaskState::Completed ||
                   task->GetState() == cyxwiz::TaskState::Failed ||
                   task->GetState() == cyxwiz::TaskState::Cancelled);
          }, std::chrono::seconds(10)),
          session_name + " task should publish a terminal state");
    return tasks.GetTask(task_id);
}

void CheckTerminalTrace(const cyxwiz::TrainingMetrics& metrics,
                        const std::string& expected_status) {
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.available && trace.status == expected_status,
          "debugger trace should publish the expected terminal status");
    Check(!trace.effective_backend.empty() && trace.execution_validated,
          "debugger trace should publish validated effective backend truth");
    Check(trace.native_cpu_fallback_count == 0,
          "supported manager session should not use native CPU fallback");

    const std::string expected_terminal_stage =
        expected_status == "early_stopped"
        ? "EarlyStopped"
        : "TrainingTerminal";
    int terminal_events = 0;
    for (const auto& event : trace.recent_events) {
        if (event.stage == expected_terminal_stage &&
            event.status == metrics.terminal_status &&
            event.terminal_reason == metrics.terminal_reason) {
            ++terminal_events;
            Check(event.epoch == metrics.current_epoch,
                  "debugger terminal event should preserve final epoch");
        }
    }
    Check(terminal_events == 1,
          "debugger trace should contain one canonical terminal event");

    const auto crash_run = cyxwiz::CrashRunRecorder::LoadLastRun();
    Check(crash_run.has_value(),
          "manager session should persist debug-run evidence");
    Check(crash_run->status == metrics.terminal_status &&
              crash_run->terminal_reason == metrics.terminal_reason,
          "debug-run status and reason should match manager metrics");
    Check(crash_run->last_executed_epoch == metrics.last_executed_epoch,
          "debug-run executed epoch should match manager metrics");
    Check(crash_run->checkpoint_used == metrics.checkpoint_used &&
              crash_run->restored_checkpoint_epoch ==
                  metrics.restored_checkpoint_epoch &&
              crash_run->restored_checkpoint_step ==
                  metrics.restored_checkpoint_step &&
              crash_run->active_model_provenance ==
                  metrics.active_model_provenance,
          "debug-run checkpoint provenance should match manager metrics");
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const auto work_dir = fs::temp_directory_path() /
        "cyxwiz_training_manager_session";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(
        work_dir / "debug_runs");
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    auto& tasks = cyxwiz::AsyncTaskManager::Instance();
    tasks.Initialize(1);
    auto& manager = cyxwiz::TrainingManager::Instance();
    manager.ClearTrainedModel();

    std::atomic<int> start_callbacks{0};
    std::atomic<int> end_callbacks{0};
    std::atomic<int> progress_callbacks{0};
    std::atomic<int> node_started{0};
    std::atomic<int> node_finished{0};
    std::atomic<bool> callback_success{false};
    manager.SetOnTrainingStart([&](const std::string&) {
        ++start_callbacks;
    });
    manager.SetOnTrainingEnd(
        [&](bool success, const cyxwiz::TrainingMetrics&) {
            callback_success.store(success);
            ++end_callbacks;
        });
    manager.SetOnProgress([&](int, float, float) {
        ++progress_callbacks;
    });

    auto panel = std::make_shared<cyxwiz::TrainingPlotPanel>();
    auto config = MakeConfig(work_dir / "checkpoints");
    const bool started = manager.StartTrainingArrow(
        config,
        MakeDataset(),
        "label",
        config.epochs,
        config.batch_size,
        panel,
        [&](bool active) {
            if (active) {
                ++node_started;
            } else {
                ++node_finished;
            }
        });
    Check(started, "TrainingManager should accept the Arrow session");
    const uint64_t task_id = manager.GetCurrentTaskId();
    Check(task_id != 0, "TrainingManager should publish a task id");
    Check(manager.IsTrainingActive(),
          "TrainingManager should report an active session after launch");

    const auto task = WaitForTerminalTask(
        manager, tasks, task_id, "TrainingManager session");

    const auto metrics = manager.GetCurrentMetrics();
    Check(metrics.is_complete && !metrics.is_training,
          "manager metrics should close the active state");
    Check(metrics.current_epoch == 1 && metrics.last_executed_epoch == 1,
          "manager metrics should report one executed epoch");
    Check(metrics.terminal_status == "completed" &&
              metrics.terminal_reason == "completed_all_epochs",
          "manager metrics should preserve exact terminal truth");
    Check(start_callbacks.load() == 1 && end_callbacks.load() == 1 &&
              callback_success.load(),
          "manager start/end callbacks should execute exactly once");
    Check(progress_callbacks.load() == 1,
          "manager progress callback should execute once per completed epoch");
    Check(node_started.load() == 1 && node_finished.load() == 1,
          "node-editor activity should open and close exactly once");

    Check(task && task->GetState() == cyxwiz::TaskState::Completed,
          "task panel state should report completed");
    const auto task_info = task->GetInfo();
    Check(task_info.status_message.find("1/1 executed epochs") !=
              std::string::npos &&
              task_info.status_message.find("completed_all_epochs") !=
                  std::string::npos,
          "task status should expose executed epoch and terminal reason");

    const auto dashboard = panel->GetStatusSnapshot();
    Check(dashboard.has_data && !dashboard.is_training,
          "dashboard should retain completed session data");
    Check(dashboard.current_epoch == metrics.current_epoch &&
              dashboard.last_executed_epoch == metrics.last_executed_epoch &&
              dashboard.total_epochs == metrics.total_epochs,
          "dashboard epoch truth should match manager metrics");
    Check(dashboard.terminal_status == metrics.terminal_status &&
              dashboard.terminal_reason == metrics.terminal_reason,
          "dashboard terminal truth should match manager metrics");
    Check(dashboard.metric_points >= 1,
          "dashboard should receive the completed epoch metrics");

    Check(manager.HasTrainedModel(),
          "manager should preserve the completed model");
    const auto active_model = manager.GetActiveModelInfo();
    Check(active_model.origin == cyxwiz::ActiveModelOrigin::TrainedInSession &&
              active_model.effective_dataset_name == config.dataset_name &&
              active_model.effective_label_column == "label",
          "active-model provenance should identify the completed session");
    CheckTerminalTrace(metrics, "completed");

    manager.ClearTrainedModel();
    start_callbacks.store(0);
    end_callbacks.store(0);
    progress_callbacks.store(0);
    node_started.store(0);
    node_finished.store(0);
    callback_success.store(true);

    auto cancelled_config = MakeConfig(work_dir / "cancelled_checkpoints");
    cancelled_config.epochs = 1000;
    auto cancelled_panel = std::make_shared<cyxwiz::TrainingPlotPanel>();
    const bool cancel_started = manager.StartTrainingArrow(
        cancelled_config,
        MakeDataset(),
        "label",
        cancelled_config.epochs,
        cancelled_config.batch_size,
        cancelled_panel,
        [&](bool active) {
            if (active) {
                ++node_started;
            } else {
                ++node_finished;
            }
        });
    Check(cancel_started,
          "TrainingManager should accept the cancellation session");
    const uint64_t cancelled_task_id = manager.GetCurrentTaskId();
    Check(cancelled_task_id != 0,
          "cancellation session should publish a task id");
    Check(WaitFor([&] {
              const auto live_metrics = manager.GetCurrentMetrics();
              return manager.IsTrainingActive() &&
                  live_metrics.current_batch > 0;
          }, std::chrono::seconds(10)),
          "cancellation session should enter a real training batch");
    Check(tasks.Cancel(cancelled_task_id),
          "task-panel cancellation request should be accepted");

    const auto cancelled_task = WaitForTerminalTask(
        manager, tasks, cancelled_task_id, "cancelled manager session");

    const auto cancelled_metrics = manager.GetCurrentMetrics();
    Check(cancelled_metrics.is_complete && !cancelled_metrics.is_training,
          "cancelled manager metrics should close the active state");
    Check(cancelled_metrics.last_executed_epoch <
              cancelled_metrics.total_epochs,
          "cancelled manager metrics should not claim all epochs executed");
    Check(cancelled_metrics.terminal_status == "cancelled" &&
              cancelled_metrics.terminal_reason == "user_cancelled",
          "cancelled manager metrics should preserve exact terminal truth");
    Check(start_callbacks.load() == 1 && end_callbacks.load() == 1 &&
              !callback_success.load(),
          "cancelled manager callbacks should close exactly once as unsuccessful");
    Check(node_started.load() == 1 && node_finished.load() == 1,
          "cancelled node-editor activity should open and close exactly once");

    Check(cancelled_task &&
              cancelled_task->GetState() == cyxwiz::TaskState::Cancelled,
          "task panel state should report cancelled");
    const auto cancelled_task_info = cancelled_task->GetInfo();
    Check(cancelled_task_info.status_message.find("executed epochs") !=
              std::string::npos &&
              cancelled_task_info.status_message.find("user_cancelled") !=
                  std::string::npos,
          "cancelled task status should expose executed epochs and reason");

    const auto cancelled_dashboard = cancelled_panel->GetStatusSnapshot();
    Check(cancelled_dashboard.has_data && !cancelled_dashboard.is_training,
          "dashboard should retain cancelled session data");
    Check(cancelled_dashboard.current_epoch ==
              cancelled_metrics.current_epoch &&
              cancelled_dashboard.last_executed_epoch ==
                  cancelled_metrics.last_executed_epoch &&
              cancelled_dashboard.total_epochs ==
                  cancelled_metrics.total_epochs,
          "cancelled dashboard epoch truth should match manager metrics");
    Check(cancelled_dashboard.terminal_status ==
              cancelled_metrics.terminal_status &&
              cancelled_dashboard.terminal_reason ==
                  cancelled_metrics.terminal_reason,
          "cancelled dashboard terminal truth should match manager metrics");
    Check(manager.HasTrainedModel(),
          "manager should preserve the partial model after cancellation");
    CheckTerminalTrace(cancelled_metrics, "cancelled");

    manager.ClearTrainedModel();
    start_callbacks.store(0);
    end_callbacks.store(0);
    progress_callbacks.store(0);
    node_started.store(0);
    node_finished.store(0);
    callback_success.store(false);

    auto early_stop_config = MakeConfig(work_dir / "early_stop_checkpoints");
    early_stop_config.epochs = 5;
    early_stop_config.learning_rate = 0.0f;
    early_stop_config.save_best_checkpoint = true;
    early_stop_config.early_stopping_patience = 1;
    auto early_stop_panel = std::make_shared<cyxwiz::TrainingPlotPanel>();
    const bool early_stop_started = manager.StartTrainingArrow(
        early_stop_config,
        MakeDataset(),
        "label",
        early_stop_config.epochs,
        early_stop_config.batch_size,
        early_stop_panel,
        [&](bool active) {
            if (active) {
                ++node_started;
            } else {
                ++node_finished;
            }
        });
    Check(early_stop_started,
          "TrainingManager should accept the early-stop session");
    const uint64_t early_stop_task_id = manager.GetCurrentTaskId();
    Check(early_stop_task_id != 0,
          "early-stop session should publish a task id");
    const auto early_stop_task = WaitForTerminalTask(
        manager, tasks, early_stop_task_id, "early-stop manager session");

    const auto early_stop_metrics = manager.GetCurrentMetrics();
    Check(early_stop_metrics.is_complete && !early_stop_metrics.is_training,
          "early-stop metrics should close the active state");
    Check(early_stop_metrics.current_epoch == 2 &&
              early_stop_metrics.last_executed_epoch == 2 &&
              early_stop_metrics.total_epochs == 5,
          "early-stop metrics should report the two executed epochs");
    Check(early_stop_metrics.terminal_status == "early_stopped" &&
              early_stop_metrics.terminal_reason ==
                  "validation_loss_plateau_patience_1",
          "early-stop metrics should preserve exact terminal truth");
    Check(!early_stop_metrics.checkpoint_used.empty() &&
              early_stop_metrics.restored_checkpoint_epoch == 1 &&
              early_stop_metrics.restored_checkpoint_step == 2 &&
              early_stop_metrics.active_model_provenance ==
                  "restored_best_checkpoint",
          "early-stop metrics should identify the restored best checkpoint");
    Check(start_callbacks.load() == 1 && end_callbacks.load() == 1 &&
              callback_success.load(),
          "early-stop callbacks should close exactly once as successful");
    Check(progress_callbacks.load() == 2,
          "early-stop progress should report exactly the executed epochs");
    Check(node_started.load() == 1 && node_finished.load() == 1,
          "early-stop node activity should open and close exactly once");
    Check(early_stop_task &&
              early_stop_task->GetState() == cyxwiz::TaskState::Completed,
          "early-stop task should be a successful terminal task");
    const auto early_stop_task_info = early_stop_task->GetInfo();
    Check(early_stop_task_info.status_message.find("Early stopped after 2/5") !=
              std::string::npos &&
              early_stop_task_info.status_message.find(
                  "validation_loss_plateau_patience_1") != std::string::npos &&
              early_stop_task_info.status_message.find(
                  "checkpoint epoch 1 step 2") != std::string::npos,
          "early-stop task should expose epoch, reason, and checkpoint truth");

    const auto early_stop_dashboard = early_stop_panel->GetStatusSnapshot();
    Check(early_stop_dashboard.has_data &&
              !early_stop_dashboard.is_training &&
              early_stop_dashboard.terminal_status ==
                  early_stop_metrics.terminal_status &&
              early_stop_dashboard.terminal_reason ==
                  early_stop_metrics.terminal_reason &&
              early_stop_dashboard.current_epoch == 2 &&
              early_stop_dashboard.last_executed_epoch == 2 &&
              early_stop_dashboard.total_epochs == 5,
          "early-stop dashboard should match manager terminal truth");
    Check(early_stop_dashboard.checkpoint_used ==
              early_stop_metrics.checkpoint_used &&
              early_stop_dashboard.checkpoint_epoch == 1 &&
              early_stop_dashboard.checkpoint_step == 2 &&
              early_stop_dashboard.active_model_provenance ==
                  "restored_best_checkpoint",
          "early-stop dashboard should expose restored checkpoint provenance");
    Check(manager.HasTrainedModel(),
          "manager should preserve the early-stopped model");
    const auto early_stop_model = manager.GetActiveModelInfo();
    Check(early_stop_model.origin ==
              cyxwiz::ActiveModelOrigin::TrainedInSession &&
              early_stop_model.checkpoint_path ==
                  early_stop_metrics.checkpoint_used,
          "active model should identify the restored checkpoint path");
    CheckTerminalTrace(early_stop_metrics, "early_stopped");

    manager.ClearTrainedModel();
    start_callbacks.store(0);
    end_callbacks.store(0);
    progress_callbacks.store(0);
    node_started.store(0);
    node_finished.store(0);
    callback_success.store(true);

    auto failure_config = MakeConfig(work_dir / "failure_checkpoints");
    failure_config.optimizer_type = gui::NodeType::Output;
    auto failure_panel = std::make_shared<cyxwiz::TrainingPlotPanel>();
    const bool failure_started = manager.StartTrainingArrow(
        failure_config,
        MakeDataset(),
        "label",
        failure_config.epochs,
        failure_config.batch_size,
        failure_panel,
        [&](bool active) {
            if (active) {
                ++node_started;
            } else {
                ++node_finished;
            }
        });
    Check(failure_started,
          "TrainingManager should accept the deterministic failure session");
    const uint64_t failure_task_id = manager.GetCurrentTaskId();
    Check(failure_task_id != 0,
          "failure session should publish a task id");
    const auto failure_task = WaitForTerminalTask(
        manager, tasks, failure_task_id, "failure manager session");

    const auto failure_metrics = manager.GetCurrentMetrics();
    Check(failure_metrics.is_complete && !failure_metrics.is_training,
          "failure metrics should close the active state");
    Check(failure_metrics.terminal_status == "failed" &&
              failure_metrics.terminal_reason.find(
                  "execution_preflight_failed") != std::string::npos,
          "failure metrics should preserve executor preflight truth");
    Check(failure_metrics.current_epoch == 0 &&
              failure_metrics.last_executed_epoch == 0,
          "preflight failure should not claim executed training");
    Check(start_callbacks.load() == 1 && end_callbacks.load() == 1 &&
              !callback_success.load(),
          "failure callbacks should close exactly once as unsuccessful");
    Check(progress_callbacks.load() == 0,
          "preflight failure should not report epoch progress");
    Check(node_started.load() == 1 && node_finished.load() == 1,
          "failure node activity should open and close exactly once");
    Check(failure_task &&
              failure_task->GetState() == cyxwiz::TaskState::Failed,
          "preflight failure should publish a failed task");
    Check(failure_task->GetInfo().status_message.find(
              failure_metrics.terminal_reason) != std::string::npos,
          "failed task should expose the exact preflight reason");

    const auto failure_dashboard = failure_panel->GetStatusSnapshot();
    Check(!failure_dashboard.has_data &&
              failure_dashboard.metric_points == 0 &&
              !failure_dashboard.is_training &&
              failure_dashboard.terminal_status == "failed" &&
              failure_dashboard.terminal_reason ==
                  failure_metrics.terminal_reason &&
              failure_dashboard.current_epoch == 0 &&
              failure_dashboard.last_executed_epoch == 0 &&
              failure_dashboard.total_epochs == 1,
          "failure dashboard should match manager preflight truth");
    Check(!manager.HasTrainedModel(),
          "manager should not preserve a model after preflight failure");
    Check(failure_metrics.checkpoint_used.empty() &&
              failure_metrics.restored_checkpoint_epoch == 0 &&
              failure_metrics.restored_checkpoint_step == 0 &&
              failure_metrics.active_model_provenance.empty(),
          "preflight failure should not claim active-model provenance");
    CheckTerminalTrace(failure_metrics, "failed");

    manager.SetOnTrainingStart(nullptr);
    manager.SetOnTrainingEnd(nullptr);
    manager.SetOnProgress(nullptr);
    manager.ClearTrainedModel();
    panel.reset();
    cancelled_panel.reset();
    early_stop_panel.reset();
    failure_panel.reset();
    tasks.Shutdown();
    fs::remove_all(work_dir);

    std::cout << "TrainingManager session orchestration passed\n";
    return 0;
}
