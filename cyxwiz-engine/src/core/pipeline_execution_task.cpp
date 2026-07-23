#include "pipeline_execution_task.h"

#include "async_task_manager.h"
#include "pipeline_executor.h"

#include <stdexcept>
#include <utility>

namespace cyxwiz {

void PipelineExecutionTracker::Record(
    int node_id,
    PipelineNodeExecutionEvent event,
    const std::string& status) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.node_states[node_id] = event;
    snapshot_.node_status[node_id] = status;
}

PipelineExecutionSnapshot PipelineExecutionTracker::GetSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

PipelineExecutionSubmission SubmitPipelineExecutionTask(
    const std::string& task_name,
    std::string pipeline_json,
    std::shared_ptr<PipelineExecutor> executor) {
    if (!executor) {
        executor = std::make_shared<PipelineExecutor>();
    }

    auto tracker = std::make_shared<PipelineExecutionTracker>();
    executor->SetNodeExecutionCallback(
        [weak_tracker = std::weak_ptr<PipelineExecutionTracker>(tracker)](
            int node_id,
            PipelineNodeExecutionEvent event,
            const std::string& status) {
            if (auto current = weak_tracker.lock()) {
                current->Record(node_id, event, status);
            }
        });

    auto task = std::make_shared<LambdaTask>(
        task_name,
        [pipeline_json = std::move(pipeline_json), executor](LambdaTask& task) {
            if (task.ShouldStop()) {
                return;
            }

            executor->SetProgressCallback(
                [&task, weak_executor = std::weak_ptr<PipelineExecutor>(executor)](
                    float progress, const std::string& status) {
                    task.ReportProgress(progress, status);
                    if (task.ShouldStop()) {
                        if (auto current = weak_executor.lock()) {
                            current->RequestCancel();
                        }
                    }
                });

            const bool success = executor->ExecutePipeline(pipeline_json);
            if (task.ShouldStop() || executor->IsCancelled()) {
                return;
            }
            if (!success) {
                const std::string error = executor->GetLastError().empty()
                    ? "Pipeline execution failed"
                    : executor->GetLastError();
                throw std::runtime_error(error);
            }
            task.MarkCompleted("Pipeline execution completed");
        });

    task->SetCancellationCallback(
        [weak_executor = std::weak_ptr<PipelineExecutor>(executor)]() {
            if (auto current = weak_executor.lock()) {
                current->RequestCancel();
            }
        });

    PipelineExecutionSubmission submission;
    submission.executor = std::move(executor);
    submission.tracker = std::move(tracker);
    submission.task_id = AsyncTaskManager::Instance().Submit(task);
    return submission;
}

bool IsPipelineExecutionTaskActive(uint64_t task_id) {
    if (task_id == 0) {
        return false;
    }
    const auto task = AsyncTaskManager::Instance().GetTask(task_id);
    if (!task) {
        return false;
    }
    const auto state = task->GetState();
    return state == TaskState::Pending || state == TaskState::Running;
}

} // namespace cyxwiz
