#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace cyxwiz {

class PipelineExecutor;
enum class PipelineNodeExecutionEvent : uint8_t;

struct PipelineExecutionSnapshot {
    std::map<int, PipelineNodeExecutionEvent> node_states;
    std::map<int, std::string> node_status;
};

class PipelineExecutionTracker {
public:
    void Record(int node_id,
                PipelineNodeExecutionEvent event,
                const std::string& status);
    PipelineExecutionSnapshot GetSnapshot() const;

private:
    mutable std::mutex mutex_;
    PipelineExecutionSnapshot snapshot_;
};

struct PipelineExecutionSubmission {
    uint64_t task_id = 0;
    std::shared_ptr<PipelineExecutor> executor;
    std::shared_ptr<PipelineExecutionTracker> tracker;
};

// Submit one pipeline snapshot to the shared background task system. The
// returned executor owns runtime/deployment state while AsyncTaskManager owns
// execution progress, cancellation, and recent-task visibility.
PipelineExecutionSubmission SubmitPipelineExecutionTask(
    const std::string& task_name,
    std::string pipeline_json,
    std::shared_ptr<PipelineExecutor> executor = nullptr);

bool IsPipelineExecutionTaskActive(uint64_t task_id);

} // namespace cyxwiz
