#pragma once

#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <unordered_map>

namespace cyxwiz {

// Forward declarations
class AsyncTask;

// Task priority levels
enum class TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3
};

// Task state
enum class TaskState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled
};

// Progress callback type
using ProgressCallback = std::function<void(float progress, const std::string& message)>;
using CompletionCallback = std::function<void(bool success, const std::string& error)>;

// Task info for UI display
struct TaskInfo {
    uint64_t id = 0;
    std::string name;
    std::string description;
    TaskState state = TaskState::Pending;
    float progress = 0.0f;
    std::string status_message;
    std::string error_message;
    bool cancellable = true;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
};

// Base class for async tasks
class AsyncTask {
public:
    AsyncTask(const std::string& name, bool cancellable = true);
    virtual ~AsyncTask() = default;

    // Execute the task (called on worker thread)
    virtual void Execute() = 0;

    // Task identification
    uint64_t GetId() const { return id_; }
    const std::string& GetName() const { return name_; }

    // State management
    TaskState GetState() const { return state_.load(); }
    float GetProgress() const { return progress_.load(); }
    const std::string& GetStatusMessage() const;
    const std::string& GetErrorMessage() const;
    bool IsCancellable() const { return cancellable_; }

    // Cancel request
    void RequestCancel();
    bool IsCancelRequested() const { return cancel_requested_.load(); }

    // Callbacks
    void SetProgressCallback(ProgressCallback callback);
    void SetCompletionCallback(CompletionCallback callback);

    // Get task info for UI
    TaskInfo GetInfo() const;

protected:
    // Call from Execute() to update progress
    void ReportProgress(float progress, const std::string& message = "");

    // Call from Execute() to check if should stop
    bool ShouldStop() const { return cancel_requested_.load(); }

    // Mark task as complete or failed
    void MarkCompleted();
    void MarkFailed(const std::string& error);

private:
    friend class AsyncTaskManager;

    uint64_t id_;
    std::string name_;
    bool cancellable_;

    std::atomic<TaskState> state_{TaskState::Pending};
    std::atomic<float> progress_{0.0f};
    std::atomic<bool> cancel_requested_{false};

    mutable std::mutex message_mutex_;
    std::string status_message_;
    std::string error_message_;

    ProgressCallback progress_callback_;
    CompletionCallback completion_callback_;

    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;

    static std::atomic<uint64_t> next_id_;
};

// Thread pool for executing async tasks
class AsyncTaskManager {
public:
    // Singleton access
    static AsyncTaskManager& Instance();

    // Initialize with number of worker threads (0 = auto)
    void Initialize(size_t num_threads = 0);
    void Shutdown();

    // Submit a task
    uint64_t Submit(std::shared_ptr<AsyncTask> task, TaskPriority priority = TaskPriority::Normal);

    // Cancel a task
    bool Cancel(uint64_t task_id);
    void CancelAll();

    // Get task info
    std::shared_ptr<AsyncTask> GetTask(uint64_t task_id);
    std::vector<TaskInfo> GetActiveTasks() const;
    std::vector<TaskInfo> GetRecentTasks(size_t count = 10) const;

    // Check if any tasks are running
    bool HasActiveTasks() const;
    size_t GetActiveTaskCount() const;

    // Process completed task callbacks (call from main thread)
    void ProcessCompletedCallbacks();

    // Convenience method to run a simple function async
    template<typename Func>
    uint64_t RunAsync(const std::string& name, Func&& func,
                      ProgressCallback progress_cb = nullptr,
                      CompletionCallback completion_cb = nullptr);

private:
    AsyncTaskManager() = default;
    ~AsyncTaskManager();

    AsyncTaskManager(const AsyncTaskManager&) = delete;
    AsyncTaskManager& operator=(const AsyncTaskManager&) = delete;

    void WorkerThread();

    std::vector<std::thread> workers_;
    std::atomic<bool> shutdown_{false};

    // Task queue with priority
    struct PrioritizedTask {
        std::shared_ptr<AsyncTask> task;
        TaskPriority priority;

        bool operator<(const PrioritizedTask& other) const {
            return priority < other.priority;
        }
    };

    mutable std::mutex queue_mutex_;
    std::priority_queue<PrioritizedTask> task_queue_;
    std::condition_variable queue_cv_;

    // Active and completed tasks
    mutable std::mutex tasks_mutex_;
    std::unordered_map<uint64_t, std::shared_ptr<AsyncTask>> active_tasks_;
    std::vector<std::shared_ptr<AsyncTask>> completed_tasks_;

    // Callbacks to process on main thread
    mutable std::mutex callback_mutex_;
    std::queue<std::function<void()>> pending_callbacks_;

    bool initialized_ = false;
};

// Lambda-based task for convenience
class LambdaTask : public AsyncTask {
public:
    using TaskFunction = std::function<void(LambdaTask&)>;

    LambdaTask(const std::string& name, TaskFunction func, bool cancellable = true);
    void Execute() override;

    // Public access to progress reporting for lambdas
    using AsyncTask::ReportProgress;
    using AsyncTask::ShouldStop;
    using AsyncTask::MarkCompleted;
    using AsyncTask::MarkFailed;

private:
    TaskFunction func_;
};

// Template implementation
template<typename Func>
uint64_t AsyncTaskManager::RunAsync(const std::string& name, Func&& func,
                                     ProgressCallback progress_cb,
                                     CompletionCallback completion_cb) {
    auto task = std::make_shared<LambdaTask>(name,
        [f = std::forward<Func>(func)](LambdaTask& task) {
            try {
                f(task);
                if (!task.IsCancelRequested()) {
                    task.MarkCompleted();
                }
            } catch (const std::exception& e) {
                task.MarkFailed(e.what());
            }
        });

    if (progress_cb) {
        task->SetProgressCallback(progress_cb);
    }
    if (completion_cb) {
        task->SetCompletionCallback(completion_cb);
    }

    return Submit(task);
}

} // namespace cyxwiz
