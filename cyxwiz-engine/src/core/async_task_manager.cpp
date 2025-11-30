#include "async_task_manager.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

// Static member initialization
std::atomic<uint64_t> AsyncTask::next_id_{1};

// ============================================================================
// AsyncTask Implementation
// ============================================================================

AsyncTask::AsyncTask(const std::string& name, bool cancellable)
    : id_(next_id_++), name_(name), cancellable_(cancellable) {
}

const std::string& AsyncTask::GetStatusMessage() const {
    std::lock_guard<std::mutex> lock(message_mutex_);
    return status_message_;
}

const std::string& AsyncTask::GetErrorMessage() const {
    std::lock_guard<std::mutex> lock(message_mutex_);
    return error_message_;
}

void AsyncTask::RequestCancel() {
    if (cancellable_) {
        cancel_requested_.store(true);
        spdlog::info("Cancel requested for task '{}' (ID: {})", name_, id_);
    }
}

void AsyncTask::SetProgressCallback(ProgressCallback callback) {
    progress_callback_ = std::move(callback);
}

void AsyncTask::SetCompletionCallback(CompletionCallback callback) {
    completion_callback_ = std::move(callback);
}

TaskInfo AsyncTask::GetInfo() const {
    TaskInfo info;
    info.id = id_;
    info.name = name_;
    info.state = state_.load();
    info.progress = progress_.load();
    info.cancellable = cancellable_;
    info.start_time = start_time_;
    info.end_time = end_time_;

    {
        std::lock_guard<std::mutex> lock(message_mutex_);
        info.status_message = status_message_;
        info.error_message = error_message_;
    }

    return info;
}

void AsyncTask::ReportProgress(float progress, const std::string& message) {
    progress_.store(std::clamp(progress, 0.0f, 1.0f));

    if (!message.empty()) {
        std::lock_guard<std::mutex> lock(message_mutex_);
        status_message_ = message;
    }

    if (progress_callback_) {
        progress_callback_(progress_.load(), message);
    }
}

void AsyncTask::MarkCompleted() {
    state_.store(TaskState::Completed);
    progress_.store(1.0f);
    end_time_ = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(message_mutex_);
        status_message_ = "Completed";
    }

    spdlog::info("Task '{}' (ID: {}) completed", name_, id_);
}

void AsyncTask::MarkFailed(const std::string& error) {
    state_.store(TaskState::Failed);
    end_time_ = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(message_mutex_);
        error_message_ = error;
        status_message_ = "Failed: " + error;
    }

    spdlog::error("Task '{}' (ID: {}) failed: {}", name_, id_, error);
}

// ============================================================================
// LambdaTask Implementation
// ============================================================================

LambdaTask::LambdaTask(const std::string& name, TaskFunction func, bool cancellable)
    : AsyncTask(name, cancellable), func_(std::move(func)) {
}

void LambdaTask::Execute() {
    if (func_) {
        func_(*this);
    }
}

// ============================================================================
// AsyncTaskManager Implementation
// ============================================================================

AsyncTaskManager& AsyncTaskManager::Instance() {
    static AsyncTaskManager instance;
    return instance;
}

AsyncTaskManager::~AsyncTaskManager() {
    Shutdown();
}

void AsyncTaskManager::Initialize(size_t num_threads) {
    if (initialized_) {
        return;
    }

    // Auto-detect thread count
    if (num_threads == 0) {
        num_threads = std::max(2u, std::thread::hardware_concurrency());
    }

    spdlog::info("Initializing AsyncTaskManager with {} worker threads", num_threads);

    shutdown_.store(false);

    // Create worker threads
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&AsyncTaskManager::WorkerThread, this);
    }

    initialized_ = true;
}

void AsyncTaskManager::Shutdown() {
    if (!initialized_) {
        return;
    }

    spdlog::info("Shutting down AsyncTaskManager...");

    // Signal shutdown
    shutdown_.store(true);
    queue_cv_.notify_all();

    // Wait for workers to finish
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    // Clear queues
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!task_queue_.empty()) {
            task_queue_.pop();
        }
    }

    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        active_tasks_.clear();
        completed_tasks_.clear();
    }

    initialized_ = false;
    spdlog::info("AsyncTaskManager shutdown complete");
}

uint64_t AsyncTaskManager::Submit(std::shared_ptr<AsyncTask> task, TaskPriority priority) {
    if (!initialized_) {
        Initialize();
    }

    uint64_t task_id = task->GetId();

    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        active_tasks_[task_id] = task;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push({task, priority});
    }

    queue_cv_.notify_one();

    spdlog::debug("Task '{}' (ID: {}) submitted with priority {}",
                  task->GetName(), task_id, static_cast<int>(priority));

    return task_id;
}

bool AsyncTaskManager::Cancel(uint64_t task_id) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);

    auto it = active_tasks_.find(task_id);
    if (it != active_tasks_.end()) {
        it->second->RequestCancel();
        return true;
    }

    return false;
}

void AsyncTaskManager::CancelAll() {
    std::lock_guard<std::mutex> lock(tasks_mutex_);

    for (auto& [id, task] : active_tasks_) {
        task->RequestCancel();
    }
}

std::shared_ptr<AsyncTask> AsyncTaskManager::GetTask(uint64_t task_id) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);

    auto it = active_tasks_.find(task_id);
    if (it != active_tasks_.end()) {
        return it->second;
    }

    return nullptr;
}

std::vector<TaskInfo> AsyncTaskManager::GetActiveTasks() const {
    std::lock_guard<std::mutex> lock(tasks_mutex_);

    std::vector<TaskInfo> infos;
    infos.reserve(active_tasks_.size());

    for (const auto& [id, task] : active_tasks_) {
        infos.push_back(task->GetInfo());
    }

    return infos;
}

std::vector<TaskInfo> AsyncTaskManager::GetRecentTasks(size_t count) const {
    std::lock_guard<std::mutex> lock(tasks_mutex_);

    std::vector<TaskInfo> infos;

    // First add active tasks
    for (const auto& [id, task] : active_tasks_) {
        infos.push_back(task->GetInfo());
    }

    // Then add completed tasks (most recent first)
    size_t completed_to_add = count > infos.size() ? count - infos.size() : 0;
    for (size_t i = 0; i < completed_to_add && i < completed_tasks_.size(); ++i) {
        size_t idx = completed_tasks_.size() - 1 - i;
        infos.push_back(completed_tasks_[idx]->GetInfo());
    }

    return infos;
}

bool AsyncTaskManager::HasActiveTasks() const {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    return !active_tasks_.empty();
}

size_t AsyncTaskManager::GetActiveTaskCount() const {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    return active_tasks_.size();
}

void AsyncTaskManager::ProcessCompletedCallbacks() {
    std::queue<std::function<void()>> callbacks;

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        std::swap(callbacks, pending_callbacks_);
    }

    while (!callbacks.empty()) {
        callbacks.front()();
        callbacks.pop();
    }
}

void AsyncTaskManager::WorkerThread() {
    while (!shutdown_.load()) {
        std::shared_ptr<AsyncTask> task;

        // Wait for a task
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return shutdown_.load() || !task_queue_.empty();
            });

            if (shutdown_.load() && task_queue_.empty()) {
                return;
            }

            if (!task_queue_.empty()) {
                task = task_queue_.top().task;
                task_queue_.pop();
            }
        }

        if (!task) {
            continue;
        }

        // Execute the task
        uint64_t task_id = task->GetId();

        task->state_.store(TaskState::Running);
        task->start_time_ = std::chrono::steady_clock::now();

        spdlog::debug("Starting task '{}' (ID: {})", task->GetName(), task_id);

        try {
            task->Execute();

            // Check if task was cancelled
            if (task->IsCancelRequested() && task->GetState() == TaskState::Running) {
                task->state_.store(TaskState::Cancelled);
                task->end_time_ = std::chrono::steady_clock::now();
                spdlog::info("Task '{}' (ID: {}) was cancelled", task->GetName(), task_id);
            }
        } catch (const std::exception& e) {
            task->MarkFailed(e.what());
        } catch (...) {
            task->MarkFailed("Unknown error");
        }

        // Move from active to completed
        {
            std::lock_guard<std::mutex> lock(tasks_mutex_);
            active_tasks_.erase(task_id);
            completed_tasks_.push_back(task);

            // Limit completed tasks history
            while (completed_tasks_.size() > 100) {
                completed_tasks_.erase(completed_tasks_.begin());
            }
        }

        // Queue completion callback for main thread
        if (task->completion_callback_) {
            bool success = task->GetState() == TaskState::Completed;
            std::string error = task->GetErrorMessage();

            std::lock_guard<std::mutex> lock(callback_mutex_);
            pending_callbacks_.push([cb = task->completion_callback_, success, error]() {
                cb(success, error);
            });
        }
    }
}

} // namespace cyxwiz
