#include "async_task_manager.h"
#include "training_trace_collector.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

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
    if (cancellable_ && !cancel_requested_.exchange(true)) {
        if (cancellation_callback_) {
            cancellation_callback_();
        }
        TrainingTraceCollector::Instance().RecordTaskProgress(
            id_,
            name_,
            "TaskCancelRequested",
            progress_.load(),
            "cancel requested",
            "cancel_requested");
        spdlog::info("Cancel requested for task '{}' (ID: {})", name_, id_);
    }
}

void AsyncTask::SetCancellationCallback(std::function<void()> callback) {
    cancellation_callback_ = std::move(callback);
}

void AsyncTask::SetProgressCallback(ProgressCallback callback) {
    progress_callback_ = std::move(callback);
}

void AsyncTask::SetCompletionCallback(CompletionCallback callback) {
    completion_callback_ = std::move(callback);
}

void AsyncTask::BindOwner(std::weak_ptr<const void> owner) {
    owner_ = std::move(owner);
    owned_ = true;
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
    float normalized = progress_.load();
    if (std::isfinite(progress)) {
        normalized = std::clamp(progress, 0.0f, 1.0f);
    } else {
        spdlog::warn(
            "Task '{}' (ID: {}) ignored non-finite progress value",
            name_, id_);
    }

    // A task's overall progress cannot move backwards when an inner phase
    // restarts or reports a local fraction. Status text may still change.
    float current = progress_.load();
    while (normalized > current &&
           !progress_.compare_exchange_weak(current, normalized)) {
    }

    if (!message.empty()) {
        std::lock_guard<std::mutex> lock(message_mutex_);
        status_message_ = message;
    }

    if (progress_callback_) {
        progress_callback_(progress_.load(), message);
    }

    TrainingTraceCollector::Instance().RecordTaskProgress(
        id_,
        name_,
        "TaskProgress",
        progress_.load(),
        message,
        "running");
}

void AsyncTask::MarkCompleted(
    const std::string& message,
    const std::string& status) {
    state_.store(TaskState::Completed);
    progress_.store(1.0f);
    end_time_ = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(message_mutex_);
        status_message_ = message;
    }

    TrainingTraceCollector::Instance().RecordTaskProgress(
        id_,
        name_,
        "TaskCompleted",
        1.0f,
        message,
        status);
    spdlog::info("Task '{}' (ID: {}) completed: {}", name_, id_, message);
}

void AsyncTask::MarkCancelled(const std::string& message) {
    state_.store(TaskState::Cancelled);
    end_time_ = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(message_mutex_);
        status_message_ = message;
    }

    TrainingTraceCollector::Instance().RecordTaskProgress(
        id_,
        name_,
        "TaskCancelled",
        progress_.load(),
        message,
        "cancelled");
    spdlog::info("Task '{}' (ID: {}) cancelled: {}", name_, id_, message);
}

void AsyncTask::MarkFailed(const std::string& error) {
    state_.store(TaskState::Failed);
    end_time_ = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(message_mutex_);
        error_message_ = error;
        status_message_ = "Failed: " + error;
    }

    TrainingTraceCollector::Instance().RecordTaskProgress(
        id_,
        name_,
        "TaskFailed",
        progress_.load(),
        error,
        "failed");
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
    shutting_down_.store(false);
    generation_.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(worker_exit_mutex_);
        live_workers_ = num_threads;
    }

    // Create worker threads
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&AsyncTaskManager::WorkerThread, this);
    }

    initialized_ = true;
}

AsyncTaskManager::ShutdownReport AsyncTaskManager::Shutdown(
    std::chrono::milliseconds drain_timeout) {
    ShutdownReport report;
    if (!initialized_) {
        return report;
    }

    spdlog::info("Shutting down AsyncTaskManager...");
    shutting_down_.store(true);  // Submit/PostToMainThread reject from here

    // 1. Queued work never starts: take it off the queue before any worker
    //    can dispatch it and retire it as Cancelled, like the dispatch gate.
    std::vector<std::shared_ptr<AsyncTask>> queued;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!task_queue_.empty()) {
            queued.push_back(task_queue_.top().task);
            task_queue_.pop();
        }
    }
    for (const auto& task : queued) {
        task->RequestCancel();  // callbacks run outside every manager lock
        task->start_time_ = std::chrono::steady_clock::now();
        task->MarkCancelled("Cancelled before execution: task manager shutting down");
        RetireTask(task);
        ++report.queued_cancelled;
    }

    // 2. Running work is asked to stop; it stays cooperative.
    std::vector<std::shared_ptr<AsyncTask>> running;
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        running.reserve(active_tasks_.size());
        for (const auto& [id, task] : active_tasks_) running.push_back(task);
    }
    for (const auto& task : running) {
        if (task->IsCancellable()) {
            task->RequestCancel();
            ++report.running_cancel_requested;
        }
    }

    // 3. Release the workers and wait, bounded, for them to retire.
    shutdown_.store(true);
    queue_cv_.notify_all();
    {
        std::unique_lock<std::mutex> lock(worker_exit_mutex_);
        report.drained = worker_exit_cv_.wait_for(
            lock, drain_timeout, [this] { return live_workers_ == 0; });
    }
    if (report.drained) {
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    } else {
        // A task that ignores cancellation must not hold process exit
        // hostage. The worker is released; it can still retire its task but
        // can never deliver into a later generation (see EnqueueMainThread).
        {
            std::lock_guard<std::mutex> lock(tasks_mutex_);
            for (const auto& [id, task] : active_tasks_) {
                report.unfinished_tasks.push_back(task->GetName());
            }
        }
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.detach();
        }
        std::string names;
        for (const auto& name : report.unfinished_tasks) {
            names += (names.empty() ? "" : ", ") + name;
        }
        spdlog::warn(
            "AsyncTaskManager shutdown: {} task(s) still running after {} ms "
            "and released without completion: {}",
            report.unfinished_tasks.size(), drain_timeout.count(), names);
    }
    workers_.clear();

    // 4. Nothing may reach a UI owner that is about to be destroyed.
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        report.callbacks_discarded += pending_callbacks_.size();
        std::queue<MainThreadCallback>().swap(pending_callbacks_);
    }

    // 5. History is cleared only when every worker retired; an unfinished
    //    task stays visible as active until its released worker retires it.
    if (report.drained) {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        active_tasks_.clear();
        completed_tasks_.clear();
    }

    initialized_ = false;
    // Close this generation: a worker released above can still retire its
    // task, but nothing it produces is delivered into the next generation.
    generation_.fetch_add(1);
    shutting_down_.store(false);
    spdlog::info(
        "AsyncTaskManager shutdown complete: queued cancelled={}, running "
        "cancel requested={}, callbacks discarded={}, drained={}",
        report.queued_cancelled, report.running_cancel_requested,
        report.callbacks_discarded, report.drained);
    return report;
}

void AsyncTaskManager::RetireTask(const std::shared_ptr<AsyncTask>& task) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    active_tasks_.erase(task->GetId());
    completed_tasks_.push_back(task);

    // Limit completed tasks history
    while (completed_tasks_.size() > 100) {
        completed_tasks_.erase(completed_tasks_.begin());
    }
}

void AsyncTaskManager::EnqueueMainThread(MainThreadCallback callback,
                                         uint64_t generation) {
    if (!callback.callback) {
        return;
    }
    // Work retiring during shutdown, or under an earlier generation, has no
    // owner left to receive it; it is dropped here, never delivered late.
    if (shutting_down_.load() || generation != generation_.load()) {
        spdlog::debug("AsyncTaskManager dropped a main-thread callback with no live owner");
        return;
    }
    std::lock_guard<std::mutex> lock(callback_mutex_);
    pending_callbacks_.push(std::move(callback));
}

uint64_t AsyncTaskManager::Submit(std::shared_ptr<AsyncTask> task, TaskPriority priority) {
    uint64_t task_id = task->GetId();
    const auto reject = [this, &task, task_id] {
        // Nothing submitted while Shutdown runs is executed; the task is kept
        // in history so a caller that polls its state sees a truthful end.
        task->start_time_ = std::chrono::steady_clock::now();
        task->MarkCancelled("Rejected: task manager is shutting down");
        RetireTask(task);
        return task_id;
    };

    if (shutting_down_.load()) {
        return reject();
    }

    if (!initialized_) {
        Initialize();
    }

    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        active_tasks_[task_id] = task;
    }

    {
        // Checked again under the queue lock: Shutdown raises the flag before
        // it drains this queue, so a task either lands before the drain (and
        // is retired by it) or is rejected here; it is never stranded.
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (shutting_down_.load()) {
            lock.unlock();
            return reject();
        }
        task_queue_.push({task, priority});
    }

    queue_cv_.notify_one();

    spdlog::debug("Task '{}' (ID: {}) submitted with priority {}",
                  task->GetName(), task_id, static_cast<int>(priority));

    return task_id;
}

bool AsyncTaskManager::Cancel(uint64_t task_id) {
    std::shared_ptr<AsyncTask> task;
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        const auto it = active_tasks_.find(task_id);
        if (it == active_tasks_.end()) return false;
        task = it->second;
    }
    // Callbacks may query this manager. Retain ownership, but never call them
    // while holding its task registry lock.
    task->RequestCancel();
    return true;
}

void AsyncTaskManager::CancelAll() {
    std::vector<std::shared_ptr<AsyncTask>> tasks;
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        tasks.reserve(active_tasks_.size());
        for (const auto& [id, task] : active_tasks_) tasks.push_back(task);
    }
    // Cancel the snapshot; callbacks can safely inspect or submit other tasks.
    for (const auto& task : tasks) task->RequestCancel();
}

size_t AsyncTaskManager::CancelOwnedBy(const std::shared_ptr<const void>& owner) {
    if (!owner) {
        return 0;
    }
    const void* key = owner.get();
    std::vector<std::shared_ptr<AsyncTask>> owned;
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        for (const auto& [id, task] : active_tasks_) {
            if (task->owned_ && task->owner_.lock().get() == key) {
                owned.push_back(task);
            }
        }
    }
    size_t requested = 0;
    for (const auto& task : owned) {
        if (task->IsCancellable()) {
            task->RequestCancel();  // outside the registry lock, as Cancel()
            ++requested;
        }
    }
    // Queued deliveries for this owner are stale from now on.
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        std::queue<MainThreadCallback> kept;
        while (!pending_callbacks_.empty()) {
            auto& entry = pending_callbacks_.front();
            if (!(entry.owned && entry.owner_key == key)) {
                kept.push(std::move(entry));
            }
            pending_callbacks_.pop();
        }
        pending_callbacks_.swap(kept);
    }
    return requested;
}

std::shared_ptr<AsyncTask> AsyncTaskManager::GetTask(uint64_t task_id) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);

    // First check active tasks
    auto it = active_tasks_.find(task_id);
    if (it != active_tasks_.end()) {
        return it->second;
    }

    // Also check completed tasks (for fast-completing tasks that get moved before UI can check)
    for (const auto& task : completed_tasks_) {
        if (task->GetId() == task_id) {
            return task;
        }
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
    std::queue<MainThreadCallback> callbacks;

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        std::swap(callbacks, pending_callbacks_);
    }

    size_t discarded = 0;
    while (!callbacks.empty()) {
        auto& entry = callbacks.front();
        // An owner that died between completion and this pump receives
        // nothing; its callback would touch destroyed state.
        if (entry.owned && entry.owner.expired()) {
            ++discarded;
        } else {
            entry.callback();
        }
        callbacks.pop();
    }
    if (discarded > 0) {
        spdlog::debug("AsyncTaskManager discarded {} main-thread callback(s) whose owner is gone",
                      discarded);
    }
}

void AsyncTaskManager::PostToMainThread(std::function<void()> callback) {
    MainThreadCallback entry;
    entry.callback = std::move(callback);
    EnqueueMainThread(std::move(entry), generation_.load());
}

void AsyncTaskManager::PostToMainThread(std::weak_ptr<const void> owner,
                                        std::function<void()> callback) {
    MainThreadCallback entry;
    entry.owned = true;
    entry.owner_key = owner.lock().get();
    entry.owner = std::move(owner);
    entry.callback = std::move(callback);
    EnqueueMainThread(std::move(entry), generation_.load());
}

void AsyncTaskManager::WorkerThread() {
    const uint64_t generation = generation_.load();
    // Counts this worker out of ITS generation only: a worker released by a
    // timed-out Shutdown that exits later must not disturb the next one.
    struct ExitSignal {
        AsyncTaskManager& manager;
        uint64_t generation;
        ~ExitSignal() {
            std::lock_guard<std::mutex> lock(manager.worker_exit_mutex_);
            if (generation == manager.generation_.load()) {
                --manager.live_workers_;
                manager.worker_exit_cv_.notify_all();
            }
        }
    } exit_signal{*this, generation};
    const auto retired = [this, generation] {
        return shutdown_.load() || generation != generation_.load();
    };

    while (!retired()) {
        std::shared_ptr<AsyncTask> task;

        // Wait for a task
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this, &retired] {
                return retired() || !task_queue_.empty();
            });

            // Shutdown drains the queue itself before it releases workers,
            // so a retiring worker never takes new work.
            if (retired()) {
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

        // Dispatch time bounds the duration of a skipped task. A cancellation
        // observed here must never invoke the user body or emit TaskStarted.
        // Cancellation after this gate remains cooperative, as for running tasks.
        task->start_time_ = std::chrono::steady_clock::now();
        if (task->IsCancelRequested()) {
            task->MarkCancelled("Cancelled before execution");
        } else {
            task->state_.store(TaskState::Running);

            TrainingTraceCollector::Instance().RecordTaskProgress(
                task_id,
                task->GetName(),
                "TaskStarted",
                task->GetProgress(),
                "started",
                "running");
            spdlog::debug("Starting task '{}' (ID: {})", task->GetName(), task_id);

            try {
                task->Execute();

                // Check if task was cancelled
                if (task->IsCancelRequested() && task->GetState() == TaskState::Running) {
                    task->MarkCancelled();
                }
            } catch (const std::exception& e) {
                task->MarkFailed(e.what());
            } catch (...) {
                task->MarkFailed("Unknown error");
            }

        }

        // Move from active to completed
        RetireTask(task);

        // Queue completion callback for main thread
        if (task->completion_callback_) {
            bool success = task->GetState() == TaskState::Completed;
            std::string error = task->GetErrorMessage();

            MainThreadCallback entry;
            entry.owned = task->owned_;
            entry.owner = task->owner_;
            entry.owner_key = task->owned_ ? task->owner_.lock().get() : nullptr;
            entry.callback = [cb = task->completion_callback_, success, error]() {
                cb(success, error);
            };
            EnqueueMainThread(std::move(entry), generation);
        }
    }
}

} // namespace cyxwiz
