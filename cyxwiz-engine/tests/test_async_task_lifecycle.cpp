// TOFIX101 package A: task-manager shutdown and owner lifetime.
// Real AsyncTaskManager; deterministic gates instead of sleeps where possible.
#include "core/async_task_manager.h"
#include "core/training_trace_collector.h"
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace cyxwiz;
using namespace std::chrono_literals;

namespace {
int checks = 0;
void Check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}

// Occupies the single worker so later submissions stay queued.
struct Blocker {
    std::promise<void> release;
    bool released = false;
    std::shared_ptr<LambdaTask> task;
    explicit Blocker(AsyncTaskManager& manager, bool cancellable = false) {
        auto entered = std::make_shared<std::promise<void>>();
        auto ready = entered->get_future();
        auto signal = release.get_future().share();
        task = std::make_shared<LambdaTask>("worker barrier", [entered, signal](auto& self) {
            entered->set_value();
            // Cooperative: stops on cancel when cancellable, else on release.
            while (signal.wait_for(1ms) != std::future_status::ready) {
                if (self.ShouldStop()) return;
            }
            self.MarkCompleted();
        }, cancellable);
        manager.Submit(task);
        Check(ready.wait_for(5s) == std::future_status::ready, "worker entered barrier");
    }
    void Release() { if (!released) { release.set_value(); released = true; } }
    ~Blocker() { Release(); }
};

void Pump(AsyncTaskManager& manager, int rounds = 20) {
    for (int i = 0; i < rounds; ++i) {
        manager.ProcessCompletedCallbacks();
        std::this_thread::sleep_for(1ms);
    }
}

bool WaitFor(const std::function<bool()>& done, std::chrono::milliseconds limit = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

// 1+2: queued work retires without running; running work is asked to stop;
// nothing is delivered to the UI after Shutdown.
void ShutdownRetiresQueuedAndStopsRunning() {
    auto& manager = AsyncTaskManager::Instance();
    manager.Initialize(1);
    Blocker running(manager, /*cancellable=*/true);
    std::atomic<int> bodies = 0, completions = 0, cancel_callbacks = 0;
    auto queued = std::make_shared<LambdaTask>("queued at shutdown", [&](auto& self) {
        ++bodies; self.MarkCompleted();
    });
    queued->SetCancellationCallback([&] { ++cancel_callbacks; });
    queued->SetCompletionCallback([&](bool, const std::string&) { ++completions; });
    running.task->SetCompletionCallback([&](bool, const std::string&) { ++completions; });
    manager.Submit(queued);

    const auto report = manager.Shutdown(2s);
    Check(report.queued_cancelled == 1, "shutdown retires the queued task");
    Check(report.running_cancel_requested == 1, "shutdown asks the running task to stop");
    Check(report.drained && report.unfinished_tasks.empty(), "cooperative task drains in time");
    Check(bodies == 0, "queued body never runs during shutdown");
    Check(queued->GetState() == TaskState::Cancelled, "queued task retires Cancelled");
    Check(queued->GetStatusMessage().find("shutting down") != std::string::npos,
          "queued cancellation names the shutdown");
    Check(cancel_callbacks == 1, "queued task cancellation callback fires once");
    Check(running.task->IsCancelRequested(), "running task observed a cancel request");
    Pump(manager);
    Check(completions == 0, "no completion reaches the UI after shutdown");
    Check(!manager.IsShuttingDown(), "shutdown is a phase, not a terminal state");
}

// 3: a completion queued before Shutdown (UI not yet pumped) is discarded.
void CompletionQueuedBeforeShutdownIsDiscarded() {
    auto& manager = AsyncTaskManager::Instance();
    manager.Initialize(1);
    std::atomic<int> completions = 0;
    const auto id = manager.RunAsync("finished, not yet delivered", [](auto&) {}, nullptr,
        [&](bool, const std::string&) { ++completions; });
    Check(WaitFor([&] {
        auto task = manager.GetTask(id);
        return task && task->GetState() == TaskState::Completed;
    }), "task completed on the worker");
    Check(WaitFor([&] { return manager.GetActiveTaskCount() == 0; }), "task retired");
    const auto report = manager.Shutdown(2s);
    Check(report.callbacks_discarded >= 1, "pending completion counted as discarded");
    Pump(manager);
    Check(completions == 0, "completion queued before shutdown is not delivered after it");
}

// 4: work submitted while Shutdown runs (here from a cancellation callback)
// is rejected, never executed.
void SubmitDuringShutdownIsRejected() {
    auto& manager = AsyncTaskManager::Instance();
    manager.Initialize(1);
    Blocker running(manager, /*cancellable=*/true);
    std::atomic<int> late_bodies = 0;
    std::shared_ptr<LambdaTask> late;
    running.task->SetCancellationCallback([&] {
        late = std::make_shared<LambdaTask>("submitted during shutdown",
            [&](auto&) { ++late_bodies; });
        manager.Submit(late);
    });
    const auto report = manager.Shutdown(2s);
    Check(report.drained, "shutdown with a reentrant submit still drains");
    Check(late != nullptr, "cancellation callback ran during shutdown");
    Check(late->GetState() == TaskState::Cancelled, "late submission retires Cancelled");
    Check(late->GetStatusMessage().find("Rejected") != std::string::npos,
          "late submission states the rejection");
    Check(late_bodies == 0, "late submission never runs");
}

// 5: a task that ignores cancellation cannot hold shutdown hostage, and its
// eventual completion is not delivered into the next generation.
void UncooperativeTaskIsReleasedNotAwaited() {
    auto& manager = AsyncTaskManager::Instance();
    manager.Initialize(1);
    std::promise<void> release;
    auto signal = release.get_future().share();
    std::promise<void> entered;
    auto ready = entered.get_future();
    std::atomic<int> completions = 0;
    auto stuck = std::make_shared<LambdaTask>("ignores cancellation", [&, signal](auto& self) {
        entered.set_value();
        signal.wait_for(10s);  // deliberately does not poll ShouldStop
        self.MarkCompleted();
    });
    stuck->SetCompletionCallback([&](bool, const std::string&) { ++completions; });
    manager.Submit(stuck);
    Check(ready.wait_for(5s) == std::future_status::ready, "stuck task started");

    const auto start = std::chrono::steady_clock::now();
    const auto report = manager.Shutdown(200ms);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    Check(!report.drained, "report says the worker did not drain");
    Check(report.unfinished_tasks.size() == 1 &&
              report.unfinished_tasks.front() == "ignores cancellation",
          "report names the unfinished task");
    Check(elapsed < 2s, "shutdown returns within its drain bound");

    release.set_value();
    Check(WaitFor([&] { return stuck->GetState() == TaskState::Completed; }),
          "released worker still retires its task");
    // A fresh generation starts; the old worker must neither deliver into it
    // nor take its work, and the new generation's accounting stays exact.
    manager.Initialize(1);
    std::atomic<int> fresh = 0;
    manager.RunAsync("fresh generation", [](auto&) {}, nullptr,
        [&](bool ok, const std::string&) { if (ok) ++fresh; });
    Check(WaitFor([&] { manager.ProcessCompletedCallbacks(); return fresh.load() == 1; }),
          "restart after a timed-out shutdown runs and delivers new work");
    Check(completions == 0, "old-generation completion is never delivered");
    const auto second = manager.Shutdown(2s);
    Check(second.drained, "next generation drains cleanly (worker accounting intact)");
}

// 6+7+8: owner-scoped cancellation and delivery.
void OwnerScopedLifetime() {
    auto& manager = AsyncTaskManager::Instance();
    manager.Initialize(1);
    auto editor = std::make_shared<int>(1);
    auto other = std::make_shared<int>(2);
    std::atomic<int> editor_bodies = 0, other_bodies = 0;
    std::atomic<int> editor_completions = 0, other_completions = 0;
    {
        Blocker blocker(manager);
        auto owned = std::make_shared<LambdaTask>("editor pipeline", [&](auto& self) {
            ++editor_bodies; self.MarkCompleted();
        });
        owned->BindOwner(editor);
        owned->SetCompletionCallback([&](bool, const std::string&) { ++editor_completions; });
        auto unowned = std::make_shared<LambdaTask>("other panel work", [&](auto& self) {
            ++other_bodies; self.MarkCompleted();
        });
        unowned->BindOwner(other);
        unowned->SetCompletionCallback([&](bool ok, const std::string&) { if (ok) ++other_completions; });
        manager.Submit(owned);
        manager.Submit(unowned);
        manager.PostToMainThread(std::weak_ptr<const void>(editor), [&] { ++editor_completions; });
        manager.PostToMainThread(std::weak_ptr<const void>(other), [&] { ++other_completions; });

        // Mirrors ~NodeEditor: cancel owned work, then the token dies with
        // the editor. The cancelled completion must not reach it.
        const size_t requested = manager.CancelOwnedBy(editor);
        Check(requested == 1, "owner close cancels exactly its own task");
        Check(owned->IsCancelRequested() && !unowned->IsCancelRequested(),
              "other owners' work is untouched");
        editor.reset();
        blocker.Release();
        Check(WaitFor([&] {
            manager.ProcessCompletedCallbacks();
            return other_completions.load() == 2;
        }), "other owner's work and posted callback are delivered");
        Pump(manager);
        Check(editor_bodies == 0, "cancelled owned task never ran");
        Check(editor_completions == 0, "owner close discards its queued deliveries");
        Check(other_bodies == 1, "other owner's task ran once");
    }

    // Owner destroyed after its task finished but before the UI pump.
    auto panel = std::make_shared<int>(3);
    std::atomic<int> panel_completions = 0;
    auto task = std::make_shared<LambdaTask>("finishes after panel closes", [](auto& self) {
        self.MarkCompleted();
    });
    task->BindOwner(panel);
    task->SetCompletionCallback([&](bool, const std::string&) { ++panel_completions; });
    manager.Submit(task);
    Check(WaitFor([&] { return task->GetState() == TaskState::Completed; }), "owned task done");
    Check(WaitFor([&] { return manager.GetActiveTaskCount() == 0; }), "owned task retired");
    panel.reset();
    Pump(manager);
    Check(panel_completions == 0, "completion for a destroyed owner is discarded at the pump");

    // Posted work for an expired owner is skipped too.
    std::atomic<int> posted = 0;
    {
        auto transient = std::make_shared<int>(4);
        manager.PostToMainThread(std::weak_ptr<const void>(transient), [&] { ++posted; });
    }
    Pump(manager);
    Check(posted == 0, "owned PostToMainThread is skipped once the owner is gone");

    // Unowned behaviour is unchanged.
    std::atomic<int> unowned_posted = 0;
    manager.PostToMainThread([&] { ++unowned_posted; });
    Pump(manager);
    Check(unowned_posted == 1, "unowned PostToMainThread still delivers");
    Check(manager.Shutdown(2s).drained, "owner scenario shuts down cleanly");
}

// 8b: RunAsync's owner parameter (used for project-session and console work)
// binds exactly like BindOwner: the session end cancels queued work and the
// finished session receives nothing.
void RunAsyncOwnerParameterScopesProjectSession() {
    auto& manager = AsyncTaskManager::Instance();
    manager.Initialize(1);
    auto session = std::make_shared<int>(5);
    std::atomic<int> bodies = 0, completions = 0;
    uint64_t id = 0;
    {
        Blocker blocker(manager);
        id = manager.RunAsync("project venv", [&](auto&) { ++bodies; }, nullptr,
            [&](bool, const std::string&) { ++completions; },
            std::weak_ptr<const void>(session));
        // Mirrors ProjectManager::EndProjectSession: cancel, then drop the token.
        Check(manager.CancelOwnedBy(session) == 1, "session end cancels its queued task");
        session.reset();
        blocker.Release();
        Check(WaitFor([&] { return manager.GetActiveTaskCount() == 0; }), "session task retired");
    }
    Pump(manager);
    Check(bodies == 0, "cancelled session task never ran");
    Check(manager.GetTask(id)->GetState() == TaskState::Cancelled, "session task retires Cancelled");
    Check(completions == 0, "ended session receives no completion");

    // An expired owner passed to RunAsync leaves the task unbound (it runs
    // and delivers) instead of silently binding to nothing.
    std::weak_ptr<const void> gone;
    { auto transient = std::make_shared<int>(6); gone = transient; }
    std::atomic<int> delivered = 0;
    manager.RunAsync("expired owner at submit", [](auto&) {}, nullptr,
        [&](bool ok, const std::string&) { if (ok) ++delivered; }, gone);
    Check(WaitFor([&] { manager.ProcessCompletedCallbacks(); return delivered.load() == 1; }),
          "expired owner at submit behaves as unowned");
    Check(manager.Shutdown(2s).drained, "session scenario shuts down cleanly");
}

// 9: after Shutdown a plain Submit re-initializes, as existing callers expect.
void SubmitAfterShutdownRestarts() {
    auto& manager = AsyncTaskManager::Instance();
    std::atomic<int> delivered = 0;
    manager.RunAsync("after shutdown", [](auto&) {}, nullptr,
        [&](bool ok, const std::string&) { if (ok) ++delivered; });
    Check(WaitFor([&] { manager.ProcessCompletedCallbacks(); return delivered.load() == 1; }),
          "submit after shutdown re-initializes and delivers");
    Check(manager.Shutdown(2s).drained, "restarted generation shuts down cleanly");
    Check(manager.Shutdown(2s).drained, "a second shutdown is a no-op");
}
}  // namespace

int main() {
    TrainingTraceSettings settings;
    settings.persist_enabled = false;
    TrainingTraceCollector::Instance().Configure(settings);
    TrainingTraceCollector::Instance().StartRun("task-lifecycle-regression");
    try {
        ShutdownRetiresQueuedAndStopsRunning();
        CompletionQueuedBeforeShutdownIsDiscarded();
        SubmitDuringShutdownIsRejected();
        UncooperativeTaskIsReleasedNotAwaited();
        OwnerScopedLifetime();
        RunAsyncOwnerParameterScopesProjectSession();
        SubmitAfterShutdownRestarts();
        std::cout << "PASS: " << checks << " task lifecycle checks\n";
        return 0;
    } catch (const std::exception& e) {
        AsyncTaskManager::Instance().Shutdown(2s);
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
