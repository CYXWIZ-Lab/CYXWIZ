#include "core/async_task_manager.h"
#include "core/training_trace_collector.h"
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>

using namespace cyxwiz;
using namespace std::chrono_literals;
namespace {
int checks = 0;
void Check(bool ok, const char* what) { ++checks; if (!ok) throw std::runtime_error(what); }
struct Blocker {
    std::promise<void> release;
    std::future<void> entered;
    std::shared_ptr<LambdaTask> task;
    bool released = false;
    Blocker(AsyncTaskManager& manager) {
        auto ready = std::make_shared<std::promise<void>>();
        entered = ready->get_future();
        auto signal = release.get_future().share();
        task = std::make_shared<LambdaTask>("worker barrier", [ready, signal](auto& self) {
            ready->set_value();
            if (signal.wait_for(10s) != std::future_status::ready) throw std::runtime_error("barrier timeout");
            self.MarkCompleted();
        }, false);
        manager.Submit(task);
        Check(entered.wait_for(5s)==std::future_status::ready, "worker entered barrier");
    }
    void Release() { if (!released) { release.set_value(); released=true; } }
    ~Blocker() { Release(); }
};
void Drain(AsyncTaskManager& manager, std::atomic<int>& callbacks, int expected) {
    const auto deadline = std::chrono::steady_clock::now()+5s;
    while (callbacks.load()!=expected && std::chrono::steady_clock::now()<deadline) {
        manager.ProcessCompletedCallbacks();
        std::this_thread::sleep_for(1ms);
    }
    Check(callbacks==expected, "completion delivered once through UI pump");
    manager.ProcessCompletedCallbacks();
    Check(callbacks==expected, "no repeated completion");
}
void Queued(bool all, bool pre_submit, bool reentrant) {
    auto& manager=AsyncTaskManager::Instance();
    Blocker blocker(manager);
    std::atomic<int> bodies=0, completions=0, cancel_callbacks=0;
    bool success=true;
    const auto owner_thread=std::this_thread::get_id();
    bool correct_thread=false;
    auto task=std::make_shared<LambdaTask>("cancelled queued preparation", [&](auto& self) {
        ++bodies; self.MarkCompleted();
    });
    const auto id=task->GetId();
    task->SetCancellationCallback([&] {
        ++cancel_callbacks;
        if (reentrant) Check(manager.GetTask(id)!=nullptr, "cancel callback can query manager");
    });
    task->SetCompletionCallback([&](bool ok, const std::string&) {
        success=ok; correct_thread=std::this_thread::get_id()==owner_thread; ++completions;
    });
    if (pre_submit) task->RequestCancel();
    manager.Submit(task);
    if (all) manager.CancelAll(); else Check(manager.Cancel(id), "cancel finds queued task");
    Check(task->GetState()==TaskState::Pending, "task is deterministically queued");
    Check(cancel_callbacks==1, "cancel callback fires once");
    Check(completions==0, "no completion before worker retirement");
    blocker.Release();
    Drain(manager,completions,1);
    Check(bodies==0, "cancelled queued task body must not run");
    Check(task->GetState()==TaskState::Cancelled, "queued task retires as Cancelled");
    Check(!success && correct_thread, "cancelled completion reports false on UI thread");
    Check(manager.GetTask(id)==task, "cancelled task retained in history");
    Check(!manager.Cancel(id), "retired task cannot be cancelled again");
    Check(task->GetStatusMessage()=="Cancelled before execution", "clear cancellation status");
    const auto info=task->GetInfo();
    Check(info.end_time>=info.start_time && info.end_time-info.start_time<1s, "bounded skipped-task duration");
    bool started=false, cancelled=false;
    for (const auto& event:TrainingTraceCollector::Instance().Snapshot().recent_events) {
        if (event.task_id!=id) continue;
        started |= event.stage=="TaskStarted";
        cancelled |= event.stage=="TaskCancelled";
    }
    Check(!started && cancelled, "trace records cancellation without false start");
}
void Controls() {
    auto& manager=AsyncTaskManager::Instance();
    Blocker blocker(manager);
    std::atomic<int> bodies=0, completions=0;
    auto task=std::make_shared<LambdaTask>("noncancellable",[&](auto& self) { ++bodies; self.MarkCompleted(); },false);
    task->SetCompletionCallback([&](bool ok,const std::string&) { Check(ok,"noncancellable completes"); ++completions; });
    manager.Submit(task); manager.Cancel(task->GetId());
    Check(!task->IsCancelRequested(),"noncancellable flag respected");
    manager.RunAsync("ordinary success",[&](auto&) { ++bodies; },nullptr,
        [&](bool ok,const std::string&) { Check(ok,"normal RunAsync success"); ++completions; });
    manager.RunAsync("ordinary failure",[](auto&) { throw std::runtime_error("expected failure"); },nullptr,
        [&](bool ok,const std::string& error) { Check(!ok && error=="expected failure","failure preserved"); ++completions; });
    blocker.Release(); Drain(manager,completions,3); Check(bodies==2,"uncancelled bodies execute once");
    std::promise<void> running, release;
    auto ready=running.get_future(); auto signal=release.get_future().share();
    const auto id=manager.RunAsync("cooperative running cancellation",[&](auto& self) {
        running.set_value(); signal.wait_for(5s); Check(self.ShouldStop(),"running task observes cancellation");
    },nullptr,[&](bool ok,const std::string&) { Check(!ok,"running cancellation preserved"); ++completions; });
    Check(ready.wait_for(5s)==std::future_status::ready,"running task entered");
    manager.Cancel(id); release.set_value(); Drain(manager,completions,4);
    Check(manager.GetTask(id)->GetState()==TaskState::Cancelled,"running task retires cancelled");
}
}
int main(int argc,char**) {
    TrainingTraceSettings settings; settings.persist_enabled=false;
    TrainingTraceCollector::Instance().Configure(settings);
    TrainingTraceCollector::Instance().StartRun("task-cancellation-regression");
    auto& manager=AsyncTaskManager::Instance(); manager.Initialize(1);
    try {
        Queued(false,false,false);
        if (argc==1) { Queued(false,false,true); Queued(true,false,true); Queued(false,true,false); Controls(); }
        manager.Shutdown(); std::cout<<"PASS: "<<checks<<" task cancellation checks\n"; return 0;
    } catch (const std::exception& e) {
        manager.Shutdown(); std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1;
    }
}
