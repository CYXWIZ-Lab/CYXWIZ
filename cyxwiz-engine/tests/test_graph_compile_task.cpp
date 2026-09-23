#include "core/graph_compile_task.h"
#include "core/debug_run_paths.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace std::chrono_literals;
void Require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
template<class Condition> void Await(Condition condition, bool pump = true) {
    const auto limit = std::chrono::steady_clock::now()+5s;
    while (!condition()) {
        Require(std::chrono::steady_clock::now()<limit,"task timed out");
        if (pump) cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
        std::this_thread::sleep_for(1ms);
    }
}
}
void RunGraphCompileTaskTests() {
    using namespace cyxwiz;
    const ScopedDebugRunRootOverrideForTesting debug(std::filesystem::current_path()/"compile_task_trace");
    auto& tasks=AsyncTaskManager::Instance(); tasks.Initialize(1);
    const auto ui=std::this_thread::get_id();
    std::atomic<bool> entered=false; bool complete=false; int heartbeats=0;
    gui::MLNode node; node.id=7;
    std::vector<gui::MLNode> nodes{node};
    const auto start=std::chrono::steady_clock::now();
    auto request=SubmitGraphCompileTask(nodes,{},
        [&](bool ok,const std::string&,TrainingConfiguration config) {
            Require(std::this_thread::get_id()==ui,"completion must run on UI thread");
            Require(ok&&config.input_size==7,"snapshot result"); complete=true;
        },[&](const auto& snapshot,const auto&) {
            Require(std::this_thread::get_id()!=ui,"compile must run on worker");
            entered=true; std::this_thread::sleep_for(400ms);
            TrainingConfiguration config;config.is_valid=true;config.input_size=snapshot.at(0).id;
            return config;
        });
    const auto submit_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    Require(submit_ms<200,"submission must not wait for compile");
    nodes[0].id=99;
    Await([&]{return entered.load();});
    for(int i=0;i<10;++i) {
        tasks.PostToMainThread([&]{++heartbeats;});tasks.ProcessCompletedCallbacks();
        std::this_thread::sleep_for(5ms);
    }
    Require(heartbeats==10&&!complete,"main callback pumping must remain responsive during compile");
    Await([&]{return complete;});
    for(int mode=0;mode<3;++mode) {
        bool delivered=false;
        auto task=SubmitGraphCompileTask({}, {},
            [&](bool ok,const std::string& error,TrainingConfiguration) {
                Require(std::this_thread::get_id()==ui&&!ok&&!error.empty(),"cancel/error delivered on UI");
                delivered=true;
            },[mode](const auto&,const auto&) -> TrainingConfiguration {
                std::this_thread::sleep_for(20ms);
                if(mode==2) throw std::runtime_error("fixture compile failure");
                TrainingConfiguration config;config.is_valid=true;return config;
            });
        if(mode==0) task->RequestCancel();
        if(mode==1) {
            Await([&]{return task->GetState()==TaskState::Completed;},false);
            task->RequestCancel(); // Terminal work, UI handoff still queued.
        }
        Await([&]{return delivered;});
    }
    // Expired UI ownership can discard a queued completion without dereference.
    auto life=std::make_shared<int>(0);std::weak_ptr<int> weak=life; bool touched=false;
    auto dead=SubmitGraphCompileTask({}, {},
        [weak,&touched](bool,const std::string&,TrainingConfiguration) { if(!weak.expired())touched=true; },
        [](const auto&,const auto&) {TrainingConfiguration c;c.is_valid=true;return c;});
    life.reset();Await([&]{return dead->GetState()==TaskState::Completed;},false);
    tasks.Shutdown();tasks.ProcessCompletedCallbacks();
    Require(!touched,"destroyed UI must not receive results");
    std::cout<<"PASS: background compilation, owned graph snapshot, UI heartbeat, cancellation, failure, lifetime; submit_ms="<<submit_ms<<"\n";
}
