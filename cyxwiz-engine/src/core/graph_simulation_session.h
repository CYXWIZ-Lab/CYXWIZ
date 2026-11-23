#pragma once

// GraphSimulationSession — one live graph simulation run (TOFIX105 / TOFIX195 S2).
//
// Owns the GraphExecutor, its worker thread, the run state and the history of
// every Signal Scope. History is taken on the worker after each tick, with the
// tick's time, so it does not depend on what the UI draws. All public members
// are safe to call from the UI thread while the worker runs.

#include "graph_executor.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cyxwiz {

enum class SimulationState { Stopped, Running, Paused, Failed };

class GraphSimulationSession {
public:
    static constexpr int kDefaultScopeWindow = 500;
    // How many samples a Scope node keeps: its window_size within 10..100000,
    // or the default when it has none or it is no number.
    static std::size_t ScopeWindow(const ::gui::MLNode& scope);

    GraphSimulationSession() = default;
    ~GraphSimulationSession();
    GraphSimulationSession(const GraphSimulationSession&) = delete;
    GraphSimulationSession& operator=(const GraphSimulationSession&) = delete;

    // Builds a copy of the graph and starts ticking at rate_hz. A run already
    // going is stopped first. False when the graph does not build: the state
    // is then Failed with the reason in GetError(). held_for_run (the plugins'
    // run lease) is kept until the worker has finished and is dropped on the
    // calling thread by Stop, Poll or the next Start.
    bool Start(const std::vector<::gui::MLNode>& nodes,
               const std::vector<::gui::NodeLink>& links,
               PluginEvalCallback plugin_eval,
               float rate_hz = 60.0f,
               std::shared_ptr<void> held_for_run = nullptr);
    // A run refused before it started (a plugin said no): Failed with the reason.
    void FailToStart(const std::string& reason);
    void Pause();
    void Resume();
    // Pauses a running graph, then runs one tick. Returns at once; the
    // worker runs the tick.
    void Step();
    // Ends the run and joins the worker. History stays until the next Start
    // or ClearHistory.
    void Stop();
    // Call on the UI thread every frame: after a failed tick it joins the
    // finished worker and drops what was held for the run.
    void Poll();

    SimulationState GetState() const { return state_.load(); }
    // Failed before the first tick (refused, or the graph did not build), not
    // during the run.
    bool FailedBeforeRunning() const { return failed_before_running_.load(); }
    // Running or paused: the worker exists and holds the graph.
    bool IsActive() const;
    std::string GetError() const;
    float GetSimTime() const;
    std::uint64_t GetRunId() const { return run_id_.load(); }
    // Ticks a second of the last run started; 0 before the first.
    float GetRateHz() const { return rate_hz_.load(); }

    bool TryGetScalar(int pin_id, float& value) const;
    // A live edit from Properties. A Scope's window_size also resizes its history.
    bool SetNodeParameter(int node_id, const std::string& key, const std::string& value);

    // A Scope's history: times and values of the same ticks. False when the
    // node is no Scope of the last run.
    bool CopyScope(int node_id, std::vector<float>& times, std::vector<float>& values) const;
    void ClearScope(int node_id);
    // How many samples a Scope keeps (its window size); 0 when the node is no
    // Scope of the last run.
    std::size_t GetScopeCapacity(int node_id) const;
    // Empty every Scope; a run goes on sampling into them.
    void ClearScopes();
    // Forget every Scope (a different graph was opened).
    void ClearHistory();

    // Runs one tick on the calling thread with no worker: tests drive the
    // session deterministically through this after BuildForTest.
    bool BuildForTest(const std::vector<::gui::MLNode>& nodes,
                      const std::vector<::gui::NodeLink>& links,
                      PluginEvalCallback plugin_eval);
    bool TickForTest(float dt);

private:
    struct ScopeChannel {
        int node_id = 0;
        int input_pin = 0;
        std::size_t capacity = kDefaultScopeWindow;
        std::deque<float> times;
        std::deque<float> values;
    };

    bool Build(const std::vector<::gui::MLNode>& nodes,
               const std::vector<::gui::NodeLink>& links,
               PluginEvalCallback plugin_eval);
    bool TickAndSample(float dt);
    void Fail(const std::string& reason);
    void WorkerLoop(float rate_hz);
    void JoinWorker();

    std::unique_ptr<GraphExecutor> executor_;
    std::thread worker_;
    std::shared_ptr<void> held_for_run_;

    std::atomic<SimulationState> state_{SimulationState::Stopped};
    std::atomic<std::uint64_t> run_id_{0};
    std::atomic<float> rate_hz_{0.0f};
    std::atomic<bool> failed_before_running_{false};

    // Guards the worker's commands and the error.
    mutable std::mutex control_mutex_;
    std::condition_variable control_cv_;
    bool stop_requested_ = false;
    int steps_requested_ = 0;
    std::string error_;

    mutable std::mutex scopes_mutex_;
    std::vector<ScopeChannel> scopes_;
};

} // namespace cyxwiz
