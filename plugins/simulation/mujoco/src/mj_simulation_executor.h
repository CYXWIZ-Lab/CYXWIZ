#pragma once

// MjSimulationExecutor — Runs the MuJoCo simulation on its own thread
//
// Manual Control: steps the loaded model in real time (or faster) with its
// current controls. Supports Play/Pause/Step.

#include "mj_env_manager.h"
#include <mutex>
#include <atomic>
#include <functional>
#include <thread>

namespace cyxwiz::plugin::mujoco {

enum class SimMode {
    Stopped,
    ManualControl   // Simulink-style: signal sources -> actuators -> step -> sensors
};

enum class SimState {
    Idle,
    Running,
    Paused,
    Stepping  // Single-step mode
};

// Run by the step loop under the physics lock (the Viewport's stress requests)
struct MjStepHooks {
    // Before every step: the pushes on the model
    std::function<void(const mjModel* m, mjData* d)> before_step;
    // An episode ended: true holds the next one until ContinueEpisode (its
    // ground is built on the UI thread first); false starts it at once
    std::function<bool()> episode_ended;
};

// Simulation metrics exposed to UI
struct SimMetrics {
    double sim_time = 0.0;      // Simulated time (seconds)
    double wall_time = 0.0;     // Wall clock time (seconds)
    double real_time_factor = 0.0;
    int step_count = 0;
};

class MjSimulationExecutor {
public:
    MjSimulationExecutor() = default;
    ~MjSimulationExecutor();

    // Non-copyable
    MjSimulationExecutor(const MjSimulationExecutor&) = delete;
    MjSimulationExecutor& operator=(const MjSimulationExecutor&) = delete;

    // --- Lifecycle ---
    // Bind to an environment manager (must already have model loaded)
    void SetEnvManager(MjEnvManager* env) { env_ = env; }
    // Set before the first start (the step loop reads them without a lock)
    void SetStepHooks(MjStepHooks hooks) { hooks_ = std::move(hooks); }
    // An ended episode waits (episode_ended said so). Read and cleared under
    // the physics lock; ContinueEpisode after the caller started the next one.
    bool HoldsEpisode() const { return held_.load(); }
    void ContinueEpisode() { held_ = false; }

    // --- Manual Control Mode ---
    void StartManualControl();
    void Stop();
    void Pause();
    void Resume();
    void SingleStep();  // Advance one timestep

    // --- State queries ---
    SimMode GetMode() const { return mode_; }
    SimState GetState() const { return state_; }
    SimMetrics GetMetrics() const;
    bool IsRunning() const { return state_ == SimState::Running || state_ == SimState::Stepping; }

    // Simulation rate control
    void SetRealtimeFactor(float factor) { realtime_factor_ = factor; }  // 1.0 = real-time, 0 = as fast as possible
    float GetRealtimeFactor() const { return realtime_factor_; }

private:
    void ManualControlLoop();
    void ExecuteOneStep();

    MjEnvManager* env_ = nullptr;
    MjStepHooks hooks_;
    std::atomic<bool> held_{false};

    std::atomic<SimMode> mode_{SimMode::Stopped};
    std::atomic<SimState> state_{SimState::Idle};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> do_single_step_{false};
    std::atomic<float> realtime_factor_{1.0f};

    // Thread for simulation loop
    std::thread sim_thread_;

    // Metrics (written by the sim thread, read by the viewport)
    mutable std::mutex io_mutex_;
    SimMetrics metrics_;
};

} // namespace cyxwiz::plugin::mujoco
