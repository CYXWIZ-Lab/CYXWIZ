#pragma once

// MjSimulationExecutor — Runs MuJoCo simulation driven by node graph signals
//
// Two modes:
//   1. Manual Control: Reads actuator values from connected signal sources,
//      calls mj_step(), pushes sensor outputs. Supports Play/Pause/Step.
//   2. RL Training: Wraps env in Gymnasium-style loop, agent receives obs,
//      produces actions, trains policy. Uses MjEnvManager's Step/Reset API.

#include "mj_env_manager.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <thread>
#include <functional>

namespace cyxwiz::plugin::mujoco {

enum class SimMode {
    Stopped,
    ManualControl,  // Simulink-style: signal sources -> actuators -> step -> sensors
    RLTraining      // RL loop: env.step(action) with agent policy
};

enum class SimState {
    Idle,
    Running,
    Paused,
    Stepping  // Single-step mode
};

// Per-actuator value from the node graph (populated by engine before each step)
struct ActuatorInput {
    std::string name;
    float value = 0.0f;
};

// Per-sensor output pushed back to the node graph after each step
struct SensorOutput {
    std::string name;
    std::vector<float> values;  // dim >= 1
};

// Simulation metrics exposed to UI
struct SimMetrics {
    double sim_time = 0.0;      // Simulated time (seconds)
    double wall_time = 0.0;     // Wall clock time (seconds)
    double real_time_factor = 0.0;
    int step_count = 0;
    int episode_count = 0;
    float episode_reward = 0.0f;
    float mean_reward = 0.0f;
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

    // --- Manual Control Mode ---
    void StartManualControl();
    void Stop();
    void Pause();
    void Resume();
    void SingleStep();  // Advance one timestep

    // Set actuator values (called by engine each frame before step)
    void SetActuatorInputs(const std::vector<ActuatorInput>& inputs);

    // Get sensor outputs (called by engine each frame after step)
    std::vector<SensorOutput> GetSensorOutputs() const;

    // --- RL Training Mode ---
    void StartRLTraining(int total_timesteps = 1000000);

    // --- State queries ---
    SimMode GetMode() const { return mode_; }
    SimState GetState() const { return state_; }
    SimMetrics GetMetrics() const;
    bool IsRunning() const { return state_ == SimState::Running || state_ == SimState::Stepping; }

    // Simulation rate control
    void SetRealtimeFactor(float factor) { realtime_factor_ = factor; }  // 1.0 = real-time, 0 = as fast as possible
    float GetRealtimeFactor() const { return realtime_factor_; }

    // Callback: called after each step with current sensor data
    using StepCallback = std::function<void(const std::vector<SensorOutput>& sensors, const SimMetrics& metrics)>;
    void SetStepCallback(StepCallback cb) { step_callback_ = std::move(cb); }

private:
    void ManualControlLoop();
    void RLTrainingLoop(int total_timesteps);
    void ExecuteOneStep();

    MjEnvManager* env_ = nullptr;

    std::atomic<SimMode> mode_{SimMode::Stopped};
    std::atomic<SimState> state_{SimState::Idle};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> do_single_step_{false};
    std::atomic<float> realtime_factor_{1.0f};

    // Thread for simulation loop
    std::thread sim_thread_;

    // Actuator inputs (written by engine, read by sim thread)
    mutable std::mutex io_mutex_;
    std::vector<ActuatorInput> actuator_inputs_;
    std::vector<SensorOutput> sensor_outputs_;
    SimMetrics metrics_;

    StepCallback step_callback_;
};

} // namespace cyxwiz::plugin::mujoco
