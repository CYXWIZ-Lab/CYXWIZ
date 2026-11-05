#include "mj_simulation_executor.h"
#include <spdlog/spdlog.h>
#include <mujoco/mujoco.h>
#include <chrono>
#include <cmath>

namespace cyxwiz::plugin::mujoco {

MjSimulationExecutor::~MjSimulationExecutor() {
    Stop();
}

void MjSimulationExecutor::StartManualControl() {
    Stop();  // Ensure any previous session is cleaned up

    if (!env_ || !env_->IsLoaded()) {
        spdlog::error("SimExecutor: Cannot start — no model loaded");
        return;
    }

    should_stop_ = false;
    mode_ = SimMode::ManualControl;
    state_ = SimState::Running;

    {
        std::lock_guard lock(io_mutex_);
        metrics_ = {};
    }

    env_->Reset();

    sim_thread_ = std::thread([this]() { ManualControlLoop(); });
    spdlog::info("SimExecutor: Manual control started");
}

void MjSimulationExecutor::Stop() {
    should_stop_ = true;
    state_ = SimState::Idle;
    if (sim_thread_.joinable()) {
        sim_thread_.join();
    }
    mode_ = SimMode::Stopped;
    spdlog::info("SimExecutor: Stopped");
}

void MjSimulationExecutor::Pause() {
    if (state_ == SimState::Running) {
        state_ = SimState::Paused;
        spdlog::info("SimExecutor: Paused");
    }
}

void MjSimulationExecutor::Resume() {
    if (state_ == SimState::Paused) {
        state_ = SimState::Running;
        spdlog::info("SimExecutor: Resumed");
    }
}

void MjSimulationExecutor::SingleStep() {
    if (state_ == SimState::Paused || state_ == SimState::Idle) {
        if (state_ == SimState::Idle && mode_ == SimMode::Stopped) {
            // Auto-start in paused manual control
            StartManualControl();
            Pause();
        }
        do_single_step_ = true;
    }
}

SimMetrics MjSimulationExecutor::GetMetrics() const {
    std::lock_guard lock(io_mutex_);
    return metrics_;
}

void MjSimulationExecutor::ManualControlLoop() {
    using clock = std::chrono::high_resolution_clock;
    auto wall_start = clock::now();

    const mjModel* m = env_->GetModel();
    double timestep = m->opt.timestep;
    int frame_skip = env_->GetFrameSkip();

    while (!should_stop_) {
        // Paused — wait for resume or single-step
        if (state_ == SimState::Paused) {
            if (do_single_step_.exchange(false)) {
                ExecuteOneStep();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            continue;
        }

        auto step_start = clock::now();

        ExecuteOneStep();

        // Real-time pacing
        float rtf = realtime_factor_.load();
        if (rtf > 0.0f) {
            double target_dt = (timestep * frame_skip) / rtf;
            auto elapsed = std::chrono::duration<double>(clock::now() - step_start).count();
            if (elapsed < target_dt) {
                auto sleep_us = static_cast<int64_t>((target_dt - elapsed) * 1e6);
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            }
        }

        // Update wall time
        {
            std::lock_guard lock(io_mutex_);
            metrics_.wall_time = std::chrono::duration<double>(clock::now() - wall_start).count();
            if (metrics_.wall_time > 0)
                metrics_.real_time_factor = metrics_.sim_time / metrics_.wall_time;
        }
    }

    state_ = SimState::Idle;
}

void MjSimulationExecutor::ExecuteOneStep() {
    if (!env_ || !env_->IsLoaded()) return;

    mjData* d = env_->GetMutableData();
    const mjModel* m = env_->GetModel();
    if (!d || !m) return;

    // Step and read the time as one physics transaction
    const int frame_skip = env_->GetFrameSkip();
    double sim_time = 0.0;
    {
        std::lock_guard phys_lock(env_->GetPhysicsMutex());
        for (int i = 0; i < frame_skip; i++) {
            mj_step(m, d);
        }
        sim_time = d->time;
    }

    std::lock_guard lock(io_mutex_);
    metrics_.step_count++;
    metrics_.sim_time = sim_time;
}

} // namespace cyxwiz::plugin::mujoco
