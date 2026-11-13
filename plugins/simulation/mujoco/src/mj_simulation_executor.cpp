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
    held_ = false;
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
    using clock = std::chrono::steady_clock;
    const auto wall_start = clock::now();

    const mjModel* m = env_->GetModel();
    const double step_seconds = m->opt.timestep * env_->GetFrameSkip();

    // Pace against a wall-clock schedule: step while the simulation is behind the clock and
    // sleep only when it is ahead. A sleep per step is rounded up to the OS timer tick
    // (~15.6 ms on Windows), which ran small timesteps far below real time.
    constexpr auto kMaxLag = std::chrono::milliseconds(250);
    auto schedule_start = clock::now();
    long long scheduled_steps = 0;
    float schedule_rtf = realtime_factor_.load();

    while (!should_stop_) {
        // An ended episode waits for its ground: no step until the UI starts the next one
        if (held_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            schedule_start = clock::now();
            scheduled_steps = 0;
            continue;
        }
        // Paused — wait for resume or single-step
        if (state_ == SimState::Paused) {
            if (do_single_step_.exchange(false)) {
                ExecuteOneStep();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            // Resuming starts a new schedule rather than catching up on the paused time
            schedule_start = clock::now();
            scheduled_steps = 0;
            continue;
        }

        const float rtf = realtime_factor_.load();
        if (rtf != schedule_rtf) {
            schedule_rtf = rtf;
            schedule_start = clock::now();
            scheduled_steps = 0;
        }

        ExecuteOneStep();
        ++scheduled_steps;

        if (rtf > 0.0f) {
            const auto due = schedule_start + std::chrono::duration_cast<clock::duration>(
                std::chrono::duration<double>(static_cast<double>(scheduled_steps) * step_seconds / rtf));
            const auto now = clock::now();
            if (due > now) {
                std::this_thread::sleep_until(due);
            } else if (now - due > kMaxLag) {
                // The model steps slower than the requested speed: run at full speed from
                // here instead of bursting to repay the backlog.
                schedule_start = now;
                scheduled_steps = 0;
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

    // Step through the environment (frame skip, episode step, reward) and read
    // the time as one physics transaction; an ended episode starts the next one.
    double sim_time = 0.0;
    {
        std::lock_guard phys_lock(env_->GetPhysicsMutex());
        if (held_.load()) return;
        mjData* d = env_->GetMutableData();
        if (!d) return;
        if (hooks_.before_step) hooks_.before_step(env_->GetModel(), d);
        std::vector<float> action = controls_;  // held controls, else zeros
        action.resize(static_cast<size_t>(env_->GetActionDim()), 0.0f);
        const StepResult result = env_->Step(action);
        if (result.terminated || result.truncated) {
            if (hooks_.episode_ended && hooks_.episode_ended()) held_ = true;
            else env_->Reset();
        }
        d = env_->GetMutableData();  // the same state (a scene change runs on the UI thread)
        sim_time = d->time;
    }

    std::lock_guard lock(io_mutex_);
    metrics_.step_count++;
    metrics_.sim_time = sim_time;
}

} // namespace cyxwiz::plugin::mujoco
