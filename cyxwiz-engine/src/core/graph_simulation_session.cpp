#include "graph_simulation_session.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace cyxwiz {

namespace {

constexpr int kMinScopeWindow = 10;
constexpr int kMaxScopeWindow = 100000;

std::size_t ScopeWindow(const std::string& text) {
    try {
        std::size_t used = 0;
        const int value = std::stoi(text, &used);
        if (used == text.size()) return static_cast<std::size_t>(std::clamp(value, kMinScopeWindow, kMaxScopeWindow));
    } catch (const std::exception&) {
    }
    return GraphSimulationSession::kDefaultScopeWindow;
}

} // namespace

GraphSimulationSession::~GraphSimulationSession() {
    Stop();
}

bool GraphSimulationSession::IsActive() const {
    const auto state = state_.load();
    return state == SimulationState::Running || state == SimulationState::Paused;
}

std::string GraphSimulationSession::GetError() const {
    std::lock_guard lock(control_mutex_);
    return error_;
}

float GraphSimulationSession::GetSimTime() const {
    return executor_ ? executor_->GetSimTime() : 0.0f;
}

bool GraphSimulationSession::Build(const std::vector<::gui::MLNode>& nodes,
                                   const std::vector<::gui::NodeLink>& links,
                                   PluginEvalCallback plugin_eval) {
    JoinWorker();
    {
        std::lock_guard lock(control_mutex_);
        stop_requested_ = false;
        steps_requested_ = 0;
        error_.clear();
    }
    ++run_id_;

    auto executor = std::make_unique<GraphExecutor>();
    executor->SetPluginEvalCallback(std::move(plugin_eval));
    if (!executor->Build(nodes, links)) {
        executor_.reset();
        Fail(executor->GetError());
        return false;
    }
    executor_ = std::move(executor);

    std::vector<ScopeChannel> scopes;
    for (const auto& node : nodes) {
        if (node.type != ::gui::NodeType::SignalScope || node.inputs.size() != 1) continue;
        ScopeChannel channel;
        channel.node_id = node.id;
        channel.input_pin = node.inputs[0].id;
        const auto window = node.parameters.find("window_size");
        if (window != node.parameters.end()) channel.capacity = ScopeWindow(window->second);
        scopes.push_back(std::move(channel));
    }
    std::lock_guard lock(scopes_mutex_);
    scopes_ = std::move(scopes);
    return true;
}

bool GraphSimulationSession::Start(const std::vector<::gui::MLNode>& nodes,
                                   const std::vector<::gui::NodeLink>& links,
                                   PluginEvalCallback plugin_eval,
                                   float rate_hz,
                                   std::shared_ptr<void> held_for_run) {
    Stop();
    if (!std::isfinite(rate_hz) || rate_hz <= 0.0f) {
        Fail("The simulation rate must be a positive number of ticks a second");
        return false;
    }
    if (!Build(nodes, links, std::move(plugin_eval))) {
        spdlog::error("Graph simulation not started: {}", GetError());
        return false;
    }
    held_for_run_ = std::move(held_for_run);
    rate_hz_ = rate_hz;
    state_ = SimulationState::Running;
    spdlog::info("Graph simulation started at {:.0f} ticks a second", rate_hz);
    worker_ = std::thread([this, rate_hz]() { WorkerLoop(rate_hz); });
    return true;
}

void GraphSimulationSession::Pause() {
    auto expected = SimulationState::Running;
    state_.compare_exchange_strong(expected, SimulationState::Paused);
}

void GraphSimulationSession::Resume() {
    auto expected = SimulationState::Paused;
    if (state_.compare_exchange_strong(expected, SimulationState::Running)) control_cv_.notify_all();
}

void GraphSimulationSession::Step() {
    // A running graph pauses first
    auto running = SimulationState::Running;
    state_.compare_exchange_strong(running, SimulationState::Paused);
    if (state_.load() != SimulationState::Paused) return;
    {
        std::lock_guard lock(control_mutex_);
        ++steps_requested_;
    }
    control_cv_.notify_all();
}

void GraphSimulationSession::Stop() {
    const bool was_active = IsActive();
    {
        std::lock_guard lock(control_mutex_);
        stop_requested_ = true;
    }
    control_cv_.notify_all();
    JoinWorker();
    held_for_run_.reset();
    if (was_active) {
        state_ = SimulationState::Stopped;
        spdlog::info("Graph simulation stopped at t={:.2f}s", GetSimTime());
    }
}

void GraphSimulationSession::Poll() {
    if (state_.load() != SimulationState::Failed || !worker_.joinable()) return;
    JoinWorker();
    held_for_run_.reset();
}

void GraphSimulationSession::FailToStart(const std::string& reason) {
    Stop();
    Fail(reason);
    spdlog::error("Graph simulation not started: {}", reason);
}

void GraphSimulationSession::JoinWorker() {
    if (worker_.joinable()) worker_.join();
}

void GraphSimulationSession::Fail(const std::string& reason) {
    {
        std::lock_guard lock(control_mutex_);
        error_ = reason;
    }
    state_ = SimulationState::Failed;
}

bool GraphSimulationSession::TickAndSample(float dt) {
    // A sample carries the time its tick evaluated at. The worker is the only
    // writer of pin values and time, so what is read after the tick belongs to it.
    const float time = executor_->GetSimTime();
    if (!executor_->Tick(dt)) {
        Fail(executor_->GetError());
        return false;
    }

    std::lock_guard lock(scopes_mutex_);
    for (auto& scope : scopes_) {
        NodeValue value;
        if (!executor_->TryGetPinValue(scope.input_pin, value)) continue;
        const auto* scalar = std::get_if<float>(&value);
        if (!scalar) continue;
        scope.times.push_back(time);
        scope.values.push_back(*scalar);
        while (scope.times.size() > scope.capacity) {
            scope.times.pop_front();
            scope.values.pop_front();
        }
    }
    return true;
}

void GraphSimulationSession::WorkerLoop(float rate_hz) {
    const float dt = 1.0f / rate_hz;
    const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<float>(dt));
    auto next = std::chrono::steady_clock::now();
    for (;;) {
        bool step_once = false;
        {
            std::unique_lock lock(control_mutex_);
            control_cv_.wait(lock, [&]() {
                return stop_requested_ || steps_requested_ > 0 || state_.load() != SimulationState::Paused;
            });
            if (stop_requested_) return;
            if (state_.load() == SimulationState::Paused) {
                --steps_requested_;
                step_once = true;
            }
        }
        if (!TickAndSample(dt)) {
            spdlog::warn("Graph simulation failed: {}", GetError());
            return;
        }
        if (step_once) {
            next = std::chrono::steady_clock::now();
            continue;
        }
        next += period;
        const auto now = std::chrono::steady_clock::now();
        if (next < now) next = now;  // a slow tick does not make the next ones race
        std::unique_lock lock(control_mutex_);
        if (control_cv_.wait_until(lock, next, [&]() { return stop_requested_; })) return;
    }
}

bool GraphSimulationSession::TryGetScalar(int pin_id, float& value) const {
    if (!executor_) return false;
    NodeValue node_value;
    if (!executor_->TryGetPinValue(pin_id, node_value)) return false;
    const auto* scalar = std::get_if<float>(&node_value);
    if (!scalar) return false;
    value = *scalar;
    return true;
}

bool GraphSimulationSession::SetNodeParameter(int node_id, const std::string& key, const std::string& value) {
    if (key == "window_size") {
        std::lock_guard lock(scopes_mutex_);
        for (auto& scope : scopes_) {
            if (scope.node_id != node_id) continue;
            scope.capacity = ScopeWindow(value);
            while (scope.times.size() > scope.capacity) {
                scope.times.pop_front();
                scope.values.pop_front();
            }
        }
    }
    return executor_ && executor_->SetNodeParameter(node_id, key, value);
}

bool GraphSimulationSession::CopyScope(int node_id, std::vector<float>& times, std::vector<float>& values) const {
    std::lock_guard lock(scopes_mutex_);
    for (const auto& scope : scopes_) {
        if (scope.node_id != node_id) continue;
        times.assign(scope.times.begin(), scope.times.end());
        values.assign(scope.values.begin(), scope.values.end());
        return true;
    }
    return false;
}

void GraphSimulationSession::ClearScope(int node_id) {
    std::lock_guard lock(scopes_mutex_);
    for (auto& scope : scopes_) {
        if (scope.node_id != node_id) continue;
        scope.times.clear();
        scope.values.clear();
    }
}

std::size_t GraphSimulationSession::GetScopeCapacity(int node_id) const {
    std::lock_guard lock(scopes_mutex_);
    for (const auto& scope : scopes_)
        if (scope.node_id == node_id) return scope.capacity;
    return 0;
}

void GraphSimulationSession::ClearScopes() {
    std::lock_guard lock(scopes_mutex_);
    for (auto& scope : scopes_) {
        scope.times.clear();
        scope.values.clear();
    }
}

void GraphSimulationSession::ClearHistory() {
    std::lock_guard lock(scopes_mutex_);
    scopes_.clear();
}

bool GraphSimulationSession::BuildForTest(const std::vector<::gui::MLNode>& nodes,
                                          const std::vector<::gui::NodeLink>& links,
                                          PluginEvalCallback plugin_eval) {
    Stop();
    return Build(nodes, links, std::move(plugin_eval));
}

bool GraphSimulationSession::TickForTest(float dt) {
    return executor_ && !worker_.joinable() && TickAndSample(dt);
}

} // namespace cyxwiz
