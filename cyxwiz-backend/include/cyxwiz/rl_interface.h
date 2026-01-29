#pragma once

#include "api_export.h"
#include <vector>
#include <string>
#include <deque>
#include <random>
#include <mutex>
#include <cstddef>

namespace cyxwiz {

/// A single RL transition (s, a, r, s', done)
struct CYXWIZ_API RLTransition {
    std::vector<float> state;
    std::vector<float> action;
    float reward = 0.0f;
    std::vector<float> next_state;
    bool done = false;
};

/// Batch of transitions for training
struct CYXWIZ_API RLBatch {
    std::vector<std::vector<float>> states;
    std::vector<std::vector<float>> actions;
    std::vector<float> rewards;
    std::vector<std::vector<float>> next_states;
    std::vector<bool> dones;
    size_t size = 0;
};

/// Environment step result
struct CYXWIZ_API StepResult {
    std::vector<float> observation;
    float reward = 0.0f;
    bool done = false;
    bool truncated = false;
    std::string info;
};

/// Environment metadata
struct CYXWIZ_API EnvInfo {
    std::string name;
    int observation_dim = 0;
    int action_dim = 0;
    bool discrete_actions = true;
    int num_actions = 0;          // For discrete action spaces
    std::vector<float> action_low;   // For continuous action spaces
    std::vector<float> action_high;
    bool valid = false;
    std::string error_message;
};

/// Circular replay buffer for experience replay
class CYXWIZ_API ReplayBuffer {
public:
    /// Create a replay buffer with given capacity
    explicit ReplayBuffer(size_t capacity = 100000, unsigned int seed = 42);

    /// Add a transition to the buffer
    void Push(const RLTransition& transition);

    /// Add a transition from components
    void Push(const std::vector<float>& state,
              const std::vector<float>& action,
              float reward,
              const std::vector<float>& next_state,
              bool done);

    /// Sample a random batch
    RLBatch Sample(size_t batch_size);

    /// Current number of stored transitions
    size_t Size() const;

    /// Maximum capacity
    size_t Capacity() const { return capacity_; }

    /// Check if buffer has enough samples
    bool CanSample(size_t batch_size) const { return Size() >= batch_size; }

    /// Clear all stored transitions
    void Clear();

private:
    size_t capacity_;
    std::deque<RLTransition> buffer_;
    std::mt19937 rng_;
    mutable std::mutex mutex_;
};

/// Epsilon-greedy exploration schedule
class CYXWIZ_API EpsilonSchedule {
public:
    /// Create a linear epsilon decay schedule
    EpsilonSchedule(float start = 1.0f, float end = 0.01f, int decay_steps = 10000);

    /// Get current epsilon value
    float GetEpsilon() const { return current_; }

    /// Step the schedule (call once per action)
    void Step();

    /// Reset schedule to initial state
    void Reset();

    /// Get current step count
    int GetStep() const { return step_; }

private:
    float start_;
    float end_;
    int decay_steps_;
    float current_;
    int step_ = 0;
};

} // namespace cyxwiz
