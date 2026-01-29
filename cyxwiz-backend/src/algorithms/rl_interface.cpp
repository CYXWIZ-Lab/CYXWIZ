#include "cyxwiz/rl_interface.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace cyxwiz {

// ============================================================================
// ReplayBuffer
// ============================================================================

ReplayBuffer::ReplayBuffer(size_t capacity, unsigned int seed)
    : capacity_(capacity), rng_(seed) {
    if (capacity == 0) {
        throw std::invalid_argument("ReplayBuffer capacity must be > 0");
    }
}

void ReplayBuffer::Push(const RLTransition& transition) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.size() >= capacity_) {
        buffer_.pop_front();
    }
    buffer_.push_back(transition);
}

void ReplayBuffer::Push(const std::vector<float>& state,
                         const std::vector<float>& action,
                         float reward,
                         const std::vector<float>& next_state,
                         bool done) {
    RLTransition t;
    t.state = state;
    t.action = action;
    t.reward = reward;
    t.next_state = next_state;
    t.done = done;
    Push(t);
}

RLBatch ReplayBuffer::Sample(size_t batch_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.size() < batch_size) {
        throw std::runtime_error("Not enough samples in replay buffer: have " +
                                 std::to_string(buffer_.size()) + ", need " +
                                 std::to_string(batch_size));
    }

    // Generate random indices
    std::vector<size_t> indices(buffer_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);
    indices.resize(batch_size);

    RLBatch batch;
    batch.size = batch_size;
    batch.states.reserve(batch_size);
    batch.actions.reserve(batch_size);
    batch.rewards.reserve(batch_size);
    batch.next_states.reserve(batch_size);
    batch.dones.reserve(batch_size);

    for (size_t idx : indices) {
        const auto& t = buffer_[idx];
        batch.states.push_back(t.state);
        batch.actions.push_back(t.action);
        batch.rewards.push_back(t.reward);
        batch.next_states.push_back(t.next_state);
        batch.dones.push_back(t.done);
    }

    return batch;
}

size_t ReplayBuffer::Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
}

void ReplayBuffer::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
}

// ============================================================================
// EpsilonSchedule
// ============================================================================

EpsilonSchedule::EpsilonSchedule(float start, float end, int decay_steps)
    : start_(start), end_(end), decay_steps_(decay_steps), current_(start) {
}

void EpsilonSchedule::Step() {
    step_++;
    if (step_ >= decay_steps_) {
        current_ = end_;
    } else {
        float fraction = static_cast<float>(step_) / static_cast<float>(decay_steps_);
        current_ = start_ + fraction * (end_ - start_);
    }
}

void EpsilonSchedule::Reset() {
    step_ = 0;
    current_ = start_;
}

} // namespace cyxwiz
