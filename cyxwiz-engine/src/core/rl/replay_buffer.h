#pragma once

// SAC's replay buffer on the training device (TOFIX216 S3, owner D3): the
// last `capacity` transitions as one ArrayFire table, written as a ring and
// sampled uniformly with replacement, as Stable-Baselines3's ReplayBuffer
// does (handle_timeout_termination: an episode cut by a time limit is not
// done; its next observation is the episode's final one). GUI-free.

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cyxwiz::rl {

class ReplayBuffer {
public:
    // On the current ArrayFire device. Throws std::runtime_error, saying how
    // much it needed, when the device cannot hold it.
    ReplayBuffer(int64_t capacity, size_t observation_size, size_t action_size);
    ~ReplayBuffer();
    ReplayBuffer(const ReplayBuffer&) = delete;
    ReplayBuffer& operator=(const ReplayBuffer&) = delete;

    // `rows` transitions (one per copy), each row-major: observations [rows,
    // obs], actions [rows, act] in [-1, 1], rewards [rows], next observations
    // [rows, obs] and terminated [rows] (1: the episode ended on its own).
    // Past the capacity the oldest rows are overwritten.
    void Add(const float* observations, const float* actions, const float* rewards, const float* next_observations,
             const uint8_t* terminated, size_t rows);

    int64_t Size() const { return size_; }
    int64_t Capacity() const { return capacity_; }
    uint64_t Bytes() const;  // what the table holds on the device

    struct Batch {
        Tensor observations;       // [B, obs]
        Tensor actions;            // [B, act]
        Tensor rewards;            // [B, 1]
        Tensor next_observations;  // [B, obs]
        Tensor dones;              // [B, 1]
    };
    // `batch` rows drawn uniformly with replacement from the stored ones;
    // one seed, one draw on every device. Throws when the buffer is empty.
    Batch Sample(size_t batch, uint64_t seed) const;

private:
    struct Table;
    std::unique_ptr<Table> table_;
    int64_t capacity_ = 0;
    size_t observation_size_ = 0;
    size_t action_size_ = 0;
    int64_t size_ = 0;
    int64_t next_ = 0;  // the row the next transition goes to
};

}  // namespace cyxwiz::rl
