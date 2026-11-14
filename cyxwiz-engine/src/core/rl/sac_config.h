#pragma once

// Soft Actor-Critic's settings (TOFIX216), as Stable-Baselines3 2.7.1's SAC
// names and defaults them; the SAC node fills them, the SAC trainer reads
// them. GUI-free.

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cyxwiz::rl {

struct SacConfig {
    int64_t total_timesteps = 1000000;
    int64_t buffer_size = 1000000;   // transitions kept (replay buffer)
    int64_t learning_starts = 100;   // uniform random actions before learning
    int batch_size = 256;
    float learning_rate = 3e-4f;     // Adam for the actor, the critics and the entropy weight
    float gamma = 0.99f;
    float tau = 0.005f;              // Polyak update of the target critics
    int train_freq = 1;              // environment steps between updates
    int gradient_steps = 1;          // updates each time
    int target_update_interval = 1;  // gradient steps between target updates
    bool ent_coef_auto = true;       // learned entropy weight (ent_coef="auto")
    float ent_coef = 1.0f;           // its start when learned, else the fixed weight
    bool target_entropy_auto = true; // -actions
    float target_entropy = 0.0f;
    uint64_t seed = 1;
    bool normalize_observations = false;
    // The mean return of the last 100 episodes that stops the run; NaN: none
    float target_return = std::numeric_limits<float>::quiet_NaN();
};

// Bytes the replay buffer holds on the device: per transition the observation
// and the next one, the action, the reward and whether the episode ended
// (float32 each)
inline uint64_t ReplayBytes(int64_t buffer_size, size_t observation_size, size_t action_size) {
    if (buffer_size <= 0) return 0;
    return static_cast<uint64_t>(buffer_size) * (2 * observation_size + action_size + 2) * sizeof(float);
}

}  // namespace cyxwiz::rl
