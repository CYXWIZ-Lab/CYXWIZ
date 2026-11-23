#pragma once

// Twin Delayed DDPG's settings (TOFIX216 TD3), as Stable-Baselines3 2.7.1's
// TD3 names and defaults them, except the exploration noise: SB3 explores
// with none unless told; the node's default is RL Zoo's Gaussian 0.1. The TD3
// node fills them, the TD3 trainer reads them. GUI-free.

#include <cstdint>
#include <limits>

namespace cyxwiz::rl {

struct Td3Config {
    int64_t total_timesteps = 1000000;
    int64_t buffer_size = 1000000;   // transitions kept (replay buffer)
    int64_t learning_starts = 100;   // uniform random actions before learning
    int batch_size = 256;
    float learning_rate = 1e-3f;     // Adam for the actor and the critics
    float gamma = 0.99f;
    float tau = 0.005f;              // Polyak update of the target actor and critics
    int train_freq = 1;              // environment steps between updates
    int gradient_steps = 1;          // updates each time
    int policy_delay = 2;            // critic updates per actor (and target) update
    bool action_noise = true;        // Gaussian exploration noise (NormalActionNoise); false: none
    float action_noise_std = 0.1f;   // its std, in the [-1, 1] action scale
    float target_policy_noise = 0.2f;  // target smoothing: noise on the target actor's actions
    float target_noise_clip = 0.5f;    // ... clipped to +-this
    uint64_t seed = 1;
    bool normalize_observations = false;
    // The mean return of the last 100 episodes that stops the run; NaN: none
    float target_return = std::numeric_limits<float>::quiet_NaN();
    int64_t report_every = 1000;     // environment steps between the dashboard's reports (not an SB3 setting)
};

}  // namespace cyxwiz::rl
