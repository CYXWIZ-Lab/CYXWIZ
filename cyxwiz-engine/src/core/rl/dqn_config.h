#pragma once

// Deep Q-Network's settings (TOFIX216 DQN), as Stable-Baselines3 2.7.1's DQN
// names and defaults them. The DQN node fills them, the DQN trainer reads
// them. GUI-free.

#include <cstdint>
#include <limits>

namespace cyxwiz::rl {

struct DqnConfig {
    int64_t total_timesteps = 1000000;
    int64_t buffer_size = 1000000;   // transitions kept (replay buffer)
    int64_t learning_starts = 100;   // random choices before learning
    int batch_size = 32;
    float learning_rate = 1e-4f;     // Adam for the Q network
    float gamma = 0.99f;
    float tau = 1.0f;                // target update: 1 copies the Q network, below 1 a Polyak step
    int train_freq = 4;              // environment steps between updates
    int gradient_steps = 1;          // updates each time
    int64_t target_update_interval = 10000;  // environment steps between target updates
    float exploration_fraction = 0.1f;       // share of the run over which epsilon falls
    float exploration_initial_eps = 1.0f;    // epsilon at the start
    float exploration_final_eps = 0.05f;     // epsilon from then on
    float max_grad_norm = 10.0f;     // the gradients' global norm is clipped to this
    uint64_t seed = 1;
    bool normalize_observations = false;
    // The mean return of the last 100 episodes that stops the run; NaN: none
    float target_return = std::numeric_limits<float>::quiet_NaN();
    int64_t report_every = 1000;     // environment steps between the dashboard's reports (not an SB3 setting)
};

}  // namespace cyxwiz::rl
