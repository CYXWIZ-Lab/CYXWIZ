#pragma once

// Pieces every Engine learner shares (PPO, SAC, TD3, DQN): its own host
// random draws and the fused-kernel form of a small network. GUI-free.

#include "../executable_model.h"

#include <cstdint>
#include <memory>
#include <string>

namespace cyxwiz {
class FusedMlp;
}

namespace cyxwiz::rl {

// SplitMix64: a learner's own host draws (action, sample and noise seeds,
// minibatch orders, DQN's random choices), the same on every compiler
class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}
    uint64_t Next() {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

private:
    uint64_t state_;
};

// A draw's bits as a uniform value in [0, 1): its top 53 bits, a double's precision
inline double UniformFromBits(uint64_t bits) { return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0); }

// A small Dense/activation network on the device provider's fused kernels:
// a few launches per forward and backward instead of several per layer.
// Null, with the reason in why, when it cannot run there.
std::unique_ptr<FusedMlp> FuseNetwork(IExecutableModel& model, std::string& why);

}  // namespace cyxwiz::rl
