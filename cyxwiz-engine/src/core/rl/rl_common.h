#pragma once

// Pieces every Engine learner shares (PPO, SAC, TD3, DQN): its own host
// random draws, the fused-kernel form of a small network, and the words for
// how a run computes (its compute path). GUI-free.

#include "../executable_model.h"

#include <cyxwiz/fused_mlp.h>
#include <cyxwiz/neural_provider_dispatch.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

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
// Null, with the reason, when it cannot run there: the network is not a
// fusable chain, a provider declines it, or no provider serves the device
// (FusedMlp::Refusal tells them apart).
std::unique_ptr<FusedMlp> FuseNetwork(IExecutableModel& model, FusedMlp::Refusal& refusal);

// How a learner's networks compute, and why when that is not a provider.
struct NetworkRoute {
    // "fused small-network kernels (<provider>)" or "ArrayFire operations (<why>)"
    std::string text = "ArrayFire operations";
    std::string provider_id;  // the provider that runs them; empty: ArrayFire
    std::string reason;       // why ArrayFire, as a reason code where one exists (BackendFallbackReasonName)
    std::string detail;       // why ArrayFire, in words
    // "<name>. <fix>" when the cause is a provider that is missing and can be
    // supplied ("NVIDIA provider not installed. Install the CUDA GPU pack.");
    // empty for every other cause (the network's shape, a failed call, a
    // device family no provider is offered for)
    std::string missing_provider;
};
NetworkRoute FusedNetworkRoute(const FusedMlp& fused);
// which: the network the refusal is about ("policy"), named when the cause is
// that network's own; empty when the learner has one kind of reason
NetworkRoute ArrayFireNetworkRoute(const FusedMlp::Refusal& refusal, const std::string& which = {});
// A fused call failed mid-run: ArrayFire from then on
NetworkRoute FailedNetworkRoute(const std::exception& e);

// One part of a run's compute path: its networks, its Adam step, a PPO op.
struct ComputePathPart {
    std::string what;                     // "networks", "Adam", "PPO sample"
    std::string provider_id;              // the provider that runs it; empty: ArrayFire
    bool mixed = false;                   // the provider runs it, and some calls ran on ArrayFire
    std::string reason;                   // why ArrayFire (BackendFallbackReasonName), where a code exists
    std::string detail;                   // why ArrayFire, in words
    std::string arrayfire = "ArrayFire";  // how it runs there ("ArrayFire, per tensor")
};
ComputePathPart NetworksPart(const NetworkRoute& route);
// The part of an op that goes through the provider dispatch, from what the
// run's tally counted: the provider when every call ran there, ArrayFire with
// the last reason when none did, both when some did. unasked: why it runs on
// ArrayFire when the dispatch was never asked.
ComputePathPart DispatchedPart(const char* what, const NeuralDispatchTally& tally, const char* layer, NeuralOp op,
                               const std::string& unasked = {});
// "<networks>; Adam: <where>; PPO sample, PPO loss, PPO GAE: <where>". It
// starts with the networks' text, so a path on the fused kernels starts with
// "fused". A reason the networks already gave is not repeated.
std::string FormatComputePath(const NetworkRoute& route, const std::vector<ComputePathPart>& ops);
// The networks' part first, then the ops
std::vector<ComputePathPart> ComputePathParts(const NetworkRoute& route, const std::vector<ComputePathPart>& ops);

// Call first in a handler that falls back from the fused kernels: a Strict
// run's NeuralProviderRouteError is not a failure to fall back from, so it is
// rethrown and stops the run (TOFIX233 B). Anything else returns.
void RethrowProviderRouteError(const std::exception& e);

}  // namespace cyxwiz::rl
