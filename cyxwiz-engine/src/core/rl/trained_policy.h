#pragma once

// A trained PPO policy (TOFIX195 S5): the networks a run leaves, how to act
// with them, and their .cyxmodel. The package is the ordinary one: the RL
// graph, the weights ("policy.*", "value.*", "log_std"), the optimizers'
// states in optimizer/, and rl/policy.json for the environment, the
// observation statistics and the PPO updates. Loading rebuilds the networks
// from the graph, so the environment must be available (for a MuJoCo task,
// its plugin loaded). GUI-free.

#include "../rl_graph_compile.h"
#include "ppo_trainer.h"

#include <cyxwiz/device.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::rl {

struct TrainedPolicy {
    RLTrainingConfiguration config;  // the compiled graph it was trained from
    std::string graph_json;          // that graph as saved (the package keeps it)
    std::unique_ptr<IExecutableModel> policy;  // observations -> logits or means
    std::unique_ptr<IExecutableModel> value;   // observations -> value
    Tensor log_std;                            // [actions] of a continuous policy
    std::unique_ptr<VecNormalize> normalizer;  // the observation statistics
    std::map<std::string, OptimizerState> optimizers;
    std::vector<PpoUpdateMetrics> updates;
    // The ArrayFire device its networks live on: playback acts there
    DeviceType device_type = DeviceType::CPU;
    int device_id = 0;
    // Steps trained and the last mean return of 100 episodes
    int64_t steps = 0;
    float mean_return = std::numeric_limits<float>::quiet_NaN();
    // Held while the networks are used: playback acts on its worker while
    // Save runs on the GUI thread, and a forward pass changes the layers'
    // caches and the weights' device layout (PolicyActions, SaveTrainedPolicy)
    mutable std::mutex in_use;
};

// ArrayFire arrays belong to the backend that made them: work on a policy's
// networks happens on its device, whatever this thread was on; the thread's
// device comes back after (and nothing changes when it already matches).
// Throws when the device does not start.
class PolicyDeviceScope {
public:
    explicit PolicyDeviceScope(const TrainedPolicy& p);
    PolicyDeviceScope(DeviceType type, int id);
    ~PolicyDeviceScope();
    PolicyDeviceScope(const PolicyDeviceScope&) = delete;
    PolicyDeviceScope& operator=(const PolicyDeviceScope&) = delete;

private:
    std::optional<std::pair<DeviceType, int>> previous_;
    bool switched_ = false;
};

// The two networks of a compiled RL graph: its layers plus each head's output
// layer, initialised as Stable-Baselines3 does (from the current ArrayFire
// random engine). Null with the reason when a network cannot be built.
bool BuildRLNetworks(const RLTrainingConfiguration& config, std::unique_ptr<IExecutableModel>& policy,
                     std::unique_ptr<IExecutableModel>& value, std::string& error);

// The policy network alone, made the same way (a training preview's own
// copy, which plays and never trains). Null with the reason.
std::unique_ptr<IExecutableModel> BuildRLPolicyNetwork(const RLTrainingConfiguration& config, std::string& error);

// The policy network alone, made the same way (a training preview's own
// copy, which plays and never trains). Null with the reason.
std::unique_ptr<IExecutableModel> BuildRLPolicyNetwork(const RLTrainingConfiguration& config, std::string& error);

// The policy network alone, made the same way (a training preview's own
// copy, which plays and never trains). Null with the reason.
std::unique_ptr<IExecutableModel> BuildRLPolicyNetwork(const RLTrainingConfiguration& config, std::string& error);

// What a trained run leaves (PpoTrainer::TakeTrained), with its graph and updates.
std::unique_ptr<TrainedPolicy> MakeTrainedPolicy(const RLTrainingConfiguration& config, std::string graph_json,
                                                 PpoTrainedState state, std::vector<PpoUpdateMetrics> updates);

// One action per row of `observations` (rows x observation size, raw from the
// environment; normalised here when the policy was trained so). best: the most
// likely action (a discrete policy's argmax, a Gaussian's mean); otherwise
// sampled as in training. Continuous actions are clipped to the bounds.
std::vector<float> PolicyActions(TrainedPolicy& policy, std::vector<float> observations, size_t rows, bool best,
                                 uint64_t seed);

// Why `policy` does not fit the compiled graph `graph` (another environment,
// or other layers), or empty when it fits.
std::string PolicyMismatch(const TrainedPolicy& policy, const RLTrainingConfiguration& graph);

bool SaveTrainedPolicy(const TrainedPolicy& policy, const std::filesystem::path& path, std::string& error);
// Null with the reason: not an RL policy, the environment is not available,
// or the graph's networks differ from the saved weights. The networks are
// built on the device the Engine would train on (the route chosen in
// Preferences > Devices when it may train, else the qualified CPU route), and
// the policy records it.
std::unique_ptr<TrainedPolicy> LoadTrainedPolicy(const std::filesystem::path& path, std::string& error);

}  // namespace cyxwiz::rl
