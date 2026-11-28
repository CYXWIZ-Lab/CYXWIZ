#pragma once

// What the RL surfaces say about the device a PPO run trains on (TOFIX206,
// mockup W1-W4): the device's route and the dashboard's note, the toolbar's word,
// the playback line, and Compile's line on how the networks will compute on a
// GPU (the actual path and its reason; the small-network advice only where
// small networks are the cause).
// Pure data in and out (no ImGui, no ArrayFire).

#include <string>
#include <vector>

namespace cyxwiz::rl_device {

// A run's device, copied from its ExecutionDeviceContext
struct RunDevice {
    bool valid = false;
    std::string requested_backend;  // "arrayfire_cuda"
    int requested_device_id = 0;
    std::string effective_backend;  // "arrayfire_cpu"
    int effective_device_id = 0;
    std::string device_name;        // "GeForce GTX 1050 Ti"
    std::string policy;             // "strict" or "compatibility"
    bool fallback = false;          // Compatibility fell back to the CPU
    std::string error;              // why it fell back
};

// "arrayfire_cuda" -> "CUDA"; CPU, OpenCL, oneAPI; unknown names as given
std::string BackendWord(const std::string& backend);

// The full route and policy ("arrayfire_cpu device 0 · Intel · policy
// Compatibility"), or the fallback's reason; empty before preflight
std::string DeviceRoute(const RunDevice& device);

// The toolbar line's end: " · CPU", " · CPU (fallback)"; empty before preflight
std::string ToolbarSuffix(const RunDevice& device);

// The dashboard's note under the header: the fallback explained; else, on a
// GPU whose networks run layer by layer, the cause: a missing provider with
// what to do about it, or the small-network advice when small networks are
// the cause; else empty. compute_path: how the run computes, as its updates
// report it ("" before the first update). missing_provider: "<name>. <fix>"
// when the networks are on ArrayFire because a provider is missing and can be
// supplied ("NVIDIA provider not installed. Install the CUDA GPU pack."),
// else empty. A run on the fused small-network kernels has no note, and
// before the first update nothing is known yet.
std::string DashboardNote(const RunDevice& device, const std::vector<int>& policy_units,
                          const std::vector<int>& value_units, const std::string& compute_path,
                          const std::string& missing_provider);

// Small networks: every hidden layer of both networks has at most 256 units
bool SmallNetworks(const std::vector<int>& policy_units, const std::vector<int>& value_units);
// "64-64", or "64-64 and 128-128" when the networks differ
std::string Widths(const std::vector<int>& policy_units, const std::vector<int>& value_units);
// How an RL graph's networks will compute on a route, as Compile forecasts
// it from the provider registry (ForecastRLNetworks).
struct NetworkForecast {
    bool fused = false;            // a provider runs them as fused small-network kernels
    std::string provider_id;       // fused: that provider
    // Not fused because a provider is missing and can be supplied: "<name>.
    // <fix>" ("NVIDIA provider not installed. Install the CUDA GPU pack.")
    std::string missing_provider;
    // Not fused for another cause, in words without a full stop: the serving
    // provider declines these networks, the networks are not a fusable chain,
    // or no provider is offered for them on this device family
    std::string why;
};
// Compile's Info line when the chosen route is a GPU (backend_word "CUDA"):
// the path the networks will take and its reason, with what to do when a
// provider is missing, and the small-network advice when the networks are
// small and the cause is not a missing provider. Empty on the CPU.
std::string CompileAdvice(const std::string& backend_word, const std::vector<int>& policy_units,
                          const std::vector<int>& value_units, const NetworkForecast& forecast);

}  // namespace cyxwiz::rl_device
