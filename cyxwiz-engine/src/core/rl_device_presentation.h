#pragma once

// What the RL surfaces say about the device a PPO run trains on (TOFIX206,
// mockup W1-W4): the dashboard's device chip and note, the toolbar's word,
// the playback line, and Compile's advice for small networks on a GPU.
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

struct Chip {
    std::string text;     // "CUDA · GeForce GTX 1050 Ti", "CPU · fallback"
    std::string tooltip;  // the full route and policy, or the fallback's reason
    bool fallback = false;
};
Chip DeviceChip(const RunDevice& device);

// The toolbar line's end: " · CPU", " · CPU (fallback)"; empty before preflight
std::string ToolbarSuffix(const RunDevice& device);

// The dashboard's note under the header: the fallback explained, else the
// small-network advice when it applies, else empty
std::string DashboardNote(const RunDevice& device, const std::vector<int>& policy_units,
                          const std::vector<int>& value_units);

// Small networks: every hidden layer of both networks has at most 256 units
bool SmallNetworks(const std::vector<int>& policy_units, const std::vector<int>& value_units);
// "64-64", or "64-64 and 128-128" when the networks differ
std::string Widths(const std::vector<int>& policy_units, const std::vector<int>& value_units);
// Compile's Info line when the chosen route is a GPU (backend_word "CUDA") and
// the networks are small; empty otherwise
std::string CompileAdvice(const std::string& backend_word, const std::vector<int>& policy_units,
                          const std::vector<int>& value_units);

}  // namespace cyxwiz::rl_device
