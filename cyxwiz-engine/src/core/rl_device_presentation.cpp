#include "rl_device_presentation.h"

#include <algorithm>

namespace cyxwiz::rl_device {

namespace {

constexpr int kSmallLayerUnits = 256;
constexpr const char* kDriverReason =
    "usually train faster on the CPU, and the many small GPU steps of PPO strain some graphics drivers.";

std::string Join(const std::vector<int>& units) {
    std::string text;
    for (const int u : units) text += (text.empty() ? "" : "-") + std::to_string(u);
    return text.empty() ? std::string("no hidden layers") : text;
}

std::string PolicyWord(const std::string& policy) {
    return policy == "strict" ? "Strict" : policy == "compatibility" ? "Compatibility" : policy;
}

}  // namespace

std::string BackendWord(const std::string& backend) {
    if (backend == "arrayfire_cpu") return "CPU";
    if (backend == "arrayfire_cuda") return "CUDA";
    if (backend == "arrayfire_opencl") return "OpenCL";
    if (backend == "arrayfire_oneapi") return "oneAPI";
    return backend;
}

Chip DeviceChip(const RunDevice& d) {
    Chip chip;
    if (!d.valid) return chip;
    const std::string word = BackendWord(d.effective_backend);
    const std::string route =
        d.effective_backend + " device " + std::to_string(d.effective_device_id) +
        (d.device_name.empty() ? "" : " \xC2\xB7 " + d.device_name) + " \xC2\xB7 policy " + PolicyWord(d.policy);
    if (d.fallback) {
        chip.fallback = true;
        chip.text = word + " \xC2\xB7 fallback";
        chip.tooltip = "The chosen " + BackendWord(d.requested_backend) + " device " +
                       std::to_string(d.requested_device_id) + " failed its check" +
                       (d.error.empty() ? "" : " (" + d.error + ")") + "; " + route + ".";
        return chip;
    }
    chip.text = d.device_name.empty() ? word : word + " \xC2\xB7 " + d.device_name;
    chip.tooltip = route;
    return chip;
}

std::string ToolbarSuffix(const RunDevice& d) {
    if (!d.valid) return {};
    return " \xC2\xB7 " + BackendWord(d.effective_backend) + (d.fallback ? " (fallback)" : "");
}

bool SmallNetworks(const std::vector<int>& policy_units, const std::vector<int>& value_units) {
    const auto small = [](const std::vector<int>& units) {
        return std::all_of(units.begin(), units.end(), [](int u) { return u <= kSmallLayerUnits; });
    };
    return small(policy_units) && small(value_units);
}

std::string Widths(const std::vector<int>& policy_units, const std::vector<int>& value_units) {
    const std::string p = Join(policy_units), v = Join(value_units);
    return p == v ? p : p + " and " + v;
}

std::string CompileAdvice(const std::string& backend_word, const std::vector<int>& policy_units,
                          const std::vector<int>& value_units) {
    if (backend_word.empty() || backend_word == "CPU" || !SmallNetworks(policy_units, value_units)) return {};
    return "Small networks (" + Widths(policy_units, value_units) + ") " + kDriverReason + " Training uses " +
           backend_word + ", as chosen in Preferences > Devices.";
}

std::string DashboardNote(const RunDevice& d, const std::vector<int>& policy_units,
                          const std::vector<int>& value_units) {
    if (!d.valid) return {};
    if (d.fallback)
        return "Trained on the " + BackendWord(d.effective_backend) + ": the chosen " +
               BackendWord(d.requested_backend) +
               " device failed its check, and the Compatibility policy fell back. Strict would have refused the "
               "run instead.";
    if (BackendWord(d.effective_backend) == "CPU" || !SmallNetworks(policy_units, value_units)) return {};
    return "Small networks (" + Widths(policy_units, value_units) + ") " + kDriverReason +
           " Preferences > Devices chooses where training runs.";
}

}  // namespace cyxwiz::rl_device
