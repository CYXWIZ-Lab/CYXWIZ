#include "rl_device_presentation.h"

#include <algorithm>

namespace cyxwiz::rl_device {

namespace {

constexpr int kSmallLayerUnits = 256;
constexpr const char* kDriverReason =
    "usually train faster on the CPU, and the many small GPU steps of RL training strain some graphics drivers.";

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

std::string DeviceRoute(const RunDevice& d) {
    if (!d.valid) return {};
    const std::string route =
        d.effective_backend + " device " + std::to_string(d.effective_device_id) +
        (d.device_name.empty() ? "" : " \xC2\xB7 " + d.device_name) + " \xC2\xB7 policy " + PolicyWord(d.policy);
    if (d.fallback)
        return "The chosen " + BackendWord(d.requested_backend) + " device " + std::to_string(d.requested_device_id) +
               " failed its check" + (d.error.empty() ? "" : " (" + d.error + ")") + "; " + route + ".";
    return route;
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

namespace {

// "The networks run as ArrayFire operations on CUDA, not as fused kernels: NVIDIA provider not installed. Install ..."
std::string MissingProviderLine(const std::string& backend_word, const std::string& missing_provider) {
    return "The networks run as ArrayFire operations on " + backend_word + ", not as fused kernels: " + missing_provider;
}

}  // namespace

std::string CompileAdvice(const std::string& backend_word, const std::vector<int>& policy_units,
                          const std::vector<int>& value_units, const NetworkForecast& forecast) {
    if (backend_word.empty() || backend_word == "CPU") return {};
    const std::string chosen = " Training uses " + backend_word + ", as chosen in Preferences > Devices.";
    if (forecast.fused)
        return "The networks run as fused small-network kernels on " + forecast.provider_id + "." + chosen;
    // A missing provider is the cause and has its own remedy: the
    // small-network advice would name the wrong one
    if (!forecast.missing_provider.empty()) return MissingProviderLine(backend_word, forecast.missing_provider) + chosen;
    std::string text = "The networks run as ArrayFire operations on " + backend_word +
                       (forecast.why.empty() ? std::string() : ": " + forecast.why) + ".";
    if (SmallNetworks(policy_units, value_units))
        text += " Small networks (" + Widths(policy_units, value_units) + ") " + kDriverReason;
    return text + chosen;
}

std::string DashboardNote(const RunDevice& d, const std::vector<int>& policy_units,
                          const std::vector<int>& value_units, const std::string& compute_path,
                          const std::string& missing_provider) {
    if (!d.valid) return {};
    if (d.fallback)
        return "Trained on the " + BackendWord(d.effective_backend) + ": the chosen " +
               BackendWord(d.requested_backend) +
               " device failed its check, and the Compatibility policy fell back. Strict would have refused the "
               "run instead.";
    const std::string backend = BackendWord(d.effective_backend);
    if (backend == "CPU") return {};
    if (compute_path.empty() || compute_path.rfind("fused", 0) == 0) return {};  // not known yet, or no launch cost
    // The cause first: a missing provider is not a matter of network size
    if (!missing_provider.empty()) return MissingProviderLine(backend, missing_provider);
    if (!SmallNetworks(policy_units, value_units)) return {};
    return "Small networks (" + Widths(policy_units, value_units) + ") " + kDriverReason +
           " Preferences > Devices chooses where training runs.";
}

}  // namespace cyxwiz::rl_device
