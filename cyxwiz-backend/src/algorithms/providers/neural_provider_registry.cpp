#include "cyxwiz/neural_provider.h"

#include <atomic>
#include <mutex>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

namespace {
std::atomic<bool> g_neural_providers_disabled_for_testing{false};
} // namespace

void SetNeuralProvidersDisabledForTesting(bool disabled) {
    g_neural_providers_disabled_for_testing.store(disabled);
}

#ifdef CYXWIZ_HAS_NVIDIA_DNN_PROVIDER
// Defined in nvidia_cublas_provider.cpp; registers the provider when its
// runtime probe succeeds. No-op stub otherwise.
void RegisterNvidiaCublasNeuralProvider(NeuralProviderRegistry& registry);
#endif
#ifdef CYXWIZ_HAS_OPENCL_DNN_PROVIDER
// Defined in opencl_cell_provider.cpp; registers when an OpenCL GPU device
// is enumerated. Tenant #2 of the device-keyed dispatch.
void RegisterOpenclCellNeuralProvider(NeuralProviderRegistry& registry);
#endif

const char* NeuralOpName(NeuralOp op) {
    switch (op) {
    case NeuralOp::RnnForward: return "rnn_forward";
    case NeuralOp::RnnBackward: return "rnn_backward";
    case NeuralOp::LstmForward: return "lstm_forward";
    case NeuralOp::LstmBackward: return "lstm_backward";
    case NeuralOp::GruForward: return "gru_forward";
    case NeuralOp::GruBackward: return "gru_backward";
    case NeuralOp::AttentionForward: return "attention_forward";
    case NeuralOp::AttentionBackward: return "attention_backward";
    }
    return "unknown_neural_op";
}

const char* NeuralDevicePlatformName(DeviceType platform) {
    switch (platform) {
    case DeviceType::CPU: return "cpu";
    case DeviceType::CUDA: return "cuda";
    case DeviceType::OPENCL: return "opencl";
    case DeviceType::METAL: return "metal";
    case DeviceType::VULKAN: return "vulkan";
    case DeviceType::ONEAPI: return "oneapi";
    }
    return "unknown";
}

NeuralDeviceTarget CaptureCurrentNeuralDeviceTarget() {
    NeuralDeviceTarget target;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        switch (af::getActiveBackend()) {
        case AF_BACKEND_CUDA: target.platform = DeviceType::CUDA; break;
        case AF_BACKEND_OPENCL: target.platform = DeviceType::OPENCL; break;
        case AF_BACKEND_ONEAPI: target.platform = DeviceType::ONEAPI; break;
        default: target.platform = DeviceType::CPU; break;
        }
        target.device_id = af::getDevice();
    } catch (...) {
        // Query failure is not a placement: fail closed to the CPU target
        // (no provider serves it), the portable path decides.
        target = NeuralDeviceTarget{};
    }
#endif
    return target;
}

struct NeuralProviderRegistry::State {
    mutable std::mutex mutex;
    std::vector<std::shared_ptr<INeuralNetworkProvider>> providers;
};

NeuralProviderRegistry::State& NeuralProviderRegistry::GetState() const {
    static State state;
    return state;
}

NeuralProviderRegistry& NeuralProviderRegistry::Instance() {
    static NeuralProviderRegistry registry;
    static std::once_flag built_ins_registered;
    std::call_once(built_ins_registered, [] {
#ifdef CYXWIZ_HAS_NVIDIA_DNN_PROVIDER
        RegisterNvidiaCublasNeuralProvider(registry);
#endif
#ifdef CYXWIZ_HAS_OPENCL_DNN_PROVIDER
        RegisterOpenclCellNeuralProvider(registry);
#endif
    });
    return registry;
}

void NeuralProviderRegistry::Register(
    std::shared_ptr<INeuralNetworkProvider> provider) {
    if (!provider) {
        return;
    }
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.providers.push_back(std::move(provider));
}

std::vector<std::shared_ptr<INeuralNetworkProvider>>
NeuralProviderRegistry::List() const {
    if (g_neural_providers_disabled_for_testing.load()) {
        return {};
    }
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.providers;
}

std::vector<std::shared_ptr<INeuralNetworkProvider>>
NeuralProviderRegistry::ListServing(const NeuralDeviceTarget& target) const {
    std::vector<std::shared_ptr<INeuralNetworkProvider>> serving;
    for (const auto& provider : List()) {
        if (provider->Platform() == target.platform) {
            serving.push_back(provider);
        }
    }
    return serving;
}

std::shared_ptr<INeuralNetworkProvider>
NeuralProviderRegistry::FindSupporting(
    const NeuralOpRequest& request) const {
    // Device-keyed dispatch: platform filter first, capability second.
    for (const auto& provider : ListServing(request.target)) {
        if (provider->QueryCapability(request).supported) {
            return provider;
        }
    }
    return nullptr;
}

} // namespace cyxwiz
