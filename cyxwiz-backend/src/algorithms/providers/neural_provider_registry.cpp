#include "cyxwiz/neural_provider.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>

#include <arrayfire.h>
#include <spdlog/spdlog.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cyxwiz {

namespace {
std::atomic<bool> g_neural_providers_disabled_for_testing{false};
} // namespace

void SetNeuralProvidersDisabledForTesting(bool disabled) {
    g_neural_providers_disabled_for_testing.store(disabled);
}

// The NVIDIA provider is a module of its own (cyxwiz-nvidia-provider,
// TOFIX178 1.0(c)), loaded at first use: installed with the CUDA GPU pack,
// or next to the backend in a build. Absent, NVIDIA devices use the
// portable path.
namespace {

#if defined(_WIN32)
constexpr const wchar_t* kNvidiaProviderModule = L"cyxwiz-nvidia-provider.dll";
#else
constexpr const char* kNvidiaProviderModule = "libcyxwiz-nvidia-provider.so";
#endif

struct NvidiaProviderModule {
    void (*register_fn)(NeuralProviderRegistry*) = nullptr;
    bool (*device_memory_fn)(size_t*, size_t*) = nullptr;
};

void* NvidiaProviderSymbol(void* module, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name));
#else
    return dlsym(module, name);
#endif
}

// The module next to this backend library (a build's bin folder).
#if defined(_WIN32)
std::filesystem::path BesideBackend() {
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&BesideBackend), &self))
        return {};
    wchar_t path[MAX_PATH];
    const DWORD length = GetModuleFileNameW(self, path, MAX_PATH);
    if (length == 0 || length == MAX_PATH) return {};
    return std::filesystem::path(path).parent_path() / kNvidiaProviderModule;
}
#else
std::filesystem::path BesideBackend() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&BesideBackend), &info) == 0 || !info.dli_fname) return {};
    return std::filesystem::path(info.dli_fname).parent_path() / kNvidiaProviderModule;
}
#endif

const NvidiaProviderModule& LoadNvidiaProviderModule() {
    static const NvidiaProviderModule module = [] {
        NvidiaProviderModule loaded;
        void* handle = nullptr;
        std::string where;
        const std::filesystem::path beside = BesideBackend();
#if defined(_WIN32)
        // The runtime's DLL folders (the GPU packs), then the backend's own.
        handle = LoadLibraryW(kNvidiaProviderModule);
        if (!handle && !beside.empty() && std::filesystem::exists(beside))
            handle = LoadLibraryExW(beside.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (handle) {
            wchar_t path[MAX_PATH];
            const DWORD length = GetModuleFileNameW(static_cast<HMODULE>(handle), path, MAX_PATH);
            where = length > 0 && length < MAX_PATH ? std::filesystem::path(path).string() : "cyxwiz-nvidia-provider.dll";
        }
#else
        handle = dlopen(kNvidiaProviderModule, RTLD_NOW | RTLD_LOCAL);
        if (!handle && !beside.empty() && std::filesystem::exists(beside))
            handle = dlopen(beside.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle) where = kNvidiaProviderModule;
#endif
        if (!handle) {
            spdlog::debug("NVIDIA neural provider module not installed ({}): NVIDIA devices use the portable path",
                          beside.empty() ? std::string("cyxwiz-nvidia-provider") : beside.filename().string());
            return loaded;
        }
        const auto abi = reinterpret_cast<int (*)()>(NvidiaProviderSymbol(handle, "cyxwiz_nvidia_provider_abi"));
        const auto register_fn = reinterpret_cast<void (*)(NeuralProviderRegistry*)>(
            NvidiaProviderSymbol(handle, "cyxwiz_nvidia_provider_register"));
        if (!abi || !register_fn) {
            spdlog::warn("NVIDIA neural provider module {} lacks its entry points; not used", where);
            return loaded;
        }
        if (abi() != CYXWIZ_NEURAL_PROVIDER_MODULE_ABI) {
            spdlog::warn("NVIDIA neural provider module {} has ABI {}, this Engine needs {}; not used (install the "
                         "CUDA GPU pack of this version)",
                         where, abi(), CYXWIZ_NEURAL_PROVIDER_MODULE_ABI);
            return loaded;
        }
        // The module stays loaded for the process: its provider lives in
        // the registry until exit.
        loaded.register_fn = register_fn;
        loaded.device_memory_fn = reinterpret_cast<bool (*)(size_t*, size_t*)>(
            NvidiaProviderSymbol(handle, "cyxwiz_nvidia_provider_device_memory"));
        spdlog::info("NVIDIA neural provider module loaded: {}", where);
        return loaded;
    }();
    return module;
}

}  // namespace

#ifdef CYXWIZ_HAS_NVIDIA_DNN_PROVIDER
bool NvidiaProviderDeviceMemoryForTesting(size_t& free_bytes, size_t& total_bytes) {
    free_bytes = total_bytes = 0;
    const auto& module = LoadNvidiaProviderModule();
    return module.device_memory_fn && module.device_memory_fn(&free_bytes, &total_bytes);
}
#endif
#ifdef CYXWIZ_HAS_OPENCL_DNN_PROVIDER
// Defined in opencl_cell_provider.cpp; registers when an OpenCL GPU device
// is enumerated. Tenant #2 of the device-keyed dispatch.
void RegisterOpenclCellNeuralProvider(NeuralProviderRegistry& registry);
#endif
#ifdef CYXWIZ_HAS_ONEAPI_DNN_PROVIDER
// Defined in oneapi_sycl_provider.cpp; registers when the SYCL kernel
// library is installed next to the backend. Tenant #3.
void RegisterOneapiSyclNeuralProvider(NeuralProviderRegistry& registry);
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
    case NeuralOp::DeviceProbe: return "device_probe";
    case NeuralOp::Conv2dForward: return "conv2d_forward";
    case NeuralOp::Conv2dBackward: return "conv2d_backward";
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
        if (const auto& nvidia = LoadNvidiaProviderModule(); nvidia.register_fn) nvidia.register_fn(&registry);
#ifdef CYXWIZ_HAS_OPENCL_DNN_PROVIDER
        RegisterOpenclCellNeuralProvider(registry);
#endif
#ifdef CYXWIZ_HAS_ONEAPI_DNN_PROVIDER
        RegisterOneapiSyclNeuralProvider(registry);
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

// Operator escape hatch: CYXWIZ_DISABLE_NEURAL_PROVIDERS=1 hides every
// provider for the process (portable path everywhere). Read once.
static bool NeuralProvidersDisabledByEnvironment() {
    static const bool disabled = [] {
        const char* value = std::getenv("CYXWIZ_DISABLE_NEURAL_PROVIDERS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return disabled;
}

std::vector<std::shared_ptr<INeuralNetworkProvider>>
NeuralProviderRegistry::List() const {
    if (g_neural_providers_disabled_for_testing.load() ||
        NeuralProvidersDisabledByEnvironment()) {
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
