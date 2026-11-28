#include "cyxwiz/neural_provider.h"

#include "neural_provider_absence.h"
#include "neural_run_target_internal.h"

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

bool NeuralProvidersDisabledForTesting() { return g_neural_providers_disabled_for_testing.load(); }

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
    // What the one load attempt of the process found (the reason an NVIDIA
    // device has no provider: WhyNoNeuralProviderServes)
    provider_absence::ModuleState state = provider_absence::ModuleState::NotFound;
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
            // Once per process (this is the one load attempt). Info where the
            // build has an ArrayFire CUDA route, so a CUDA run on ArrayFire
            // alone is explained in the log; Debug in a build without one,
            // where no NVIDIA provider could ever be used.
            const std::string name = beside.empty() ? std::string("cyxwiz-nvidia-provider") : beside.filename().string();
#ifdef CYXWIZ_HAS_NVIDIA_DNN_PROVIDER
            const BackendFallbackReason reason = BackendFallbackReason::ProviderNotInstalled;
#else
            const BackendFallbackReason reason = BackendFallbackReason::ProviderNotBuilt;
#endif
            const NeuralProviderAbsence absence = DescribeNeuralProviderAbsence(reason, DeviceType::CUDA);
#ifdef CYXWIZ_ENABLE_CUDA
            spdlog::info("NVIDIA neural provider module not found ({}): NVIDIA devices run on ArrayFire alone "
                         "(reason={}). {}",
                         name, BackendFallbackReasonName(reason), absence.fix);
#else
            spdlog::debug("NVIDIA neural provider module not found ({}): this build has no CUDA route (reason={})",
                          name, BackendFallbackReasonName(reason));
#endif
            return loaded;
        }
        const auto abi = reinterpret_cast<int (*)()>(NvidiaProviderSymbol(handle, "cyxwiz_nvidia_provider_abi"));
        const auto register_fn = reinterpret_cast<void (*)(NeuralProviderRegistry*)>(
            NvidiaProviderSymbol(handle, "cyxwiz_nvidia_provider_register"));
        if (!abi || !register_fn) {
            loaded.state = provider_absence::ModuleState::WrongAbi;
            spdlog::warn("NVIDIA neural provider module {} lacks its entry points; not used (reason={})", where,
                         BackendFallbackReasonName(BackendFallbackReason::ProviderAbiMismatch));
            return loaded;
        }
        if (abi() != CYXWIZ_NEURAL_PROVIDER_MODULE_ABI) {
            loaded.state = provider_absence::ModuleState::WrongAbi;
            spdlog::warn("NVIDIA neural provider module {} has ABI {}, this Engine needs {}; not used (reason={}). {}",
                         where, abi(), CYXWIZ_NEURAL_PROVIDER_MODULE_ABI,
                         BackendFallbackReasonName(BackendFallbackReason::ProviderAbiMismatch),
                         DescribeNeuralProviderAbsence(BackendFallbackReason::ProviderAbiMismatch, DeviceType::CUDA).fix);
            return loaded;
        }
        // The module stays loaded for the process: its provider lives in
        // the registry until exit.
        loaded.state = provider_absence::ModuleState::Present;
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
// How the process found the kernel library (oneapi_sycl_provider.cpp).
provider_absence::ModuleState OneapiKernelLibraryState();
#endif

static bool NeuralProvidersDisabledByEnvironment();

bool NeuralProviderOfferedFor(DeviceType platform, NeuralOp op) {
    switch (platform) {
    case DeviceType::CUDA:
        return true;  // the NVIDIA provider implements every op
    case DeviceType::OPENCL:
        switch (op) {
        case NeuralOp::RnnForward:
        case NeuralOp::RnnBackward:
        case NeuralOp::LstmForward:
        case NeuralOp::LstmBackward:
        case NeuralOp::GruForward:
        case NeuralOp::GruBackward:
        case NeuralOp::AttentionForward:
        case NeuralOp::AttentionBackward:
        case NeuralOp::DeviceProbe:
            return true;
        default:
            return false;
        }
    case DeviceType::ONEAPI:
        return op == NeuralOp::AttentionForward || op == NeuralOp::AttentionBackward || op == NeuralOp::DeviceProbe;
    default:
        return false;  // no provider serves the CPU, Metal or Vulkan
    }
}

namespace {

const char* PlatformWord(DeviceType platform) {
    switch (platform) {
    case DeviceType::CPU: return "the CPU";
    case DeviceType::CUDA: return "CUDA";
    case DeviceType::OPENCL: return "OpenCL";
    case DeviceType::ONEAPI: return "oneAPI";
    case DeviceType::METAL: return "Metal";
    case DeviceType::VULKAN: return "Vulkan";
    }
    return "this device";
}

// The provider family of a device family, as the user knows it.
const char* ProviderWord(DeviceType platform) {
    switch (platform) {
    case DeviceType::CUDA: return "NVIDIA";
    case DeviceType::OPENCL: return "OpenCL";
    case DeviceType::ONEAPI: return "oneAPI";
    default: return "neural";
    }
}

}  // namespace

NeuralProviderAbsence DescribeNeuralProviderAbsence(BackendFallbackReason reason, DeviceType platform) {
    using R = BackendFallbackReason;
    NeuralProviderAbsence absence;
    absence.reason = reason;
    absence.fixable = true;
    const std::string provider = ProviderWord(platform);
    const bool cuda = platform == DeviceType::CUDA;
    const bool oneapi = platform == DeviceType::ONEAPI;
    switch (reason) {
    case R::ProviderNotBuilt:
        absence.name = provider + " provider not built";
        absence.fix = cuda     ? "This build has no NVIDIA provider: install the CUDA GPU pack, or build with the "
                                 "gpu-dev preset."
                      : oneapi ? "This build has no oneAPI provider: build with the gpu-dev preset and the DPC++ "
                                 "compiler."
                               : "This build has no " + provider + " provider: build with the gpu-dev preset.";
        break;
    case R::ProviderNotInstalled:
        absence.name = oneapi ? std::string("oneAPI kernel library not installed") : provider + " provider not installed";
        absence.fix = cuda     ? "Install the CUDA GPU pack."
                      : oneapi ? "Put cyxwiz-oneapi-kernels beside the Engine's backend library (a gpu-dev build "
                                 "makes it), then restart the Engine."
                               : "Reinstall the Engine: its " + provider + " provider is missing.";
        break;
    case R::ProviderAbiMismatch:
        absence.name = (oneapi ? std::string("oneAPI kernel library") : provider + " provider") +
                       " from another Engine version";
        absence.fix = cuda     ? "Install the CUDA GPU pack of this Engine version."
                      : oneapi ? "Rebuild or reinstall cyxwiz-oneapi-kernels for this Engine version, then restart "
                                 "the Engine."
                               : "Reinstall the " + provider + " provider of this Engine version.";
        break;
    case R::ProviderDisabled:
        absence.name = "Neural providers turned off";
        absence.fix = "Unset CYXWIZ_DISABLE_NEURAL_PROVIDERS and start again to use them.";
        break;
    case R::NvidiaProviderUnavailable:
        absence.name = "NVIDIA provider found no CUDA device";
        absence.fix = "Check the NVIDIA driver, then start the Engine again.";
        break;
    case R::OpenclProviderUnavailable:
        absence.name = "OpenCL provider found no GPU device";
        absence.fix = "Check the GPU's OpenCL driver, then start the Engine again.";
        break;
    case R::OneapiProviderUnavailable:
        absence.name = "oneAPI provider not available";
        absence.fix = "Check the oneAPI runtime and the GPU driver, then start the Engine again.";
        break;
    case R::TargetMismatch:
        absence.name = "Not on the run's device";
        absence.fix = "ArrayFire runs it where its data is; start the run again, and report it if this repeats.";
        break;
    case R::ProviderLatched:
        absence.name = std::string("Turned off for this run after a provider failure");
        absence.fix = "The next run asks the provider again; under Strict a provider failure stops the run instead.";
        break;
    default:
        absence.reason = R::ProviderNotOffered;
        absence.fixable = false;
        absence.name = std::string("No provider is offered for this on ") + PlatformWord(platform);
        absence.fix = "ArrayFire runs it by design; there is nothing to install.";
        break;
    }
    return absence;
}

NeuralProviderAbsence WhyNoNeuralProviderServes(DeviceType platform, NeuralOp op) {
    provider_absence::Facts facts;
    facts.disabled = g_neural_providers_disabled_for_testing.load() || NeuralProvidersDisabledByEnvironment();
#ifdef CYXWIZ_HAS_NVIDIA_DNN_PROVIDER
    facts.nvidia_built = true;
#endif
#ifdef CYXWIZ_HAS_OPENCL_DNN_PROVIDER
    facts.opencl_built = true;
#endif
#ifdef CYXWIZ_HAS_ONEAPI_DNN_PROVIDER
    facts.oneapi_built = true;
    if (platform == DeviceType::ONEAPI) facts.oneapi_kernels = OneapiKernelLibraryState();
#endif
    // The one load attempt of the process, made here if nothing asked before
    if (platform == DeviceType::CUDA && !facts.disabled) facts.nvidia_module = LoadNvidiaProviderModule().state;
    return DescribeNeuralProviderAbsence(provider_absence::Reason(platform, op, facts), platform);
}

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
    case NeuralOp::MlpForward: return "mlp_forward";
    case NeuralOp::MlpBackward: return "mlp_backward";
    case NeuralOp::AdamStep: return "adam_step";
    case NeuralOp::PpoSample: return "ppo_sample";
    case NeuralOp::PpoLoss: return "ppo_loss";
    case NeuralOp::PpoGae: return "ppo_gae";
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

namespace {

// The bound target of this thread's run (ScopedNeuralRunTarget), and whether
// the thread's ArrayFire device was found to be it in the innermost scope
thread_local bool g_run_target_bound = false;
thread_local NeuralDeviceTarget g_run_target;
thread_local bool g_run_thread_verified = false;
std::atomic<bool> g_target_query_failure_logged{false};

}  // namespace

bool detail::NeuralRunThreadVerified() { return g_run_thread_verified; }
void detail::MarkNeuralRunThreadVerified() { g_run_thread_verified = g_run_target_bound; }

ScopedNeuralRunTarget::ScopedNeuralRunTarget(NeuralDeviceTarget target)
    : had_previous_(g_run_target_bound), previous_(g_run_target) {
    g_run_target_bound = true;
    g_run_target = target;
    g_run_thread_verified = false;
}

ScopedNeuralRunTarget::~ScopedNeuralRunTarget() {
    g_run_target_bound = had_previous_;
    g_run_target = previous_;
    g_run_thread_verified = false;  // the scope around it reads the thread again
}

bool BoundNeuralRunTarget(NeuralDeviceTarget& target) {
    if (!g_run_target_bound) return false;
    target = g_run_target;
    return true;
}

bool QueryArrayFireNeuralTarget(NeuralDeviceTarget& target, std::string& error) {
    target = NeuralDeviceTarget{};
    try {
        switch (af::getActiveBackend()) {
        case AF_BACKEND_CUDA: target.platform = DeviceType::CUDA; break;
        case AF_BACKEND_OPENCL: target.platform = DeviceType::OPENCL; break;
        case AF_BACKEND_ONEAPI: target.platform = DeviceType::ONEAPI; break;
        default: target.platform = DeviceType::CPU; break;
        }
        target.device_id = af::getDevice();
        return true;
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown ArrayFire error";
    }
    target = NeuralDeviceTarget{};
    return false;
}

NeuralDeviceTarget CaptureCurrentNeuralDeviceTarget() {
    // A run's target is the device it bound, not the thread's ArrayFire
    // state; the dispatch checks that the two agree
    NeuralDeviceTarget target;
    if (BoundNeuralRunTarget(target)) return target;
    std::string error;
    if (QueryArrayFireNeuralTarget(target, error)) return target;
    // Not a placement, and never silent: the CPU target (no provider serves
    // it), and a warning that says why
    if (!g_target_query_failure_logged.exchange(true))
        spdlog::warn("Neural providers: ArrayFire's active device could not be read ({}); no provider is used for "
                     "requests made outside a run until it can (reason={})",
                     error, BackendFallbackReasonName(BackendFallbackReason::TargetMismatch));
    return NeuralDeviceTarget{};
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

void NeuralProviderRegistry::UnregisterForTesting(const std::string& provider_id) {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::erase_if(state.providers, [&](const std::shared_ptr<INeuralNetworkProvider>& provider) {
        return provider->ProviderId() == provider_id;
    });
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
