// oneAPI (SYCL) neural provider, device-resident attention (tofix112 phase
// 5b, tenant #3 of the device-keyed dispatch).
//
// The kernels live in cyxwiz-oneapi-kernels, a separate library built with
// the Intel DPC++ compiler (cyxwiz-backend/oneapi). This file is plain C++:
// it loads that library at first use through its C ABI, so the backend
// builds with any compiler and users without oneAPI never load SYCL.
//
// The library is loaded only while ArrayFire's oneAPI backend is loaded, so
// it binds the SYCL runtime ArrayFire already uses (same sycl8.dll). Data is
// host-staged by ExecuteNeuralOpOnDevice (ArrayFire 3.10 exposes no SYCL
// queue or context to share buffers safely), so buffer handles here are
// host float arrays; the kernels still run on the oneAPI device.
#include "cyxwiz/neural_provider.h"

#include "../../../oneapi/cyxwiz_oneapi_kernels.h"

#include <spdlog/spdlog.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <filesystem>
#include <limits>
#include <mutex>
#include <string>

namespace cyxwiz {

namespace {

constexpr const char* kProviderId = "cyxwiz.oneapi-sycl-attention";
constexpr const char* kProviderSemver = "0.1.0";

#ifdef _WIN32
constexpr const char* kKernelLibrary = "cyxwiz-oneapi-kernels.dll";
constexpr const char* kArrayFireOneapiLibrary = "afoneapi.dll";
#else
constexpr const char* kKernelLibrary = "libcyxwiz-oneapi-kernels.so";
constexpr const char* kArrayFireOneapiLibrary = "libafoneapi.so";
#endif

// Directory of the module holding this code (the backend library).
std::filesystem::path BackendDirectory() {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&BackendDirectory), &module)) {
        return {};
    }
    wchar_t path[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(module, path, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) return {};
    return std::filesystem::path(path).parent_path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&BackendDirectory), &info) == 0 || !info.dli_fname) return {};
    return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

bool ArrayFireOneapiLoaded() {
#ifdef _WIN32
    return GetModuleHandleA(kArrayFireOneapiLibrary) != nullptr;
#else
    void* handle = dlopen(kArrayFireOneapiLibrary, RTLD_LAZY | RTLD_NOLOAD);
    if (handle) dlclose(handle);
    return handle != nullptr;
#endif
}

// Lazily loaded kernel library (never unloaded: SYCL objects it owns live
// for the process).
class KernelLibrary {
public:
    explicit KernelLibrary(std::filesystem::path path) : path_(std::move(path)) {}

    const std::filesystem::path& Path() const { return path_; }

    // True once the entry points are bound; failure is sticky and explained.
    bool Ensure(std::string& failure) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (loaded_) return true;
        if (!failure_.empty()) {
            failure = failure_;
            return false;
        }
        if (!ArrayFireOneapiLoaded()) {
            // Not sticky: ArrayFire loads its oneAPI backend on first use.
            failure = "ArrayFire's oneAPI backend is not loaded in this process";
            return false;
        }
#ifdef _WIN32
        HMODULE module = LoadLibraryExW(path_.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        const auto symbol = [&](const char* name) {
            return module ? reinterpret_cast<void*>(GetProcAddress(module, name)) : nullptr;
        };
#else
        void* module = dlopen(path_.string().c_str(), RTLD_NOW | RTLD_LOCAL);
        const auto symbol = [&](const char* name) { return module ? dlsym(module, name) : nullptr; };
#endif
        if (!module) {
            failure_ = "could not load " + path_.string();
            failure = failure_;
            return false;
        }
        abi_ = reinterpret_cast<int (*)()>(symbol("cyxwiz_oneapi_kernels_abi"));
        info_ = reinterpret_cast<int (*)(char*, size_t)>(symbol("cyxwiz_oneapi_runtime_info"));
        probe_ = reinterpret_cast<decltype(probe_)>(symbol("cyxwiz_oneapi_device_probe"));
        forward_ = reinterpret_cast<decltype(forward_)>(symbol("cyxwiz_oneapi_attention_forward"));
        backward_ = reinterpret_cast<decltype(backward_)>(symbol("cyxwiz_oneapi_attention_backward"));
        if (!abi_ || !info_ || !probe_ || !forward_ || !backward_) {
            failure_ = path_.string() + " is missing entry points";
        } else if (abi_() != CYXWIZ_ONEAPI_KERNELS_ABI) {
            failure_ = path_.string() + " has ABI " + std::to_string(abi_()) + ", expected " +
                       std::to_string(CYXWIZ_ONEAPI_KERNELS_ABI);
        }
        if (!failure_.empty()) {
            failure = failure_;
            return false;
        }
        char info[2048] = {};
        if (info_(info, sizeof(info)) == 0) runtime_info_ = info;
        spdlog::info("oneAPI neural provider kernels loaded: {}", runtime_info_);
        loaded_ = true;
        return true;
    }

    std::string RuntimeInfo() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_ ? runtime_info_ : std::string("kernels not loaded yet");
    }

    int (*abi_)() = nullptr;
    int (*info_)(char*, size_t) = nullptr;
    int (*probe_)(const CyxOneapiDevice*, CyxOneapiBuffer, CyxOneapiBuffer, char*, size_t) = nullptr;
    int (*forward_)(const CyxOneapiDevice*, const CyxOneapiAttentionArgs*, char*, size_t) = nullptr;
    int (*backward_)(const CyxOneapiDevice*, const CyxOneapiAttentionArgs*, char*, size_t) = nullptr;

private:
    std::filesystem::path path_;
    mutable std::mutex mutex_;
    bool loaded_ = false;
    std::string failure_;
    std::string runtime_info_;
};

NeuralOpStatus Fail(BackendFallbackReason reason, std::string detail) {
    NeuralOpStatus status;
    status.reason = reason;
    status.detail = std::move(detail);
    return status;
}

NeuralOpStatus Ok() {
    NeuralOpStatus status;
    status.ok = true;
    status.reason = BackendFallbackReason::BackendInternalError;
    return status;
}

std::string AttentionContractError(const NeuralOpRequest& r) {
    if (r.dtype != DataType::Float32) return "attention contract is Float32 only";
    if (r.batch == 0 || r.heads == 0 || r.kv_heads == 0 || r.seq == 0 || r.kv_seq == 0) {
        return "attention dimensions must be positive";
    }
    if (r.heads % r.kv_heads != 0) return "heads must be a multiple of kv_heads";
    if (r.head_dim == 0 || r.head_dim > 128) return "attention head_dim must be 1..128";
    if (r.position_strategy != NeuralPositionStrategy::None && r.position_strategy != NeuralPositionStrategy::Alibi) {
        return "attention kernels take positions as none or alibi (RoPE is applied before the call)";
    }
    if (!(r.softmax_scale > 0.0f) || !(r.logit_softcap >= 0.0f)) return "attention scale must be positive";
    if (!(r.attention_dropout >= 0.0f) || !(r.attention_dropout < 1.0f)) return "attention dropout must be in [0, 1)";
    const size_t limit = static_cast<size_t>(std::numeric_limits<int>::max());
    if (r.batch > limit || r.heads > limit || r.seq > limit || r.kv_seq > limit || r.query_offset > limit ||
        r.sliding_window > limit) {
        return "attention sizes exceed the int range";
    }
    return {};
}

class OneapiSyclProvider final : public INeuralNetworkProvider {
public:
    explicit OneapiSyclProvider(std::filesystem::path library) : library_(std::move(library)) {}

    const char* ProviderId() const override { return kProviderId; }
    DeviceType Platform() const override { return DeviceType::ONEAPI; }

    std::string Version() const override {
        return std::string(kProviderId) + " " + kProviderSemver + " / kernels " + library_.Path().string() + " / " +
               library_.RuntimeInfo();
    }

    NeuralCapability QueryCapability(const NeuralOpRequest& request) const override {
        NeuralCapability capability;
        capability.reason = BackendFallbackReason::OneapiProviderUnsupportedContract;
        if (request.target.platform != DeviceType::ONEAPI) {
            capability.detail = std::string("request targets ") + NeuralDevicePlatformName(request.target.platform) +
                                "; this provider serves oneapi only";
            return capability;
        }
        if (!request.device_resident) {
            capability.detail = "the oneAPI provider runs device-resident requests only";
            return capability;
        }
        if (request.op == NeuralOp::AttentionForward || request.op == NeuralOp::AttentionBackward) {
            capability.detail = AttentionContractError(request);
            if (!capability.detail.empty()) return capability;
        } else if (request.op == NeuralOp::DeviceProbe) {
            if (request.dtype != DataType::Float32 || request.elements == 0) {
                capability.detail = "device_probe needs Float32 and at least one element";
                return capability;
            }
        } else {
            capability.detail = "the oneAPI provider covers device_probe and attention";
            return capability;
        }
        capability.supported = true;
        capability.reason = BackendFallbackReason::BackendInternalError;
        return capability;
    }

    NeuralResourceEstimate EstimateResources(const NeuralOpRequest&) const override { return {}; }

    NeuralOpStatus Execute(const NeuralOpRequest&, NeuralOpBuffers&) override {
        return Fail(BackendFallbackReason::OneapiProviderUnsupportedContract,
                    "the oneAPI provider has no host-copy (v1) path; device-resident requests run through "
                    "ExecuteDevice");
    }

    NeuralOpStatus ExecuteDevice(const NeuralOpRequest& request, NeuralDeviceOpBuffers& buffers) override {
        const NeuralCapability capability = QueryCapability(request);
        if (!capability.supported) return Fail(capability.reason, capability.detail);
        if (buffers.queue.platform != DeviceType::ONEAPI || buffers.queue.oneapi_device_name.empty()) {
            return Fail(BackendFallbackReason::OneapiProviderUnsupportedContract,
                        "device-resident execution needs the ArrayFire oneAPI device identity");
        }
        std::string failure;
        if (!library_.Ensure(failure)) return Fail(BackendFallbackReason::OneapiProviderUnavailable, failure);
        CyxOneapiDevice device{};
        device.name = buffers.queue.oneapi_device_name.c_str();
        device.platform = buffers.queue.oneapi_platform_name.c_str();
        device.is_gpu = -1;  // match by name; names do not collide across types
        char error[1024] = {};
        const auto buffer = [](const NeuralDeviceBuffer& b) { return CyxOneapiBuffer{b.handle, b.elements}; };
        if (request.op == NeuralOp::DeviceProbe) {
            if (buffers.inputs.size() != 1 || buffers.outputs.size() != 1 ||
                buffers.inputs[0].elements != request.elements || buffers.outputs[0].elements != request.elements) {
                return Fail(BackendFallbackReason::OneapiProviderUnsupportedContract,
                            "device_probe needs one input and one output of request.elements");
            }
            if (library_.probe_(&device, buffer(buffers.inputs[0]), buffer(buffers.outputs[0]), error,
                                sizeof(error)) != 0) {
                return Fail(BackendFallbackReason::OneapiProviderExecutionFailed, error);
            }
            return Ok();
        }
        const bool backward = request.op == NeuralOp::AttentionBackward;
        const bool alibi = request.position_strategy == NeuralPositionStrategy::Alibi;
        const auto& r = request;
        const size_t q_elements = r.head_dim * r.seq * r.batch * r.heads;
        const size_t kv_elements = r.head_dim * r.kv_seq * r.batch * r.kv_heads;
        const size_t rows = r.seq * r.batch * r.heads;
        const auto& in = buffers.inputs;
        const auto& out = buffers.outputs;
        const bool shapes_ok = backward
            ? (in.size() == (alibi ? 7u : 6u) && out.size() == 4 && in[0].elements == q_elements &&
               in[1].elements == kv_elements && in[2].elements == kv_elements && in[3].elements == q_elements &&
               in[4].elements == q_elements && in[5].elements == rows && (!alibi || in[6].elements == r.heads) &&
               out[0].elements == q_elements && out[1].elements == kv_elements && out[2].elements == kv_elements &&
               out[3].elements == rows)
            : (in.size() == (alibi ? 4u : 3u) && out.size() == 2 && in[0].elements == q_elements &&
               in[1].elements == kv_elements && in[2].elements == kv_elements &&
               (!alibi || in[3].elements == r.heads) && out[0].elements == q_elements && out[1].elements == rows);
        if (!shapes_ok) {
            return Fail(BackendFallbackReason::OneapiProviderUnsupportedContract,
                        "attention buffers do not match the request");
        }
        CyxOneapiAttentionArgs args{};
        args.q = buffer(in[0]);
        args.k = buffer(in[1]);
        args.v = buffer(in[2]);
        if (alibi) args.slopes = buffer(in[backward ? 6 : 3]);
        if (backward) {
            args.o = buffer(in[3]);
            args.d_o = buffer(in[4]);
            args.lse = buffer(in[5]);
            args.d_q = buffer(out[0]);
            args.d_k = buffer(out[1]);
            args.d_v = buffer(out[2]);
            args.delta = buffer(out[3]);
        } else {
            args.o = buffer(out[0]);
            args.lse = buffer(out[1]);
        }
        args.batch = static_cast<int>(r.batch);
        args.heads = static_cast<int>(r.heads);
        args.kv_heads = static_cast<int>(r.kv_heads);
        args.seq = static_cast<int>(r.seq);
        args.kv_seq = static_cast<int>(r.kv_seq);
        args.head_dim = static_cast<int>(r.head_dim);
        args.query_offset = static_cast<int>(r.query_offset);
        args.causal = r.causal ? 1 : 0;
        args.window = static_cast<int>(r.sliding_window);
        args.softcap = r.logit_softcap;
        args.scale = r.softmax_scale;
        args.dropout = r.training ? r.attention_dropout : 0.0f;
        args.seed = r.dropout_seed;
        const int status = backward ? library_.backward_(&device, &args, error, sizeof(error))
                                    : library_.forward_(&device, &args, error, sizeof(error));
        if (status != 0) return Fail(BackendFallbackReason::OneapiProviderExecutionFailed, error);
        return Ok();
    }

private:
    mutable KernelLibrary library_;
};

}  // namespace

void RegisterOneapiSyclNeuralProvider(NeuralProviderRegistry& registry) {
    const std::filesystem::path library = BackendDirectory() / kKernelLibrary;
    std::error_code ec;
    if (!std::filesystem::exists(library, ec)) {
        spdlog::info("oneAPI neural provider not registered (reason={}): {} is not installed",
                     BackendFallbackReasonName(BackendFallbackReason::OneapiProviderUnavailable), library.string());
        return;
    }
    auto provider = std::make_shared<OneapiSyclProvider>(library);
    spdlog::info("oneAPI neural provider registered: {}", provider->Version());
    registry.Register(std::move(provider));
}

}  // namespace cyxwiz
