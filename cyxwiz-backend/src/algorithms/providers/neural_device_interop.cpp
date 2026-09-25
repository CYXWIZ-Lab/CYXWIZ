// v2 device-resident provider execution (tofix112 phase 5b, package 1).
//
// ArrayFire owns device memory and the queue; providers enqueue kernels on
// that same queue using the arrays' raw device buffers, so no host copy or
// synchronization separates ArrayFire work from provider work. The backend
// links ArrayFire's unified library, so the per-backend queue accessors
// (afcu_get_stream in afcuda, afcl_get_* in afopencl) are resolved from the
// backend libraries the unified loader has already loaded.
//
// oneAPI is staged instead: ArrayFire 3.10 exposes neither its SYCL queue
// nor its context, and a provider queue in another context cannot safely
// share ArrayFire's pooled sycl::buffers (the SYCL runtime's per-context
// coherence goes stale when ArrayFire reuses a buffer the provider wrote;
// measured on the oneAPI CPU device, tofix112). Inputs are read out through
// ArrayFire, the provider runs on its own buffers, and outputs come back as
// new ArrayFire arrays; handles are host float* for that platform.
#include "cyxwiz/neural_provider.h"
#include "../arrayfire_backend_utils.h"
#include "../arrayfire_host_materialization.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

#ifdef CYXWIZ_HAS_ARRAYFIRE
namespace {

void* FindLoadedSymbol(const char* windows_library, const char* posix_library, const char* symbol) {
#ifdef _WIN32
    (void)posix_library;
    HMODULE module = GetModuleHandleA(windows_library);
    return module ? reinterpret_cast<void*>(GetProcAddress(module, symbol)) : nullptr;
#else
    (void)windows_library;
    void* handle = dlopen(posix_library, RTLD_LAZY | RTLD_NOLOAD);
    void* found = handle ? dlsym(handle, symbol) : nullptr;
    if (handle) dlclose(handle);
    return found;
#endif
}

// Parses the active device's line of af_info_string: the active device is
// marked "[N]" (others "-N-"), formatted "<platform>: <name>, <memory> MB (...)".
bool ActiveOneapiDeviceIdentity(std::string& name, std::string& platform) {
    char* info = nullptr;
    if (af_info_string(&info, false) != AF_SUCCESS || !info) return false;
    const std::string text(info);
    af_free_host(info);
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(start, end - start);
        start = end + 1;
        if (line.empty() || line[0] != '[') continue;
        const size_t marker = line.find("] ");
        const size_t colon = line.find(": ", marker == std::string::npos ? 0 : marker);
        const size_t memory = line.rfind(" MB");
        const size_t comma = memory == std::string::npos ? std::string::npos : line.rfind(", ", memory);
        if (marker == std::string::npos || colon == std::string::npos || comma == std::string::npos ||
            comma <= colon) {
            return false;
        }
        platform = line.substr(marker + 2, colon - marker - 2);
        name = line.substr(colon + 2, comma - colon - 2);
        return !name.empty();
    }
    return false;
}

bool CaptureArrayFireQueue(NeuralDeviceQueue& queue, std::string& error) {
    const af::Backend backend = af::getActiveBackend();
    const int device = af::getDevice();
    if (backend == AF_BACKEND_CUDA) {
        using GetStream = af_err (*)(void**, int);
        using GetNativeId = af_err (*)(int*, int);
        const auto get_stream = reinterpret_cast<GetStream>(
            FindLoadedSymbol("afcuda.dll", "libafcuda.so", "afcu_get_stream"));
        const auto get_native = reinterpret_cast<GetNativeId>(
            FindLoadedSymbol("afcuda.dll", "libafcuda.so", "afcu_get_native_id"));
        if (!get_stream || !get_native) {
            error = "ArrayFire CUDA queue accessors (afcu_get_stream/afcu_get_native_id) are unavailable";
            return false;
        }
        if (get_stream(&queue.cuda_stream, device) != AF_SUCCESS ||
            get_native(&queue.native_device, device) != AF_SUCCESS) {
            error = "ArrayFire CUDA stream query failed";
            return false;
        }
        queue.platform = DeviceType::CUDA;
        return true;
    }
    if (backend == AF_BACKEND_OPENCL) {
        using GetHandle = af_err (*)(void**, bool);
        using GetDevice = af_err (*)(void**);
        const auto get_context = reinterpret_cast<GetHandle>(
            FindLoadedSymbol("afopencl.dll", "libafopencl.so", "afcl_get_context"));
        const auto get_queue = reinterpret_cast<GetHandle>(
            FindLoadedSymbol("afopencl.dll", "libafopencl.so", "afcl_get_queue"));
        const auto get_device = reinterpret_cast<GetDevice>(
            FindLoadedSymbol("afopencl.dll", "libafopencl.so", "afcl_get_device_id"));
        if (!get_context || !get_queue || !get_device) {
            error = "ArrayFire OpenCL queue accessors (afcl_get_context/queue/device_id) are unavailable";
            return false;
        }
        if (get_context(&queue.cl_context, false) != AF_SUCCESS ||
            get_queue(&queue.cl_queue, false) != AF_SUCCESS ||
            get_device(&queue.cl_device) != AF_SUCCESS) {
            error = "ArrayFire OpenCL context/queue query failed";
            return false;
        }
        queue.platform = DeviceType::OPENCL;
        queue.native_device = device;
        return true;
    }
    if (backend == AF_BACKEND_ONEAPI) {
        // No queue accessor on this backend (af/oneapi.h is empty) and no
        // af::deviceInfo either; the provider selects the device by the
        // identity in ArrayFire's device listing.
        if (!ActiveOneapiDeviceIdentity(queue.oneapi_device_name, queue.oneapi_platform_name)) {
            error = "could not identify the active ArrayFire oneAPI device";
            return false;
        }
        queue.platform = DeviceType::ONEAPI;
        queue.native_device = device;
        return true;
    }
    error = "device-resident execution needs the ArrayFire CUDA, OpenCL or oneAPI backend";
    return false;
}

// Locks an ArrayFire array's device buffer for the lifetime of the object.
class LockedDeviceArray {
public:
    explicit LockedDeviceArray(const af::array& array) : array_(array) {
        void* pointer = nullptr;
        if (af_get_device_ptr(&pointer, array_.get()) == AF_SUCCESS) {
            buffer_.handle = pointer;
            buffer_.elements = static_cast<size_t>(array_.elements());
        }
    }
    ~LockedDeviceArray() {
        if (buffer_.handle) af_unlock_array(array_.get());
    }
    LockedDeviceArray(const LockedDeviceArray&) = delete;
    LockedDeviceArray& operator=(const LockedDeviceArray&) = delete;
    const NeuralDeviceBuffer& Buffer() const { return buffer_; }
    const af::array& Array() const { return array_; }

private:
    af::array array_;
    NeuralDeviceBuffer buffer_;
};

NeuralOpStatus Failed(const std::string& detail) {
    NeuralOpStatus status;
    status.reason = BackendFallbackReason::BackendInternalError;
    status.detail = detail;
    return status;
}

af::dim4 SemanticDims(const std::vector<size_t>& shape) {
    af::dim4 dims(1, 1, 1, 1);
    for (size_t d = 0; d < shape.size() && d < 4; ++d) dims[static_cast<unsigned>(d)] = static_cast<dim_t>(shape[d]);
    return dims;
}

// oneAPI: inputs through host memory, provider-owned device buffers, outputs
// uploaded as new ArrayFire arrays (see the file comment).
NeuralOpStatus ExecuteHostStaged(INeuralNetworkProvider& provider, const NeuralOpRequest& request,
                                 const std::vector<const Tensor*>& inputs,
                                 const std::vector<std::vector<size_t>>& output_shapes,
                                 NeuralDeviceOpBuffers& buffers, std::vector<Tensor>& outputs) {
    std::vector<std::vector<float>> host_inputs;
    host_inputs.reserve(inputs.size());
    for (const Tensor* input : inputs) {
        if (!input || input->GetDataType() != DataType::Float32) {
            return Failed("device-resident inputs must be Float32 tensors");
        }
        const af::array& array = input->GetSemanticArray();
        host_inputs.emplace_back(static_cast<size_t>(array.elements()));
        MaterializeArrayFireToHost(array, host_inputs.back().data(), ArrayFireHostSyncCategory::ProviderHostStaging,
                                   std::string("ExecuteNeuralOpOnDevice:") + NeuralOpName(request.op), "semantic",
                                   "oneapi_provider_host_staging");
        buffers.inputs.push_back({host_inputs.back().data(), host_inputs.back().size()});
    }
    std::vector<std::vector<float>> host_outputs;
    host_outputs.reserve(output_shapes.size());
    for (const auto& shape : output_shapes) {
        size_t count = 1;
        for (size_t d : shape) count *= d;
        host_outputs.emplace_back(count);
        buffers.outputs.push_back({host_outputs.back().data(), count});
    }
    const NeuralOpStatus status = provider.ExecuteDevice(request, buffers);
    if (!status.ok) return status;
    outputs.clear();
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        outputs.push_back(Tensor::FromSemanticArray(af::array(SemanticDims(output_shapes[i]), host_outputs[i].data()),
                                                    output_shapes[i]));
    }
    return status;
}

}  // namespace
#endif

NeuralOpStatus ExecuteNeuralOpOnDevice(INeuralNetworkProvider& provider, const NeuralOpRequest& request,
                                       const std::vector<const Tensor*>& inputs,
                                       const std::vector<std::vector<size_t>>& output_shapes,
                                       std::vector<Tensor>& outputs) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!request.device_resident) {
        return Failed("ExecuteNeuralOpOnDevice needs a request with device_resident=true");
    }
    try {
        NeuralDeviceOpBuffers buffers;
        std::string error;
        if (!CaptureArrayFireQueue(buffers.queue, error)) return Failed(error);
        if (buffers.queue.platform != provider.Platform()) {
            return Failed(std::string("provider serves ") + NeuralDevicePlatformName(provider.Platform()) +
                          " but the active ArrayFire backend is " +
                          NeuralDevicePlatformName(buffers.queue.platform));
        }
        if (buffers.queue.platform == DeviceType::ONEAPI) {
            return ExecuteHostStaged(provider, request, inputs, output_shapes, buffers, outputs);
        }
        std::vector<std::unique_ptr<LockedDeviceArray>> locked;
        for (const Tensor* input : inputs) {
            if (!input || input->GetDataType() != DataType::Float32) {
                return Failed("device-resident inputs must be Float32 tensors");
            }
            locked.push_back(std::make_unique<LockedDeviceArray>(input->GetSemanticArray()));
            if (!locked.back()->Buffer().handle) return Failed("input device buffer could not be locked");
            buffers.inputs.push_back(locked.back()->Buffer());
        }
        std::vector<std::unique_ptr<LockedDeviceArray>> locked_outputs;
        for (const auto& shape : output_shapes) {
            dim_t count = 1;
            for (size_t d : shape) count *= static_cast<dim_t>(d);
            af::array storage(count, f32);
            locked_outputs.push_back(std::make_unique<LockedDeviceArray>(storage));
            if (!locked_outputs.back()->Buffer().handle) return Failed("output device buffer could not be locked");
            buffers.outputs.push_back(locked_outputs.back()->Buffer());
        }
        const NeuralOpStatus status = provider.ExecuteDevice(request, buffers);
        if (!status.ok) return status;
        outputs.clear();
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            // Semantic arrays of CyxWiz Tensors use the shape's own dims
            // (dim0 = shape[0]); providers write in that element order.
            const auto& shape = output_shapes[i];
            af::dim4 dims(1, 1, 1, 1);
            for (size_t d = 0; d < shape.size() && d < 4; ++d) dims[static_cast<unsigned>(d)] = static_cast<dim_t>(shape[d]);
            outputs.push_back(Tensor::FromSemanticArray(af::moddims(locked_outputs[i]->Array(), dims), shape));
        }
        return status;
    } catch (const af::exception& e) {
        // Not a native-CPU fallback: the caller keeps its ArrayFire path.
        NeuralOpStatus status = Failed(std::string("ArrayFire error in device-resident execution: ") + e.what());
        status.reason = ClassifyArrayFireBackendFallbackReason(e.what());
        return status;
    }
#else
    (void)provider; (void)request; (void)inputs; (void)output_shapes; (void)outputs;
    NeuralOpStatus status;
    status.reason = BackendFallbackReason::BackendInternalError;
    status.detail = "ArrayFire is not compiled into this backend";
    return status;
#endif
}

}  // namespace cyxwiz
