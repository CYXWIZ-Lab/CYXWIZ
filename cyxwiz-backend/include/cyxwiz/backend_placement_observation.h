#pragma once

#include "api_export.h"
#include "recurrent_cuda_placement.h"

#include <string>
#include <vector>

namespace cyxwiz {

namespace BackendPlacementObservationReason {
inline constexpr const char* CudaJitParamOverflow = "cuda_jit_param_overflow";
inline constexpr const char* ArrayFireJitCompileFailure =
    "arrayfire_jit_compile_failure";
inline constexpr const char* GpuBackendException = "gpu_backend_exception";
inline constexpr const char* GpuOutOfMemory = "gpu_out_of_memory";
inline constexpr const char* UnsupportedDtype = "unsupported_dtype";
inline constexpr const char* UnsupportedShape = "unsupported_shape";
inline constexpr const char* BackendCompileTimeout = "backend_compile_timeout";
inline constexpr const char* BackendInternalError = "backend_internal_error";
} // namespace BackendPlacementObservationReason

namespace BackendPlacementObservationSource {
inline constexpr const char* RuntimeFallback = "runtime_fallback";
inline constexpr const char* PreflightProbe = "preflight_probe";
inline constexpr const char* Test = "test";
} // namespace BackendPlacementObservationSource

struct BackendPlacementObservation {
    std::string op_type;
    std::string backend;
    std::string device;
    std::string dtype;
    std::string shape_signature;
    std::string reason_code;
    std::string source = BackendPlacementObservationSource::RuntimeFallback;
    std::string detail;
};

CYXWIZ_API std::string BuildRecurrentCudaPlacementShapeSignature(
    const RecurrentCudaPlacementRequest& request);

CYXWIZ_API std::string BuildDensePlacementShapeSignature(
    const std::vector<size_t>& input_shape,
    size_t out_features);

CYXWIZ_API std::string BuildTensorLayerPlacementShapeSignature(
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& output_shape);

CYXWIZ_API std::string CurrentBackendPlacementDeviceSignature();

CYXWIZ_API void RecordBackendPlacementObservation(
    const BackendPlacementObservation& observation);

CYXWIZ_API void RecordBackendPlacementObservationForActiveDevice(
    const std::string& op_type,
    const std::string& backend,
    const std::string& dtype,
    const std::string& shape_signature,
    const std::string& reason_code,
    const std::string& source,
    const std::string& detail);

CYXWIZ_API bool TryGetBackendPlacementObservation(
    const std::string& op_type,
    const std::string& backend,
    const std::string& device,
    const std::string& dtype,
    const std::string& shape_signature,
    BackendPlacementObservation& observation);

CYXWIZ_API bool TryGetBackendPlacementObservationForActiveDevice(
    const std::string& op_type,
    const std::string& backend,
    const std::string& dtype,
    const std::string& shape_signature,
    BackendPlacementObservation& observation);

CYXWIZ_API void RecordRecurrentCudaPlacementObservation(
    const RecurrentCudaPlacementRequest& request,
    const std::string& reason_code,
    const std::string& source,
    const std::string& detail);

CYXWIZ_API void RecordRecurrentCudaPreflightProbeFailure(
    const RecurrentCudaPlacementRequest& request,
    const std::string& reason_code,
    const std::string& detail);

CYXWIZ_API bool TryRunRecurrentCudaPreflightProbe(
    const RecurrentCudaPlacementRequest& request,
    BackendPlacementObservation& failure_observation);

CYXWIZ_API bool TryGetRecurrentCudaPlacementObservation(
    const RecurrentCudaPlacementRequest& request,
    BackendPlacementObservation& observation);

CYXWIZ_API void ClearBackendPlacementObservationCacheForTesting();

} // namespace cyxwiz
