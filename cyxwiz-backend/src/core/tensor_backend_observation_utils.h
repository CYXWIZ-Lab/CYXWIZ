#pragma once

#include "cyxwiz/backend_placement_observation.h"
#include "cyxwiz/tensor.h"

#include "../algorithms/arrayfire_backend_utils.h"

#include <string>
#include <vector>

namespace cyxwiz::tensor_backend_observation {

inline const char* DataTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float64: return "float64";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
    }
    return "unknown";
}

inline void RecordArrayFireFallback(
    const char* operation_name,
    const std::string& dtype,
    const std::string& shape_signature,
    const char* error_message) {
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context =
        BuildArrayFireBackendFallbackContext(shape_signature);
    const std::string message = BuildArrayFireBackendFallbackMessage(
        operation_name,
        reason,
        reason != BackendFallbackReason::CudaJitParamOverflow,
        error_message,
        context);
    RecordBackendPlacementObservationForActiveDevice(
        operation_name,
        "cuda",
        dtype,
        shape_signature,
        BackendFallbackReasonName(reason),
        BackendPlacementObservationSource::RuntimeFallback,
        message);
}

inline std::string BuildTensorOpSignature(
    const std::vector<std::vector<size_t>>& input_shapes,
    const std::vector<size_t>& output_shape,
    DataType dtype,
    const std::string& attributes) {
    return BuildTensorOpPlacementShapeSignature(
        input_shapes,
        output_shape,
        DataTypeName(dtype),
        attributes);
}

} // namespace cyxwiz::tensor_backend_observation
