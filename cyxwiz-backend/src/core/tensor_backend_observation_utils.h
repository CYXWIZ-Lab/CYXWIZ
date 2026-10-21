#pragma once

#include "cyxwiz/tensor.h"

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

} // namespace cyxwiz::tensor_backend_observation
