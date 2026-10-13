#include "optimizer_utils.h"

#include "cyxwiz/tensor.h"

#include <stdexcept>

#include <arrayfire.h>

namespace cyxwiz {
namespace optimizer_detail {

void ThrowOptimizerDeviceError(const char* operation_name, const af::exception& error) {
    throw std::runtime_error(std::string(operation_name) + " failed on the ArrayFire device: " + error.what());
}

void ValidateOptimizerStepTensors(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    const Tensor& gradient) {
    const std::string operation =
        operation_name == nullptr ? "Optimizer::Step" : operation_name;
    const std::string name =
        parameter_name.empty() ? "parameter" : parameter_name;
    if (parameter.GetDataType() != DataType::Float32 ||
        gradient.GetDataType() != DataType::Float32) {
        throw std::invalid_argument(
            operation + " requires Float32 parameter and gradient tensors for '" +
            name + "'.");
    }
    if (parameter.Shape() != gradient.Shape()) {
        throw std::invalid_argument(
            operation + " gradient shape does not match parameter '" + name +
            "'.");
    }
}

} // namespace optimizer_detail
} // namespace cyxwiz
