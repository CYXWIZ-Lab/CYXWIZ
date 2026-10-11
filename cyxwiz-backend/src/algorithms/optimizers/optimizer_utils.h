#pragma once

#include <string>

#include <arrayfire.h>

namespace cyxwiz {

class Tensor;

namespace optimizer_detail {

// One ArrayFire path (the CPU is ArrayFire's CPU backend): a device error is
// reported, not hidden behind host loops.
[[noreturn]] void ThrowOptimizerDeviceError(const char* operation_name, const af::exception& error);
void ValidateOptimizerStepTensors(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    const Tensor& gradient);

} // namespace optimizer_detail
} // namespace cyxwiz
