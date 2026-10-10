#pragma once

#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

class Tensor;

namespace optimizer_detail {

// One ArrayFire path (the CPU is ArrayFire's CPU backend): a device error is
// reported, not hidden behind host loops; a build without ArrayFire refuses.
[[noreturn]] void ThrowOptimizerNeedsArrayFire(const char* operation_name);
#ifdef CYXWIZ_HAS_ARRAYFIRE
[[noreturn]] void ThrowOptimizerDeviceError(const char* operation_name, const af::exception& error);
#endif
void ValidateOptimizerStepTensors(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    const Tensor& gradient);

} // namespace optimizer_detail
} // namespace cyxwiz
