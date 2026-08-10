#pragma once

#include <string>

namespace cyxwiz {

class Tensor;

namespace optimizer_detail {

bool OptimizerArrayFireAvailable();
void LogOptimizerFallbackOnce(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    const char* error_message);

} // namespace optimizer_detail
} // namespace cyxwiz
