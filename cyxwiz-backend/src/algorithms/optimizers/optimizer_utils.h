#pragma once

#include <string>

namespace cyxwiz {

enum class BackendFallbackReason;
class Tensor;

namespace optimizer_detail {

bool OptimizerArrayFireAvailable();
void LogOptimizerFallbackOnce(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    const char* error_message);
void LogOptimizerFallbackOnce(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    BackendFallbackReason reason,
    const char* error_message);
void ValidateOptimizerStepTensors(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    const Tensor& gradient);

} // namespace optimizer_detail
} // namespace cyxwiz
