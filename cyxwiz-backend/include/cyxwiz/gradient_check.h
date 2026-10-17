#pragma once

// Numerical gradient check for a module's hand-written Backward (TOFIX125
// P2 step 2.4, decision D1: extension authors write the gradient and the
// engine checks it). Central differences of L = sum(Forward(x) * r) for a
// fixed random r, compared with Backward(r) for the input and GetGradients()
// for every parameter. Lives in the backend so both routes run the same
// code: the Engine (C++ plugin nodes, Properties) and pycyxwiz.gradcheck.

#include "api_export.h"
#include "sequential.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

struct GradientCheckOptions {
    // Shapes checked, each including the batch dimension.
    std::vector<std::vector<size_t>> input_shapes = {{2, 3}, {3, 4, 5}};
    uint32_t seed = 125;
    float input_range = 2.0f;      // inputs uniform in [-range, range]
    double step = 1e-2;            // finite-difference step
    double tolerance = 2e-2;       // largest accepted relative error
    // Relative error is |analytic - numeric| / max(floor, |analytic| + |numeric|).
    // The floor keeps 32-bit rounding on a gradient near zero (Snake where
    // sin(2ax) is close to -1: true value 4e-4, rounding 6e-5) from reading
    // as a 6% error; wrong gradients differ by 0.1 or more.
    double absolute_floor = 5e-2;
    size_t max_elements = 48;      // elements probed per tensor
};

struct GradientCheckResult {
    bool passed = false;
    double tolerance = 0.0;
    double max_relative_error = 0.0;
    std::string worst;   // where the largest error is ("input", "parameter a")
    std::string error;   // set when the module could not be checked at all
};

// make: creates a fresh module for an input shape that includes the batch
// dimension. Callers that use a factory pass it the shape without the batch
// (CreateModule takes one sample's shape, as the model builder passes it).
// The module runs in evaluation mode.
CYXWIZ_API GradientCheckResult CheckModuleGradients(
    const std::function<std::unique_ptr<Module>(const std::vector<size_t>& input_shape)>& make,
    const GradientCheckOptions& options = {});

// "Passed: largest relative error 3.1e-07 (tolerance 2.0e-02)".
CYXWIZ_API std::string DescribeGradientCheck(const GradientCheckResult& result);

}  // namespace cyxwiz
