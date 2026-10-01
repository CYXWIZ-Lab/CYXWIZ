#pragma once

// Numerical gradient check for a module's hand-written Backward (TOFIX125
// P2 step 2.4, decision D1: extension authors write the gradient and the
// engine checks it). Central differences of L = sum(Forward(x) * r) for a
// fixed random r, compared with Backward(r) for the input and GetGradients()
// for every parameter. Used by both routes: C++ plugins and Python nodes.

#include <cyxwiz/sequential.h>

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
    size_t max_elements = 48;      // elements probed per tensor
};

struct GradientCheckResult {
    bool passed = false;
    double tolerance = 0.0;
    double max_relative_error = 0.0;
    std::string worst;   // where the largest error is ("input", "parameter a")
    std::string error;   // set when the module could not be checked at all
};

// make: creates a fresh module for an input shape (the factory's CreateModule
// with the node's parameters). The module runs in evaluation mode.
GradientCheckResult CheckModuleGradients(
    const std::function<std::unique_ptr<Module>(const std::vector<size_t>& input_shape)>& make,
    const GradientCheckOptions& options = {});

// "Passed: largest relative error 3.1e-07 (tolerance 2.0e-02)".
std::string DescribeGradientCheck(const GradientCheckResult& result);

}  // namespace cyxwiz
