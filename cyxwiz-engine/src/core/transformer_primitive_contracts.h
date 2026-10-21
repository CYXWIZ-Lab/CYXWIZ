#pragma once

#include <cstddef>
#include <vector>

namespace cyxwiz {

struct LayerNormBackwardResult {
    std::vector<float> grad_input;
    std::vector<float> grad_gamma;
    std::vector<float> grad_beta;
};

std::vector<float> LayerNormForwardCpu(
    const std::vector<float>& input,
    size_t outer_size,
    size_t normalized_size,
    const std::vector<float>& gamma = {},
    const std::vector<float>& beta = {},
    float epsilon = 1.0e-5f);

LayerNormBackwardResult LayerNormBackwardCpu(
    const std::vector<float>& input,
    const std::vector<float>& grad_output,
    size_t outer_size,
    size_t normalized_size,
    const std::vector<float>& gamma = {},
    float epsilon = 1.0e-5f);

} // namespace cyxwiz
