#include "core/transformer_primitive_contracts.h"

#include <cmath>
#include <stdexcept>

namespace cyxwiz {

std::vector<float> LayerNormForwardCpu(
    const std::vector<float>& input,
    size_t outer_size,
    size_t normalized_size,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    float epsilon) {

    if (outer_size == 0) {
        throw std::invalid_argument("LayerNormForwardCpu requires outer_size > 0");
    }
    if (normalized_size == 0) {
        throw std::invalid_argument(
            "LayerNormForwardCpu requires normalized_size > 0");
    }
    if (input.size() != outer_size * normalized_size) {
        throw std::invalid_argument(
            "LayerNormForwardCpu input shape mismatch");
    }
    if (!gamma.empty() && gamma.size() != normalized_size) {
        throw std::invalid_argument(
            "LayerNormForwardCpu gamma shape mismatch");
    }
    if (!beta.empty() && beta.size() != normalized_size) {
        throw std::invalid_argument(
            "LayerNormForwardCpu beta shape mismatch");
    }
    if (epsilon <= 0.0f) {
        throw std::invalid_argument(
            "LayerNormForwardCpu requires epsilon > 0");
    }

    std::vector<float> output(input.size(), 0.0f);
    for (size_t row = 0; row < outer_size; ++row) {
        const size_t offset = row * normalized_size;

        double mean = 0.0;
        for (size_t col = 0; col < normalized_size; ++col) {
            mean += input[offset + col];
        }
        mean /= static_cast<double>(normalized_size);

        double variance = 0.0;
        for (size_t col = 0; col < normalized_size; ++col) {
            const double centered =
                static_cast<double>(input[offset + col]) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(normalized_size);

        const float inv_std =
            1.0f / std::sqrt(static_cast<float>(variance) + epsilon);
        for (size_t col = 0; col < normalized_size; ++col) {
            float value =
                (input[offset + col] - static_cast<float>(mean)) * inv_std;
            if (!gamma.empty()) {
                value *= gamma[col];
            }
            if (!beta.empty()) {
                value += beta[col];
            }
            output[offset + col] = value;
        }
    }

    return output;
}

LayerNormBackwardResult LayerNormBackwardCpu(
    const std::vector<float>& input,
    const std::vector<float>& grad_output,
    size_t outer_size,
    size_t normalized_size,
    const std::vector<float>& gamma,
    float epsilon) {

    if (outer_size == 0) {
        throw std::invalid_argument("LayerNormBackwardCpu requires outer_size > 0");
    }
    if (normalized_size == 0) {
        throw std::invalid_argument(
            "LayerNormBackwardCpu requires normalized_size > 0");
    }
    if (input.size() != outer_size * normalized_size ||
        grad_output.size() != input.size()) {
        throw std::invalid_argument(
            "LayerNormBackwardCpu input/gradient shape mismatch");
    }
    if (!gamma.empty() && gamma.size() != normalized_size) {
        throw std::invalid_argument(
            "LayerNormBackwardCpu gamma shape mismatch");
    }
    if (epsilon <= 0.0f) {
        throw std::invalid_argument(
            "LayerNormBackwardCpu requires epsilon > 0");
    }

    LayerNormBackwardResult result;
    result.grad_input.assign(input.size(), 0.0f);
    result.grad_gamma.assign(normalized_size, 0.0f);
    result.grad_beta.assign(normalized_size, 0.0f);

    for (size_t row = 0; row < outer_size; ++row) {
        const size_t offset = row * normalized_size;

        double mean = 0.0;
        for (size_t col = 0; col < normalized_size; ++col) {
            mean += input[offset + col];
        }
        mean /= static_cast<double>(normalized_size);

        double variance = 0.0;
        for (size_t col = 0; col < normalized_size; ++col) {
            const double centered =
                static_cast<double>(input[offset + col]) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(normalized_size);

        const float inv_std =
            1.0f / std::sqrt(static_cast<float>(variance) + epsilon);
        std::vector<float> normalized(normalized_size, 0.0f);
        std::vector<float> grad_normalized(normalized_size, 0.0f);

        float grad_normalized_sum = 0.0f;
        float grad_normalized_dot_normalized = 0.0f;
        for (size_t col = 0; col < normalized_size; ++col) {
            normalized[col] =
                (input[offset + col] - static_cast<float>(mean)) * inv_std;
            const float upstream = grad_output[offset + col];
            result.grad_beta[col] += upstream;
            result.grad_gamma[col] += upstream * normalized[col];

            const float scale = gamma.empty() ? 1.0f : gamma[col];
            grad_normalized[col] = upstream * scale;
            grad_normalized_sum += grad_normalized[col];
            grad_normalized_dot_normalized +=
                grad_normalized[col] * normalized[col];
        }

        const float n = static_cast<float>(normalized_size);
        for (size_t col = 0; col < normalized_size; ++col) {
            result.grad_input[offset + col] =
                (inv_std / n) *
                ((n * grad_normalized[col]) -
                 grad_normalized_sum -
                 (normalized[col] * grad_normalized_dot_normalized));
        }
    }

    return result;
}

} // namespace cyxwiz
