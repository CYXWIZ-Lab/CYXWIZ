#pragma once

#include "cyxwiz/tensor.h"
#include <cstddef>

namespace cyxwiz {

size_t Pool4DIndex(size_t h, size_t w, size_t c, size_t b,
                   size_t width, size_t channels, size_t batch_size);

void ValidateSpatial4DInput(const Tensor& input, const char* name);
void ValidatePoolInput(const Tensor& input, const char* name);

struct ResizeLinearSample {
    size_t lower = 0;
    size_t upper = 0;
    float upper_weight = 0.0f;
};

ResizeLinearSample ComputeResizeLinearSample(size_t out_index, size_t in_size, int scale_factor);

} // namespace cyxwiz
