#pragma once

#include "cyxwiz/tensor.h"
#include <cstddef>

namespace cyxwiz {

enum class BackendFallbackReason;

size_t Pool4DIndex(size_t h, size_t w, size_t c, size_t b,
                   size_t width, size_t channels, size_t batch_size);

void ValidateSpatial4DInput(const Tensor& input, const char* name);
void ValidatePoolInput(const Tensor& input, const char* name);
size_t CheckedSpatialPaddedExtent(size_t input_extent,
                                  int padding,
                                  const char* layer_name);
size_t CheckedLayerProduct(size_t left,
                           size_t right,
                           const char* layer_name,
                           const char* quantity);
void RecordLayerArrayFireFallback(const char* operation_name,
                                  BackendFallbackReason reason,
                                  const char* error_message,
                                  const Tensor& tensor,
                                  const char* tensor_name);
void RecordLayerArrayFireFallback(const char* operation_name,
                                  const char* error_message,
                                  const Tensor& tensor,
                                  const char* tensor_name);

struct ResizeLinearSample {
    size_t lower = 0;
    size_t upper = 0;
    float upper_weight = 0.0f;
};

ResizeLinearSample ComputeResizeLinearSample(size_t out_index, size_t in_size, int scale_factor);

} // namespace cyxwiz
