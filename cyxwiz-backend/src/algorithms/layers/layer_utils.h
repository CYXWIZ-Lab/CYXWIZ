#pragma once

#include "cyxwiz/tensor.h"
#include <cstddef>
#include <stdexcept>

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

// tofix67 slice 7: like RecordLayerArrayFireFallback, but ALSO records a
// placement observation the compiler can consume. node_type_name must be
// the compiler's node-type spelling (e.g. "Conv2D", not
// "Conv2DLayer::Forward"); layer_input must be the layer's runtime input
// (Backward sites pass their cached forward input) so the input-only key
// matches the compile-time lookup. Exported for placement-evidence tests.
CYXWIZ_API void RecordLayerArrayFireFallbackObservation(
    const char* operation_name,
    const char* node_type_name,
    BackendFallbackReason reason,
    const char* error_message,
    const Tensor& layer_input,
    const char* tensor_name);
CYXWIZ_API void RecordLayerArrayFireFallbackObservation(
    const char* operation_name,
    const char* node_type_name,
    const char* error_message,
    const Tensor& layer_input,
    const char* tensor_name);

struct ResizeLinearSample {
    size_t lower = 0;
    size_t upper = 0;
    float upper_weight = 0.0f;
};

// Keep integer coordinates exact even beyond Float32's consecutive-integer
// range. constexpr permits allocation-free geometry checks at this private
// boundary without exporting a test-only DLL interface.
constexpr ResizeLinearSample ComputeResizeLinearSample(size_t out_index, size_t in_size,
                                                       int scale_factor) {
    if (in_size == 0 || scale_factor <= 0) {
        throw std::invalid_argument("Resize requires a positive extent and scale");
    }
    const size_t scale = static_cast<size_t>(scale_factor);
    const size_t base = out_index / scale;
    if (base >= in_size) {
        throw std::out_of_range("Resize output coordinate is outside the scaled extent");
    }
    const float delta = (static_cast<float>(out_index % scale) + 0.5f) /
                            static_cast<float>(scale_factor) - 0.5f;
    if (delta < 0.0f) {
        return base == 0 ? ResizeLinearSample{0, 0, 0.0f}
                         : ResizeLinearSample{base - 1, base, 1.0f + delta};
    }
    return base == in_size - 1 ? ResizeLinearSample{base, base, 0.0f}
                               : ResizeLinearSample{base, base + 1, delta};
}

} // namespace cyxwiz
