#include "cyxwiz/layers/pooling.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

namespace {

struct Pool2DGeometry {
    size_t in_h;
    size_t in_w;
    size_t channels;
    size_t batch_size;
    size_t padded_h;
    size_t padded_w;
    size_t out_h;
    size_t out_w;
};

Pool2DGeometry ValidatePoolForwardInput(const Tensor& input,
                                        int pool_size,
                                        int stride,
                                        int padding,
                                        const char* layer_name) {
    ValidatePoolInput(input, layer_name);
    const std::vector<size_t>& shape = input.Shape();
    const size_t padded_h = CheckedSpatialPaddedExtent(
        shape[0], padding, layer_name);
    const size_t padded_w = CheckedSpatialPaddedExtent(
        shape[1], padding, layer_name);
    const size_t window = static_cast<size_t>(pool_size);
    if (padded_h < window || padded_w < window) {
        throw std::runtime_error(
            std::string(layer_name) +
            " pool window is larger than padded input");
    }
    const size_t step = static_cast<size_t>(stride);
    return {
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        padded_h,
        padded_w,
        (padded_h - window) / step + 1,
        (padded_w - window) / step + 1,
    };
}

std::vector<size_t> PoolOutputShape(const Pool2DGeometry& geometry) {
    return {
        geometry.out_h,
        geometry.out_w,
        geometry.channels,
        geometry.batch_size,
    };
}

Pool2DGeometry ValidatePoolBackwardInput(const Tensor& cached_input,
                                         const Tensor& grad_output,
                                         int pool_size,
                                         int stride,
                                         int padding,
                                         const char* layer_name) {
    const Pool2DGeometry geometry = ValidatePoolForwardInput(
        cached_input, pool_size, stride, padding, layer_name);
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error(
            std::string(layer_name) +
            " backward requires Float32 grad_output");
    }
    if (grad_output.Shape() != PoolOutputShape(geometry)) {
        throw std::runtime_error(
            std::string(layer_name) +
            " backward gradient shape does not match Forward output");
    }
    return geometry;
}

#ifndef CYXWIZ_HAS_ARRAYFIRE
[[noreturn]] void ThrowPoolingNeedsArrayFire(const char* layer) {
    throw std::runtime_error(std::string(layer) + " runs on ArrayFire, and this build has no ArrayFire");
}
#endif

#ifdef CYXWIZ_HAS_ARRAYFIRE

// torch's adaptive pooling bins: [floor(i*in/out), ceil((i+1)*in/out)).
size_t AdaptiveBinStart(size_t i, size_t in, size_t out) { return (i * in) / out; }
size_t AdaptiveBinEnd(size_t i, size_t in, size_t out) { return ((i + 1) * in + out - 1) / out; }

// The [out, in] matrix whose row i averages bin i: Y = P_h X P_w^T.
af::array AdaptiveAverageMatrix(size_t in, size_t out) {
    std::vector<float> values(out * in, 0.0f);
    for (size_t i = 0; i < out; ++i) {
        const size_t start = AdaptiveBinStart(i, in, out);
        const size_t end = AdaptiveBinEnd(i, in, out);
        const float weight = 1.0f / static_cast<float>(end - start);
        for (size_t j = start; j < end; ++j) values[i + out * j] = weight;  // column-major
    }
    return af::array(static_cast<dim_t>(out), static_cast<dim_t>(in), values.data());
}

// Every pool runs on the device only (one ArrayFire path; ArrayFire's CPU
// backend is the CPU): an ArrayFire error is reported, not hidden behind a
// host loop.
[[noreturn]] void ThrowPoolingDeviceError(const char* operation, const af::exception& error) {
    throw std::runtime_error(std::string(operation) + " failed on the ArrayFire device: " + error.what());
}

af::array PadSpatialWithValue(const af::array& input,
                              const Pool2DGeometry& geometry,
                              int padding,
                              float value) {
    if (padding == 0) {
        return input;
    }

    af::array padded = af::constant(
        value,
        af::dim4(
            static_cast<dim_t>(geometry.padded_h),
            static_cast<dim_t>(geometry.padded_w),
            static_cast<dim_t>(geometry.channels),
            static_cast<dim_t>(geometry.batch_size)),
        af::dtype::f32);
    const double pad = static_cast<double>(padding);
    padded(
        af::seq(pad, pad + static_cast<double>(geometry.in_h) - 1.0),
        af::seq(pad, pad + static_cast<double>(geometry.in_w) - 1.0),
        af::span,
        af::span) = input;
    padded.eval();
    return padded;
}

// ArrayFire flattens each [window_h, window_w] patch with dimension 0
// varying fastest. CyxWiz and PyTorch expose row-major first-maximum tie
// behavior, so transpose the square patch axes before/after max selection.
af::array TransposeSquarePoolingPatchOrder(
    const af::array& patches,
    dim_t window,
    dim_t patch_count,
    dim_t channels,
    dim_t batch_size) {
    af::array spatial = af::moddims(
        patches,
        af::dim4(window, window, patch_count, channels * batch_size));
    spatial = af::reorder(spatial, 1, 0, 2, 3);
    return af::moddims(
        spatial,
        af::dim4(
            window * window, patch_count, channels, batch_size));
}

#endif

} // namespace

MaxPool2DLayer::MaxPool2DLayer(int pool_size, int stride, int padding)
    : pool_size_(pool_size), stride_(stride == -1 ? pool_size : stride), padding_(padding) {
    if (pool_size_ <= 0 || stride_ <= 0 || padding_ < 0 ||
        padding_ > pool_size_ / 2) {
        throw std::invalid_argument(
            "MaxPool2D requires positive pool_size/stride and padding no "
            "larger than half the pool size");
    }
}

Tensor MaxPool2DLayer::Forward(const Tensor& input) {
    has_forward_ = false;
    const Pool2DGeometry geometry = ValidatePoolForwardInput(
        input, pool_size_, stride_, padding_, "MaxPool2D");
    const std::vector<size_t> output_shape = PoolOutputShape(geometry);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t window = static_cast<dim_t>(pool_size_);
        const dim_t patch_count =
            static_cast<dim_t>(geometry.out_h * geometry.out_w);
        const dim_t channels = static_cast<dim_t>(geometry.channels);
        const dim_t batch_size = static_cast<dim_t>(geometry.batch_size);
        af::array x = PadSpatialWithValue(
            TensorToAf(input), geometry, padding_,
            -std::numeric_limits<float>::infinity());
        af::array patches = af::unwrap(
            x, window, window, stride_, stride_);
        patches = TransposeSquarePoolingPatchOrder(
            patches, window, patch_count, channels, batch_size);

        af::array max_values;
        af::array max_indices;
        af::max(max_values, max_indices, patches, 0);
        max_values = af::moddims(
            max_values,
            af::dim4(
                static_cast<dim_t>(geometry.out_h),
                static_cast<dim_t>(geometry.out_w),
                channels,
                batch_size));
        max_indices = af::moddims(
            max_indices.as(af::dtype::s32), max_values.dims());
        max_values.eval();
        max_indices.eval();

        Tensor result = Tensor::FromSemanticArray(
            max_values, output_shape);
        Tensor index_cache = Tensor::FromSemanticArray(
            max_indices, output_shape);
        cached_input_ = input;
        max_indices_ = std::move(index_cache);
        has_forward_ = true;
        return result;
    } catch (const af::exception& e) {
        ThrowPoolingDeviceError("MaxPool2DLayer::Forward", e);
    }
#else
    ThrowPoolingNeedsArrayFire("Max Pool 2D");
#endif
}

Tensor MaxPool2DLayer::Backward(const Tensor& grad_output) {
    if (!has_forward_) {
        throw std::logic_error(
            "MaxPool2DLayer::Backward requires a successful Forward call");
    }
    const Pool2DGeometry geometry = ValidatePoolBackwardInput(
        cached_input_, grad_output, pool_size_, stride_, padding_,
        "MaxPool2D");
    if (max_indices_.Shape() != grad_output.Shape() ||
        max_indices_.GetDataType() != DataType::Int32) {
        throw std::runtime_error(
            "MaxPool2D backward index cache does not match Forward output");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t window = static_cast<dim_t>(pool_size_);
        const dim_t window_elements = window * window;
        const dim_t patch_count =
            static_cast<dim_t>(geometry.out_h * geometry.out_w);
        const dim_t channels = static_cast<dim_t>(geometry.channels);
        const dim_t batch_size = static_cast<dim_t>(geometry.batch_size);
        af::array grad_columns = af::moddims(
            TensorToAf(grad_output),
            af::dim4(1, patch_count, channels, batch_size));
        af::array index_columns = af::moddims(
            TensorToAf(max_indices_).as(af::dtype::s32),
            af::dim4(1, patch_count, channels, batch_size));
        af::array positions = af::range(
            af::dim4(
                window_elements, patch_count, channels, batch_size),
            0,
            af::dtype::s32);
        af::array row_major_patch_gradients =
            af::tile(
                grad_columns,
                af::dim4(window_elements, 1, 1, 1)) *
            (positions == af::tile(
                index_columns,
                af::dim4(window_elements, 1, 1, 1))).as(
                    af::dtype::f32);
        af::array native_patch_gradients =
            TransposeSquarePoolingPatchOrder(
                row_major_patch_gradients,
                window,
                patch_count,
                channels,
                batch_size);
        af::array dx = af::wrap(
            native_patch_gradients,
            static_cast<dim_t>(geometry.in_h),
            static_cast<dim_t>(geometry.in_w),
            window,
            window,
            stride_,
            stride_,
            padding_,
            padding_);
        dx.eval();
        return Tensor::FromSemanticArray(dx, cached_input_.Shape());
    } catch (const af::exception& e) {
        ThrowPoolingDeviceError("MaxPool2DLayer::Backward", e);
    }
#else
    ThrowPoolingNeedsArrayFire("Max Pool 2D");
#endif
}

// ============================================================================
// AvgPool2D Layer Implementation
// ============================================================================

AvgPool2DLayer::AvgPool2DLayer(int pool_size, int stride, int padding)
    : pool_size_(pool_size), stride_(stride == -1 ? pool_size : stride), padding_(padding) {
    if (pool_size_ <= 0 || stride_ <= 0 || padding_ < 0 ||
        padding_ > pool_size_ / 2) {
        throw std::invalid_argument(
            "AvgPool2D requires positive pool_size/stride and padding no "
            "larger than half the pool size");
    }
}

Tensor AvgPool2DLayer::Forward(const Tensor& input) {
    has_forward_ = false;
    const Pool2DGeometry geometry = ValidatePoolForwardInput(
        input, pool_size_, stride_, padding_, "AvgPool2D");
    const std::vector<size_t> output_shape = PoolOutputShape(geometry);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t window = static_cast<dim_t>(pool_size_);
        af::array patches = af::unwrap(
            TensorToAf(input),
            window,
            window,
            stride_,
            stride_,
            padding_,
            padding_);
        af::array output = af::mean(patches, 0);
        output = af::moddims(
            output,
            af::dim4(
                static_cast<dim_t>(geometry.out_h),
                static_cast<dim_t>(geometry.out_w),
                static_cast<dim_t>(geometry.channels),
                static_cast<dim_t>(geometry.batch_size)));
        output.eval();

        Tensor result = Tensor::FromSemanticArray(output, output_shape);
        cached_input_ = input;
        has_forward_ = true;
        return result;
    } catch (const af::exception& e) {
        ThrowPoolingDeviceError("AvgPool2DLayer::Forward", e);
    }
#else
    ThrowPoolingNeedsArrayFire("Average Pool 2D");
#endif
}

Tensor AvgPool2DLayer::Backward(const Tensor& grad_output) {
    if (!has_forward_) {
        throw std::logic_error(
            "AvgPool2DLayer::Backward requires a successful Forward call");
    }
    const Pool2DGeometry geometry = ValidatePoolBackwardInput(
        cached_input_, grad_output, pool_size_, stride_, padding_,
        "AvgPool2D");

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t window = static_cast<dim_t>(pool_size_);
        const dim_t window_elements = window * window;
        const dim_t patch_count =
            static_cast<dim_t>(geometry.out_h * geometry.out_w);
        const dim_t channels = static_cast<dim_t>(geometry.channels);
        const dim_t batch_size = static_cast<dim_t>(geometry.batch_size);
        af::array grad_columns = af::moddims(
            TensorToAf(grad_output),
            af::dim4(1, patch_count, channels, batch_size));
        af::array patch_gradients = af::tile(
            grad_columns,
            af::dim4(window_elements, 1, 1, 1));
        patch_gradients *=
            1.0f / (static_cast<float>(pool_size_) *
                    static_cast<float>(pool_size_));
        af::array dx = af::wrap(
            patch_gradients,
            static_cast<dim_t>(geometry.in_h),
            static_cast<dim_t>(geometry.in_w),
            window,
            window,
            stride_,
            stride_,
            padding_,
            padding_);
        dx.eval();
        return Tensor::FromSemanticArray(dx, cached_input_.Shape());
    } catch (const af::exception& e) {
        ThrowPoolingDeviceError("AvgPool2DLayer::Backward", e);
    }
#else
    ThrowPoolingNeedsArrayFire("Average Pool 2D");
#endif
}

// ============================================================================
// GlobalAvgPool2D Layer Implementation
// ============================================================================

Tensor GlobalAvgPool2DLayer::Forward(const Tensor& input) {
    has_forward_ = false;
    ValidatePoolInput(input, "GlobalAvgPool2D");
    const std::vector<size_t>& shape = input.Shape();
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array output = af::mean(
            af::mean(TensorToAf(input), 0), 1);
        output = af::moddims(
            output,
            af::dim4(
                static_cast<dim_t>(channels),
                static_cast<dim_t>(batch_size)));
        output.eval();

        Tensor result = Tensor::FromSemanticArray(
            output, {channels, batch_size});
        cached_input_ = input;
        has_forward_ = true;
        return result;
    } catch (const af::exception& e) {
        ThrowPoolingDeviceError("GlobalAvgPool2DLayer::Forward", e);
    }
#else
    ThrowPoolingNeedsArrayFire("Global Average Pool");
#endif
}

Tensor GlobalAvgPool2DLayer::Backward(const Tensor& grad_output) {
    if (!has_forward_) {
        throw std::logic_error(
            "GlobalAvgPool2DLayer::Backward requires a successful Forward call");
    }
    ValidatePoolInput(cached_input_, "GlobalAvgPool2D");
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error(
            "GlobalAvgPool2D backward requires Float32 grad_output");
    }
    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    if (grad_output.Shape() != std::vector<size_t>{channels, batch_size}) {
        throw std::runtime_error("GlobalAvgPool2D backward gradient shape mismatch");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad = af::moddims(
            TensorToAf(grad_output),
            af::dim4(
                1,
                1,
                static_cast<dim_t>(channels),
                static_cast<dim_t>(batch_size)));
        af::array grad_input = af::tile(
            grad,
            af::dim4(
                static_cast<dim_t>(in_h),
                static_cast<dim_t>(in_w),
                1,
                1));
        grad_input *=
            1.0f / (static_cast<float>(in_h) *
                    static_cast<float>(in_w));
        grad_input.eval();
        return Tensor::FromSemanticArray(
            grad_input, cached_input_.Shape());
    } catch (const af::exception& e) {
        ThrowPoolingDeviceError("GlobalAvgPool2DLayer::Backward", e);
    }
#else
    ThrowPoolingNeedsArrayFire("Global Average Pool");
#endif
}

// ============================================================================
// GlobalMaxPool2D Layer Implementation
// ============================================================================

Tensor GlobalMaxPool2DLayer::Forward(const Tensor& input) {
    has_forward_ = false;
    ValidatePoolInput(input, "GlobalMaxPool2D");
    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array positions = af::moddims(
            TensorToAf(input),
            af::dim4(static_cast<dim_t>(in_h * in_w),
                     static_cast<dim_t>(channels),
                     static_cast<dim_t>(batch_size)));
        af::array output = af::moddims(
            af::max(positions, 0),
            af::dim4(static_cast<dim_t>(channels),
                     static_cast<dim_t>(batch_size)));
        output.eval();
        Tensor result = Tensor::FromSemanticArray(output, {channels, batch_size});
        cached_input_ = input;
        has_forward_ = true;
        return result;
    } catch (const af::exception& e) {
        ThrowPoolingDeviceError("GlobalMaxPool2DLayer::Forward", e);
    }
#else
    (void)in_h; (void)in_w; (void)channels; (void)batch_size;
    ThrowPoolingNeedsArrayFire("Global Max Pool");
#endif
}

Tensor GlobalMaxPool2DLayer::Backward(const Tensor& grad_output) {
    if (!has_forward_) {
        throw std::logic_error(
            "GlobalMaxPool2DLayer::Backward requires a successful Forward call");
    }
    ValidatePoolInput(cached_input_, "GlobalMaxPool2D");
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error(
            "GlobalMaxPool2D backward requires Float32 grad_output");
    }
    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    if (grad_output.Shape() != std::vector<size_t>{channels, batch_size}) {
        throw std::runtime_error("GlobalMaxPool2D backward gradient shape mismatch");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t count = static_cast<dim_t>(in_h * in_w);
        const dim_t c_dim = static_cast<dim_t>(channels);
        const dim_t n_dim = static_cast<dim_t>(batch_size);
        const af::array positions = af::moddims(
            TensorToAf(cached_input_), af::dim4(count, c_dim, n_dim));
        af::array maxima;
        af::array argmax;
        af::max(maxima, argmax, positions, 0);  // the first maximum, as torch
        // One-hot over the H*W positions at each channel's argmax.
        const af::array iota = af::iota(af::dim4(count), af::dim4(1, c_dim, n_dim), u32);
        const af::array mask =
            (iota == af::tile(argmax, static_cast<unsigned>(count))).as(f32);
        const af::array grad = af::moddims(
            TensorToAf(grad_output), af::dim4(1, c_dim, n_dim));
        af::array grad_input = af::moddims(
            mask * af::tile(grad, static_cast<unsigned>(count)),
            af::dim4(static_cast<dim_t>(in_h), static_cast<dim_t>(in_w), c_dim, n_dim));
        grad_input.eval();
        return Tensor::FromSemanticArray(grad_input, cached_input_.Shape());
    } catch (const af::exception& e) {
        ThrowPoolingDeviceError("GlobalMaxPool2DLayer::Backward", e);
    }
#else
    (void)in_h; (void)in_w; (void)channels; (void)batch_size;
    ThrowPoolingNeedsArrayFire("Global Max Pool");
#endif
}

// ============================================================================
// AdaptiveAvgPool2D Layer Implementation
// ============================================================================

AdaptiveAvgPool2DLayer::AdaptiveAvgPool2DLayer(int out_h, int out_w)
    : out_h_(out_h), out_w_(out_w) {
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("AdaptiveAvgPool2D output size must be positive");
    }
}

Tensor AdaptiveAvgPool2DLayer::Forward(const Tensor& input) {
    has_forward_ = false;
    ValidatePoolInput(input, "AdaptiveAvgPool2D");
    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    const size_t out_h = static_cast<size_t>(out_h_);
    const size_t out_w = static_cast<size_t>(out_w_);
    const std::vector<size_t> output_shape{out_h, out_w, channels, batch_size};
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Y = P_h X P_w^T, the bin-averaging matrices batched over C and N.
        const unsigned c = static_cast<unsigned>(channels);
        const unsigned n = static_cast<unsigned>(batch_size);
        const af::array rows = af::tile(AdaptiveAverageMatrix(in_h, out_h), 1, 1, c, n);
        const af::array cols_t = af::tile(af::transpose(AdaptiveAverageMatrix(in_w, out_w)), 1, 1, c, n);
        af::array output = af::matmul(af::matmul(rows, TensorToAf(input)), cols_t);
        output.eval();
        Tensor result = Tensor::FromSemanticArray(output, output_shape);
        cached_input_ = input;
        has_forward_ = true;
        return result;
    } catch (const af::exception& e) {
        ThrowPoolingDeviceError("AdaptiveAvgPool2DLayer::Forward", e);
    }
#else
    (void)in_h; (void)in_w; (void)channels; (void)batch_size; (void)output_shape;
    ThrowPoolingNeedsArrayFire("Adaptive Average Pool");
#endif
}

Tensor AdaptiveAvgPool2DLayer::Backward(const Tensor& grad_output) {
    if (!has_forward_) {
        throw std::logic_error(
            "AdaptiveAvgPool2DLayer::Backward requires a successful Forward call");
    }
    ValidatePoolInput(cached_input_, "AdaptiveAvgPool2D");
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error(
            "AdaptiveAvgPool2D backward requires Float32 grad_output");
    }
    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    const size_t out_h = static_cast<size_t>(out_h_);
    const size_t out_w = static_cast<size_t>(out_w_);
    if (grad_output.Shape() != std::vector<size_t>{out_h, out_w, channels, batch_size}) {
        throw std::runtime_error("AdaptiveAvgPool2D backward gradient shape mismatch");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // dX = P_h^T dY P_w
        const unsigned c = static_cast<unsigned>(channels);
        const unsigned n = static_cast<unsigned>(batch_size);
        const af::array rows_t = af::tile(af::transpose(AdaptiveAverageMatrix(in_h, out_h)), 1, 1, c, n);
        const af::array cols = af::tile(AdaptiveAverageMatrix(in_w, out_w), 1, 1, c, n);
        af::array grad_input = af::matmul(af::matmul(rows_t, TensorToAf(grad_output)), cols);
        grad_input.eval();
        return Tensor::FromSemanticArray(grad_input, input_shape);
    } catch (const af::exception& e) {
        ThrowPoolingDeviceError("AdaptiveAvgPool2DLayer::Backward", e);
    }
#else
    (void)in_h; (void)in_w; (void)channels; (void)batch_size;
    ThrowPoolingNeedsArrayFire("Adaptive Average Pool");
#endif
}

} // namespace cyxwiz
