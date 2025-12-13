#include "cyxwiz/layer.h"
#include "cyxwiz/tensor.h"
#include <stdexcept>
#include <cmath>
#include <random>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Undefine Windows macros that conflict with ArrayFire functions
// Must be AFTER all includes (Windows headers define these)
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

// ============================================================================
// Helper Functions for ArrayFire Integration
// ============================================================================

#ifdef CYXWIZ_HAS_ARRAYFIRE

// Helper: Convert CyxWiz DataType to ArrayFire dtype
static af::dtype ToAfType(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return af::dtype::f32;
        case DataType::Float64: return af::dtype::f64;
        case DataType::Int32: return af::dtype::s32;
        case DataType::Int64: return af::dtype::s64;
        case DataType::UInt8: return af::dtype::u8;
        default: throw std::runtime_error("Unsupported DataType for ArrayFire");
    }
}

// Helper: Create ArrayFire array from Tensor
// Note: CyxWiz Tensor uses row-major (C-style), ArrayFire uses column-major (Fortran-style)
// For 2D arrays [rows, cols], we need to transpose after loading row-major data
static af::array TensorToAf(const Tensor& t) {
    const auto& shape = t.Shape();
    af::dim4 dims(1, 1, 1, 1);
    for (size_t i = 0; i < shape.size() && i < 4; i++) {
        dims[static_cast<unsigned int>(i)] = static_cast<dim_t>(shape[i]);
    }

    // For 2D arrays, swap dimensions to account for row-major input
    // We load as [cols, rows] then transpose to get [rows, cols] in column-major
    if (shape.size() == 2) {
        af::dim4 swapped_dims(dims[1], dims[0], 1, 1);
        af::array arr(swapped_dims, ToAfType(t.GetDataType()));
        arr.write(t.Data(), arr.bytes(), afHost);
        return af::transpose(arr);  // Now [rows, cols] in column-major
    }

    af::array arr(dims, ToAfType(t.GetDataType()));
    arr.write(t.Data(), arr.bytes(), afHost);
    return arr;
}

// Helper: Create Tensor from ArrayFire array
// Note: Transpose 2D arrays back to row-major for CyxWiz Tensor
static Tensor AfToTensor(const af::array& arr) {
    // Count significant dimensions
    int ndims = 0;
    for (unsigned int i = 0; i < 4; i++) {
        if (arr.dims(i) > 1) ndims = i + 1;
        else if (i == 0) ndims = 1;
    }

    DataType dtype = DataType::Float32;
    switch (arr.type()) {
        case af::dtype::f32: dtype = DataType::Float32; break;
        case af::dtype::f64: dtype = DataType::Float64; break;
        case af::dtype::s32: dtype = DataType::Int32; break;
        case af::dtype::s64: dtype = DataType::Int64; break;
        case af::dtype::u8: dtype = DataType::UInt8; break;
        default: dtype = DataType::Float32;
    }

    // For 2D arrays, transpose to row-major before copying to Tensor
    if (ndims == 2) {
        af::array transposed = af::transpose(arr);
        std::vector<size_t> shape = {
            static_cast<size_t>(arr.dims(0)),
            static_cast<size_t>(arr.dims(1))
        };
        Tensor result(shape, dtype);
        transposed.host(result.Data());
        return result;
    }

    // For other dimensions, copy directly
    std::vector<size_t> shape;
    for (int i = 0; i < ndims; i++) {
        shape.push_back(static_cast<size_t>(arr.dims(i)));
    }
    if (shape.empty()) shape.push_back(1);

    Tensor result(shape, dtype);
    arr.host(result.Data());
    return result;
}

// Helper: Xavier/Glorot uniform initialization
static af::array XavierUniform(int fan_in, int fan_out, af::dim4 dims) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    return af::randu(dims, af::dtype::f32) * 2.0f * limit - limit;
}

// Helper: Kaiming/He initialization for ReLU layers
static af::array KaimingUniform(int fan_in, af::dim4 dims) {
    float limit = std::sqrt(6.0f / fan_in);
    return af::randu(dims, af::dtype::f32) * 2.0f * limit - limit;
}

#endif // CYXWIZ_HAS_ARRAYFIRE

// ============================================================================
// Dense (Fully Connected) Layer Implementation
// ============================================================================

DenseLayer::DenseLayer(int in_features, int out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Initialize weights using Xavier initialization
    af::dim4 weight_dims(out_features, in_features);
    af::array w = XavierUniform(in_features, out_features, weight_dims);
    weights_ = AfToTensor(w);

    if (use_bias_) {
        // Initialize bias to zeros
        af::array b = af::constant(0.0f, af::dim4(out_features));
        bias_ = AfToTensor(b);
    }

    // Initialize gradient accumulators
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(out_features),
                                    static_cast<size_t>(in_features)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_features)});
    }
#else
    // CPU fallback: simple random initialization
    weights_ = Tensor::Random({static_cast<size_t>(out_features),
                                static_cast<size_t>(in_features)});
    if (use_bias_) {
        bias_ = Tensor::Zeros({static_cast<size_t>(out_features)});
    }
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(out_features),
                                    static_cast<size_t>(in_features)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_features)});
    }
#endif
}

Tensor DenseLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        af::array w = TensorToAf(weights_);

        // Ensure x is 2D: [batch_size, in_features]
        // Matrix multiply: output = x @ W^T
        // Where W is [out_features, in_features]
        af::array output = af::matmul(x, af::transpose(w));

        if (use_bias_) {
            af::array b = TensorToAf(bias_);
            // Broadcast bias across batch dimension
            output = output + af::tile(b, static_cast<unsigned int>(x.dims(0)));
        }

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire DenseLayer::Forward failed: {}", e.what());
    }
#endif

    // CPU fallback would go here
    throw std::runtime_error("Dense forward requires ArrayFire");
}

Tensor DenseLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);
        af::array w = TensorToAf(weights_);

        // Gradient w.r.t weights: dW = grad_out^T @ x
        af::array dW = af::matmul(af::transpose(grad_out), x);
        grad_weights_ = AfToTensor(dW);

        // Gradient w.r.t bias: db = sum(grad_out, axis=0)
        if (use_bias_) {
            af::array db = af::sum(grad_out, 0);
            db = af::moddims(db, af::dim4(db.elements()));
            grad_bias_ = AfToTensor(db);
        }

        // Gradient w.r.t input: dx = grad_out @ W
        af::array dx = af::matmul(grad_out, w);

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire DenseLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Dense backward requires ArrayFire");
}

std::map<std::string, Tensor> DenseLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void DenseLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("weights")) {
        weights_ = params.at("weights");
    }
    if (params.count("bias") && use_bias_) {
        bias_ = params.at("bias");
    }
}

// ============================================================================
// Conv2D Layer Implementation
// ============================================================================

Conv2DLayer::Conv2DLayer(int in_channels, int out_channels, int kernel_size,
                         int stride, int padding, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      use_bias_(use_bias) {

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Initialize weights using Kaiming initialization
    // Shape: [kernel_size, kernel_size, in_channels, out_channels] for ArrayFire
    // (ArrayFire uses column-major order)
    int fan_in = in_channels * kernel_size * kernel_size;
    af::dim4 weight_dims(kernel_size, kernel_size, in_channels, out_channels);
    af::array w = KaimingUniform(fan_in, weight_dims);
    weights_ = AfToTensor(w);

    if (use_bias_) {
        af::array b = af::constant(0.0f, af::dim4(out_channels));
        bias_ = AfToTensor(b);
    }

    // Initialize gradient accumulators
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(in_channels),
                                    static_cast<size_t>(out_channels)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
#else
    weights_ = Tensor::Random({static_cast<size_t>(kernel_size),
                                static_cast<size_t>(kernel_size),
                                static_cast<size_t>(in_channels),
                                static_cast<size_t>(out_channels)});
    if (use_bias_) {
        bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(in_channels),
                                    static_cast<size_t>(out_channels)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
#endif
}

Tensor Conv2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Input shape: [H, W, C, N] for ArrayFire (column-major)
        // or [batch, channels, height, width] in standard ML format
        af::array x = TensorToAf(input);
        af::array w = TensorToAf(weights_);

        // Apply padding if needed
        if (padding_ > 0) {
            // Pad height and width dimensions
            x = af::pad(x, af::dim4(padding_, padding_, 0, 0),
                        af::dim4(padding_, padding_, 0, 0), AF_PAD_ZERO);
        }

        // Perform convolution using ArrayFire
        // af::convolve2 performs 2D convolution for each channel
        af::array output = af::constant(0.0f, 1, 1, 1, 1);

        // Get dimensions
        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        // Calculate output dimensions
        dim_t out_h = (in_h - kernel_size_) / stride_ + 1;
        dim_t out_w = (in_w - kernel_size_) / stride_ + 1;

        // Initialize output
        output = af::constant(0.0f, af::dim4(out_h, out_w, out_channels_, batch_size));

        // Convolve each output channel
        for (int oc = 0; oc < out_channels_; oc++) {
            af::array channel_out = af::constant(0.0f, af::dim4(out_h, out_w, 1, batch_size));

            for (int ic = 0; ic < in_channels_; ic++) {
                // Get filter for this input/output channel pair
                af::array filter = w(af::span, af::span, ic, oc);

                // Get input channel for all batches
                af::array input_channel = x(af::span, af::span, ic, af::span);

                // Perform 2D convolution using af::convolve2
                // Need to handle stride manually if stride > 1
                af::array conv_result = af::convolve2(input_channel, filter, AF_CONV_DEFAULT);

                // Apply striding if needed
                if (stride_ > 1) {
                    conv_result = conv_result(af::seq(0, static_cast<double>(out_h - 1) * stride_, stride_),
                                               af::seq(0, static_cast<double>(out_w - 1) * stride_, stride_),
                                               af::span, af::span);
                }

                // Accumulate
                channel_out += conv_result;
            }

            // Store in output
            output(af::span, af::span, oc, af::span) = channel_out;
        }

        // Add bias if needed
        if (use_bias_) {
            af::array b = TensorToAf(bias_);
            // Reshape bias for broadcasting: [1, 1, out_channels, 1]
            b = af::moddims(b, af::dim4(1, 1, out_channels_, 1));
            output = output + af::tile(b, static_cast<unsigned int>(out_h),
                                        static_cast<unsigned int>(out_w), 1,
                                        static_cast<unsigned int>(batch_size));
        }

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire Conv2DLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Conv2D forward requires ArrayFire");
}

Tensor Conv2DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);
        af::array w = TensorToAf(weights_);

        // Dimensions
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;
        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);

        // Apply padding to input if needed
        if (padding_ > 0) {
            x = af::pad(x, af::dim4(padding_, padding_, 0, 0),
                        af::dim4(padding_, padding_, 0, 0), AF_PAD_ZERO);
        }

        // 1. Gradient w.r.t. bias: sum over all spatial and batch dimensions
        if (use_bias_) {
            af::array db = af::sum(af::sum(af::sum(grad_out, 0), 1), 3);
            db = af::moddims(db, af::dim4(out_channels_));
            grad_bias_ = AfToTensor(db);
        }

        // 2. Gradient w.r.t. weights: dW = conv(input, grad_output)
        af::array dW = af::constant(0.0f, af::dim4(kernel_size_, kernel_size_,
                                                    in_channels_, out_channels_));

        for (int oc = 0; oc < out_channels_; oc++) {
            for (int ic = 0; ic < in_channels_; ic++) {
                af::array grad_channel = grad_out(af::span, af::span, oc, af::span);
                af::array input_channel = x(af::span, af::span, ic, af::span);

                // Correlate input with grad_output to get weight gradient
                af::array dw_single = af::constant(0.0f, af::dim4(kernel_size_, kernel_size_));

                for (dim_t b = 0; b < batch_size; b++) {
                    af::array g = grad_channel(af::span, af::span, af::span, b);
                    af::array i = input_channel(af::span, af::span, af::span, b);
                    dw_single += af::convolve2(i, g, AF_CONV_DEFAULT)(
                        af::seq(0, kernel_size_ - 1), af::seq(0, kernel_size_ - 1));
                }

                dW(af::span, af::span, ic, oc) = dw_single;
            }
        }
        grad_weights_ = AfToTensor(dW);

        // 3. Gradient w.r.t. input: dx = full_conv(grad_output, flipped_weights)
        // Pad gradient output for full convolution
        dim_t pad_h = kernel_size_ - 1;
        dim_t pad_w = kernel_size_ - 1;

        af::array grad_padded = af::pad(grad_out,
                                        af::dim4(pad_h, pad_w, 0, 0),
                                        af::dim4(pad_h, pad_w, 0, 0), AF_PAD_ZERO);

        af::array dx = af::constant(0.0f, x.dims());

        for (int ic = 0; ic < in_channels_; ic++) {
            for (int oc = 0; oc < out_channels_; oc++) {
                // Flip kernel (rotate 180 degrees)
                af::array filter = w(af::span, af::span, ic, oc);
                af::array flipped = af::flip(af::flip(filter, 0), 1);

                af::array grad_channel = grad_padded(af::span, af::span, oc, af::span);

                // Convolve
                af::array dx_single = af::convolve2(grad_channel, flipped, AF_CONV_DEFAULT);

                // Extract valid region
                dx(af::span, af::span, ic, af::span) += dx_single(
                    af::seq(0, x.dims(0) - 1), af::seq(0, x.dims(1) - 1), af::span, af::span);
            }
        }

        // Remove padding from gradient if padding was applied
        if (padding_ > 0) {
            dx = dx(af::seq(padding_, in_h + padding_ - 1),
                    af::seq(padding_, in_w + padding_ - 1),
                    af::span, af::span);
        }

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire Conv2DLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Conv2D backward requires ArrayFire");
}

std::map<std::string, Tensor> Conv2DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void Conv2DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("weights")) {
        weights_ = params.at("weights");
    }
    if (params.count("bias") && use_bias_) {
        bias_ = params.at("bias");
    }
}

// ============================================================================
// MaxPool2D Layer Implementation
// ============================================================================

MaxPool2DLayer::MaxPool2DLayer(int pool_size, int stride, int padding)
    : pool_size_(pool_size), stride_(stride > 0 ? stride : pool_size), padding_(padding) {
}

Tensor MaxPool2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        // Apply padding if needed
        if (padding_ > 0) {
            // Pad with -infinity for max pooling
            x = af::pad(x, af::dim4(padding_, padding_, 0, 0),
                        af::dim4(padding_, padding_, 0, 0), AF_PAD_ZERO);
            // Note: For max pooling with zero padding, zeros will participate
            // in max computation but won't affect results if inputs are positive
        }

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t channels = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        // Calculate output dimensions
        dim_t out_h = (in_h - pool_size_) / stride_ + 1;
        dim_t out_w = (in_w - pool_size_) / stride_ + 1;

        // Use af::unwrap to extract patches, then max
        // unwrap extracts patches into columns
        af::array output = af::constant(0.0f, af::dim4(out_h, out_w, channels, batch_size));
        af::array indices = af::constant(0, af::dim4(out_h, out_w, channels, batch_size), af::dtype::s32);

        for (dim_t c = 0; c < channels; c++) {
            for (dim_t b = 0; b < batch_size; b++) {
                af::array channel = x(af::span, af::span, c, b);

                // Extract patches using unwrap
                af::array patches = af::unwrap(channel, pool_size_, pool_size_,
                                                stride_, stride_);

                // patches shape: [pool_size*pool_size, num_patches]
                // Take max along first dimension
                af::array max_vals, max_idx;
                af::max(max_vals, max_idx, patches, 0);

                // Reshape to output spatial dimensions
                max_vals = af::moddims(max_vals, af::dim4(out_h, out_w));

                output(af::span, af::span, c, b) = max_vals;
                indices(af::span, af::span, c, b) = af::moddims(max_idx, af::dim4(out_h, out_w));
            }
        }

        max_indices_ = AfToTensor(indices);
        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire MaxPool2DLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("MaxPool2D forward requires ArrayFire");
}

Tensor MaxPool2DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);
        af::array indices = TensorToAf(max_indices_);

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t channels = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        dim_t out_h = grad_out.dims(0);
        dim_t out_w = grad_out.dims(1);

        // Initialize gradient w.r.t. input
        af::array dx = af::constant(0.0f, x.dims());

        // Scatter gradients back to max positions
        for (dim_t c = 0; c < channels; c++) {
            for (dim_t b = 0; b < batch_size; b++) {
                for (dim_t oh = 0; oh < out_h; oh++) {
                    for (dim_t ow = 0; ow < out_w; ow++) {
                        // Get the max index within the pool window
                        int idx = indices(oh, ow, c, b).scalar<int>();
                        int pool_h = idx / pool_size_;
                        int pool_w = idx % pool_size_;

                        // Calculate input position
                        dim_t ih = oh * stride_ + pool_h;
                        dim_t iw = ow * stride_ + pool_w;

                        // Add gradient
                        dx(ih, iw, c, b) += grad_out(oh, ow, c, b);
                    }
                }
            }
        }

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire MaxPool2DLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("MaxPool2D backward requires ArrayFire");
}

// ============================================================================
// AvgPool2D Layer Implementation
// ============================================================================

AvgPool2DLayer::AvgPool2DLayer(int pool_size, int stride, int padding)
    : pool_size_(pool_size), stride_(stride > 0 ? stride : pool_size), padding_(padding) {
}

Tensor AvgPool2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        // Apply padding if needed
        if (padding_ > 0) {
            x = af::pad(x, af::dim4(padding_, padding_, 0, 0),
                        af::dim4(padding_, padding_, 0, 0), AF_PAD_ZERO);
        }

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t channels = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        // Calculate output dimensions
        dim_t out_h = (in_h - pool_size_) / stride_ + 1;
        dim_t out_w = (in_w - pool_size_) / stride_ + 1;

        af::array output = af::constant(0.0f, af::dim4(out_h, out_w, channels, batch_size));

        for (dim_t c = 0; c < channels; c++) {
            for (dim_t b = 0; b < batch_size; b++) {
                af::array channel = x(af::span, af::span, c, b);

                // Extract patches using unwrap
                af::array patches = af::unwrap(channel, pool_size_, pool_size_,
                                                stride_, stride_);

                // Take mean along first dimension
                af::array mean_vals = af::mean(patches, 0);

                // Reshape to output spatial dimensions
                mean_vals = af::moddims(mean_vals, af::dim4(out_h, out_w));

                output(af::span, af::span, c, b) = mean_vals;
            }
        }

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire AvgPool2DLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("AvgPool2D forward requires ArrayFire");
}

Tensor AvgPool2DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t channels = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        dim_t out_h = grad_out.dims(0);
        dim_t out_w = grad_out.dims(1);

        // For average pooling, gradient is distributed equally
        float scale = 1.0f / (pool_size_ * pool_size_);

        af::array dx = af::constant(0.0f, x.dims());

        for (dim_t c = 0; c < channels; c++) {
            for (dim_t b = 0; b < batch_size; b++) {
                for (dim_t oh = 0; oh < out_h; oh++) {
                    for (dim_t ow = 0; ow < out_w; ow++) {
                        float grad_val = grad_out(oh, ow, c, b).scalar<float>() * scale;

                        // Distribute gradient to all positions in the pool window
                        for (int ph = 0; ph < pool_size_; ph++) {
                            for (int pw = 0; pw < pool_size_; pw++) {
                                dim_t ih = oh * stride_ + ph;
                                dim_t iw = ow * stride_ + pw;
                                dx(ih, iw, c, b) += grad_val;
                            }
                        }
                    }
                }
            }
        }

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire AvgPool2DLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("AvgPool2D backward requires ArrayFire");
}

// ============================================================================
// GlobalAvgPool2D Layer Implementation
// ============================================================================

Tensor GlobalAvgPool2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        // Input: [H, W, C, N]
        // Output: [1, 1, C, N] or [C, N] (flattened)
        dim_t channels = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        // Global average over spatial dimensions
        af::array output = af::mean(af::mean(x, 0), 0);

        // Reshape to [C, N]
        output = af::moddims(output, af::dim4(channels, batch_size));

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire GlobalAvgPool2DLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("GlobalAvgPool2D forward requires ArrayFire");
}

Tensor GlobalAvgPool2DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t channels = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        // Scale factor for distributing gradient
        float scale = 1.0f / (in_h * in_w);

        // Reshape grad_output to [1, 1, C, N]
        af::array grad_reshaped = af::moddims(grad_out, af::dim4(1, 1, channels, batch_size));

        // Tile to match input shape and scale
        af::array dx = af::tile(grad_reshaped, static_cast<unsigned int>(in_h),
                                 static_cast<unsigned int>(in_w), 1, 1) * scale;

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire GlobalAvgPool2DLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("GlobalAvgPool2D backward requires ArrayFire");
}

// ============================================================================
// BatchNorm2D Layer Implementation
// ============================================================================

BatchNorm2DLayer::BatchNorm2DLayer(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum) {

    // Initialize gamma (scale) to ones
    gamma_ = Tensor::Ones({static_cast<size_t>(num_features)});

    // Initialize beta (shift) to zeros
    beta_ = Tensor::Zeros({static_cast<size_t>(num_features)});

    // Initialize running statistics
    running_mean_ = Tensor::Zeros({static_cast<size_t>(num_features)});
    running_var_ = Tensor::Ones({static_cast<size_t>(num_features)});

    // Initialize gradient accumulators
    grad_gamma_ = Tensor::Zeros({static_cast<size_t>(num_features)});
    grad_beta_ = Tensor::Zeros({static_cast<size_t>(num_features)});
}

Tensor BatchNorm2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        af::array gamma = TensorToAf(gamma_);
        af::array beta = TensorToAf(beta_);

        // Input: [H, W, C, N] for ArrayFire
        dim_t height = x.dims(0);
        dim_t width = x.dims(1);
        dim_t channels = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        af::array mean, var, normalized;

        if (training_) {
            // Compute batch statistics
            // Mean over H, W, N dimensions for each channel
            mean = af::mean(af::mean(af::mean(x, 0), 1), 3);
            mean = af::moddims(mean, af::dim4(channels));

            // Variance over H, W, N dimensions for each channel
            af::array x_centered = x - af::tile(
                af::moddims(mean, af::dim4(1, 1, channels, 1)),
                static_cast<unsigned int>(height),
                static_cast<unsigned int>(width), 1,
                static_cast<unsigned int>(batch_size));

            var = af::mean(af::mean(af::mean(x_centered * x_centered, 0), 1), 3);
            var = af::moddims(var, af::dim4(channels));

            // Update running statistics
            af::array rm = TensorToAf(running_mean_);
            af::array rv = TensorToAf(running_var_);

            rm = (1.0f - momentum_) * rm + momentum_ * mean;
            rv = (1.0f - momentum_) * rv + momentum_ * var;

            running_mean_ = AfToTensor(rm);
            running_var_ = AfToTensor(rv);
        } else {
            // Use running statistics during inference
            mean = TensorToAf(running_mean_);
            var = TensorToAf(running_var_);
        }

        // Normalize: (x - mean) / sqrt(var + eps)
        af::array std_inv = 1.0f / af::sqrt(var + eps_);
        std_inv_ = AfToTensor(std_inv);

        // Reshape for broadcasting
        af::array mean_bc = af::moddims(mean, af::dim4(1, 1, channels, 1));
        af::array std_inv_bc = af::moddims(std_inv, af::dim4(1, 1, channels, 1));
        af::array gamma_bc = af::moddims(gamma, af::dim4(1, 1, channels, 1));
        af::array beta_bc = af::moddims(beta, af::dim4(1, 1, channels, 1));

        // Tile for full shape
        mean_bc = af::tile(mean_bc, static_cast<unsigned int>(height),
                            static_cast<unsigned int>(width), 1,
                            static_cast<unsigned int>(batch_size));
        std_inv_bc = af::tile(std_inv_bc, static_cast<unsigned int>(height),
                               static_cast<unsigned int>(width), 1,
                               static_cast<unsigned int>(batch_size));
        gamma_bc = af::tile(gamma_bc, static_cast<unsigned int>(height),
                             static_cast<unsigned int>(width), 1,
                             static_cast<unsigned int>(batch_size));
        beta_bc = af::tile(beta_bc, static_cast<unsigned int>(height),
                            static_cast<unsigned int>(width), 1,
                            static_cast<unsigned int>(batch_size));

        // Normalize and scale
        normalized = (x - mean_bc) * std_inv_bc;
        normalized_ = AfToTensor(normalized);

        af::array output = gamma_bc * normalized + beta_bc;

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire BatchNorm2DLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("BatchNorm2D forward requires ArrayFire");
}

Tensor BatchNorm2DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);
        af::array normalized = TensorToAf(normalized_);
        af::array gamma = TensorToAf(gamma_);
        af::array std_inv = TensorToAf(std_inv_);

        dim_t height = x.dims(0);
        dim_t width = x.dims(1);
        dim_t channels = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        float N = static_cast<float>(height * width * batch_size);

        // Gradient w.r.t. gamma: sum(grad_out * normalized)
        af::array dg = af::sum(af::sum(af::sum(grad_out * normalized, 0), 1), 3);
        dg = af::moddims(dg, af::dim4(channels));
        grad_gamma_ = AfToTensor(dg);

        // Gradient w.r.t. beta: sum(grad_out)
        af::array db = af::sum(af::sum(af::sum(grad_out, 0), 1), 3);
        db = af::moddims(db, af::dim4(channels));
        grad_beta_ = AfToTensor(db);

        // Gradient w.r.t. input (using simplified formula for efficiency)
        // dx = (1/N) * gamma * std_inv * (N * dy - sum(dy) - normalized * sum(dy * normalized))

        // Reshape gamma and std_inv for broadcasting
        af::array gamma_bc = af::moddims(gamma, af::dim4(1, 1, channels, 1));
        gamma_bc = af::tile(gamma_bc, static_cast<unsigned int>(height),
                             static_cast<unsigned int>(width), 1,
                             static_cast<unsigned int>(batch_size));

        af::array std_inv_bc = af::moddims(std_inv, af::dim4(1, 1, channels, 1));
        std_inv_bc = af::tile(std_inv_bc, static_cast<unsigned int>(height),
                               static_cast<unsigned int>(width), 1,
                               static_cast<unsigned int>(batch_size));

        // sum(dy) per channel
        af::array sum_dy = af::sum(af::sum(af::sum(grad_out, 0), 1), 3);
        sum_dy = af::moddims(sum_dy, af::dim4(1, 1, channels, 1));
        sum_dy = af::tile(sum_dy, static_cast<unsigned int>(height),
                           static_cast<unsigned int>(width), 1,
                           static_cast<unsigned int>(batch_size));

        // sum(dy * normalized) per channel
        af::array sum_dy_norm = af::sum(af::sum(af::sum(grad_out * normalized, 0), 1), 3);
        sum_dy_norm = af::moddims(sum_dy_norm, af::dim4(1, 1, channels, 1));
        sum_dy_norm = af::tile(sum_dy_norm, static_cast<unsigned int>(height),
                                static_cast<unsigned int>(width), 1,
                                static_cast<unsigned int>(batch_size));

        // Compute dx
        af::array dx = (1.0f / N) * gamma_bc * std_inv_bc *
                       (N * grad_out - sum_dy - normalized * sum_dy_norm);

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire BatchNorm2DLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("BatchNorm2D backward requires ArrayFire");
}

std::map<std::string, Tensor> BatchNorm2DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["gamma"] = gamma_;
    params["beta"] = beta_;
    params["running_mean"] = running_mean_;
    params["running_var"] = running_var_;
    params["grad_gamma"] = grad_gamma_;
    params["grad_beta"] = grad_beta_;
    return params;
}

void BatchNorm2DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) {
        gamma_ = params.at("gamma");
    }
    if (params.count("beta")) {
        beta_ = params.at("beta");
    }
    if (params.count("running_mean")) {
        running_mean_ = params.at("running_mean");
    }
    if (params.count("running_var")) {
        running_var_ = params.at("running_var");
    }
}

// ============================================================================
// Flatten Layer Implementation
// ============================================================================

Tensor FlattenLayer::Forward(const Tensor& input) {
    input_shape_ = input.Shape();

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        // Flatten all dimensions except batch
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;
        dim_t flat_size = x.elements() / batch_size;

        af::array output = af::moddims(x, af::dim4(flat_size, batch_size));

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire FlattenLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Flatten forward requires ArrayFire");
}

Tensor FlattenLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);

        // Reshape back to original shape
        af::dim4 dims(1, 1, 1, 1);
        for (size_t i = 0; i < input_shape_.size() && i < 4; i++) {
            dims[static_cast<unsigned int>(i)] = static_cast<dim_t>(input_shape_[i]);
        }

        af::array dx = af::moddims(grad_out, dims);

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire FlattenLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Flatten backward requires ArrayFire");
}

// ============================================================================
// Dropout Layer Implementation
// ============================================================================

DropoutLayer::DropoutLayer(float p) : p_(p) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument("Dropout probability must be in [0, 1)");
    }
}

Tensor DropoutLayer::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        if (training_ && p_ > 0.0f) {
            // Generate random mask
            af::array rand_mask = af::randu(x.dims(), af::dtype::f32);
            af::array mask = (rand_mask > p_).as(af::dtype::f32);

            // Scale by 1/(1-p) to maintain expected value
            float scale = 1.0f / (1.0f - p_);
            af::array output = x * mask * scale;

            mask_ = AfToTensor(mask);
            return AfToTensor(output);
        } else {
            // During inference, just pass through
            return input;
        }
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire DropoutLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Dropout forward requires ArrayFire");
}

Tensor DropoutLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        if (training_ && p_ > 0.0f) {
            af::array grad_out = TensorToAf(grad_output);
            af::array mask = TensorToAf(mask_);

            // Apply same mask and scaling
            float scale = 1.0f / (1.0f - p_);
            af::array dx = grad_out * mask * scale;

            return AfToTensor(dx);
        } else {
            return grad_output;
        }
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire DropoutLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Dropout backward requires ArrayFire");
}

} // namespace cyxwiz
