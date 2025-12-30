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

                for (int b = 0; b < static_cast<int>(batch_size); b++) {
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
                    af::seq(0, static_cast<double>(x.dims(0) - 1)), af::seq(0, static_cast<double>(x.dims(1) - 1)), af::span, af::span);
            }
        }

        // Remove padding from gradient if padding was applied
        if (padding_ > 0) {
            dx = dx(af::seq(static_cast<double>(padding_), static_cast<double>(in_h + padding_ - 1)),
                    af::seq(static_cast<double>(padding_), static_cast<double>(in_w + padding_ - 1)),
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
        int channels = static_cast<int>(x.dims(2));
        int batch_size = static_cast<int>((x.numdims() > 3) ? x.dims(3) : 1);

        // Calculate output dimensions
        dim_t out_h = (in_h - pool_size_) / stride_ + 1;
        dim_t out_w = (in_w - pool_size_) / stride_ + 1;

        // Use af::unwrap to extract patches, then max
        // unwrap extracts patches into columns
        af::array output = af::constant(0.0f, af::dim4(out_h, out_w, channels, batch_size));
        af::array indices = af::constant(0, af::dim4(out_h, out_w, channels, batch_size), af::dtype::s32);

        for (int c = 0; c < channels; c++) {
            for (int b = 0; b < batch_size; b++) {
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

        int in_h = static_cast<int>(x.dims(0));
        int in_w = static_cast<int>(x.dims(1));
        int channels = static_cast<int>(x.dims(2));
        int batch_size = static_cast<int>((x.numdims() > 3) ? x.dims(3) : 1);

        int out_h = static_cast<int>(grad_out.dims(0));
        int out_w = static_cast<int>(grad_out.dims(1));

        // Suppress unused variable warnings
        (void)in_h;
        (void)in_w;

        // Initialize gradient w.r.t. input
        af::array dx = af::constant(0.0f, x.dims());

        // Scatter gradients back to max positions
        for (int c = 0; c < channels; c++) {
            for (int b = 0; b < batch_size; b++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        // Get the max index within the pool window
                        int idx = indices(oh, ow, c, b).scalar<int>();
                        int pool_h = idx / pool_size_;
                        int pool_w = idx % pool_size_;

                        // Calculate input position
                        int ih = oh * stride_ + pool_h;
                        int iw = ow * stride_ + pool_w;

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
        int channels = static_cast<int>(x.dims(2));
        int batch_size = static_cast<int>((x.numdims() > 3) ? x.dims(3) : 1);

        // Calculate output dimensions
        dim_t out_h = (in_h - pool_size_) / stride_ + 1;
        dim_t out_w = (in_w - pool_size_) / stride_ + 1;

        af::array output = af::constant(0.0f, af::dim4(out_h, out_w, channels, batch_size));

        for (int c = 0; c < channels; c++) {
            for (int b = 0; b < batch_size; b++) {
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

        int in_h = static_cast<int>(x.dims(0));
        int in_w = static_cast<int>(x.dims(1));
        int channels = static_cast<int>(x.dims(2));
        int batch_size = static_cast<int>((x.numdims() > 3) ? x.dims(3) : 1);

        int out_h = static_cast<int>(grad_out.dims(0));
        int out_w = static_cast<int>(grad_out.dims(1));

        // Suppress unused variable warnings
        (void)in_h;
        (void)in_w;

        // For average pooling, gradient is distributed equally
        float scale = 1.0f / (pool_size_ * pool_size_);

        af::array dx = af::constant(0.0f, x.dims());

        for (int c = 0; c < channels; c++) {
            for (int b = 0; b < batch_size; b++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        float grad_val = grad_out(oh, ow, c, b).scalar<float>() * scale;

                        // Distribute gradient to all positions in the pool window
                        for (int ph = 0; ph < pool_size_; ph++) {
                            for (int pw = 0; pw < pool_size_; pw++) {
                                int ih = oh * stride_ + ph;
                                int iw = ow * stride_ + pw;
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

// ============================================================================
// Embedding Layer Implementation
// ============================================================================

EmbeddingLayer::EmbeddingLayer(int num_embeddings, int embedding_dim,
                               int padding_idx, float max_norm)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim),
      padding_idx_(padding_idx), max_norm_(max_norm) {

    InitializeWeights();
}

void EmbeddingLayer::InitializeWeights() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Initialize with normal distribution N(0, 1)
    af::array w = af::randn(af::dim4(num_embeddings_, embedding_dim_), af::dtype::f32);
    weight_ = AfToTensor(w);

    // Zero out padding index if specified
    if (padding_idx_ >= 0 && padding_idx_ < num_embeddings_) {
        float* data = static_cast<float*>(weight_.Data());
        for (int i = 0; i < embedding_dim_; i++) {
            data[padding_idx_ * embedding_dim_ + i] = 0.0f;
        }
    }
#else
    weight_ = Tensor::Random({static_cast<size_t>(num_embeddings_),
                               static_cast<size_t>(embedding_dim_)});
#endif

    grad_weight_ = Tensor::Zeros({static_cast<size_t>(num_embeddings_),
                                   static_cast<size_t>(embedding_dim_)});
}

void EmbeddingLayer::NormalizeEmbeddings() {
    if (max_norm_ <= 0.0f) return;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array w = TensorToAf(weight_);

        // Compute L2 norm for each embedding
        af::array norms = af::sqrt(af::sum(w * w, 1));

        // Create scaling factors (clip to max_norm)
        af::array scale = af::min(max_norm_ / (norms + 1e-8f), 1.0f);

        // Apply scaling
        w = w * af::tile(scale, 1, embedding_dim_);

        weight_ = AfToTensor(w);
    } catch (const af::exception& e) {
        spdlog::warn("EmbeddingLayer::NormalizeEmbeddings failed: {}", e.what());
    }
#endif
}

Tensor EmbeddingLayer::Forward(const Tensor& input) {
    // Cache indices for backward pass
    cached_indices_ = input.Clone();

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Apply max_norm if specified
        if (max_norm_ > 0.0f) {
            NormalizeEmbeddings();
        }

        const auto& shape = input.Shape();
        bool is_batched = shape.size() == 2;

        // For batched input (3D output), skip ArrayFire to avoid memory layout issues
        if (is_batched) {
            throw af::exception("Use CPU for batched embedding");
        }

        dim_t batch_size = is_batched ? shape[0] : 1;
        dim_t seq_len = is_batched ? shape[1] : shape[0];
        dim_t total_indices = batch_size * seq_len;

        // Get indices as int32
        const int32_t* indices_ptr = input.Data<int32_t>();

        // Get weight matrix
        af::array w = TensorToAf(weight_);  // [num_embeddings, embedding_dim]

        // Gather embeddings using advanced indexing
        // Create index array
        af::array indices_af(total_indices, indices_ptr);

        // Use af::rows to select embeddings (vectorized lookup)
        // This avoids explicit loops by using ArrayFire's indexing
        af::array output_flat = af::constant(0.0f, af::dim4(total_indices, embedding_dim_));

        // Vectorized gather: for each index, get the corresponding row
        // ArrayFire doesn't have direct gather, but we can use batch indexing
        for (dim_t i = 0; i < total_indices; i++) {
            int32_t idx = indices_ptr[i];
            if (idx >= 0 && idx < num_embeddings_) {
                output_flat(i, af::span) = w(idx, af::span);
            }
            // If idx == padding_idx or out of bounds, leave as zero
        }

        // Reshape to [batch, seq_len, embedding_dim] or [seq_len, embedding_dim]
        af::array output;
        if (is_batched) {
            output = af::moddims(output_flat, af::dim4(batch_size, seq_len, embedding_dim_));
        } else {
            output = af::moddims(output_flat, af::dim4(seq_len, embedding_dim_));
        }

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire EmbeddingLayer::Forward failed: {}", e.what());
    }
#endif

    // CPU fallback
    const auto& shape = input.Shape();
    bool is_batched = shape.size() == 2;

    size_t batch_size = is_batched ? shape[0] : 1;
    size_t seq_len = is_batched ? shape[1] : shape[0];

    std::vector<size_t> out_shape;
    if (is_batched) {
        out_shape = {batch_size, seq_len, static_cast<size_t>(embedding_dim_)};
    } else {
        out_shape = {seq_len, static_cast<size_t>(embedding_dim_)};
    }

    Tensor output(out_shape, DataType::Float32);
    float* out_data = static_cast<float*>(output.Data());
    const float* weight_data = weight_.Data<float>();
    const int32_t* indices = input.Data<int32_t>();

    size_t total = batch_size * seq_len;
    for (size_t i = 0; i < total; i++) {
        int32_t idx = indices[i];
        if (idx >= 0 && idx < num_embeddings_ && idx != padding_idx_) {
            std::memcpy(out_data + i * embedding_dim_,
                       weight_data + idx * embedding_dim_,
                       embedding_dim_ * sizeof(float));
        } else {
            std::memset(out_data + i * embedding_dim_, 0, embedding_dim_ * sizeof(float));
        }
    }

    return output;
}

Tensor EmbeddingLayer::Backward(const Tensor& grad_output) {
    if (frozen_) {
        // Return empty tensor - no gradient needed for frozen embeddings
        return Tensor();
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const auto& shape = cached_indices_.Shape();
        bool is_batched = shape.size() == 2;

        dim_t batch_size = is_batched ? shape[0] : 1;
        dim_t seq_len = is_batched ? shape[1] : shape[0];
        dim_t total_indices = batch_size * seq_len;

        const int32_t* indices_ptr = cached_indices_.Data<int32_t>();

        // Initialize gradient accumulator
        af::array dw = af::constant(0.0f, af::dim4(num_embeddings_, embedding_dim_));

        // Get flattened gradient output
        af::array grad = TensorToAf(grad_output);
        grad = af::moddims(grad, af::dim4(total_indices, embedding_dim_));

        // Scatter-add gradients to the weight matrix
        // For each position, add gradient to the corresponding embedding
        for (dim_t i = 0; i < total_indices; i++) {
            int32_t idx = indices_ptr[i];
            if (idx >= 0 && idx < num_embeddings_ && idx != padding_idx_) {
                dw(idx, af::span) += grad(i, af::span);
            }
        }

        grad_weight_ = AfToTensor(dw);

        // Return empty tensor (no gradient w.r.t. integer indices)
        return Tensor();
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire EmbeddingLayer::Backward failed: {}", e.what());
    }
#endif

    // CPU fallback
    const auto& shape = cached_indices_.Shape();
    bool is_batched = shape.size() == 2;

    size_t batch_size = is_batched ? shape[0] : 1;
    size_t seq_len = is_batched ? shape[1] : shape[0];
    size_t total = batch_size * seq_len;

    // Zero out gradient
    grad_weight_ = Tensor::Zeros({static_cast<size_t>(num_embeddings_),
                                   static_cast<size_t>(embedding_dim_)});
    float* dw = static_cast<float*>(grad_weight_.Data());
    const float* grad_data = grad_output.Data<float>();
    const int32_t* indices = cached_indices_.Data<int32_t>();

    // Scatter-add gradients
    for (size_t i = 0; i < total; i++) {
        int32_t idx = indices[i];
        if (idx >= 0 && idx < num_embeddings_ && idx != padding_idx_) {
            for (int j = 0; j < embedding_dim_; j++) {
                dw[idx * embedding_dim_ + j] += grad_data[i * embedding_dim_ + j];
            }
        }
    }

    return Tensor();
}

Tensor EmbeddingLayer::GetEmbedding(int index) const {
    if (index < 0 || index >= num_embeddings_) {
        throw std::out_of_range("Embedding index out of range");
    }

    Tensor result({static_cast<size_t>(embedding_dim_)}, DataType::Float32);
    const float* src = weight_.Data<float>() + index * embedding_dim_;
    std::memcpy(result.Data(), src, embedding_dim_ * sizeof(float));
    return result;
}

void EmbeddingLayer::SetEmbedding(int index, const Tensor& embedding) {
    if (index < 0 || index >= num_embeddings_) {
        throw std::out_of_range("Embedding index out of range");
    }
    if (embedding.NumElements() != static_cast<size_t>(embedding_dim_)) {
        throw std::invalid_argument("Embedding dimension mismatch");
    }

    float* dst = static_cast<float*>(weight_.Data()) + index * embedding_dim_;
    std::memcpy(dst, embedding.Data<float>(), embedding_dim_ * sizeof(float));
}

void EmbeddingLayer::LoadPretrainedWeights(const Tensor& weights, bool freeze) {
    const auto& shape = weights.Shape();
    if (shape.size() != 2 ||
        shape[0] != static_cast<size_t>(num_embeddings_) ||
        shape[1] != static_cast<size_t>(embedding_dim_)) {
        throw std::invalid_argument("Weight shape mismatch");
    }

    weight_ = weights.Clone();
    frozen_ = freeze;

    // Ensure padding index is zero
    if (padding_idx_ >= 0 && padding_idx_ < num_embeddings_) {
        float* data = static_cast<float*>(weight_.Data());
        for (int i = 0; i < embedding_dim_; i++) {
            data[padding_idx_ * embedding_dim_ + i] = 0.0f;
        }
    }
}

std::map<std::string, Tensor> EmbeddingLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weight"] = weight_;
    params["grad_weight"] = grad_weight_;
    return params;
}

void EmbeddingLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("weight")) {
        weight_ = params.at("weight");
    }
}


// ============================================================================
// LayerNorm Layer Implementation
// ============================================================================

LayerNormLayer::LayerNormLayer(const std::vector<int>& normalized_shape,
                               float eps, bool elementwise_affine)
    : normalized_shape_(normalized_shape), eps_(eps),
      elementwise_affine_(elementwise_affine) {

    // Calculate total size of normalized dimensions
    size_t norm_size = 1;
    for (int dim : normalized_shape) {
        norm_size *= static_cast<size_t>(dim);
    }

    if (elementwise_affine) {
        gamma_ = Tensor::Ones({norm_size});
        beta_ = Tensor::Zeros({norm_size});
        grad_gamma_ = Tensor::Zeros({norm_size});
        grad_beta_ = Tensor::Zeros({norm_size});
    }
}

Tensor LayerNormLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        const auto& shape = input.Shape();

        // Calculate the size of normalized dimensions
        size_t norm_size = 1;
        for (int dim : normalized_shape_) {
            norm_size *= static_cast<size_t>(dim);
        }

        // Reshape to [batch_dims, norm_size]
        size_t batch_size = input.NumElements() / norm_size;
        af::array x_reshaped = af::moddims(x, af::dim4(norm_size, batch_size));

        // Compute mean and variance along normalized dimension (dim 0)
        af::array mean = af::mean(x_reshaped, 0);
        af::array var = af::var(x_reshaped, AF_VARIANCE_POPULATION, 0);

        // Broadcast mean and var
        af::array mean_bc = af::tile(mean, af::dim4(norm_size, 1));
        af::array var_bc = af::tile(var, af::dim4(norm_size, 1));

        // Normalize
        af::array std_inv = 1.0f / af::sqrt(var_bc + eps_);
        af::array normalized = (x_reshaped - mean_bc) * std_inv;

        // Store for backward pass
        normalized_ = AfToTensor(normalized);
        std_inv_ = AfToTensor(std_inv);

        // Apply affine transformation if enabled
        if (elementwise_affine_) {
            af::array gamma = TensorToAf(gamma_);
            af::array beta = TensorToAf(beta_);
            af::array gamma_bc = af::tile(gamma, af::dim4(1, batch_size));
            af::array beta_bc = af::tile(beta, af::dim4(1, batch_size));
            normalized = gamma_bc * normalized + beta_bc;
        }

        // Reshape back to original shape
        af::array output = af::moddims(normalized, x.dims());
        return AfToTensor(output);

    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire LayerNormLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("LayerNorm forward requires ArrayFire");
}

Tensor LayerNormLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        size_t norm_size = 1;
        for (int dim : normalized_shape_) {
            norm_size *= static_cast<size_t>(dim);
        }
        size_t batch_size = grad_output.NumElements() / norm_size;

        af::array grad_out = TensorToAf(grad_output);
        af::array grad_reshaped = af::moddims(grad_out, af::dim4(norm_size, batch_size));

        af::array normalized = TensorToAf(normalized_);
        af::array std_inv = TensorToAf(std_inv_);

        if (elementwise_affine_) {
            af::array gamma = TensorToAf(gamma_);

            // Compute gradients for gamma and beta
            af::array grad_gamma_arr = af::sum(grad_reshaped * normalized, 1);
            af::array grad_beta_arr = af::sum(grad_reshaped, 1);

            grad_gamma_ = AfToTensor(grad_gamma_arr);
            grad_beta_ = AfToTensor(grad_beta_arr);

            // Scale grad by gamma for input gradient
            af::array gamma_bc = af::tile(gamma, af::dim4(1, batch_size));
            grad_reshaped = grad_reshaped * gamma_bc;
        }

        // Compute input gradient
        float N = static_cast<float>(norm_size);
        af::array sum_dy = af::tile(af::sum(grad_reshaped, 0), af::dim4(norm_size, 1));
        af::array sum_dy_norm = af::tile(af::sum(grad_reshaped * normalized, 0), af::dim4(norm_size, 1));

        af::array dx = (1.0f / N) * std_inv * (N * grad_reshaped - sum_dy - normalized * sum_dy_norm);

        af::array dx_output = af::moddims(dx, grad_out.dims());
        return AfToTensor(dx_output);

    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire LayerNormLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("LayerNorm backward requires ArrayFire");
}

std::map<std::string, Tensor> LayerNormLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    if (elementwise_affine_) {
        params["gamma"] = gamma_;
        params["beta"] = beta_;
        params["grad_gamma"] = grad_gamma_;
        params["grad_beta"] = grad_beta_;
    }
    return params;
}

void LayerNormLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) gamma_ = params.at("gamma");
    if (params.count("beta")) beta_ = params.at("beta");
}

// ============================================================================
// InstanceNorm2D Layer Implementation
// ============================================================================

InstanceNorm2DLayer::InstanceNorm2DLayer(int num_features, float eps, bool affine)
    : num_features_(num_features), eps_(eps), affine_(affine) {

    if (affine) {
        gamma_ = Tensor::Ones({static_cast<size_t>(num_features)});
        beta_ = Tensor::Zeros({static_cast<size_t>(num_features)});
        grad_gamma_ = Tensor::Zeros({static_cast<size_t>(num_features)});
        grad_beta_ = Tensor::Zeros({static_cast<size_t>(num_features)});
    }
}

Tensor InstanceNorm2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // Input shape: [N, C, H, W] -> AF: [W, H, C, N]
        dim_t W = x.dims(0);
        dim_t H = x.dims(1);
        dim_t C = x.dims(2);
        dim_t N = x.dims(3);

        // Reshape to [H*W, C, N] for per-instance normalization
        af::array x_reshaped = af::moddims(x, af::dim4(W * H, C, N));

        // Compute mean and variance per (C, N) instance
        af::array mean = af::mean(x_reshaped, 0);  // [1, C, N]
        af::array var = af::var(x_reshaped, AF_VARIANCE_POPULATION, 0);

        // Broadcast and normalize
        af::array mean_bc = af::tile(mean, af::dim4(W * H, 1, 1));
        af::array var_bc = af::tile(var, af::dim4(W * H, 1, 1));

        af::array std_inv = 1.0f / af::sqrt(var_bc + eps_);
        af::array normalized = (x_reshaped - mean_bc) * std_inv;

        // Store for backward
        normalized_ = AfToTensor(af::moddims(normalized, x.dims()));
        std_inv_ = AfToTensor(std_inv);

        // Apply affine if enabled
        if (affine_) {
            af::array gamma = TensorToAf(gamma_);
            af::array beta = TensorToAf(beta_);
            // Reshape to [1, C, 1] for broadcasting
            af::array gamma_bc = af::tile(af::moddims(gamma, af::dim4(1, C, 1)), af::dim4(W * H, 1, N));
            af::array beta_bc = af::tile(af::moddims(beta, af::dim4(1, C, 1)), af::dim4(W * H, 1, N));
            normalized = gamma_bc * normalized + beta_bc;
        }

        af::array output = af::moddims(normalized, x.dims());
        return AfToTensor(output);

    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire InstanceNorm2DLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("InstanceNorm2D forward requires ArrayFire");
}

Tensor InstanceNorm2DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);
        af::array normalized = TensorToAf(normalized_);
        af::array std_inv = TensorToAf(std_inv_);

        dim_t W = grad_out.dims(0);
        dim_t H = grad_out.dims(1);
        dim_t C = grad_out.dims(2);
        dim_t N = grad_out.dims(3);

        af::array grad_reshaped = af::moddims(grad_out, af::dim4(W * H, C, N));
        af::array norm_reshaped = af::moddims(normalized, af::dim4(W * H, C, N));

        if (affine_) {
            af::array gamma = TensorToAf(gamma_);

            // Gradients for gamma and beta
            af::array grad_gamma_arr = af::sum(af::sum(grad_reshaped * norm_reshaped, 0), 2);
            af::array grad_beta_arr = af::sum(af::sum(grad_reshaped, 0), 2);

            grad_gamma_ = AfToTensor(af::moddims(grad_gamma_arr, af::dim4(C)));
            grad_beta_ = AfToTensor(af::moddims(grad_beta_arr, af::dim4(C)));

            // Scale by gamma
            af::array gamma_bc = af::tile(af::moddims(gamma, af::dim4(1, C, 1)), af::dim4(W * H, 1, N));
            grad_reshaped = grad_reshaped * gamma_bc;
        }

        // Input gradient
        float M = static_cast<float>(W * H);
        af::array sum_dy = af::tile(af::sum(grad_reshaped, 0), af::dim4(W * H, 1, 1));
        af::array sum_dy_norm = af::tile(af::sum(grad_reshaped * norm_reshaped, 0), af::dim4(W * H, 1, 1));

        af::array dx = (1.0f / M) * std_inv * (M * grad_reshaped - sum_dy - norm_reshaped * sum_dy_norm);

        return AfToTensor(af::moddims(dx, grad_out.dims()));

    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire InstanceNorm2DLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("InstanceNorm2D backward requires ArrayFire");
}

std::map<std::string, Tensor> InstanceNorm2DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    if (affine_) {
        params["gamma"] = gamma_;
        params["beta"] = beta_;
        params["grad_gamma"] = grad_gamma_;
        params["grad_beta"] = grad_beta_;
    }
    return params;
}

void InstanceNorm2DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) gamma_ = params.at("gamma");
    if (params.count("beta")) beta_ = params.at("beta");
}

// ============================================================================
// GroupNorm Layer Implementation
// ============================================================================

GroupNormLayer::GroupNormLayer(int num_groups, int num_channels, float eps, bool affine)
    : num_groups_(num_groups), num_channels_(num_channels), eps_(eps), affine_(affine) {

    if (num_channels % num_groups != 0) {
        throw std::invalid_argument("num_channels must be divisible by num_groups");
    }

    if (affine) {
        gamma_ = Tensor::Ones({static_cast<size_t>(num_channels)});
        beta_ = Tensor::Zeros({static_cast<size_t>(num_channels)});
        grad_gamma_ = Tensor::Zeros({static_cast<size_t>(num_channels)});
        grad_beta_ = Tensor::Zeros({static_cast<size_t>(num_channels)});
    }
}

Tensor GroupNormLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // Input: [N, C, H, W] -> AF: [W, H, C, N]
        dim_t W = x.dims(0);
        dim_t H = x.dims(1);
        dim_t C = x.dims(2);
        dim_t N = x.dims(3);

        int channels_per_group = num_channels_ / num_groups_;

        // Reshape to [W*H*channels_per_group, num_groups, N]
        af::array x_reshaped = af::moddims(x, af::dim4(W * H * channels_per_group, num_groups_, N));

        // Normalize per group
        af::array mean = af::mean(x_reshaped, 0);  // [1, num_groups, N]
        af::array var = af::var(x_reshaped, AF_VARIANCE_POPULATION, 0);

        af::array mean_bc = af::tile(mean, af::dim4(W * H * channels_per_group, 1, 1));
        af::array var_bc = af::tile(var, af::dim4(W * H * channels_per_group, 1, 1));

        af::array std_inv = 1.0f / af::sqrt(var_bc + eps_);
        af::array normalized = (x_reshaped - mean_bc) * std_inv;

        // Reshape back
        normalized = af::moddims(normalized, x.dims());

        // Store for backward
        normalized_ = AfToTensor(normalized);
        std_inv_ = AfToTensor(af::moddims(std_inv, af::dim4(W * H * channels_per_group, num_groups_, N)));

        // Apply affine
        if (affine_) {
            af::array gamma = TensorToAf(gamma_);
            af::array beta = TensorToAf(beta_);
            // Reshape to [1, 1, C, 1] for proper broadcasting
            af::array gamma_bc = af::tile(af::moddims(gamma, af::dim4(1, 1, C, 1)), af::dim4(W, H, 1, N));
            af::array beta_bc = af::tile(af::moddims(beta, af::dim4(1, 1, C, 1)), af::dim4(W, H, 1, N));
            normalized = gamma_bc * normalized + beta_bc;
        }

        return AfToTensor(normalized);

    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire GroupNormLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("GroupNorm forward requires ArrayFire");
}

Tensor GroupNormLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array normalized = TensorToAf(normalized_);

        dim_t W = grad_out.dims(0);
        dim_t H = grad_out.dims(1);
        dim_t C = grad_out.dims(2);
        dim_t N = grad_out.dims(3);

        int channels_per_group = num_channels_ / num_groups_;

        if (affine_) {
            af::array gamma = TensorToAf(gamma_);

            // Gradients for gamma and beta
            af::array grad_gamma_arr = af::sum(af::sum(af::sum(grad_out * normalized, 0), 1), 3);
            af::array grad_beta_arr = af::sum(af::sum(af::sum(grad_out, 0), 1), 3);

            grad_gamma_ = AfToTensor(af::moddims(grad_gamma_arr, af::dim4(C)));
            grad_beta_ = AfToTensor(af::moddims(grad_beta_arr, af::dim4(C)));

            // Scale grad by gamma
            af::array gamma_bc = af::tile(af::moddims(gamma, af::dim4(1, 1, C, 1)), af::dim4(W, H, 1, N));
            grad_out = grad_out * gamma_bc;
        }

        // Reshape for group computation
        af::array grad_reshaped = af::moddims(grad_out, af::dim4(W * H * channels_per_group, num_groups_, N));
        af::array norm_reshaped = af::moddims(normalized, af::dim4(W * H * channels_per_group, num_groups_, N));
        af::array std_inv = TensorToAf(std_inv_);

        float M = static_cast<float>(W * H * channels_per_group);
        af::array sum_dy = af::tile(af::sum(grad_reshaped, 0), af::dim4(W * H * channels_per_group, 1, 1));
        af::array sum_dy_norm = af::tile(af::sum(grad_reshaped * norm_reshaped, 0), af::dim4(W * H * channels_per_group, 1, 1));

        af::array dx = (1.0f / M) * std_inv * (M * grad_reshaped - sum_dy - norm_reshaped * sum_dy_norm);

        return AfToTensor(af::moddims(dx, grad_out.dims()));

    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire GroupNormLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("GroupNorm backward requires ArrayFire");
}

std::map<std::string, Tensor> GroupNormLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    if (affine_) {
        params["gamma"] = gamma_;
        params["beta"] = beta_;
        params["grad_gamma"] = grad_gamma_;
        params["grad_beta"] = grad_beta_;
    }
    return params;
}

void GroupNormLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) gamma_ = params.at("gamma");
    if (params.count("beta")) beta_ = params.at("beta");
}

// ============================================================================
// Conv1D Layer Implementation
// ============================================================================

Conv1DLayer::Conv1DLayer(int in_channels, int out_channels, int kernel_size,
                         int stride, int padding, int dilation, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      dilation_(dilation), use_bias_(use_bias) {

    // Xavier initialization for weights
    float stddev = std::sqrt(2.0f / (in_channels * kernel_size + out_channels));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);

    weights_ = Tensor({static_cast<size_t>(out_channels),
                       static_cast<size_t>(in_channels),
                       static_cast<size_t>(kernel_size)}, DataType::Float32);

    float* w_data = weights_.Data<float>();
    for (size_t i = 0; i < weights_.NumElements(); ++i) {
        w_data[i] = dist(gen);
    }

    if (use_bias) {
        bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }

    grad_weights_ = Tensor::Zeros(weights_.Shape());
    grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
}

Tensor Conv1DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Input: [batch, in_channels, length] -> AF: [length, in_channels, batch]
        af::array x = TensorToAf(input);
        af::array w = TensorToAf(weights_);

        dim_t L = x.dims(0);
        dim_t batch = x.dims(2);

        // Apply padding if needed
        if (padding_ > 0) {
            af::array padded = af::constant(0.0f, L + 2 * padding_, x.dims(1), x.dims(2));
            padded(af::seq(padding_, padding_ + L - 1), af::span, af::span) = x;
            x = padded;
            L = x.dims(0);
        }

        // Output length
        dim_t L_out = (L - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;

        // Simple implementation: loop over output positions
        af::array output = af::constant(0.0f, L_out, out_channels_, batch);

        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int ic = 0; ic < in_channels_; ++ic) {
                // Get kernel for this input-output channel pair
                af::array kernel = w(af::span, ic, oc);  // [kernel_size]

                // Convolve each batch sample
                for (dim_t b = 0; b < batch; ++b) {
                    af::array x_channel = x(af::span, ic, b);  // [L]

                    // Use ArrayFire convolve1
                    af::array conv_result = af::convolve1(x_channel, kernel, AF_CONV_DEFAULT);

                    // Handle stride and dilation (simplified)
                    if (stride_ > 1) {
                        conv_result = conv_result(af::seq(0, L_out * stride_ - 1, stride_));
                    }

                    // Accumulate
                    output(af::span, oc, b) += conv_result(af::seq(0, L_out - 1));
                }
            }
        }

        // Add bias
        if (use_bias_) {
            af::array b = TensorToAf(bias_);
            for (int oc = 0; oc < out_channels_; ++oc) {
                output(af::span, oc, af::span) += b(oc);
            }
        }

        return AfToTensor(output);

    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire Conv1DLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Conv1D forward requires ArrayFire");
}

Tensor Conv1DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);
        af::array w = TensorToAf(weights_);

        dim_t L_out = grad_out.dims(0);
        dim_t batch = grad_out.dims(2);
        dim_t L_in = x.dims(0);

        // Gradient w.r.t. bias
        if (use_bias_) {
            af::array grad_b = af::sum(af::sum(grad_out, 0), 2);
            grad_bias_ = AfToTensor(af::moddims(grad_b, af::dim4(out_channels_)));
        }

        // Gradient w.r.t. weights - convolution of input with grad_output
        af::array grad_w = af::constant(0.0f, kernel_size_, in_channels_, out_channels_);

        // Gradient w.r.t. input - transposed convolution
        af::array grad_x = af::constant(0.0f, L_in, in_channels_, batch);

        // Simplified gradient computation
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int ic = 0; ic < in_channels_; ++ic) {
                af::array kernel = w(af::span, ic, oc);

                for (dim_t b = 0; b < batch; ++b) {
                    af::array grad_o = grad_out(af::span, oc, b);
                    af::array x_channel = x(af::span, ic, b);

                    // Grad w.r.t. weights
                    af::array gw = af::convolve1(x_channel, grad_o, AF_CONV_DEFAULT);
                    grad_w(af::span, ic, oc) += gw(af::seq(0, kernel_size_ - 1));

                    // Grad w.r.t. input (transposed convolution)
                    af::array flipped_kernel = af::flip(kernel, 0);
                    af::array gx = af::convolve1(grad_o, flipped_kernel, AF_CONV_EXPAND);
                    grad_x(af::span, ic, b) += gx(af::seq(0, L_in - 1));
                }
            }
        }

        grad_weights_ = AfToTensor(grad_w);
        return AfToTensor(grad_x);

    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire Conv1DLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Conv1D backward requires ArrayFire");
}

std::map<std::string, Tensor> Conv1DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void Conv1DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("weights")) weights_ = params.at("weights");
    if (params.count("bias")) bias_ = params.at("bias");
}

// ============================================================================
// LSTM Layer Implementation
// ============================================================================

LSTMLayer::LSTMLayer(int input_size, int hidden_size, int num_layers,
                     bool batch_first, bool bidirectional, float dropout)
    : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
      batch_first_(batch_first), bidirectional_(bidirectional), dropout_(dropout) {

    InitializeWeights();
}

void LSTMLayer::InitializeWeights() {
    int num_directions = bidirectional_ ? 2 : 1;

    W_ih_.resize(num_layers_);
    W_hh_.resize(num_layers_);
    b_ih_.resize(num_layers_);
    b_hh_.resize(num_layers_);
    grad_W_ih_.resize(num_layers_);
    grad_W_hh_.resize(num_layers_);
    grad_b_ih_.resize(num_layers_);
    grad_b_hh_.resize(num_layers_);

    if (bidirectional_) {
        W_ih_reverse_.resize(num_layers_);
        W_hh_reverse_.resize(num_layers_);
        b_ih_reverse_.resize(num_layers_);
        b_hh_reverse_.resize(num_layers_);
        grad_W_ih_reverse_.resize(num_layers_);
        grad_W_hh_reverse_.resize(num_layers_);
        grad_b_ih_reverse_.resize(num_layers_);
        grad_b_hh_reverse_.resize(num_layers_);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    for (int layer = 0; layer < num_layers_; layer++) {
        // Input size for this layer
        int layer_input_size = (layer == 0) ? input_size_ : hidden_size_ * num_directions;
        int gate_size = 4 * hidden_size_;

        // Xavier initialization for input-hidden weights
        float limit_ih = std::sqrt(6.0f / (layer_input_size + hidden_size_));
        af::array w_ih = af::randu(af::dim4(gate_size, layer_input_size), af::dtype::f32) * 2.0f * limit_ih - limit_ih;
        W_ih_[layer] = AfToTensor(w_ih);

        // Xavier initialization for hidden-hidden weights
        float limit_hh = std::sqrt(6.0f / (hidden_size_ + hidden_size_));
        af::array w_hh = af::randu(af::dim4(gate_size, hidden_size_), af::dtype::f32) * 2.0f * limit_hh - limit_hh;
        W_hh_[layer] = AfToTensor(w_hh);

        // Initialize biases to zero (with forget gate bias = 1 for better gradient flow)
        af::array b_ih = af::constant(0.0f, af::dim4(gate_size));
        af::array b_hh = af::constant(0.0f, af::dim4(gate_size));
        // Set forget gate bias to 1
        b_ih(af::seq(hidden_size_, 2 * hidden_size_ - 1)) = 1.0f;
        b_ih_[layer] = AfToTensor(b_ih);
        b_hh_[layer] = AfToTensor(b_hh);

        // Initialize gradient accumulators
        grad_W_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
        grad_W_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
        grad_b_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        grad_b_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});

        if (bidirectional_) {
            af::array w_ih_r = af::randu(af::dim4(gate_size, layer_input_size), af::dtype::f32) * 2.0f * limit_ih - limit_ih;
            af::array w_hh_r = af::randu(af::dim4(gate_size, hidden_size_), af::dtype::f32) * 2.0f * limit_hh - limit_hh;
            af::array b_ih_r = af::constant(0.0f, af::dim4(gate_size));
            af::array b_hh_r = af::constant(0.0f, af::dim4(gate_size));
            b_ih_r(af::seq(hidden_size_, 2 * hidden_size_ - 1)) = 1.0f;

            W_ih_reverse_[layer] = AfToTensor(w_ih_r);
            W_hh_reverse_[layer] = AfToTensor(w_hh_r);
            b_ih_reverse_[layer] = AfToTensor(b_ih_r);
            b_hh_reverse_[layer] = AfToTensor(b_hh_r);

            grad_W_ih_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
            grad_W_hh_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
            grad_b_ih_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
            grad_b_hh_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        }
    }
#else
    // CPU fallback initialization
    for (int layer = 0; layer < num_layers_; layer++) {
        int layer_input_size = (layer == 0) ? input_size_ : hidden_size_ * num_directions;
        int gate_size = 4 * hidden_size_;

        W_ih_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
        W_hh_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
        b_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        b_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        grad_W_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
        grad_W_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
        grad_b_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        grad_b_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});

        if (bidirectional_) {
            W_ih_reverse_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
            W_hh_reverse_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
            b_ih_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
            b_hh_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        }
    }
#endif
}

void LSTMLayer::ResetState() {
    h_n_ = Tensor();
    c_n_ = Tensor();
}

void LSTMLayer::SetHiddenState(const Tensor& h0) {
    h_n_ = h0.Clone();
}

void LSTMLayer::SetCellState(const Tensor& c0) {
    c_n_ = c0.Clone();
}

Tensor LSTMLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        // Handle batch_first format
        // Input: [batch, seq_len, input_size] if batch_first
        // Convert to: [seq_len, batch, input_size] for processing
        dim_t batch_size, seq_len, input_dim;

        if (batch_first_) {
            batch_size = x.dims(0);
            seq_len = x.dims(1);
            input_dim = x.dims(2);
            // Transpose to [seq_len, batch, input_size]
            x = af::reorder(x, 1, 0, 2);
        } else {
            seq_len = x.dims(0);
            batch_size = x.dims(1);
            input_dim = x.dims(2);
        }

        int num_directions = bidirectional_ ? 2 : 1;

        // Initialize hidden and cell states if not set
        if (h_n_.NumElements() == 0) {
            h_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                                   static_cast<size_t>(batch_size),
                                   static_cast<size_t>(hidden_size_)});
        }
        if (c_n_.NumElements() == 0) {
            c_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                                   static_cast<size_t>(batch_size),
                                   static_cast<size_t>(hidden_size_)});
        }

        // Clear caches
        cached_inputs_.clear();
        cached_gates_.clear();
        cached_cell_states_.clear();
        cached_hidden_states_.clear();

        // Output container: [seq_len, batch, hidden_size * num_directions]
        af::array output = af::constant(0.0f, af::dim4(seq_len, batch_size, hidden_size_ * num_directions));

        af::array layer_input = x;

        for (int layer = 0; layer < num_layers_; layer++) {
            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);
            af::array b_ih = TensorToAf(b_ih_[layer]);
            af::array b_hh = TensorToAf(b_hh_[layer]);

            // Get initial hidden/cell state for this layer
            af::array h_full = TensorToAf(h_n_);
            af::array c_full = TensorToAf(c_n_);
            af::array h = h_full(layer, af::span, af::span);
            af::array c = c_full(layer, af::span, af::span);
            h = af::moddims(h, af::dim4(batch_size, hidden_size_));
            c = af::moddims(c, af::dim4(batch_size, hidden_size_));

            // Pre-compute input projections for ALL timesteps at once
            // layer_input: [seq_len, batch, input_size]
            // Reshape to [seq_len * batch, input_size] for batch matmul
            dim_t layer_input_size = layer_input.dims(2);
            af::array input_flat = af::moddims(layer_input, af::dim4(seq_len * batch_size, layer_input_size));

            // Compute all input projections at once: [seq_len * batch, 4 * hidden_size]
            // W_ih: [4 * hidden_size, input_size]
            af::array input_proj = af::matmul(input_flat, af::transpose(W_ih));
            // Add bias (broadcast)
            input_proj = input_proj + af::tile(af::transpose(b_ih), static_cast<unsigned int>(seq_len * batch_size));
            // Reshape back: [seq_len, batch, 4 * hidden_size]
            input_proj = af::moddims(input_proj, af::dim4(seq_len, batch_size, 4 * hidden_size_));

            // Cache for backward
            cached_inputs_.push_back(AfToTensor(layer_input));

            // Storage for hidden states and cell states over time
            af::array h_states = af::constant(0.0f, af::dim4(seq_len + 1, batch_size, hidden_size_));
            af::array c_states = af::constant(0.0f, af::dim4(seq_len + 1, batch_size, hidden_size_));
            af::array all_gates = af::constant(0.0f, af::dim4(seq_len, batch_size, 4 * hidden_size_));

            // Store initial states
            h_states(0, af::span, af::span) = h;
            c_states(0, af::span, af::span) = c;

            // Forward pass through time using vectorized operations per timestep
            // Note: The recurrent dependency requires sequential processing,
            // but each timestep is fully vectorized across the batch
            for (dim_t t = 0; t < seq_len; t++) {
                // Get input projection for this timestep: [batch, 4 * hidden_size]
                af::array x_t = input_proj(t, af::span, af::span);
                x_t = af::moddims(x_t, af::dim4(batch_size, 4 * hidden_size_));

                // Compute hidden projection: h @ W_hh^T + b_hh
                af::array h_proj = af::matmul(h, af::transpose(W_hh));
                h_proj = h_proj + af::tile(af::transpose(b_hh), static_cast<unsigned int>(batch_size));

                // Combined gates: [batch, 4 * hidden_size]
                af::array gates = x_t + h_proj;

                // Split into individual gates and apply activations
                // Order: input, forget, cell, output
                af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, hidden_size_ - 1)));
                af::array f_gate = af::sigmoid(gates(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1)));
                af::array g_gate = af::tanh(gates(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1)));
                af::array o_gate = af::sigmoid(gates(af::span, af::seq(3 * hidden_size_, 4 * hidden_size_ - 1)));

                // Update cell state: c_t = f * c_{t-1} + i * g
                c = f_gate * c + i_gate * g_gate;

                // Update hidden state: h_t = o * tanh(c_t)
                h = o_gate * af::tanh(c);

                // Store states
                h_states(t + 1, af::span, af::span) = h;
                c_states(t + 1, af::span, af::span) = c;

                // Store gates for backward pass (pre-activation for efficiency)
                all_gates(t, af::span, af::span) = gates;
            }

            // Cache for backward
            cached_gates_.push_back(AfToTensor(all_gates));
            cached_hidden_states_.push_back(AfToTensor(h_states));
            cached_cell_states_.push_back(AfToTensor(c_states));

            // Extract output hidden states [seq_len, batch, hidden_size]
            af::array layer_output = h_states(af::seq(1, static_cast<double>(seq_len)), af::span, af::span);

            // Handle bidirectional
            if (bidirectional_) {
                af::array W_ih_r = TensorToAf(W_ih_reverse_[layer]);
                af::array W_hh_r = TensorToAf(W_hh_reverse_[layer]);
                af::array b_ih_r = TensorToAf(b_ih_reverse_[layer]);
                af::array b_hh_r = TensorToAf(b_hh_reverse_[layer]);

                // Get reverse initial state
                af::array h_r = h_full(num_layers_ + layer, af::span, af::span);
                af::array c_r = c_full(num_layers_ + layer, af::span, af::span);
                h_r = af::moddims(h_r, af::dim4(batch_size, hidden_size_));
                c_r = af::moddims(c_r, af::dim4(batch_size, hidden_size_));

                // Pre-compute reverse input projections
                af::array input_proj_r = af::matmul(input_flat, af::transpose(W_ih_r));
                input_proj_r = input_proj_r + af::tile(af::transpose(b_ih_r), static_cast<unsigned int>(seq_len * batch_size));
                input_proj_r = af::moddims(input_proj_r, af::dim4(seq_len, batch_size, 4 * hidden_size_));

                af::array h_states_r = af::constant(0.0f, af::dim4(seq_len + 1, batch_size, hidden_size_));
                af::array c_states_r = af::constant(0.0f, af::dim4(seq_len + 1, batch_size, hidden_size_));

                h_states_r(seq_len, af::span, af::span) = h_r;
                c_states_r(seq_len, af::span, af::span) = c_r;

                // Backward through time (reverse direction)
                for (dim_t t = seq_len - 1; t >= 0; t--) {
                    af::array x_t = input_proj_r(t, af::span, af::span);
                    x_t = af::moddims(x_t, af::dim4(batch_size, 4 * hidden_size_));

                    af::array h_proj = af::matmul(h_r, af::transpose(W_hh_r));
                    h_proj = h_proj + af::tile(af::transpose(b_hh_r), static_cast<unsigned int>(batch_size));

                    af::array gates = x_t + h_proj;

                    af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, hidden_size_ - 1)));
                    af::array f_gate = af::sigmoid(gates(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1)));
                    af::array g_gate = af::tanh(gates(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1)));
                    af::array o_gate = af::sigmoid(gates(af::span, af::seq(3 * hidden_size_, 4 * hidden_size_ - 1)));

                    c_r = f_gate * c_r + i_gate * g_gate;
                    h_r = o_gate * af::tanh(c_r);

                    h_states_r(t, af::span, af::span) = h_r;
                    c_states_r(t, af::span, af::span) = c_r;
                }

                // Extract reverse output and concatenate
                af::array layer_output_r = h_states_r(af::seq(0, static_cast<double>(seq_len - 1)), af::span, af::span);
                layer_output = af::join(2, layer_output, layer_output_r);

                // Update final states
                h_full(num_layers_ + layer, af::span, af::span) = h_r;
                c_full(num_layers_ + layer, af::span, af::span) = c_r;
            }

            // Update final hidden and cell states for forward direction
            h_full(layer, af::span, af::span) = h;
            c_full(layer, af::span, af::span) = c;

            // Update stored states
            h_n_ = AfToTensor(h_full);
            c_n_ = AfToTensor(c_full);

            // Apply dropout between layers (not on last layer)
            if (layer < num_layers_ - 1 && dropout_ > 0.0f && training_) {
                af::array mask = (af::randu(layer_output.dims()) > dropout_).as(af::dtype::f32);
                layer_output = layer_output * mask / (1.0f - dropout_);
            }

            // Use this layer's output as next layer's input
            layer_input = layer_output;
        }

        output = layer_input;

        // Convert back to batch_first if needed
        if (batch_first_) {
            output = af::reorder(output, 1, 0, 2);
        }

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire LSTMLayer::Forward failed: {}, falling back to CPU", e.what());
    }
#endif

    // CPU fallback implementation
    const auto& shape = input.Shape();
    size_t batch_size, seq_len, input_dim;

    if (batch_first_) {
        batch_size = shape[0];
        seq_len = shape[1];
        input_dim = shape[2];
    } else {
        seq_len = shape[0];
        batch_size = shape[1];
        input_dim = shape[2];
    }

    int num_directions = bidirectional_ ? 2 : 1;

    // Reinitialize weights with CPU if they have null data (ArrayFire init failed)
    if (W_ih_.empty() || W_ih_[0].Data<float>() == nullptr) {
        for (int layer = 0; layer < num_layers_; layer++) {
            size_t layer_input_size = (layer == 0) ? input_dim : static_cast<size_t>(hidden_size_ * num_directions);
            size_t gate_size = static_cast<size_t>(4 * hidden_size_);
            W_ih_[layer] = Tensor::Random({gate_size, layer_input_size});
            W_hh_[layer] = Tensor::Random({gate_size, static_cast<size_t>(hidden_size_)});
            b_ih_[layer] = Tensor::Zeros({gate_size});
            b_hh_[layer] = Tensor::Zeros({gate_size});
        }
    }

    if (h_n_.NumElements() == 0 || h_n_.Data<float>() == nullptr) {
            h_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                               batch_size, static_cast<size_t>(hidden_size_)});
    }
    if (c_n_.NumElements() == 0 || c_n_.Data<float>() == nullptr) {
            c_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                               batch_size, static_cast<size_t>(hidden_size_)});
    }

    size_t out_dim0 = batch_first_ ? batch_size : seq_len;
    size_t out_dim1 = batch_first_ ? seq_len : batch_size;
    size_t out_features = static_cast<size_t>(hidden_size_ * num_directions);
    Tensor output = Tensor::Zeros({out_dim0, out_dim1, out_features});

    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();
    float* h_data = h_n_.Data<float>();
    float* c_data = c_n_.Data<float>();

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto tanh_f = [](float x) { return std::tanh(x); };

    Tensor layer_input = input;
    size_t layer_input_size = input_dim;

    for (int layer = 0; layer < num_layers_; layer++) {
        const float* W_ih = W_ih_[layer].Data<float>();
        const float* W_hh = W_hh_[layer].Data<float>();
        const float* b_ih = b_ih_[layer].Data<float>();
        const float* b_hh = b_hh_[layer].Data<float>();
        int gate_size = 4 * hidden_size_;

        Tensor layer_output = Tensor::Zeros({seq_len, batch_size, static_cast<size_t>(hidden_size_)});
        float* layer_out = layer_output.Data<float>();
        const float* layer_in = layer_input.Data<float>();

        for (size_t b = 0; b < batch_size; b++) {
            std::vector<float> h(hidden_size_), c(hidden_size_);
            for (int i = 0; i < hidden_size_; i++) {
                h[i] = h_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i];
                c[i] = c_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i];
            }

            for (size_t t = 0; t < seq_len; t++) {
                std::vector<float> gates(gate_size, 0.0f);
                const float* x_ptr;
                if (layer == 0) {
                    if (batch_first_) x_ptr = input_data + b * seq_len * input_dim + t * input_dim;
                    else x_ptr = input_data + t * batch_size * input_dim + b * input_dim;
                } else {
                    x_ptr = layer_in + t * batch_size * layer_input_size + b * layer_input_size;
                }

                for (int g = 0; g < gate_size; g++) {
                    gates[g] = b_ih[g] + b_hh[g];
                    for (size_t k = 0; k < layer_input_size; k++)
                        gates[g] += W_ih[g * layer_input_size + k] * x_ptr[k];
                    for (int k = 0; k < hidden_size_; k++)
                        gates[g] += W_hh[g * hidden_size_ + k] * h[k];
                }

                for (int i = 0; i < hidden_size_; i++) {
                    float i_gate = sigmoid(gates[i]);
                    float f_gate = sigmoid(gates[hidden_size_ + i]);
                    float g_gate = tanh_f(gates[2 * hidden_size_ + i]);
                    float o_gate = sigmoid(gates[3 * hidden_size_ + i]);
                    c[i] = f_gate * c[i] + i_gate * g_gate;
                    h[i] = o_gate * tanh_f(c[i]);
                }

                for (int i = 0; i < hidden_size_; i++)
                    layer_out[t * batch_size * hidden_size_ + b * hidden_size_ + i] = h[i];
            }

            for (int i = 0; i < hidden_size_; i++) {
                h_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i] = h[i];
                c_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i] = c[i];
            }
        }
        layer_input = layer_output;
        layer_input_size = static_cast<size_t>(hidden_size_);
    }

    const float* final_out = layer_input.Data<float>();
    for (size_t t = 0; t < seq_len; t++) {
        for (size_t b = 0; b < batch_size; b++) {
            for (int f = 0; f < hidden_size_; f++) {
                float val = final_out[t * batch_size * hidden_size_ + b * hidden_size_ + f];
                if (batch_first_) output_data[b * seq_len * out_features + t * out_features + f] = val;
                else output_data[t * batch_size * out_features + b * out_features + f] = val;
            }
        }
    }
    return output;
}

Tensor LSTMLayer::Backward(const Tensor& grad_output) {
    // Check if caches are populated (Forward must have been called with ArrayFire success)
    if (cached_inputs_.empty() || cached_gates_.empty() ||
        cached_hidden_states_.empty() || cached_cell_states_.empty()) {
        // CPU fallback Forward doesn't populate caches, so return zero gradients
        spdlog::warn("LSTMLayer::Backward called without cached data from Forward, returning zero gradients");
        return Tensor::Zeros(grad_output.Shape());
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array dout = TensorToAf(grad_output);

        // Convert to seq_first if batch_first
        if (batch_first_) {
            dout = af::reorder(dout, 1, 0, 2);
        }

        dim_t seq_len = dout.dims(0);
        dim_t batch_size = dout.dims(1);
        int num_directions = bidirectional_ ? 2 : 1;

        // Gradient w.r.t. input (will accumulate from all layers)
        af::array dx;

        // Process layers in reverse order
        af::array layer_grad = dout;

        for (int layer = num_layers_ - 1; layer >= 0; layer--) {
            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);
            af::array cached_input = TensorToAf(cached_inputs_[layer]);
            af::array cached_gates = TensorToAf(cached_gates_[layer]);
            af::array cached_h = TensorToAf(cached_hidden_states_[layer]);
            af::array cached_c = TensorToAf(cached_cell_states_[layer]);

            dim_t layer_input_size = cached_input.dims(2);
            int gate_size = 4 * hidden_size_;

            // Initialize gradient accumulators
            af::array dW_ih = af::constant(0.0f, W_ih.dims());
            af::array dW_hh = af::constant(0.0f, W_hh.dims());
            af::array db_ih = af::constant(0.0f, af::dim4(gate_size));
            af::array db_hh = af::constant(0.0f, af::dim4(gate_size));

            // Gradient w.r.t. next hidden and cell state
            af::array dh_next = af::constant(0.0f, af::dim4(batch_size, hidden_size_));
            af::array dc_next = af::constant(0.0f, af::dim4(batch_size, hidden_size_));

            // Gradient w.r.t. layer input
            af::array d_layer_input = af::constant(0.0f, cached_input.dims());

            // Split layer_grad for bidirectional
            af::array grad_forward, grad_backward;
            if (bidirectional_) {
                grad_forward = layer_grad(af::span, af::span, af::seq(0, hidden_size_ - 1));
                grad_backward = layer_grad(af::span, af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1));
            } else {
                grad_forward = layer_grad;
            }

            // Backward through time (BPTT) for forward direction
            for (dim_t t = seq_len - 1; t >= 0; t--) {
                // Get cached values
                af::array h_prev = cached_h(t, af::span, af::span);
                h_prev = af::moddims(h_prev, af::dim4(batch_size, hidden_size_));
                af::array c_prev = cached_c(t, af::span, af::span);
                c_prev = af::moddims(c_prev, af::dim4(batch_size, hidden_size_));
                af::array c_t = cached_c(t + 1, af::span, af::span);
                c_t = af::moddims(c_t, af::dim4(batch_size, hidden_size_));

                af::array gates = cached_gates(t, af::span, af::span);
                gates = af::moddims(gates, af::dim4(batch_size, gate_size));

                // Recompute gate activations
                af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, hidden_size_ - 1)));
                af::array f_gate = af::sigmoid(gates(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1)));
                af::array g_gate = af::tanh(gates(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1)));
                af::array o_gate = af::sigmoid(gates(af::span, af::seq(3 * hidden_size_, 4 * hidden_size_ - 1)));

                // Get output gradient for this timestep
                af::array dh = grad_forward(t, af::span, af::span);
                dh = af::moddims(dh, af::dim4(batch_size, hidden_size_));
                dh = dh + dh_next;

                // Gradient through output gate: h = o * tanh(c)
                af::array tanh_c = af::tanh(c_t);
                af::array do_gate = dh * tanh_c * o_gate * (1.0f - o_gate);  // sigmoid derivative
                af::array dc = dh * o_gate * (1.0f - tanh_c * tanh_c) + dc_next;  // tanh derivative

                // Gradient through cell update: c = f * c_prev + i * g
                af::array df_gate = dc * c_prev * f_gate * (1.0f - f_gate);
                af::array di_gate = dc * g_gate * i_gate * (1.0f - i_gate);
                af::array dg_gate = dc * i_gate * (1.0f - g_gate * g_gate);
                dc_next = dc * f_gate;

                // Combine gate gradients: [batch, 4 * hidden_size]
                af::array dgates = af::join(1, di_gate, df_gate, dg_gate, do_gate);

                // Gradient w.r.t. weights and biases (accumulate)
                // dW_ih += dgates^T @ x_t
                af::array x_t = cached_input(t, af::span, af::span);
                x_t = af::moddims(x_t, af::dim4(batch_size, layer_input_size));
                dW_ih = dW_ih + af::matmul(af::transpose(dgates), x_t);

                // dW_hh += dgates^T @ h_prev
                dW_hh = dW_hh + af::matmul(af::transpose(dgates), h_prev);

                // db_ih += sum(dgates, batch)
                db_ih = db_ih + af::sum(dgates, 0);
                db_hh = db_hh + af::sum(dgates, 0);

                // Gradient w.r.t. input: dx = dgates @ W_ih
                af::array dx_t = af::matmul(dgates, W_ih);
                d_layer_input(t, af::span, af::span) = dx_t;

                // Gradient w.r.t. previous hidden: dh_prev = dgates @ W_hh
                dh_next = af::matmul(dgates, W_hh);
            }

            // Store gradients
            grad_W_ih_[layer] = AfToTensor(dW_ih);
            grad_W_hh_[layer] = AfToTensor(dW_hh);
            grad_b_ih_[layer] = AfToTensor(af::moddims(db_ih, af::dim4(gate_size)));
            grad_b_hh_[layer] = AfToTensor(af::moddims(db_hh, af::dim4(gate_size)));

            // TODO: Handle bidirectional backward pass similarly

            // Pass gradient to previous layer
            layer_grad = d_layer_input;
        }

        dx = layer_grad;

        // Convert back to batch_first if needed
        if (batch_first_) {
            dx = af::reorder(dx, 1, 0, 2);
        }

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire LSTMLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("LSTM backward requires ArrayFire");
}

std::map<std::string, Tensor> LSTMLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    for (int layer = 0; layer < num_layers_; layer++) {
        std::string prefix = "layer" + std::to_string(layer) + "_";
        params[prefix + "W_ih"] = W_ih_[layer];
        params[prefix + "W_hh"] = W_hh_[layer];
        params[prefix + "b_ih"] = b_ih_[layer];
        params[prefix + "b_hh"] = b_hh_[layer];
        params[prefix + "grad_W_ih"] = grad_W_ih_[layer];
        params[prefix + "grad_W_hh"] = grad_W_hh_[layer];
        params[prefix + "grad_b_ih"] = grad_b_ih_[layer];
        params[prefix + "grad_b_hh"] = grad_b_hh_[layer];

        if (bidirectional_) {
            params[prefix + "W_ih_reverse"] = W_ih_reverse_[layer];
            params[prefix + "W_hh_reverse"] = W_hh_reverse_[layer];
            params[prefix + "b_ih_reverse"] = b_ih_reverse_[layer];
            params[prefix + "b_hh_reverse"] = b_hh_reverse_[layer];
        }
    }
    return params;
}

void LSTMLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    for (int layer = 0; layer < num_layers_; layer++) {
        std::string prefix = "layer" + std::to_string(layer) + "_";
        if (params.count(prefix + "W_ih")) W_ih_[layer] = params.at(prefix + "W_ih");
        if (params.count(prefix + "W_hh")) W_hh_[layer] = params.at(prefix + "W_hh");
        if (params.count(prefix + "b_ih")) b_ih_[layer] = params.at(prefix + "b_ih");
        if (params.count(prefix + "b_hh")) b_hh_[layer] = params.at(prefix + "b_hh");

        if (bidirectional_) {
            if (params.count(prefix + "W_ih_reverse")) W_ih_reverse_[layer] = params.at(prefix + "W_ih_reverse");
            if (params.count(prefix + "W_hh_reverse")) W_hh_reverse_[layer] = params.at(prefix + "W_hh_reverse");
            if (params.count(prefix + "b_ih_reverse")) b_ih_reverse_[layer] = params.at(prefix + "b_ih_reverse");
            if (params.count(prefix + "b_hh_reverse")) b_hh_reverse_[layer] = params.at(prefix + "b_hh_reverse");
        }
    }
}

// ============================================================================
// GRU Layer Implementation
// ============================================================================

GRULayer::GRULayer(int input_size, int hidden_size, int num_layers,
                   bool batch_first, bool bidirectional, float dropout)
    : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
      batch_first_(batch_first), bidirectional_(bidirectional), dropout_(dropout) {

    InitializeWeights();
}

void GRULayer::InitializeWeights() {
    int num_directions = bidirectional_ ? 2 : 1;

    W_ih_.resize(num_layers_);
    W_hh_.resize(num_layers_);
    b_ih_.resize(num_layers_);
    b_hh_.resize(num_layers_);
    grad_W_ih_.resize(num_layers_);
    grad_W_hh_.resize(num_layers_);
    grad_b_ih_.resize(num_layers_);
    grad_b_hh_.resize(num_layers_);

    if (bidirectional_) {
        W_ih_reverse_.resize(num_layers_);
        W_hh_reverse_.resize(num_layers_);
        b_ih_reverse_.resize(num_layers_);
        b_hh_reverse_.resize(num_layers_);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    for (int layer = 0; layer < num_layers_; layer++) {
        int layer_input_size = (layer == 0) ? input_size_ : hidden_size_ * num_directions;
        int gate_size = 3 * hidden_size_;  // GRU has 3 gates

        float limit_ih = std::sqrt(6.0f / (layer_input_size + hidden_size_));
        float limit_hh = std::sqrt(6.0f / (hidden_size_ + hidden_size_));

        af::array w_ih = af::randu(af::dim4(gate_size, layer_input_size), af::dtype::f32) * 2.0f * limit_ih - limit_ih;
        af::array w_hh = af::randu(af::dim4(gate_size, hidden_size_), af::dtype::f32) * 2.0f * limit_hh - limit_hh;
        af::array b_ih = af::constant(0.0f, af::dim4(gate_size));
        af::array b_hh = af::constant(0.0f, af::dim4(gate_size));

        W_ih_[layer] = AfToTensor(w_ih);
        W_hh_[layer] = AfToTensor(w_hh);
        b_ih_[layer] = AfToTensor(b_ih);
        b_hh_[layer] = AfToTensor(b_hh);

        grad_W_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
        grad_W_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
        grad_b_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        grad_b_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});

        if (bidirectional_) {
            af::array w_ih_r = af::randu(af::dim4(gate_size, layer_input_size), af::dtype::f32) * 2.0f * limit_ih - limit_ih;
            af::array w_hh_r = af::randu(af::dim4(gate_size, hidden_size_), af::dtype::f32) * 2.0f * limit_hh - limit_hh;

            W_ih_reverse_[layer] = AfToTensor(w_ih_r);
            W_hh_reverse_[layer] = AfToTensor(w_hh_r);
            b_ih_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
            b_hh_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        }
    }
#else
    for (int layer = 0; layer < num_layers_; layer++) {
        int layer_input_size = (layer == 0) ? input_size_ : hidden_size_ * num_directions;
        int gate_size = 3 * hidden_size_;

        W_ih_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
        W_hh_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
        b_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        b_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
    }
#endif
}

void GRULayer::ResetState() {
    h_n_ = Tensor();
}

void GRULayer::SetHiddenState(const Tensor& h0) {
    h_n_ = h0.Clone();
}

Tensor GRULayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        dim_t batch_size, seq_len, input_dim;

        if (batch_first_) {
            batch_size = x.dims(0);
            seq_len = x.dims(1);
            input_dim = x.dims(2);
            x = af::reorder(x, 1, 0, 2);
        } else {
            seq_len = x.dims(0);
            batch_size = x.dims(1);
            input_dim = x.dims(2);
        }

        int num_directions = bidirectional_ ? 2 : 1;

        if (h_n_.NumElements() == 0) {
            h_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                                   static_cast<size_t>(batch_size),
                                   static_cast<size_t>(hidden_size_)});
        }

        cached_inputs_.clear();
        cached_gates_.clear();
        cached_hidden_states_.clear();

        af::array output = af::constant(0.0f, af::dim4(seq_len, batch_size, hidden_size_ * num_directions));
        af::array layer_input = x;

        for (int layer = 0; layer < num_layers_; layer++) {
            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);
            af::array b_ih = TensorToAf(b_ih_[layer]);
            af::array b_hh = TensorToAf(b_hh_[layer]);

            af::array h_full = TensorToAf(h_n_);
            af::array h = h_full(layer, af::span, af::span);
            h = af::moddims(h, af::dim4(batch_size, hidden_size_));

            dim_t layer_input_size = layer_input.dims(2);
            af::array input_flat = af::moddims(layer_input, af::dim4(seq_len * batch_size, layer_input_size));

            // Pre-compute all input projections
            af::array input_proj = af::matmul(input_flat, af::transpose(W_ih));
            input_proj = input_proj + af::tile(af::transpose(b_ih), static_cast<unsigned int>(seq_len * batch_size));
            input_proj = af::moddims(input_proj, af::dim4(seq_len, batch_size, 3 * hidden_size_));

            cached_inputs_.push_back(AfToTensor(layer_input));

            af::array h_states = af::constant(0.0f, af::dim4(seq_len + 1, batch_size, hidden_size_));
            af::array all_gates = af::constant(0.0f, af::dim4(seq_len, batch_size, 3 * hidden_size_));

            h_states(0, af::span, af::span) = h;

            // GRU forward pass - vectorized per timestep
            for (dim_t t = 0; t < seq_len; t++) {
                af::array x_t = input_proj(t, af::span, af::span);
                x_t = af::moddims(x_t, af::dim4(batch_size, 3 * hidden_size_));

                af::array h_proj = af::matmul(h, af::transpose(W_hh));
                h_proj = h_proj + af::tile(af::transpose(b_hh), static_cast<unsigned int>(batch_size));

                // GRU gates: reset, update, new
                af::array r_gate = af::sigmoid(x_t(af::span, af::seq(0, hidden_size_ - 1)) +
                                                h_proj(af::span, af::seq(0, hidden_size_ - 1)));
                af::array z_gate = af::sigmoid(x_t(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1)) +
                                                h_proj(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1)));
                af::array n_gate = af::tanh(x_t(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1)) +
                                             r_gate * h_proj(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1)));

                // Update hidden state: h = (1 - z) * n + z * h
                h = (1.0f - z_gate) * n_gate + z_gate * h;

                h_states(t + 1, af::span, af::span) = h;
                af::array gates = af::join(1, r_gate, z_gate, n_gate);
                all_gates(t, af::span, af::span) = gates;
            }

            cached_gates_.push_back(AfToTensor(all_gates));
            cached_hidden_states_.push_back(AfToTensor(h_states));

            af::array layer_output = h_states(af::seq(1, static_cast<double>(seq_len)), af::span, af::span);

            // Handle bidirectional (similar to LSTM)
            if (bidirectional_) {
                // ... reverse direction processing (similar to LSTM)
            }

            h_full(layer, af::span, af::span) = h;
            h_n_ = AfToTensor(h_full);

            if (layer < num_layers_ - 1 && dropout_ > 0.0f && training_) {
                af::array mask = (af::randu(layer_output.dims()) > dropout_).as(af::dtype::f32);
                layer_output = layer_output * mask / (1.0f - dropout_);
            }

            layer_input = layer_output;
        }

        output = layer_input;

        if (batch_first_) {
            output = af::reorder(output, 1, 0, 2);
        }

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire GRULayer::Forward failed: {}, falling back to CPU", e.what());
    }
#endif

    // CPU fallback implementation
    const auto& shape = input.Shape();
    size_t batch_size, seq_len, input_dim;

    if (batch_first_) {
        batch_size = shape[0];
        seq_len = shape[1];
        input_dim = shape[2];
    } else {
        seq_len = shape[0];
        batch_size = shape[1];
        input_dim = shape[2];
    }

    int num_directions = bidirectional_ ? 2 : 1;

    // Reinitialize weights with CPU if they have null data (ArrayFire init failed)
    if (W_ih_.empty() || W_ih_[0].Data<float>() == nullptr) {
        for (int layer = 0; layer < num_layers_; layer++) {
            size_t layer_input_size = (layer == 0) ? input_dim : static_cast<size_t>(hidden_size_ * num_directions);
            size_t gate_size = static_cast<size_t>(3 * hidden_size_);
            W_ih_[layer] = Tensor::Random({gate_size, layer_input_size});
            W_hh_[layer] = Tensor::Random({gate_size, static_cast<size_t>(hidden_size_)});
            b_ih_[layer] = Tensor::Zeros({gate_size});
            b_hh_[layer] = Tensor::Zeros({gate_size});
        }
    }

    if (h_n_.NumElements() == 0 || h_n_.Data<float>() == nullptr) {
        h_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                               batch_size, static_cast<size_t>(hidden_size_)});
    }

    size_t out_dim0 = batch_first_ ? batch_size : seq_len;
    size_t out_dim1 = batch_first_ ? seq_len : batch_size;
    size_t out_features = static_cast<size_t>(hidden_size_ * num_directions);
    Tensor output = Tensor::Zeros({out_dim0, out_dim1, out_features});

    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();
    float* h_data = h_n_.Data<float>();

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto tanh_f = [](float x) { return std::tanh(x); };

    Tensor layer_input = input;
    size_t layer_input_size = input_dim;

    for (int layer = 0; layer < num_layers_; layer++) {
        const float* W_ih = W_ih_[layer].Data<float>();
        const float* W_hh = W_hh_[layer].Data<float>();
        const float* b_ih = b_ih_[layer].Data<float>();
        const float* b_hh = b_hh_[layer].Data<float>();
        int gate_size = 3 * hidden_size_;

        Tensor layer_output = Tensor::Zeros({seq_len, batch_size, static_cast<size_t>(hidden_size_)});
        float* layer_out = layer_output.Data<float>();
        const float* layer_in = layer_input.Data<float>();

        for (size_t b = 0; b < batch_size; b++) {
            std::vector<float> h(hidden_size_);
            for (int i = 0; i < hidden_size_; i++) {
                h[i] = h_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i];
            }

            for (size_t t = 0; t < seq_len; t++) {
                std::vector<float> gates(gate_size, 0.0f);
                const float* x_ptr;
                if (layer == 0) {
                    if (batch_first_) x_ptr = input_data + b * seq_len * input_dim + t * input_dim;
                    else x_ptr = input_data + t * batch_size * input_dim + b * input_dim;
                } else {
                    x_ptr = layer_in + t * batch_size * layer_input_size + b * layer_input_size;
                }

                // Compute input projections
                for (int g = 0; g < gate_size; g++) {
                    gates[g] = b_ih[g];
                    for (size_t k = 0; k < layer_input_size; k++)
                        gates[g] += W_ih[g * layer_input_size + k] * x_ptr[k];
                }

                std::vector<float> r_gate(hidden_size_), z_gate(hidden_size_), n_gate(hidden_size_);
                for (int i = 0; i < hidden_size_; i++) {
                    float r_input = gates[i];
                    float r_hidden = b_hh[i];
                    for (int k = 0; k < hidden_size_; k++)
                        r_hidden += W_hh[i * hidden_size_ + k] * h[k];
                    r_gate[i] = sigmoid(r_input + r_hidden);

                    float z_input = gates[hidden_size_ + i];
                    float z_hidden = b_hh[hidden_size_ + i];
                    for (int k = 0; k < hidden_size_; k++)
                        z_hidden += W_hh[(hidden_size_ + i) * hidden_size_ + k] * h[k];
                    z_gate[i] = sigmoid(z_input + z_hidden);

                    float n_input = gates[2 * hidden_size_ + i];
                    float n_hidden = b_hh[2 * hidden_size_ + i];
                    for (int k = 0; k < hidden_size_; k++)
                        n_hidden += W_hh[(2 * hidden_size_ + i) * hidden_size_ + k] * h[k];
                    n_gate[i] = tanh_f(n_input + r_gate[i] * n_hidden);
                }

                for (int i = 0; i < hidden_size_; i++) {
                    h[i] = (1.0f - z_gate[i]) * n_gate[i] + z_gate[i] * h[i];
                }

                for (int i = 0; i < hidden_size_; i++)
                    layer_out[t * batch_size * hidden_size_ + b * hidden_size_ + i] = h[i];
            }

            for (int i = 0; i < hidden_size_; i++) {
                h_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i] = h[i];
            }
        }
        layer_input = layer_output;
        layer_input_size = static_cast<size_t>(hidden_size_);
    }

    const float* final_out = layer_input.Data<float>();
    for (size_t t = 0; t < seq_len; t++) {
        for (size_t b = 0; b < batch_size; b++) {
            for (int f = 0; f < hidden_size_; f++) {
                float val = final_out[t * batch_size * hidden_size_ + b * hidden_size_ + f];
                if (batch_first_) output_data[b * seq_len * out_features + t * out_features + f] = val;
                else output_data[t * batch_size * out_features + b * out_features + f] = val;
            }
        }
    }
    return output;
}

Tensor GRULayer::Backward(const Tensor& grad_output) {
    // Check if caches are populated (Forward must have been called with ArrayFire success)
    if (cached_inputs_.empty() || cached_gates_.empty() || cached_hidden_states_.empty()) {
        // CPU fallback Forward doesn't populate caches, so return zero gradients
        spdlog::warn("GRULayer::Backward called without cached data from Forward, returning zero gradients");
        return Tensor::Zeros(grad_output.Shape());
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array dout = TensorToAf(grad_output);

        if (batch_first_) {
            dout = af::reorder(dout, 1, 0, 2);
        }

        dim_t seq_len = dout.dims(0);
        dim_t batch_size = dout.dims(1);

        af::array dx;
        af::array layer_grad = dout;

        for (int layer = num_layers_ - 1; layer >= 0; layer--) {
            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);
            af::array cached_input = TensorToAf(cached_inputs_[layer]);
            af::array cached_gates = TensorToAf(cached_gates_[layer]);
            af::array cached_h = TensorToAf(cached_hidden_states_[layer]);

            dim_t layer_input_size = cached_input.dims(2);
            int gate_size = 3 * hidden_size_;

            af::array dW_ih = af::constant(0.0f, W_ih.dims());
            af::array dW_hh = af::constant(0.0f, W_hh.dims());
            af::array db_ih = af::constant(0.0f, af::dim4(gate_size));
            af::array db_hh = af::constant(0.0f, af::dim4(gate_size));

            af::array dh_next = af::constant(0.0f, af::dim4(batch_size, hidden_size_));
            af::array d_layer_input = af::constant(0.0f, cached_input.dims());

            // BPTT for GRU
            for (dim_t t = seq_len - 1; t >= 0; t--) {
                af::array h_prev = cached_h(t, af::span, af::span);
                h_prev = af::moddims(h_prev, af::dim4(batch_size, hidden_size_));

                af::array gates = cached_gates(t, af::span, af::span);
                gates = af::moddims(gates, af::dim4(batch_size, gate_size));

                af::array r_gate = gates(af::span, af::seq(0, hidden_size_ - 1));
                af::array z_gate = gates(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1));
                af::array n_gate = gates(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1));

                af::array dh = layer_grad(t, af::span, af::span);
                dh = af::moddims(dh, af::dim4(batch_size, hidden_size_));
                dh = dh + dh_next;

                // GRU backward equations
                af::array dn_gate = dh * (1.0f - z_gate) * (1.0f - n_gate * n_gate);
                af::array dz_gate = dh * (h_prev - n_gate) * z_gate * (1.0f - z_gate);
                dh_next = dh * z_gate;

                af::array dgates = af::join(1, af::constant(0.0f, af::dim4(batch_size, hidden_size_)),
                                             dz_gate, dn_gate);

                af::array x_t = cached_input(t, af::span, af::span);
                x_t = af::moddims(x_t, af::dim4(batch_size, layer_input_size));

                dW_ih = dW_ih + af::matmul(af::transpose(dgates), x_t);
                dW_hh = dW_hh + af::matmul(af::transpose(dgates), h_prev);
                db_ih = db_ih + af::sum(dgates, 0);

                af::array dx_t = af::matmul(dgates, W_ih);
                d_layer_input(t, af::span, af::span) = dx_t;
            }

            grad_W_ih_[layer] = AfToTensor(dW_ih);
            grad_W_hh_[layer] = AfToTensor(dW_hh);
            grad_b_ih_[layer] = AfToTensor(af::moddims(db_ih, af::dim4(gate_size)));
            grad_b_hh_[layer] = AfToTensor(af::moddims(db_hh, af::dim4(gate_size)));

            layer_grad = d_layer_input;
        }

        dx = layer_grad;

        if (batch_first_) {
            dx = af::reorder(dx, 1, 0, 2);
        }

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire GRULayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("GRU backward requires ArrayFire");
}

std::map<std::string, Tensor> GRULayer::GetParameters() {
    std::map<std::string, Tensor> params;
    for (int layer = 0; layer < num_layers_; layer++) {
        std::string prefix = "layer" + std::to_string(layer) + "_";
        params[prefix + "W_ih"] = W_ih_[layer];
        params[prefix + "W_hh"] = W_hh_[layer];
        params[prefix + "b_ih"] = b_ih_[layer];
        params[prefix + "b_hh"] = b_hh_[layer];
        params[prefix + "grad_W_ih"] = grad_W_ih_[layer];
        params[prefix + "grad_W_hh"] = grad_W_hh_[layer];
        params[prefix + "grad_b_ih"] = grad_b_ih_[layer];
        params[prefix + "grad_b_hh"] = grad_b_hh_[layer];
    }
    return params;
}

void GRULayer::SetParameters(const std::map<std::string, Tensor>& params) {
    for (int layer = 0; layer < num_layers_; layer++) {
        std::string prefix = "layer" + std::to_string(layer) + "_";
        if (params.count(prefix + "W_ih")) W_ih_[layer] = params.at(prefix + "W_ih");
        if (params.count(prefix + "W_hh")) W_hh_[layer] = params.at(prefix + "W_hh");
        if (params.count(prefix + "b_ih")) b_ih_[layer] = params.at(prefix + "b_ih");
        if (params.count(prefix + "b_hh")) b_hh_[layer] = params.at(prefix + "b_hh");
    }
}

// ============================================================================
// MultiHeadAttention Layer Implementation
// ============================================================================

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int embed_dim, int num_heads,
                                                   float dropout, bool use_bias)
    : embed_dim_(embed_dim), num_heads_(num_heads), dropout_(dropout), use_bias_(use_bias) {

    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }

    head_dim_ = embed_dim / num_heads;
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    InitializeWeights();
}

void MultiHeadAttentionLayer::InitializeWeights() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Xavier initialization for projection weights
        float limit = std::sqrt(6.0f / (embed_dim_ + embed_dim_));

        af::array w_q = af::randu(af::dim4(embed_dim_, embed_dim_)) * 2.0f * limit - limit;
        af::array w_k = af::randu(af::dim4(embed_dim_, embed_dim_)) * 2.0f * limit - limit;
        af::array w_v = af::randu(af::dim4(embed_dim_, embed_dim_)) * 2.0f * limit - limit;
        af::array w_o = af::randu(af::dim4(embed_dim_, embed_dim_)) * 2.0f * limit - limit;

        W_q_ = AfToTensor(w_q);
        W_k_ = AfToTensor(w_k);
        W_v_ = AfToTensor(w_v);
        W_o_ = AfToTensor(w_o);

        if (use_bias_) {
            b_q_ = Tensor({static_cast<size_t>(embed_dim_)}, DataType::Float32);
            b_k_ = Tensor({static_cast<size_t>(embed_dim_)}, DataType::Float32);
            b_v_ = Tensor({static_cast<size_t>(embed_dim_)}, DataType::Float32);
            b_o_ = Tensor({static_cast<size_t>(embed_dim_)}, DataType::Float32);
            std::memset(b_q_.Data(), 0, embed_dim_ * sizeof(float));
            std::memset(b_k_.Data(), 0, embed_dim_ * sizeof(float));
            std::memset(b_v_.Data(), 0, embed_dim_ * sizeof(float));
            std::memset(b_o_.Data(), 0, embed_dim_ * sizeof(float));
        }

        // Initialize gradient tensors
        grad_W_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
        grad_W_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
        grad_W_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
        grad_W_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});

        if (use_bias_) {
            grad_b_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            grad_b_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            grad_b_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            grad_b_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        }

        return;
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire init failed: {}, using CPU", e.what());
    }
#endif

    // CPU fallback
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (embed_dim_ + embed_dim_));
    std::uniform_real_distribution<float> dist(-limit, limit);

    W_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    W_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    W_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    W_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});

    float* wq = W_q_.Data<float>();
    float* wk = W_k_.Data<float>();
    float* wv = W_v_.Data<float>();
    float* wo = W_o_.Data<float>();

    for (int i = 0; i < embed_dim_ * embed_dim_; i++) {
        wq[i] = dist(gen);
        wk[i] = dist(gen);
        wv[i] = dist(gen);
        wo[i] = dist(gen);
    }

    if (use_bias_) {
        b_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        b_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        b_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        b_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        std::memset(b_q_.Data(), 0, embed_dim_ * sizeof(float));
        std::memset(b_k_.Data(), 0, embed_dim_ * sizeof(float));
        std::memset(b_v_.Data(), 0, embed_dim_ * sizeof(float));
        std::memset(b_o_.Data(), 0, embed_dim_ * sizeof(float));
    }

    grad_W_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    grad_W_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    grad_W_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    grad_W_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});

    if (use_bias_) {
        grad_b_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        grad_b_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        grad_b_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        grad_b_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
    }
}

Tensor MultiHeadAttentionLayer::Forward(const Tensor& input) {
    // Self-attention: Q = K = V = input
    return Forward(input, input, input, nullptr);
}

Tensor MultiHeadAttentionLayer::Forward(const Tensor& query, const Tensor& key,
                                         const Tensor& value, const Tensor* attn_mask) {
    // Cache inputs for backward
    cached_query_ = query;
    cached_key_ = key;
    cached_value_ = value;

    const auto& q_shape = query.Shape();
    if (q_shape.size() != 3) {
        throw std::invalid_argument("Input must be 3D [batch, seq_len, embed_dim]");
    }

    int batch_size = static_cast<int>(q_shape[0]);
    int seq_len_q = static_cast<int>(q_shape[1]);
    int seq_len_kv = static_cast<int>(key.Shape()[1]);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Load input tensors - [batch, seq, embed] stored as [embed, seq, batch] in AF
        af::array q_in(af::dim4(embed_dim_, seq_len_q, batch_size), query.Data<float>());
        af::array k_in(af::dim4(embed_dim_, seq_len_kv, batch_size), key.Data<float>());
        af::array v_in(af::dim4(embed_dim_, seq_len_kv, batch_size), value.Data<float>());

        // Load weights [embed_dim, embed_dim]
        af::array wq = TensorToAf(W_q_);
        af::array wk = TensorToAf(W_k_);
        af::array wv = TensorToAf(W_v_);
        af::array wo = TensorToAf(W_o_);

        // Linear projections: Q, K, V
        // For each batch, we do: proj = W @ x  (matmul along embed dimension)
        // Using gfor for batched operations
        af::array Q = af::constant(0.0f, af::dim4(embed_dim_, seq_len_q, batch_size));
        af::array K = af::constant(0.0f, af::dim4(embed_dim_, seq_len_kv, batch_size));
        af::array V = af::constant(0.0f, af::dim4(embed_dim_, seq_len_kv, batch_size));

        for (int b = 0; b < batch_size; b++) {
            af::array q_b = q_in(af::span, af::span, b);
            af::array k_b = k_in(af::span, af::span, b);
            af::array v_b = v_in(af::span, af::span, b);

            Q(af::span, af::span, b) = af::matmul(wq, q_b);
            K(af::span, af::span, b) = af::matmul(wk, k_b);
            V(af::span, af::span, b) = af::matmul(wv, v_b);
        }

        // Add bias if enabled
        if (use_bias_) {
            af::array bq = af::array(af::dim4(embed_dim_), b_q_.Data<float>());
            af::array bk = af::array(af::dim4(embed_dim_), b_k_.Data<float>());
            af::array bv = af::array(af::dim4(embed_dim_), b_v_.Data<float>());

            Q = Q + af::tile(bq, 1, seq_len_q, batch_size);
            K = K + af::tile(bk, 1, seq_len_kv, batch_size);
            V = V + af::tile(bv, 1, seq_len_kv, batch_size);
        }

        // Cache projected Q, K, V for backward
        cached_Q_ = Tensor(std::vector<size_t>{static_cast<size_t>(batch_size), static_cast<size_t>(seq_len_q),
                            static_cast<size_t>(embed_dim_)});
        cached_K_ = Tensor(std::vector<size_t>{static_cast<size_t>(batch_size), static_cast<size_t>(seq_len_kv),
                            static_cast<size_t>(embed_dim_)});
        cached_V_ = Tensor(std::vector<size_t>{static_cast<size_t>(batch_size), static_cast<size_t>(seq_len_kv),
                            static_cast<size_t>(embed_dim_)});
        Q.host(cached_Q_.Data());
        K.host(cached_K_.Data());
        V.host(cached_V_.Data());

        // Reshape for multi-head: [embed, seq, batch] -> [head_dim, seq, num_heads, batch]
        Q = af::moddims(Q, af::dim4(head_dim_, num_heads_, seq_len_q, batch_size));
        K = af::moddims(K, af::dim4(head_dim_, num_heads_, seq_len_kv, batch_size));
        V = af::moddims(V, af::dim4(head_dim_, num_heads_, seq_len_kv, batch_size));

        // Reorder to [head_dim, seq, batch, num_heads] for batch matmul
        Q = af::reorder(Q, 0, 2, 3, 1);  // [head_dim, seq_q, batch, num_heads]
        K = af::reorder(K, 0, 2, 3, 1);  // [head_dim, seq_kv, batch, num_heads]
        V = af::reorder(V, 0, 2, 3, 1);  // [head_dim, seq_kv, batch, num_heads]

        // Scaled dot-product attention
        // scores = Q^T @ K / sqrt(head_dim)  -> [seq_q, seq_kv, batch, num_heads]
        af::array scores = af::constant(0.0f, af::dim4(seq_len_q, seq_len_kv, batch_size, num_heads_));

        for (int h = 0; h < num_heads_; h++) {
            for (int b = 0; b < batch_size; b++) {
                af::array q_bh = Q(af::span, af::span, b, h);  // [head_dim, seq_q]
                af::array k_bh = K(af::span, af::span, b, h);  // [head_dim, seq_kv]

                // scores = Q^T @ K = [seq_q, head_dim] @ [head_dim, seq_kv] = [seq_q, seq_kv]
                af::array s = af::matmul(af::transpose(q_bh), k_bh) * scale_;
                scores(af::span, af::span, b, h) = s;
            }
        }

        // Apply attention mask if provided
        if (attn_mask != nullptr) {
            af::array mask(af::dim4(seq_len_q, seq_len_kv), attn_mask->Data<float>());
            scores = scores + af::tile(mask, 1, 1, batch_size, num_heads_);
        }

        // Softmax along seq_kv dimension (dim 1)
        // ArrayFire doesn't have softmax, implement manually: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        af::array max_scores = af::max(scores, 1);
        af::array scores_shifted = scores - af::tile(max_scores, 1, seq_len_kv, 1, 1);
        af::array exp_scores = af::exp(scores_shifted);
        af::array sum_exp = af::sum(exp_scores, 1);
        af::array attn_weights = exp_scores / af::tile(sum_exp, 1, seq_len_kv, 1, 1);

        // Cache attention weights for backward and visualization
        cached_attn_weights_ = Tensor(std::vector<size_t>{static_cast<size_t>(batch_size), static_cast<size_t>(num_heads_),
                                        static_cast<size_t>(seq_len_q), static_cast<size_t>(seq_len_kv)});
        af::array attn_reordered = af::reorder(attn_weights, 0, 1, 3, 2);
        attn_reordered.host(cached_attn_weights_.Data());

        // Apply dropout if training
        if (training_ && dropout_ > 0.0f) {
            af::array mask = af::randu(attn_weights.dims()) > dropout_;
            dropout_mask_ = Tensor(std::vector<size_t>{static_cast<size_t>(seq_len_q), static_cast<size_t>(seq_len_kv),
                                     static_cast<size_t>(batch_size), static_cast<size_t>(num_heads_)});
            mask.as(af::dtype::f32).host(dropout_mask_.Data());
            attn_weights = attn_weights * mask.as(af::dtype::f32) / (1.0f - dropout_);
        }

        // Weighted sum: context = attn_weights @ V
        // [seq_q, seq_kv] @ [head_dim, seq_kv]^T -> need [seq_q, head_dim]
        af::array context = af::constant(0.0f, af::dim4(head_dim_, seq_len_q, batch_size, num_heads_));

        for (int h = 0; h < num_heads_; h++) {
            for (int b = 0; b < batch_size; b++) {
                af::array a_bh = attn_weights(af::span, af::span, b, h);  // [seq_q, seq_kv]
                af::array v_bh = V(af::span, af::span, b, h);             // [head_dim, seq_kv]

                // context = V @ attn^T = [head_dim, seq_kv] @ [seq_kv, seq_q] = [head_dim, seq_q]
                af::array c = af::matmul(v_bh, af::transpose(a_bh));
                context(af::span, af::span, b, h) = c;
            }
        }

        // Reshape back: [head_dim, seq_q, batch, num_heads] -> [embed, seq_q, batch]
        context = af::reorder(context, 0, 3, 1, 2);  // [head_dim, num_heads, seq_q, batch]
        context = af::moddims(context, af::dim4(embed_dim_, seq_len_q, batch_size));

        // Cache context for backward
        cached_context_ = Tensor(std::vector<size_t>{static_cast<size_t>(batch_size), static_cast<size_t>(seq_len_q),
                                   static_cast<size_t>(embed_dim_)});
        context.host(cached_context_.Data());

        // Output projection
        af::array output = af::constant(0.0f, af::dim4(embed_dim_, seq_len_q, batch_size));
        for (int b = 0; b < batch_size; b++) {
            output(af::span, af::span, b) = af::matmul(wo, context(af::span, af::span, b));
        }

        if (use_bias_) {
            af::array bo = af::array(af::dim4(embed_dim_), b_o_.Data<float>());
            output = output + af::tile(bo, 1, seq_len_q, batch_size);
        }

        // Convert to output tensor [batch, seq, embed]
        std::vector<size_t> result_shape = {static_cast<size_t>(batch_size), static_cast<size_t>(seq_len_q),
                       static_cast<size_t>(embed_dim_)};
        Tensor result(result_shape, DataType::Float32);
        output.host(result.Data());
        return result;

    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire MultiHeadAttention forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("MultiHeadAttention forward requires ArrayFire");
}

Tensor MultiHeadAttentionLayer::Backward(const Tensor& grad_output) {
    const auto& shape = grad_output.Shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len_q = static_cast<int>(shape[1]);
    int seq_len_kv = static_cast<int>(cached_key_.Shape()[1]);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Load gradients [batch, seq, embed] -> [embed, seq, batch]
        af::array grad_out(af::dim4(embed_dim_, seq_len_q, batch_size), grad_output.Data<float>());

        // Load weights
        af::array wq = TensorToAf(W_q_);
        af::array wk = TensorToAf(W_k_);
        af::array wv = TensorToAf(W_v_);
        af::array wo = TensorToAf(W_o_);

        // Load cached values
        af::array context(af::dim4(embed_dim_, seq_len_q, batch_size), cached_context_.Data<float>());
        af::array Q(af::dim4(embed_dim_, seq_len_q, batch_size), cached_Q_.Data<float>());
        af::array K(af::dim4(embed_dim_, seq_len_kv, batch_size), cached_K_.Data<float>());
        af::array V(af::dim4(embed_dim_, seq_len_kv, batch_size), cached_V_.Data<float>());

        // Gradient through output projection
        af::array grad_context = af::constant(0.0f, af::dim4(embed_dim_, seq_len_q, batch_size));
        af::array dWo = af::constant(0.0f, wo.dims());
        af::array dbo = af::constant(0.0f, af::dim4(embed_dim_));

        for (int b = 0; b < batch_size; b++) {
            af::array grad_b = grad_out(af::span, af::span, b);
            af::array ctx_b = context(af::span, af::span, b);

            grad_context(af::span, af::span, b) = af::matmul(af::transpose(wo), grad_b);
            dWo = dWo + af::matmul(grad_b, af::transpose(ctx_b));
        }

        if (use_bias_) {
            dbo = af::sum(af::sum(grad_out, 1), 2);
        }

        // Reshape for multi-head attention backward
        Q = af::moddims(Q, af::dim4(head_dim_, num_heads_, seq_len_q, batch_size));
        K = af::moddims(K, af::dim4(head_dim_, num_heads_, seq_len_kv, batch_size));
        V = af::moddims(V, af::dim4(head_dim_, num_heads_, seq_len_kv, batch_size));
        grad_context = af::moddims(grad_context, af::dim4(head_dim_, num_heads_, seq_len_q, batch_size));

        Q = af::reorder(Q, 0, 2, 3, 1);
        K = af::reorder(K, 0, 2, 3, 1);
        V = af::reorder(V, 0, 2, 3, 1);
        grad_context = af::reorder(grad_context, 0, 2, 3, 1);

        // Load cached attention weights
        af::array attn_weights(af::dim4(seq_len_q, seq_len_kv, batch_size, num_heads_),
                               cached_attn_weights_.Data<float>());
        attn_weights = af::reorder(attn_weights, 0, 1, 3, 2);  // Reorder to match

        // Gradient through attention
        af::array grad_Q = af::constant(0.0f, Q.dims());
        af::array grad_K = af::constant(0.0f, K.dims());
        af::array grad_V = af::constant(0.0f, V.dims());

        for (int h = 0; h < num_heads_; h++) {
            for (int b = 0; b < batch_size; b++) {
                af::array v_bh = V(af::span, af::span, b, h);
                af::array q_bh = Q(af::span, af::span, b, h);
                af::array k_bh = K(af::span, af::span, b, h);
                af::array a_bh = attn_weights(af::span, af::span, b, h);
                af::array gc_bh = grad_context(af::span, af::span, b, h);

                // Gradient w.r.t. V: dV = attn @ grad_context^T
                af::array dV = af::matmul(gc_bh, a_bh);
                grad_V(af::span, af::span, b, h) = dV;

                // Gradient w.r.t. attention weights
                af::array grad_attn = af::matmul(v_bh, af::transpose(gc_bh));
                grad_attn = af::transpose(grad_attn);  // [seq_q, seq_kv]

                // Softmax backward: d_scores = attn * (d_attn - sum(d_attn * attn))
                af::array sum_grad = af::sum(grad_attn * a_bh, 1);
                af::array grad_scores = a_bh * (grad_attn - af::tile(sum_grad, 1, seq_len_kv));

                // Scale
                grad_scores = grad_scores * scale_;

                // Gradient w.r.t. Q and K from scores = Q^T @ K
                // dQ = K @ grad_scores^T, dK = Q @ grad_scores
                af::array dQ = af::matmul(k_bh, af::transpose(grad_scores));
                af::array dK = af::matmul(q_bh, grad_scores);

                grad_Q(af::span, af::span, b, h) = dQ;
                grad_K(af::span, af::span, b, h) = dK;
            }
        }

        // Reshape gradients back
        grad_Q = af::reorder(grad_Q, 0, 3, 1, 2);
        grad_K = af::reorder(grad_K, 0, 3, 1, 2);
        grad_V = af::reorder(grad_V, 0, 3, 1, 2);

        grad_Q = af::moddims(grad_Q, af::dim4(embed_dim_, seq_len_q, batch_size));
        grad_K = af::moddims(grad_K, af::dim4(embed_dim_, seq_len_kv, batch_size));
        grad_V = af::moddims(grad_V, af::dim4(embed_dim_, seq_len_kv, batch_size));

        // Gradient through input projections
        af::array query(af::dim4(embed_dim_, seq_len_q, batch_size), cached_query_.Data<float>());
        af::array key(af::dim4(embed_dim_, seq_len_kv, batch_size), cached_key_.Data<float>());
        af::array value(af::dim4(embed_dim_, seq_len_kv, batch_size), cached_value_.Data<float>());

        af::array dWq = af::constant(0.0f, wq.dims());
        af::array dWk = af::constant(0.0f, wk.dims());
        af::array dWv = af::constant(0.0f, wv.dims());
        af::array dbq = af::constant(0.0f, af::dim4(embed_dim_));
        af::array dbk = af::constant(0.0f, af::dim4(embed_dim_));
        af::array dbv = af::constant(0.0f, af::dim4(embed_dim_));

        af::array grad_query = af::constant(0.0f, query.dims());
        af::array grad_key = af::constant(0.0f, key.dims());
        af::array grad_value = af::constant(0.0f, value.dims());

        for (int b = 0; b < batch_size; b++) {
            af::array gq_b = grad_Q(af::span, af::span, b);
            af::array gk_b = grad_K(af::span, af::span, b);
            af::array gv_b = grad_V(af::span, af::span, b);
            af::array q_b = query(af::span, af::span, b);
            af::array k_b = key(af::span, af::span, b);
            af::array v_b = value(af::span, af::span, b);

            dWq = dWq + af::matmul(gq_b, af::transpose(q_b));
            dWk = dWk + af::matmul(gk_b, af::transpose(k_b));
            dWv = dWv + af::matmul(gv_b, af::transpose(v_b));

            grad_query(af::span, af::span, b) = af::matmul(af::transpose(wq), gq_b);
            grad_key(af::span, af::span, b) = af::matmul(af::transpose(wk), gk_b);
            grad_value(af::span, af::span, b) = af::matmul(af::transpose(wv), gv_b);
        }

        if (use_bias_) {
            dbq = af::sum(af::sum(grad_Q, 1), 2);
            dbk = af::sum(af::sum(grad_K, 1), 2);
            dbv = af::sum(af::sum(grad_V, 1), 2);
        }

        // Store gradients
        grad_W_q_ = AfToTensor(dWq);
        grad_W_k_ = AfToTensor(dWk);
        grad_W_v_ = AfToTensor(dWv);
        grad_W_o_ = AfToTensor(dWo);

        if (use_bias_) {
            grad_b_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            grad_b_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            grad_b_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            grad_b_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            dbq.host(grad_b_q_.Data());
            dbk.host(grad_b_k_.Data());
            dbv.host(grad_b_v_.Data());
            dbo.host(grad_b_o_.Data());
        }

        // Return gradient w.r.t. query (for self-attention, this is the input gradient)
        // For cross-attention, caller needs to handle key/value gradients separately
        std::vector<size_t> result_shape = {static_cast<size_t>(batch_size), static_cast<size_t>(seq_len_q),
                       static_cast<size_t>(embed_dim_)};
        Tensor result(result_shape, DataType::Float32);
        grad_query.host(result.Data());
        return result;

    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire MultiHeadAttention backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("MultiHeadAttention backward requires ArrayFire");
}

std::map<std::string, Tensor> MultiHeadAttentionLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["W_q"] = W_q_;
    params["W_k"] = W_k_;
    params["W_v"] = W_v_;
    params["W_o"] = W_o_;
    params["grad_W_q"] = grad_W_q_;
    params["grad_W_k"] = grad_W_k_;
    params["grad_W_v"] = grad_W_v_;
    params["grad_W_o"] = grad_W_o_;

    if (use_bias_) {
        params["b_q"] = b_q_;
        params["b_k"] = b_k_;
        params["b_v"] = b_v_;
        params["b_o"] = b_o_;
        params["grad_b_q"] = grad_b_q_;
        params["grad_b_k"] = grad_b_k_;
        params["grad_b_v"] = grad_b_v_;
        params["grad_b_o"] = grad_b_o_;
    }

    return params;
}

void MultiHeadAttentionLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("W_q")) W_q_ = params.at("W_q");
    if (params.count("W_k")) W_k_ = params.at("W_k");
    if (params.count("W_v")) W_v_ = params.at("W_v");
    if (params.count("W_o")) W_o_ = params.at("W_o");

    if (use_bias_) {
        if (params.count("b_q")) b_q_ = params.at("b_q");
        if (params.count("b_k")) b_k_ = params.at("b_k");
        if (params.count("b_v")) b_v_ = params.at("b_v");
        if (params.count("b_o")) b_o_ = params.at("b_o");
    }
}

// ============================================================================
// TransformerEncoderLayer Implementation
// ============================================================================

TransformerEncoderLayer::TransformerEncoderLayer(int d_model, int nhead,
                                                   int dim_feedforward, float dropout,
                                                   bool norm_first)
    : d_model_(d_model), nhead_(nhead), dim_feedforward_(dim_feedforward),
      dropout_(dropout), norm_first_(norm_first) {

    self_attn_ = std::make_unique<MultiHeadAttentionLayer>(d_model, nhead, dropout);
    norm1_ = std::make_unique<LayerNormLayer>(std::vector<int>{d_model});
    norm2_ = std::make_unique<LayerNormLayer>(std::vector<int>{d_model});
    linear1_ = std::make_unique<DenseLayer>(d_model, dim_feedforward);
    linear2_ = std::make_unique<DenseLayer>(dim_feedforward, d_model);
    dropout1_ = std::make_unique<DropoutLayer>(dropout);
    dropout2_ = std::make_unique<DropoutLayer>(dropout);
}

Tensor TransformerEncoderLayer::Forward(const Tensor& input) {
    return Forward(input, nullptr);
}

Tensor TransformerEncoderLayer::Forward(const Tensor& input, const Tensor* src_mask) {
    cached_input_ = input;

    if (norm_first_) {
        // Pre-LN: x + attn(norm(x))
        Tensor normed = norm1_->Forward(input);
        Tensor attn_out = self_attn_->Forward(normed, normed, normed, src_mask);
        attn_out = dropout1_->Forward(attn_out);
        cached_residual1_ = input;

        // Add residual
        const auto& shape = input.Shape();
        std::vector<size_t> tensor_shape = {shape[0], shape[1], shape[2]};
        Tensor x(tensor_shape, DataType::Float32);
        const float* in_data = input.Data<float>();
        const float* attn_data = attn_out.Data<float>();
        float* out_data = x.Data<float>();
        size_t total = shape[0] * shape[1] * shape[2];
        for (size_t i = 0; i < total; i++) {
            out_data[i] = in_data[i] + attn_data[i];
        }
        cached_attn_output_ = x;

        // FFN
        Tensor normed2 = norm2_->Forward(x);
        Tensor ffn_out = linear1_->Forward(normed2);

        // ReLU activation
        float* ffn_data = ffn_out.Data<float>();
        size_t ffn_total = ffn_out.NumElements();
        for (size_t i = 0; i < ffn_total; i++) {
            ffn_data[i] = std::max(0.0f, ffn_data[i]);
        }
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = dropout2_->Forward(ffn_out);
        cached_residual2_ = x;

        // Add residual
        Tensor result(tensor_shape, DataType::Float32);
        const float* x_data = x.Data<float>();
        const float* ffn_out_data = ffn_out.Data<float>();
        float* result_data = result.Data<float>();
        for (size_t i = 0; i < total; i++) {
            result_data[i] = x_data[i] + ffn_out_data[i];
        }

        return result;
    } else {
        // Post-LN: norm(x + attn(x))
        Tensor attn_out = self_attn_->Forward(input, input, input, src_mask);
        attn_out = dropout1_->Forward(attn_out);
        cached_residual1_ = input;

        // Add residual and norm
        const auto& shape = input.Shape();
        std::vector<size_t> tensor_shape = {shape[0], shape[1], shape[2]};
        Tensor x(tensor_shape, DataType::Float32);
        const float* in_data = input.Data<float>();
        const float* attn_data = attn_out.Data<float>();
        float* out_data = x.Data<float>();
        size_t total = shape[0] * shape[1] * shape[2];
        for (size_t i = 0; i < total; i++) {
            out_data[i] = in_data[i] + attn_data[i];
        }

        x = norm1_->Forward(x);
        cached_attn_output_ = x;

        // FFN
        Tensor ffn_out = linear1_->Forward(x);

        // ReLU activation
        float* ffn_data = ffn_out.Data<float>();
        size_t ffn_total = ffn_out.NumElements();
        for (size_t i = 0; i < ffn_total; i++) {
            ffn_data[i] = std::max(0.0f, ffn_data[i]);
        }
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = dropout2_->Forward(ffn_out);
        cached_residual2_ = x;

        // Add residual and norm
        Tensor result(tensor_shape, DataType::Float32);
        const float* x_data = x.Data<float>();
        const float* ffn_out_data = ffn_out.Data<float>();
        float* result_data = result.Data<float>();
        for (size_t i = 0; i < total; i++) {
            result_data[i] = x_data[i] + ffn_out_data[i];
        }

        return norm2_->Forward(result);
    }
}

Tensor TransformerEncoderLayer::Backward(const Tensor& grad_output) {
    // Simplified backward - full implementation would track all intermediate gradients
    Tensor grad = grad_output;

    if (!norm_first_) {
        grad = norm2_->Backward(grad);
    }

    // FFN backward
    Tensor grad_ffn = dropout2_->Backward(grad);
    grad_ffn = linear2_->Backward(grad_ffn);

    // ReLU backward
    const float* mid_data = cached_ffn_mid_.Data<float>();
    float* grad_ffn_data = grad_ffn.Data<float>();
    size_t total = grad_ffn.NumElements();
    for (size_t i = 0; i < total; i++) {
        if (mid_data[i] <= 0.0f) {
            grad_ffn_data[i] = 0.0f;
        }
    }

    grad_ffn = linear1_->Backward(grad_ffn);

    if (norm_first_) {
        grad_ffn = norm2_->Backward(grad_ffn);
    }

    // Add residual gradient
    const auto& shape = grad_output.Shape();
    std::vector<size_t> tensor_shape = {shape[0], shape[1], shape[2]};
    Tensor grad_sum(tensor_shape, DataType::Float32);
    const float* grad_data = grad.Data<float>();
    const float* grad_ffn_ptr = grad_ffn.Data<float>();
    float* sum_data = grad_sum.Data<float>();
    size_t n = shape[0] * shape[1] * shape[2];
    for (size_t i = 0; i < n; i++) {
        sum_data[i] = grad_data[i] + grad_ffn_ptr[i];
    }

    // Attention backward
    Tensor grad_attn = dropout1_->Backward(grad_sum);
    grad_attn = self_attn_->Backward(grad_attn);

    if (norm_first_) {
        grad_attn = norm1_->Backward(grad_attn);
    }

    // Add residual gradient
    Tensor result(tensor_shape, DataType::Float32);
    const float* sum_ptr = grad_sum.Data<float>();
    const float* attn_ptr = grad_attn.Data<float>();
    float* result_data = result.Data<float>();
    for (size_t i = 0; i < n; i++) {
        result_data[i] = sum_ptr[i] + attn_ptr[i];
    }

    if (!norm_first_) {
        result = norm1_->Backward(result);
    }

    return result;
}

std::map<std::string, Tensor> TransformerEncoderLayer::GetParameters() {
    std::map<std::string, Tensor> params;

    auto attn_params = self_attn_->GetParameters();
    for (const auto& [key, val] : attn_params) {
        params["self_attn." + key] = val;
    }

    auto norm1_params = norm1_->GetParameters();
    for (const auto& [key, val] : norm1_params) {
        params["norm1." + key] = val;
    }

    auto norm2_params = norm2_->GetParameters();
    for (const auto& [key, val] : norm2_params) {
        params["norm2." + key] = val;
    }

    auto linear1_params = linear1_->GetParameters();
    for (const auto& [key, val] : linear1_params) {
        params["linear1." + key] = val;
    }

    auto linear2_params = linear2_->GetParameters();
    for (const auto& [key, val] : linear2_params) {
        params["linear2." + key] = val;
    }

    return params;
}

void TransformerEncoderLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    std::map<std::string, Tensor> attn_params, norm1_params, norm2_params;
    std::map<std::string, Tensor> linear1_params, linear2_params;

    for (const auto& [key, val] : params) {
        if (key.find("self_attn.") == 0) {
            attn_params[key.substr(10)] = val;
        } else if (key.find("norm1.") == 0) {
            norm1_params[key.substr(6)] = val;
        } else if (key.find("norm2.") == 0) {
            norm2_params[key.substr(6)] = val;
        } else if (key.find("linear1.") == 0) {
            linear1_params[key.substr(8)] = val;
        } else if (key.find("linear2.") == 0) {
            linear2_params[key.substr(8)] = val;
        }
    }

    self_attn_->SetParameters(attn_params);
    norm1_->SetParameters(norm1_params);
    norm2_->SetParameters(norm2_params);
    linear1_->SetParameters(linear1_params);
    linear2_->SetParameters(linear2_params);
}

void TransformerEncoderLayer::SetTraining(bool training) {
    training_ = training;
    self_attn_->SetTraining(training);
    norm1_->SetTraining(training);
    norm2_->SetTraining(training);
    linear1_->SetTraining(training);
    linear2_->SetTraining(training);
    dropout1_->SetTraining(training);
    dropout2_->SetTraining(training);
}

// ============================================================================
// TransformerDecoderLayer Implementation
// ============================================================================

TransformerDecoderLayer::TransformerDecoderLayer(int d_model, int nhead,
                                                   int dim_feedforward, float dropout,
                                                   bool norm_first)
    : d_model_(d_model), nhead_(nhead), dim_feedforward_(dim_feedforward),
      dropout_(dropout), norm_first_(norm_first) {

    self_attn_ = std::make_unique<MultiHeadAttentionLayer>(d_model, nhead, dropout);
    cross_attn_ = std::make_unique<MultiHeadAttentionLayer>(d_model, nhead, dropout);
    norm1_ = std::make_unique<LayerNormLayer>(std::vector<int>{d_model});
    norm2_ = std::make_unique<LayerNormLayer>(std::vector<int>{d_model});
    norm3_ = std::make_unique<LayerNormLayer>(std::vector<int>{d_model});
    linear1_ = std::make_unique<DenseLayer>(d_model, dim_feedforward);
    linear2_ = std::make_unique<DenseLayer>(dim_feedforward, d_model);
    dropout1_ = std::make_unique<DropoutLayer>(dropout);
    dropout2_ = std::make_unique<DropoutLayer>(dropout);
    dropout3_ = std::make_unique<DropoutLayer>(dropout);
}

Tensor TransformerDecoderLayer::Forward(const Tensor& input) {
    // Self-attention only mode (no encoder memory)
    return Forward(input, input, nullptr, nullptr);
}

Tensor TransformerDecoderLayer::Forward(const Tensor& tgt, const Tensor& memory,
                                         const Tensor* tgt_mask, const Tensor* memory_mask) {
    cached_input_ = tgt;
    cached_memory_ = memory;

    const auto& shape = tgt.Shape();
    size_t total = shape[0] * shape[1] * shape[2];
    std::vector<size_t> tensor_shape = {shape[0], shape[1], shape[2]};

    if (norm_first_) {
        // Pre-LN decoder

        // Self-attention
        Tensor normed = norm1_->Forward(tgt);
        Tensor self_attn_out = self_attn_->Forward(normed, normed, normed, tgt_mask);
        self_attn_out = dropout1_->Forward(self_attn_out);

        // Residual
        Tensor x(tensor_shape, DataType::Float32);
        const float* tgt_data = tgt.Data<float>();
        const float* sa_data = self_attn_out.Data<float>();
        float* x_data = x.Data<float>();
        for (size_t i = 0; i < total; i++) {
            x_data[i] = tgt_data[i] + sa_data[i];
        }
        cached_self_attn_output_ = x;

        // Cross-attention
        Tensor normed2 = norm2_->Forward(x);
        Tensor cross_attn_out = cross_attn_->Forward(normed2, memory, memory, memory_mask);
        cross_attn_out = dropout2_->Forward(cross_attn_out);

        // Residual
        Tensor x2(tensor_shape, DataType::Float32);
        const float* x_ptr = x.Data<float>();
        const float* ca_data = cross_attn_out.Data<float>();
        float* x2_data = x2.Data<float>();
        for (size_t i = 0; i < total; i++) {
            x2_data[i] = x_ptr[i] + ca_data[i];
        }
        cached_cross_attn_output_ = x2;

        // FFN
        Tensor normed3 = norm3_->Forward(x2);
        Tensor ffn_out = linear1_->Forward(normed3);

        // ReLU
        float* ffn_data = ffn_out.Data<float>();
        size_t ffn_total = ffn_out.NumElements();
        for (size_t i = 0; i < ffn_total; i++) {
            ffn_data[i] = std::max(0.0f, ffn_data[i]);
        }
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = dropout3_->Forward(ffn_out);

        // Residual
        Tensor result(tensor_shape, DataType::Float32);
        const float* x2_ptr = x2.Data<float>();
        const float* ffn_ptr = ffn_out.Data<float>();
        float* result_data = result.Data<float>();
        for (size_t i = 0; i < total; i++) {
            result_data[i] = x2_ptr[i] + ffn_ptr[i];
        }

        return result;
    } else {
        // Post-LN decoder

        // Self-attention
        Tensor self_attn_out = self_attn_->Forward(tgt, tgt, tgt, tgt_mask);
        self_attn_out = dropout1_->Forward(self_attn_out);

        // Residual + norm
        Tensor x(tensor_shape, DataType::Float32);
        const float* tgt_data = tgt.Data<float>();
        const float* sa_data = self_attn_out.Data<float>();
        float* x_data = x.Data<float>();
        for (size_t i = 0; i < total; i++) {
            x_data[i] = tgt_data[i] + sa_data[i];
        }
        x = norm1_->Forward(x);
        cached_self_attn_output_ = x;

        // Cross-attention
        Tensor cross_attn_out = cross_attn_->Forward(x, memory, memory, memory_mask);
        cross_attn_out = dropout2_->Forward(cross_attn_out);

        // Residual + norm
        Tensor x2(tensor_shape, DataType::Float32);
        const float* x_ptr = x.Data<float>();
        const float* ca_data = cross_attn_out.Data<float>();
        float* x2_data = x2.Data<float>();
        for (size_t i = 0; i < total; i++) {
            x2_data[i] = x_ptr[i] + ca_data[i];
        }
        x2 = norm2_->Forward(x2);
        cached_cross_attn_output_ = x2;

        // FFN
        Tensor ffn_out = linear1_->Forward(x2);

        // ReLU
        float* ffn_data = ffn_out.Data<float>();
        size_t ffn_total = ffn_out.NumElements();
        for (size_t i = 0; i < ffn_total; i++) {
            ffn_data[i] = std::max(0.0f, ffn_data[i]);
        }
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = dropout3_->Forward(ffn_out);

        // Residual + norm
        Tensor result(tensor_shape, DataType::Float32);
        const float* x2_ptr = x2.Data<float>();
        const float* ffn_ptr = ffn_out.Data<float>();
        float* result_data = result.Data<float>();
        for (size_t i = 0; i < total; i++) {
            result_data[i] = x2_ptr[i] + ffn_ptr[i];
        }

        return norm3_->Forward(result);
    }
}

Tensor TransformerDecoderLayer::Backward(const Tensor& grad_output) {
    // Simplified backward - similar to encoder
    Tensor grad = grad_output;

    if (!norm_first_) {
        grad = norm3_->Backward(grad);
    }

    // FFN backward
    Tensor grad_ffn = dropout3_->Backward(grad);
    grad_ffn = linear2_->Backward(grad_ffn);

    // ReLU backward
    const float* mid_data = cached_ffn_mid_.Data<float>();
    float* grad_ffn_data = grad_ffn.Data<float>();
    size_t total = grad_ffn.NumElements();
    for (size_t i = 0; i < total; i++) {
        if (mid_data[i] <= 0.0f) {
            grad_ffn_data[i] = 0.0f;
        }
    }

    grad_ffn = linear1_->Backward(grad_ffn);

    if (norm_first_) {
        grad_ffn = norm3_->Backward(grad_ffn);
    }

    // Add residual gradient
    const auto& shape = grad_output.Shape();
    size_t n = shape[0] * shape[1] * shape[2];
    std::vector<size_t> tensor_shape = {shape[0], shape[1], shape[2]};
    Tensor grad_sum(tensor_shape, DataType::Float32);
    const float* grad_data = grad.Data<float>();
    const float* grad_ffn_ptr = grad_ffn.Data<float>();
    float* sum_data = grad_sum.Data<float>();
    for (size_t i = 0; i < n; i++) {
        sum_data[i] = grad_data[i] + grad_ffn_ptr[i];
    }

    // Cross-attention backward
    Tensor grad_cross = dropout2_->Backward(grad_sum);
    grad_cross = cross_attn_->Backward(grad_cross);

    if (norm_first_) {
        grad_cross = norm2_->Backward(grad_cross);
    } else {
        grad_sum = norm2_->Backward(grad_sum);
    }

    // Add residual
    Tensor grad_sum2(tensor_shape, DataType::Float32);
    const float* sum_ptr = grad_sum.Data<float>();
    const float* cross_ptr = grad_cross.Data<float>();
    float* sum2_data = grad_sum2.Data<float>();
    for (size_t i = 0; i < n; i++) {
        sum2_data[i] = sum_ptr[i] + cross_ptr[i];
    }

    // Self-attention backward
    Tensor grad_self = dropout1_->Backward(grad_sum2);
    grad_self = self_attn_->Backward(grad_self);

    if (norm_first_) {
        grad_self = norm1_->Backward(grad_self);
    }

    // Add residual
    std::vector<size_t> result_shape = {shape[0], shape[1], shape[2]};
    Tensor result(result_shape, DataType::Float32);
    const float* sum2_ptr = grad_sum2.Data<float>();
    const float* self_ptr = grad_self.Data<float>();
    float* result_data = result.Data<float>();
    for (size_t i = 0; i < n; i++) {
        result_data[i] = sum2_ptr[i] + self_ptr[i];
    }

    if (!norm_first_) {
        result = norm1_->Backward(result);
    }

    return result;
}

std::map<std::string, Tensor> TransformerDecoderLayer::GetParameters() {
    std::map<std::string, Tensor> params;

    auto self_attn_params = self_attn_->GetParameters();
    for (const auto& [key, val] : self_attn_params) {
        params["self_attn." + key] = val;
    }

    auto cross_attn_params = cross_attn_->GetParameters();
    for (const auto& [key, val] : cross_attn_params) {
        params["cross_attn." + key] = val;
    }

    auto norm1_params = norm1_->GetParameters();
    for (const auto& [key, val] : norm1_params) {
        params["norm1." + key] = val;
    }

    auto norm2_params = norm2_->GetParameters();
    for (const auto& [key, val] : norm2_params) {
        params["norm2." + key] = val;
    }

    auto norm3_params = norm3_->GetParameters();
    for (const auto& [key, val] : norm3_params) {
        params["norm3." + key] = val;
    }

    auto linear1_params = linear1_->GetParameters();
    for (const auto& [key, val] : linear1_params) {
        params["linear1." + key] = val;
    }

    auto linear2_params = linear2_->GetParameters();
    for (const auto& [key, val] : linear2_params) {
        params["linear2." + key] = val;
    }

    return params;
}

void TransformerDecoderLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    std::map<std::string, Tensor> self_attn_params, cross_attn_params;
    std::map<std::string, Tensor> norm1_params, norm2_params, norm3_params;
    std::map<std::string, Tensor> linear1_params, linear2_params;

    for (const auto& [key, val] : params) {
        if (key.find("self_attn.") == 0) {
            self_attn_params[key.substr(10)] = val;
        } else if (key.find("cross_attn.") == 0) {
            cross_attn_params[key.substr(11)] = val;
        } else if (key.find("norm1.") == 0) {
            norm1_params[key.substr(6)] = val;
        } else if (key.find("norm2.") == 0) {
            norm2_params[key.substr(6)] = val;
        } else if (key.find("norm3.") == 0) {
            norm3_params[key.substr(6)] = val;
        } else if (key.find("linear1.") == 0) {
            linear1_params[key.substr(8)] = val;
        } else if (key.find("linear2.") == 0) {
            linear2_params[key.substr(8)] = val;
        }
    }

    self_attn_->SetParameters(self_attn_params);
    cross_attn_->SetParameters(cross_attn_params);
    norm1_->SetParameters(norm1_params);
    norm2_->SetParameters(norm2_params);
    norm3_->SetParameters(norm3_params);
    linear1_->SetParameters(linear1_params);
    linear2_->SetParameters(linear2_params);
}

void TransformerDecoderLayer::SetTraining(bool training) {
    training_ = training;
    self_attn_->SetTraining(training);
    cross_attn_->SetTraining(training);
    norm1_->SetTraining(training);
    norm2_->SetTraining(training);
    norm3_->SetTraining(training);
    linear1_->SetTraining(training);
    linear2_->SetTraining(training);
    dropout1_->SetTraining(training);
    dropout2_->SetTraining(training);
    dropout3_->SetTraining(training);
}

Tensor TransformerDecoderLayer::GenerateCausalMask(int size) {
    // Create upper triangular mask with -inf above diagonal
    std::vector<size_t> shape = {static_cast<size_t>(size), static_cast<size_t>(size)};
    Tensor mask(shape, DataType::Float32);
    float* data = mask.Data<float>();

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (j > i) {
                data[i * size + j] = -1e9f;  // Large negative for softmax
            } else {
                data[i * size + j] = 0.0f;
            }
        }
    }

    return mask;
}

} // namespace cyxwiz
