#include "cyxwiz/layer.h"
#include "cyxwiz/debug_hooks.h"
#include "cyxwiz/tensor.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <random>
#include <atomic>
#include <limits>
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

namespace {

struct LSTMDirectionForwardResult {
    Tensor output;
    Tensor input_cache;
    Tensor gate_cache;
    Tensor hidden_cache;
    Tensor cell_cache;
    Tensor final_hidden;
    Tensor final_cell;
};

struct LSTMDirectionBackwardResult {
    Tensor input_grad;
    Tensor grad_W_ih;
    Tensor grad_W_hh;
    Tensor grad_b_ih;
    Tensor grad_b_hh;
};

#ifdef CYXWIZ_HAS_ARRAYFIRE
struct LSTMAfDirectionResult {
    af::array output;
    Tensor input_cache;
    Tensor gate_cache;
    Tensor hidden_cache;
    Tensor cell_cache;
    Tensor final_hidden;
    Tensor final_cell;
};
#endif

LSTMDirectionForwardResult RunLSTMCpuDirectionForward(
    const Tensor& layer_input,
    const Tensor& W_ih,
    const Tensor& W_hh,
    const Tensor& b_ih,
    const Tensor& b_hh,
    const Tensor& init_hidden,
    const Tensor& init_cell,
    int hidden_size,
    bool batch_first,
    bool reverse_time) {

    const auto& shape = layer_input.Shape();
    size_t batch_size = batch_first ? shape[0] : shape[1];
    size_t seq_len = batch_first ? shape[1] : shape[0];
    size_t input_size = shape[2];
    const int gate_size = 4 * hidden_size;

    LSTMDirectionForwardResult result;
    result.output = Tensor::Zeros({seq_len, batch_size, static_cast<size_t>(hidden_size)});
    result.input_cache = Tensor::Zeros({seq_len, batch_size, input_size});
    result.gate_cache = Tensor::Zeros({seq_len, batch_size, static_cast<size_t>(gate_size)});
    result.hidden_cache = Tensor::Zeros({seq_len + 1, batch_size, static_cast<size_t>(hidden_size)});
    result.cell_cache = Tensor::Zeros({seq_len + 1, batch_size, static_cast<size_t>(hidden_size)});
    result.final_hidden = Tensor::Zeros({batch_size, static_cast<size_t>(hidden_size)});
    result.final_cell = Tensor::Zeros({batch_size, static_cast<size_t>(hidden_size)});

    const float* input_data = layer_input.Data<float>();
    const float* Wih = W_ih.Data<float>();
    const float* Whh = W_hh.Data<float>();
    const float* bih = b_ih.Data<float>();
    const float* bhh = b_hh.Data<float>();
    float* output_data = result.output.Data<float>();
    float* input_cache_data = result.input_cache.Data<float>();
    float* gate_cache_data = result.gate_cache.Data<float>();
    float* h_cache_data = result.hidden_cache.Data<float>();
    float* c_cache_data = result.cell_cache.Data<float>();
    float* final_h_data = result.final_hidden.Data<float>();
    float* final_c_data = result.final_cell.Data<float>();

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto tanh_f = [](float x) { return std::tanh(x); };

    std::vector<float> h(batch_size * hidden_size, 0.0f);
    std::vector<float> c(batch_size * hidden_size, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
        for (int i = 0; i < hidden_size; ++i) {
            const float h0 = init_hidden.Data<float>()[b * hidden_size + i];
            const float c0 = init_cell.Data<float>()[b * hidden_size + i];
            h[b * hidden_size + i] = h0;
            c[b * hidden_size + i] = c0;
            h_cache_data[b * hidden_size + i] = h0;
            c_cache_data[b * hidden_size + i] = c0;
        }
    }

    for (size_t step = 0; step < seq_len; ++step) {
        const size_t src_t = reverse_time ? (seq_len - 1 - step) : step;
        for (size_t b = 0; b < batch_size; ++b) {
            const float* x_ptr = batch_first
                ? input_data + b * seq_len * input_size + src_t * input_size
                : input_data + src_t * batch_size * input_size + b * input_size;

            float* in_cache = input_cache_data + step * batch_size * input_size + b * input_size;
            float* gate_cache = gate_cache_data + step * batch_size * gate_size + b * gate_size;
            float* h_prev_ptr = h.data() + b * hidden_size;
            float* c_prev_ptr = c.data() + b * hidden_size;

            std::copy(x_ptr, x_ptr + input_size, in_cache);

            for (int g = 0; g < gate_size; ++g) {
                float sum = bih[g] + bhh[g];
                for (size_t k = 0; k < input_size; ++k) {
                    sum += Wih[g * input_size + k] * x_ptr[k];
                }
                for (int k = 0; k < hidden_size; ++k) {
                    sum += Whh[g * hidden_size + k] * h_prev_ptr[k];
                }
                gate_cache[g] = sum;
            }

            for (int i = 0; i < hidden_size; ++i) {
                const float i_gate = sigmoid(gate_cache[i]);
                const float f_gate = sigmoid(gate_cache[hidden_size + i]);
                const float g_gate = tanh_f(gate_cache[2 * hidden_size + i]);
                const float o_gate = sigmoid(gate_cache[3 * hidden_size + i]);
                c_prev_ptr[i] = f_gate * c_prev_ptr[i] + i_gate * g_gate;
                h_prev_ptr[i] = o_gate * tanh_f(c_prev_ptr[i]);
            }

            for (int i = 0; i < hidden_size; ++i) {
                const float h_val = h_prev_ptr[i];
                output_data[src_t * batch_size * hidden_size + b * hidden_size + i] = h_val;
                h_cache_data[(step + 1) * batch_size * hidden_size + b * hidden_size + i] = h_val;
                c_cache_data[(step + 1) * batch_size * hidden_size + b * hidden_size + i] = c_prev_ptr[i];
                final_h_data[b * hidden_size + i] = h_val;
                final_c_data[b * hidden_size + i] = c_prev_ptr[i];
            }
        }
    }

    return result;
}

LSTMDirectionBackwardResult RunLSTMCpuDirectionBackward(
    const Tensor& grad_output,
    size_t feature_offset,
    const Tensor& input_cache,
    const Tensor& gate_cache,
    const Tensor& hidden_cache,
    const Tensor& cell_cache,
    const Tensor& W_ih,
    const Tensor& W_hh,
    int hidden_size,
    bool batch_first,
    bool reverse_time) {

    const auto& input_shape = input_cache.Shape();
    size_t seq_len = input_shape[0];
    size_t batch_size = input_shape[1];
    size_t input_size = input_shape[2];
    const int gate_size = 4 * hidden_size;

    LSTMDirectionBackwardResult result;
    result.input_grad = batch_first
        ? Tensor::Zeros({batch_size, seq_len, input_size})
        : Tensor::Zeros({seq_len, batch_size, input_size});
    result.grad_W_ih = Tensor::Zeros({static_cast<size_t>(gate_size), input_size});
    result.grad_W_hh = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size)});
    result.grad_b_ih = Tensor::Zeros({static_cast<size_t>(gate_size)});
    result.grad_b_hh = Tensor::Zeros({static_cast<size_t>(gate_size)});

    const float* dout = grad_output.Data<float>();
    const float* in_cache = input_cache.Data<float>();
    const float* g_cache = gate_cache.Data<float>();
    const float* h_cache = hidden_cache.Data<float>();
    const float* c_cache = cell_cache.Data<float>();
    const float* Wih = W_ih.Data<float>();
    const float* Whh = W_hh.Data<float>();
    float* dx_data = result.input_grad.Data<float>();
    float* dW_ih_data = result.grad_W_ih.Data<float>();
    float* dW_hh_data = result.grad_W_hh.Data<float>();
    float* db_ih_data = result.grad_b_ih.Data<float>();
    float* db_hh_data = result.grad_b_hh.Data<float>();

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto tanh_f  = [](float x) { return std::tanh(x); };

    std::vector<float> layer_grad(seq_len * batch_size * hidden_size, 0.0f);
    const size_t output_features = grad_output.Shape().size() >= 3
        ? grad_output.Shape()[2]
        : static_cast<size_t>(hidden_size);

    for (size_t step = 0; step < seq_len; ++step) {
        const size_t src_t = reverse_time ? (seq_len - 1 - step) : step;
        for (size_t b = 0; b < batch_size; ++b) {
            for (int i = 0; i < hidden_size; ++i) {
                const float g = batch_first
                    ? dout[b * seq_len * output_features + src_t * output_features + feature_offset + i]
                    : dout[src_t * batch_size * output_features + b * output_features + feature_offset + i];
                layer_grad[step * batch_size * hidden_size + b * hidden_size + i] = g;
            }
        }
    }

    for (size_t b = 0; b < batch_size; ++b) {
        std::vector<float> dh_next(hidden_size, 0.0f);
        std::vector<float> dc_next(hidden_size, 0.0f);

        for (int64_t step = static_cast<int64_t>(seq_len) - 1; step >= 0; --step) {
            const size_t src_t = reverse_time ? (seq_len - 1 - static_cast<size_t>(step))
                                              : static_cast<size_t>(step);
            const size_t gate_off = static_cast<size_t>(step) * batch_size * gate_size + b * gate_size;
            const size_t h_prev_off = static_cast<size_t>(step) * batch_size * hidden_size + b * hidden_size;
            const size_t c_prev_off = static_cast<size_t>(step) * batch_size * hidden_size + b * hidden_size;
            const size_t c_t_off = (static_cast<size_t>(step) + 1) * batch_size * hidden_size + b * hidden_size;
            const size_t in_off = static_cast<size_t>(step) * batch_size * input_size + b * input_size;
            const size_t lg_off = static_cast<size_t>(step) * batch_size * hidden_size + b * hidden_size;

            std::vector<float> dh(hidden_size, 0.0f);
            for (int i = 0; i < hidden_size; ++i) {
                dh[i] = layer_grad[lg_off + i] + dh_next[i];
            }

            std::vector<float> i_g(hidden_size), f_g(hidden_size), g_g(hidden_size), o_g(hidden_size);
            for (int i = 0; i < hidden_size; ++i) {
                i_g[i] = sigmoid(g_cache[gate_off + i]);
                f_g[i] = sigmoid(g_cache[gate_off + hidden_size + i]);
                g_g[i] = tanh_f (g_cache[gate_off + 2 * hidden_size + i]);
                o_g[i] = sigmoid(g_cache[gate_off + 3 * hidden_size + i]);
            }

            std::vector<float> dgates(gate_size, 0.0f);
            for (int i = 0; i < hidden_size; ++i) {
                const float c_t = c_cache[c_t_off + i];
                const float c_prev = c_cache[c_prev_off + i];
                const float tanh_c = tanh_f(c_t);

                const float do_pre = dh[i] * tanh_c * o_g[i] * (1.0f - o_g[i]);
                const float dc = dh[i] * o_g[i] * (1.0f - tanh_c * tanh_c) + dc_next[i];
                const float df_pre = dc * c_prev * f_g[i] * (1.0f - f_g[i]);
                const float di_pre = dc * g_g[i] * i_g[i] * (1.0f - i_g[i]);
                const float dg_pre = dc * i_g[i] * (1.0f - g_g[i] * g_g[i]);

                dc_next[i] = dc * f_g[i];

                dgates[i] = di_pre;
                dgates[hidden_size + i] = df_pre;
                dgates[2 * hidden_size + i] = dg_pre;
                dgates[3 * hidden_size + i] = do_pre;
            }

            for (int g = 0; g < gate_size; ++g) {
                const float dg = dgates[g];
                db_ih_data[g] += dg;
                db_hh_data[g] += dg;
                for (size_t k = 0; k < input_size; ++k) {
                    dW_ih_data[g * input_size + k] += dg * in_cache[in_off + k];
                }
                for (int k = 0; k < hidden_size; ++k) {
                    dW_hh_data[g * hidden_size + k] += dg * h_cache[h_prev_off + k];
                }
            }

            for (size_t k = 0; k < input_size; ++k) {
                float s = 0.0f;
                for (int g = 0; g < gate_size; ++g) {
                    s += dgates[g] * Wih[g * input_size + k];
                }
                if (batch_first) {
                    dx_data[b * seq_len * input_size + src_t * input_size + k] = s;
                } else {
                    dx_data[src_t * batch_size * input_size + b * input_size + k] = s;
                }
            }
            for (int k = 0; k < hidden_size; ++k) {
                float s = 0.0f;
                for (int g = 0; g < gate_size; ++g) {
                    s += dgates[g] * Whh[g * hidden_size + k];
                }
                dh_next[k] = s;
            }
        }
    }

    return result;
}

size_t Pool4DIndex(size_t h, size_t w, size_t c, size_t b,
                   size_t width, size_t channels, size_t batch_size) {
    return ((h * width + w) * channels + c) * batch_size + b;
}

void ValidatePoolInput(const Tensor& input, const char* name) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " CPU fallback requires Float32 input");
    }
    if (input.Shape().size() != 4) {
        throw std::runtime_error(std::string(name) + " CPU fallback expects [H, W, C, N] input");
    }
}

} // namespace

// ============================================================================
// Helper Functions for ArrayFire Integration
// ============================================================================

#ifdef CYXWIZ_HAS_ARRAYFIRE

// Semantic Tensor-to-ArrayFire bridge for layer code.
// Tensor owns layout conversion; layers request the layout they need.
static af::array TensorToAf(const Tensor& t) {
    return t.Shape().size() == 2 ? t.GetArrayRowMajor2D() : t.GetArray();
}

// 3D row-major → AF column-major with matching semantic axes.
// Rationale: bare TensorToAf on 3D scrambles semantics — CyxWiz stores
// Tensor owns the 3D row-major conversion; this layer helper names the
// semantic layout expected by recurrent kernels.
static af::array TensorToAf3DRowMajor(const Tensor& t) {
    return t.Shape().size() == 3 ? t.GetArrayRowMajor3D() : TensorToAf(t);
}

// Tensor owns the inverse 3D row-major conversion as well.
// Forward-declare AfToTensor so AfToTensor3DRowMajor can fall back to
// it for 4D inputs. AfToTensor's full definition is later in this file.
static Tensor AfToTensor(const af::array& arr);

static Tensor AfToTensor3DRowMajor(const af::array& arr) {
    if (arr.numdims() > 3) {
        // Fall back to existing path for 4D; caller owns correctness.
        return AfToTensor(arr);
    }
    return Tensor::FromArrayRowMajor3D(arr);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
static LSTMAfDirectionResult RunLSTMAfDirectionForward(
    const af::array& seq_input,
    const af::array& W_ih,
    const af::array& W_hh,
    const af::array& b_ih,
    const af::array& b_hh,
    const af::array& init_h,
    const af::array& init_c,
    size_t seq_len,
    size_t batch_size,
    size_t input_dim,
    int hidden_size) {

    const int gate_size = 4 * hidden_size;
    LSTMAfDirectionResult result;

    af::array h = af::moddims(init_h, af::dim4(batch_size, hidden_size));
    af::array c = af::moddims(init_c, af::dim4(batch_size, hidden_size));

    af::array input_flat = af::moddims(seq_input, af::dim4(seq_len * batch_size, input_dim));
    af::array input_proj = af::matmul(input_flat, af::transpose(W_ih));
    input_proj = input_proj + af::tile(af::transpose(b_ih), static_cast<unsigned int>(seq_len * batch_size));
    input_proj = af::moddims(input_proj, af::dim4(seq_len, batch_size, 4 * hidden_size));

    af::array h_states = af::constant(0.0f, af::dim4(seq_len + 1, batch_size, hidden_size));
    af::array c_states = af::constant(0.0f, af::dim4(seq_len + 1, batch_size, hidden_size));
    af::array all_gates = af::constant(0.0f, af::dim4(seq_len, batch_size, 4 * hidden_size));

    h_states(0, af::span, af::span) = af::moddims(h, af::dim4(1, batch_size, hidden_size));
    c_states(0, af::span, af::span) = af::moddims(c, af::dim4(1, batch_size, hidden_size));

    for (size_t t = 0; t < seq_len; ++t) {
        af::array x_t = input_proj(t, af::span, af::span);
        x_t = af::moddims(x_t, af::dim4(batch_size, 4 * hidden_size));

        af::array h_proj = af::matmul(h, af::transpose(W_hh));
        h_proj = h_proj + af::tile(af::transpose(b_hh), static_cast<unsigned int>(batch_size));

        af::array gates = x_t + h_proj;
        af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, hidden_size - 1)));
        af::array f_gate = af::sigmoid(gates(af::span, af::seq(hidden_size, 2 * hidden_size - 1)));
        af::array g_gate = af::tanh(gates(af::span, af::seq(2 * hidden_size, 3 * hidden_size - 1)));
        af::array o_gate = af::sigmoid(gates(af::span, af::seq(3 * hidden_size, 4 * hidden_size - 1)));

        c = f_gate * c + i_gate * g_gate;
        h = o_gate * af::tanh(c);

        h_states(t + 1, af::span, af::span) = af::moddims(h, af::dim4(1, batch_size, hidden_size));
        c_states(t + 1, af::span, af::span) = af::moddims(c, af::dim4(1, batch_size, hidden_size));
        all_gates(t, af::span, af::span) = af::moddims(gates, af::dim4(1, batch_size, 4 * hidden_size));
    }

    result.output = h_states(af::seq(1, static_cast<double>(seq_len)), af::span, af::span);
    result.input_cache = AfToTensor3DRowMajor(seq_input);
    result.gate_cache = AfToTensor3DRowMajor(all_gates);
    result.hidden_cache = AfToTensor3DRowMajor(h_states);
    result.cell_cache = AfToTensor3DRowMajor(c_states);
    result.final_hidden = AfToTensor(h);
    result.final_cell = AfToTensor(c);
    return result;
}
#endif

// Helper: Create Tensor from ArrayFire array
// Note: Transpose 2D arrays back to row-major for CyxWiz Tensor
static Tensor AfToTensor(const af::array& arr) {
    // Count significant dimensions
    int ndims = 0;
    for (unsigned int i = 0; i < 4; i++) {
        if (arr.dims(i) > 1) ndims = i + 1;
        else if (i == 0) ndims = 1;
    }

    // For 2D arrays, transpose to row-major before copying to Tensor
    if (ndims == 2) {
        return Tensor::FromArrayRowMajor2D(arr);
    }

    // For other dimensions, keep the ArrayFire result resident until host data is requested.
    return Tensor(arr);
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
            // Broadcast row bias across the batch dimension. `output` is
            // semantic row-major [batch, out_features].
            output = output + af::tile(af::transpose(b),
                                       static_cast<unsigned int>(output.dims(0)));
        }

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire DenseLayer::Forward failed: {}", e.what());
    }
#endif

    const std::vector<size_t>& input_shape = input.Shape();
    const bool is_batched = input_shape.size() == 2;
    if (!is_batched && input_shape.size() != 1) {
        throw std::runtime_error("Dense forward expects a 1D or 2D Float32 input tensor");
    }
    if (input.GetDataType() != DataType::Float32 ||
        weights_.GetDataType() != DataType::Float32 ||
        (use_bias_ && bias_.GetDataType() != DataType::Float32)) {
        throw std::runtime_error("Dense forward CPU fallback requires Float32 tensors");
    }

    const size_t batch_size = is_batched ? input_shape[0] : 1;
    const size_t input_features = is_batched ? input_shape[1] : input_shape[0];
    if (input_features != static_cast<size_t>(in_features_)) {
        throw std::runtime_error("Dense forward input feature mismatch");
    }

    Tensor output(is_batched
                      ? std::vector<size_t>{batch_size, static_cast<size_t>(out_features_)}
                      : std::vector<size_t>{static_cast<size_t>(out_features_)},
                  DataType::Float32);
    const float* input_data = input.Data<float>();
    const float* weight_data = weights_.Data<float>();
    const float* bias_data = use_bias_ ? bias_.Data<float>() : nullptr;
    float* output_data = output.Data<float>();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t out = 0; out < static_cast<size_t>(out_features_); ++out) {
            float sum = use_bias_ ? bias_data[out] : 0.0f;
            for (size_t in = 0; in < static_cast<size_t>(in_features_); ++in) {
                const size_t input_index = is_batched ? batch * static_cast<size_t>(in_features_) + in : in;
                sum += input_data[input_index] *
                       weight_data[out * static_cast<size_t>(in_features_) + in];
            }
            const size_t output_index = is_batched ? batch * static_cast<size_t>(out_features_) + out : out;
            output_data[output_index] = sum;
        }
    }

    return output;
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

    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const std::vector<size_t>& grad_shape = grad_output.Shape();
    const bool is_batched = input_shape.size() == 2;
    if (!is_batched && input_shape.size() != 1) {
        throw std::runtime_error("Dense backward expects cached 1D or 2D input");
    }
    const std::vector<size_t> expected_grad_shape =
        is_batched ? std::vector<size_t>{input_shape[0], static_cast<size_t>(out_features_)}
                   : std::vector<size_t>{static_cast<size_t>(out_features_)};
    if (grad_shape != expected_grad_shape) {
        throw std::runtime_error("Dense backward gradient shape mismatch");
    }
    if (grad_output.GetDataType() != DataType::Float32 ||
        cached_input_.GetDataType() != DataType::Float32 ||
        weights_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Dense backward CPU fallback requires Float32 tensors");
    }

    const size_t batch_size = is_batched ? input_shape[0] : 1;
    Tensor grad_input(is_batched
                          ? std::vector<size_t>{batch_size, static_cast<size_t>(in_features_)}
                          : std::vector<size_t>{static_cast<size_t>(in_features_)},
                      DataType::Float32);
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(out_features_),
                                   static_cast<size_t>(in_features_)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_features_)});
    }

    const float* grad_output_data = grad_output.Data<float>();
    const float* input_data = cached_input_.Data<float>();
    const float* weight_data = weights_.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    float* grad_weight_data = grad_weights_.Data<float>();
    float* grad_bias_data = use_bias_ ? grad_bias_.Data<float>() : nullptr;

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t in = 0; in < static_cast<size_t>(in_features_); ++in) {
            float grad_sum = 0.0f;
            for (size_t out = 0; out < static_cast<size_t>(out_features_); ++out) {
                const size_t grad_output_index =
                    is_batched ? batch * static_cast<size_t>(out_features_) + out : out;
                grad_sum += grad_output_data[grad_output_index] *
                            weight_data[out * static_cast<size_t>(in_features_) + in];
            }
            const size_t grad_input_index =
                is_batched ? batch * static_cast<size_t>(in_features_) + in : in;
            grad_input_data[grad_input_index] = grad_sum;
        }

        for (size_t out = 0; out < static_cast<size_t>(out_features_); ++out) {
            const size_t grad_output_index =
                is_batched ? batch * static_cast<size_t>(out_features_) + out : out;
            if (use_bias_) {
                grad_bias_data[out] += grad_output_data[grad_output_index];
            }
            for (size_t in = 0; in < static_cast<size_t>(in_features_); ++in) {
                const size_t input_index = is_batched ? batch * static_cast<size_t>(in_features_) + in : in;
                grad_weight_data[out * static_cast<size_t>(in_features_) + in] +=
                    grad_output_data[grad_output_index] * input_data[input_index];
            }
        }
    }

    return grad_input;
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
            output = output + af::tile(b, af::dim4(out_h, out_w, 1, batch_size));
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

    ValidatePoolInput(input, "MaxPool2D");
    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    const size_t padded_h = in_h + static_cast<size_t>(2 * padding_);
    const size_t padded_w = in_w + static_cast<size_t>(2 * padding_);
    if (padded_h < static_cast<size_t>(pool_size_) || padded_w < static_cast<size_t>(pool_size_)) {
        throw std::runtime_error("MaxPool2D pool window is larger than padded input");
    }
    const size_t out_h = (padded_h - static_cast<size_t>(pool_size_)) / static_cast<size_t>(stride_) + 1;
    const size_t out_w = (padded_w - static_cast<size_t>(pool_size_)) / static_cast<size_t>(stride_) + 1;

    Tensor output({out_h, out_w, channels, batch_size}, DataType::Float32);
    max_indices_ = Tensor({out_h, out_w, channels, batch_size}, DataType::Int32);
    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();
    int32_t* index_data = max_indices_.Data<int32_t>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float max_value = -std::numeric_limits<float>::infinity();
                    int32_t max_index = 0;
                    for (int ph = 0; ph < pool_size_; ++ph) {
                        for (int pw = 0; pw < pool_size_; ++pw) {
                            const int ih = static_cast<int>(oh * static_cast<size_t>(stride_)) + ph - padding_;
                            const int iw = static_cast<int>(ow * static_cast<size_t>(stride_)) + pw - padding_;
                            const float value = ih >= 0 && iw >= 0 &&
                                                        ih < static_cast<int>(in_h) &&
                                                        iw < static_cast<int>(in_w)
                                                    ? input_data[Pool4DIndex(static_cast<size_t>(ih),
                                                                             static_cast<size_t>(iw),
                                                                             c, b, in_w, channels, batch_size)]
                                                    : 0.0f;
                            if (value > max_value) {
                                max_value = value;
                                max_index = static_cast<int32_t>(ph * pool_size_ + pw);
                            }
                        }
                    }
                    const size_t out_index = Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size);
                    output_data[out_index] = max_value;
                    index_data[out_index] = max_index;
                }
            }
        }
    }

    return output;
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

    ValidatePoolInput(cached_input_, "MaxPool2D");
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error("MaxPool2D backward CPU fallback requires Float32 grad_output");
    }
    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const std::vector<size_t>& grad_shape = grad_output.Shape();
    if (grad_shape.size() != 4) {
        throw std::runtime_error("MaxPool2D backward expects [out_h, out_w, C, N] grad_output");
    }
    if (grad_shape[2] != input_shape[2] || grad_shape[3] != input_shape[3] ||
        max_indices_.Shape() != grad_shape) {
        throw std::runtime_error("MaxPool2D backward gradient/cache shape mismatch");
    }

    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    const size_t out_h = grad_shape[0];
    const size_t out_w = grad_shape[1];
    Tensor grad_input = Tensor::Zeros(input_shape, DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    const int32_t* index_data = max_indices_.Data<int32_t>();
    float* grad_input_data = grad_input.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    const size_t grad_index = Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size);
                    const int32_t local_index = index_data[grad_index];
                    const int ph = local_index / pool_size_;
                    const int pw = local_index % pool_size_;
                    const int ih = static_cast<int>(oh * static_cast<size_t>(stride_)) + ph - padding_;
                    const int iw = static_cast<int>(ow * static_cast<size_t>(stride_)) + pw - padding_;
                    if (ih >= 0 && iw >= 0 && ih < static_cast<int>(in_h) && iw < static_cast<int>(in_w)) {
                        grad_input_data[Pool4DIndex(static_cast<size_t>(ih),
                                                    static_cast<size_t>(iw),
                                                    c, b, in_w, channels, batch_size)] += grad_data[grad_index];
                    }
                }
            }
        }
    }

    return grad_input;
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

    ValidatePoolInput(input, "AvgPool2D");
    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    const size_t padded_h = in_h + static_cast<size_t>(2 * padding_);
    const size_t padded_w = in_w + static_cast<size_t>(2 * padding_);
    if (padded_h < static_cast<size_t>(pool_size_) || padded_w < static_cast<size_t>(pool_size_)) {
        throw std::runtime_error("AvgPool2D pool window is larger than padded input");
    }
    const size_t out_h = (padded_h - static_cast<size_t>(pool_size_)) / static_cast<size_t>(stride_) + 1;
    const size_t out_w = (padded_w - static_cast<size_t>(pool_size_)) / static_cast<size_t>(stride_) + 1;

    Tensor output({out_h, out_w, channels, batch_size}, DataType::Float32);
    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();
    const float scale = 1.0f / static_cast<float>(pool_size_ * pool_size_);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    for (int ph = 0; ph < pool_size_; ++ph) {
                        for (int pw = 0; pw < pool_size_; ++pw) {
                            const int ih = static_cast<int>(oh * static_cast<size_t>(stride_)) + ph - padding_;
                            const int iw = static_cast<int>(ow * static_cast<size_t>(stride_)) + pw - padding_;
                            if (ih >= 0 && iw >= 0 &&
                                ih < static_cast<int>(in_h) && iw < static_cast<int>(in_w)) {
                                sum += input_data[Pool4DIndex(static_cast<size_t>(ih),
                                                              static_cast<size_t>(iw),
                                                              c, b, in_w, channels, batch_size)];
                            }
                        }
                    }
                    output_data[Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size)] = sum * scale;
                }
            }
        }
    }

    return output;
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

    ValidatePoolInput(cached_input_, "AvgPool2D");
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error("AvgPool2D backward CPU fallback requires Float32 grad_output");
    }
    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const std::vector<size_t>& grad_shape = grad_output.Shape();
    if (grad_shape.size() != 4 || grad_shape[2] != input_shape[2] || grad_shape[3] != input_shape[3]) {
        throw std::runtime_error("AvgPool2D backward gradient shape mismatch");
    }

    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    const size_t out_h = grad_shape[0];
    const size_t out_w = grad_shape[1];
    Tensor grad_input = Tensor::Zeros(input_shape, DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    const float scale = 1.0f / static_cast<float>(pool_size_ * pool_size_);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    const float grad_value =
                        grad_data[Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size)] * scale;
                    for (int ph = 0; ph < pool_size_; ++ph) {
                        for (int pw = 0; pw < pool_size_; ++pw) {
                            const int ih = static_cast<int>(oh * static_cast<size_t>(stride_)) + ph - padding_;
                            const int iw = static_cast<int>(ow * static_cast<size_t>(stride_)) + pw - padding_;
                            if (ih >= 0 && iw >= 0 &&
                                ih < static_cast<int>(in_h) && iw < static_cast<int>(in_w)) {
                                grad_input_data[Pool4DIndex(static_cast<size_t>(ih),
                                                            static_cast<size_t>(iw),
                                                            c, b, in_w, channels, batch_size)] += grad_value;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

// ============================================================================
// GlobalAvgPool2D Layer Implementation
// ============================================================================

Tensor GlobalAvgPool2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

    ValidatePoolInput(input, "GlobalAvgPool2D");
    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    Tensor output({channels, batch_size}, DataType::Float32);
    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();
    const float scale = 1.0f / static_cast<float>(in_h * in_w);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (size_t h = 0; h < in_h; ++h) {
                for (size_t w = 0; w < in_w; ++w) {
                    sum += input_data[Pool4DIndex(h, w, c, b, in_w, channels, batch_size)];
                }
            }
            output_data[c * batch_size + b] = sum * scale;
        }
    }

    return output;
}

Tensor GlobalAvgPool2DLayer::Backward(const Tensor& grad_output) {
    ValidatePoolInput(cached_input_, "GlobalAvgPool2D");
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error("GlobalAvgPool2D backward CPU fallback requires Float32 grad_output");
    }
    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    if (grad_output.Shape() != std::vector<size_t>{channels, batch_size}) {
        throw std::runtime_error("GlobalAvgPool2D backward gradient shape mismatch");
    }

    Tensor grad_input(input_shape, DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    const float scale = 1.0f / static_cast<float>(in_h * in_w);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            const float grad_value = grad_data[c * batch_size + b] * scale;
            for (size_t h = 0; h < in_h; ++h) {
                for (size_t w = 0; w < in_w; ++w) {
                    grad_input_data[Pool4DIndex(h, w, c, b, in_w, channels, batch_size)] = grad_value;
                }
            }
        }
    }

    return grad_input;
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
        mean_bc = af::tile(mean_bc, af::dim4(height, width, 1, batch_size));
        std_inv_bc = af::tile(std_inv_bc, af::dim4(height, width, 1, batch_size));
        gamma_bc = af::tile(gamma_bc, af::dim4(height, width, 1, batch_size));
        beta_bc = af::tile(beta_bc, af::dim4(height, width, 1, batch_size));

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
        gamma_bc = af::tile(gamma_bc, af::dim4(height, width, 1, batch_size));

        af::array std_inv_bc = af::moddims(std_inv, af::dim4(1, 1, channels, 1));
        std_inv_bc = af::tile(std_inv_bc, af::dim4(height, width, 1, batch_size));

        // sum(dy) per channel
        af::array sum_dy = af::sum(af::sum(af::sum(grad_out, 0), 1), 3);
        sum_dy = af::moddims(sum_dy, af::dim4(1, 1, channels, 1));
        sum_dy = af::tile(sum_dy, af::dim4(height, width, 1, batch_size));

        // sum(dy * normalized) per channel
        af::array sum_dy_norm = af::sum(af::sum(af::sum(grad_out * normalized, 0), 1), 3);
        sum_dy_norm = af::moddims(sum_dy_norm, af::dim4(1, 1, channels, 1));
        sum_dy_norm = af::tile(sum_dy_norm, af::dim4(height, width, 1, batch_size));

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
    const auto& shape = input.Shape();

    // Flatten is a pure reshape — no GPU computation needed. We just
    // change the shape from [batch, d1, d2, ...] to [batch, d1*d2*...]
    // and keep the data buffer in place (row-major layout stays correct
    // for LinearLayer which expects row-major [batch, features]).
    //
    // Going through ArrayFire for this was wrong: the generic
    // TensorToAf/AfToTensor round-trip scrambles the data layout
    // (column-major vs row-major mismatch with LinearLayer's manual
    // transpose in its Forward). Pure CPU reshape avoids the issue.
    if (shape.size() <= 2) {
        return input;  // already 1D or 2D, nothing to flatten
    }

    size_t batch = shape[0];
    size_t flat = 1;
    for (size_t i = 1; i < shape.size(); i++) {
        flat *= shape[i];
    }

    return Tensor({batch, flat}, input.Data(), input.GetDataType());
}

Tensor FlattenLayer::Backward(const Tensor& grad_output) {
    // Pure CPU reshape back to the original shape saved in Forward.
    // Same reasoning as Forward: no GPU needed, just change the shape.
    return Tensor(input_shape_, grad_output.Data(), grad_output.GetDataType());
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

    if (!training_ || p_ <= 0.0f) {
        return input;
    }
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Dropout forward CPU fallback requires Float32 input");
    }

    Tensor output(input.Shape(), DataType::Float32);
    mask_ = Tensor(input.Shape(), DataType::Float32);
    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();
    float* mask_data = mask_.Data<float>();
    const float scale = 1.0f / (1.0f - p_);

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < input.NumElements(); ++i) {
        mask_data[i] = dist(rng) > p_ ? 1.0f : 0.0f;
        output_data[i] = input_data[i] * mask_data[i] * scale;
    }

    return output;
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

    if (!training_ || p_ <= 0.0f) {
        return grad_output;
    }
    if (grad_output.GetDataType() != DataType::Float32 || mask_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Dropout backward CPU fallback requires Float32 tensors");
    }
    if (mask_.Shape() != grad_output.Shape()) {
        throw std::runtime_error("Dropout backward requires a forward mask matching grad_output");
    }

    Tensor grad_input(grad_output.Shape(), DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    const float* mask_data = mask_.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    const float scale = 1.0f / (1.0f - p_);
    for (size_t i = 0; i < grad_output.NumElements(); ++i) {
        grad_input_data[i] = grad_data[i] * mask_data[i] * scale;
    }

    return grad_input;
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
    // ArrayFire path is only used for the unbatched case (shape.size()==1).
    // Batched input deliberately falls through to the CPU fallback below
    // because AF's column-major scatter gives the wrong data layout for
    // the next layer. The previous version did `try { throw } catch` per
    // batch which spammed warnings 600+ times per epoch and burned CPU
    // on the exception throw — this gate avoids both.
    if (input.Shape().size() != 2) try {
        // Apply max_norm if specified (single-sample path only)
        if (max_norm_ > 0.0f) {
            NormalizeEmbeddings();
        }

        const auto& shape = input.Shape();
        dim_t seq_len = shape[0];
        dim_t total_indices = seq_len;

        // Get indices as int32
        const int32_t* indices_ptr = input.Data<int32_t>();

        // Get weight matrix
        af::array w = TensorToAf(weight_);  // [num_embeddings, embedding_dim]

        // Vectorized gather: for each index, get the corresponding row
        af::array output_flat = af::constant(0.0f, af::dim4(total_indices, embedding_dim_));
        for (dim_t i = 0; i < total_indices; i++) {
            int32_t idx = indices_ptr[i];
            if (idx >= 0 && idx < num_embeddings_) {
                output_flat(i, af::span) = w(idx, af::span);
            }
            // If idx == padding_idx or out of bounds, leave as zero
        }

        // Reshape to [seq_len, embedding_dim]
        af::array output = af::moddims(output_flat, af::dim4(seq_len, embedding_dim_));
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
    // Same gating as Forward: batched input goes to the CPU fallback
    // because AF's column-major moddims silently scrambles the row
    // ordering of [batch, seq_len, embed_dim] gradient tensors. Caught
    // when wiring text training — the AF backward returned valid-shaped
    // but content-wrong gradients, leading to slow / unstable learning.
    if (cached_indices_.Shape().size() != 2) try {
        const auto& shape = cached_indices_.Shape();
        dim_t seq_len = shape[0];
        dim_t total_indices = seq_len;

        const int32_t* indices_ptr = cached_indices_.Data<int32_t>();

        // Initialize gradient accumulator
        af::array dw = af::constant(0.0f, af::dim4(num_embeddings_, embedding_dim_));

        // Get flattened gradient output
        af::array grad = TensorToAf(grad_output);
        grad = af::moddims(grad, af::dim4(total_indices, embedding_dim_));

        // Scatter-add gradients to the weight matrix
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
    return params;
}

std::map<std::string, Tensor> EmbeddingLayer::GetGradients() {
    // Match LinearLayer convention: one entry per trainable parameter.
    // Used by EmbeddingModule so the optimizer can update weights through
    // the standard Module::GetGradients() path.
    std::map<std::string, Tensor> grads;
    grads["weight"] = grad_weight_;
    return grads;
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

    // Hoisted from the CPU fallback path: ensure weights are valid
    // BEFORE either path runs. The AF path was tripping
    // af_write_array "Expected: (data != nullptr)" because LSTMLayer's
    // constructor calls InitializeWeights() which uses the AF backend
    // (TensorToAf, etc.); when AF init failed silently the W_ih_ /
    // W_hh_ / b_ih_ / b_hh_ tensors were registered but had nullptr
    // data. The CPU fallback re-init path was only reached if AF
    // Forward also failed, but with the AF path fixed at the boundary
    // it now gets further and trips on the null weights.
    {
        const auto& shape = input.Shape();
        size_t input_dim = shape.size() == 3
            ? (batch_first_ ? shape[2] : shape[2]) : 0;
        int num_directions = bidirectional_ ? 2 : 1;
        if (W_ih_.empty() || W_ih_[0].Data<float>() == nullptr) {
            for (int layer = 0; layer < num_layers_; layer++) {
                size_t layer_input_size = (layer == 0) ? input_dim
                    : static_cast<size_t>(hidden_size_ * num_directions);
                size_t gate_size = static_cast<size_t>(4 * hidden_size_);
                W_ih_[layer] = Tensor::Random({gate_size, layer_input_size});
                W_hh_[layer] = Tensor::Random({gate_size, static_cast<size_t>(hidden_size_)});
                b_ih_[layer] = Tensor::Zeros({gate_size});
                b_hh_[layer] = Tensor::Zeros({gate_size});
            }
        }
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // The ArrayFire path below is currently broken for [batch, seq,
    // features] 3D input. On first call it throws ArrayFire Exception
    // (Invalid input argument:202); on subsequent calls Invalid input
    // size:203. Same bug family as the pre-fix EmbeddingLayer backward
    // (84ef7211): the `af::reorder` / `moddims` math assumes row-major
    // dim ordering but AF is column-major, so the dims read here
    // (`batch_size = x.dims(0)` etc.) are interpreting the tensor
    // incorrectly and every downstream shape check eventually trips.
    //
    // AF Forward is operational as of 2026-04-16 after four stacked fixes:
    //   1. 3D column-major scrambling at TensorToAf boundary —
    //      TensorToAf3DRowMajor / AfToTensor3DRowMajor helpers handle the
    //      row-major → column-major conversion via dim-reversal +
    //      af::reorder(2, 1, 0).
    //   2. Slice assignment shape mismatch — h_states(t, span, span) = h
    //      had rank-3 proxy on LHS but rank-2 RHS; wrapped each RHS in
    //      af::moddims(..., dim4(1, batch, hidden)) to match.
    //   3. Hoisted weight init guard — covers the case where the AF
    //      backend silently failed to initialize W_ih_/W_hh_/biases.
    //   4. h_n_/c_n_ init check now matches CPU fallback exactly. The
    //      previous AF-only check `NumElements() == 0` doesn't fire for
    //      a default-constructed Tensor() (shape={} → product = 1, so
    //      NumElements returns 1), but Data() is still null and
    //      TensorToAf trips. Mirror the CPU's
    //      `Data<float>() == nullptr` clause.
    //
    // AF Forward + CPU Backward form a working hybrid: AF gives GPU-speed
    // forward, CPU gives correct gradients via row-major caches that both
    // paths populate identically (via AfToTensor3DRowMajor in AF Forward).
    // Loss numerically matches CPU within fp32 noise (~10ppm at the loss
    // level) on the mini sentiment LSTM smoke test.
    //
    // AF Backward under `#if 0` below is the next perf upgrade — would
    // skip the Backward Tensor↔CPU round-trip but needs its own column-
    // major audit using the now-working AF Forward as the oracle.
        // Bidirectional LSTM is currently supported by the CPU path only.
        // The ArrayFire branch still has a shape/layout failure on this
        // configuration, so skip it entirely instead of trying-and-falling-
        // back after the fact.
        constexpr bool kAfPathEnabled = true;
        if (kAfPathEnabled) try {
        // Use the 3D-aware helper — bare TensorToAf on [batch, seq, feat]
        // produces a column-major AF array with scrambled semantic axes.
        af::array x = TensorToAf3DRowMajor(input);

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

        af::array semantic_input = x;

        int num_directions = bidirectional_ ? 2 : 1;

        // Initialize hidden and cell states if not set. The CPU fallback's
        // matching check (line 2378+) ANDs `Data<float>() == nullptr` —
        // this AF check used to omit it. The bug: a default-constructed
        // Tensor has shape={} which yields NumElements()=1 (product of
        // empty range = 1) BUT Data() returns nullptr, so the
        // NumElements==0 check passes through and h_n_/c_n_ stay
        // unallocated, then TensorToAf trips on null data. Mirror the
        // CPU check exactly to keep both paths consistent.
        if (h_n_.NumElements() == 0 || h_n_.Data<float>() == nullptr) {
            h_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                                   static_cast<size_t>(batch_size),
                                   static_cast<size_t>(hidden_size_)});
        }
        if (c_n_.NumElements() == 0 || c_n_.Data<float>() == nullptr) {
            c_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                                   static_cast<size_t>(batch_size),
                                   static_cast<size_t>(hidden_size_)});
        }

        if (bidirectional_) {
            af::array h_full = TensorToAf(h_n_);
            af::array c_full = TensorToAf(c_n_);

            cached_inputs_.clear();
            cached_gates_.clear();
            cached_cell_states_.clear();
            cached_hidden_states_.clear();
            cached_inputs_.reserve(static_cast<size_t>(num_layers_ * 2));
            cached_gates_.reserve(static_cast<size_t>(num_layers_ * 2));
            cached_hidden_states_.reserve(static_cast<size_t>(num_layers_ * 2));
            cached_cell_states_.reserve(static_cast<size_t>(num_layers_ * 2));

            af::array layer_input = semantic_input;

            for (int layer = 0; layer < num_layers_; ++layer) {
                af::array W_ih = TensorToAf(W_ih_[layer]);
                af::array W_hh = TensorToAf(W_hh_[layer]);
                af::array b_ih = TensorToAf(b_ih_[layer]);
                af::array b_hh = TensorToAf(b_hh_[layer]);

                af::array init_h_fwd = af::moddims(h_full(layer, af::span, af::span),
                                                   af::dim4(batch_size, hidden_size_));
                af::array init_c_fwd = af::moddims(c_full(layer, af::span, af::span),
                                                   af::dim4(batch_size, hidden_size_));
                af::array init_h_rev = af::moddims(h_full(num_layers_ + layer, af::span, af::span),
                                                   af::dim4(batch_size, hidden_size_));
                af::array init_c_rev = af::moddims(c_full(num_layers_ + layer, af::span, af::span),
                                                   af::dim4(batch_size, hidden_size_));

                auto fwd = RunLSTMAfDirectionForward(
                    layer_input, W_ih, W_hh, b_ih, b_hh,
                    init_h_fwd, init_c_fwd,
                    seq_len, batch_size, static_cast<size_t>(layer_input.dims(2)),
                    hidden_size_);

                af::array reversed_input = af::flip(layer_input, 0);
                af::array W_ih_r = TensorToAf(W_ih_reverse_[layer]);
                af::array W_hh_r = TensorToAf(W_hh_reverse_[layer]);
                af::array b_ih_r = TensorToAf(b_ih_reverse_[layer]);
                af::array b_hh_r = TensorToAf(b_hh_reverse_[layer]);
                auto rev = RunLSTMAfDirectionForward(
                    reversed_input, W_ih_r, W_hh_r, b_ih_r, b_hh_r,
                    init_h_rev, init_c_rev,
                    seq_len, batch_size, static_cast<size_t>(reversed_input.dims(2)),
                    hidden_size_);

                af::array layer_output = af::join(2, fwd.output, af::flip(rev.output, 0));

                cached_inputs_.push_back(std::move(fwd.input_cache));
                cached_gates_.push_back(std::move(fwd.gate_cache));
                cached_hidden_states_.push_back(std::move(fwd.hidden_cache));
                cached_cell_states_.push_back(std::move(fwd.cell_cache));
                cached_inputs_.push_back(std::move(rev.input_cache));
                cached_gates_.push_back(std::move(rev.gate_cache));
                cached_hidden_states_.push_back(std::move(rev.hidden_cache));
                cached_cell_states_.push_back(std::move(rev.cell_cache));

                h_full(layer, af::span, af::span) = af::moddims(TensorToAf(fwd.final_hidden),
                                                                af::dim4(1, batch_size, hidden_size_));
                c_full(layer, af::span, af::span) = af::moddims(TensorToAf(fwd.final_cell),
                                                                af::dim4(1, batch_size, hidden_size_));
                h_full(num_layers_ + layer, af::span, af::span) = af::moddims(TensorToAf(rev.final_hidden),
                                                                               af::dim4(1, batch_size, hidden_size_));
                c_full(num_layers_ + layer, af::span, af::span) = af::moddims(TensorToAf(rev.final_cell),
                                                                               af::dim4(1, batch_size, hidden_size_));

                h_n_ = AfToTensor3DRowMajor(h_full);
                c_n_ = AfToTensor3DRowMajor(c_full);

                if (layer < num_layers_ - 1 && dropout_ > 0.0f && training_) {
                    af::array mask = (af::randu(layer_output.dims()) > dropout_).as(af::dtype::f32);
                    layer_output = layer_output * mask / (1.0f - dropout_);
                }

                layer_input = layer_output;
            }

            af::array output = layer_input;
            if (batch_first_) {
                output = af::reorder(output, 1, 0, 2);
            }
            return AfToTensor3DRowMajor(output);
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

            // Cache for backward — use 3D-aware helper so the resulting
            // Tensor lands in row-major [seq_len, batch, input_size]
            // layout matching what CPU BPTT (LSTMLayer::Backward) reads.
            cached_inputs_.push_back(AfToTensor3DRowMajor(layer_input));

            // Storage for hidden states and cell states over time
            af::array h_states = af::constant(0.0f, af::dim4(seq_len + 1, batch_size, hidden_size_));
            af::array c_states = af::constant(0.0f, af::dim4(seq_len + 1, batch_size, hidden_size_));
            af::array all_gates = af::constant(0.0f, af::dim4(seq_len, batch_size, 4 * hidden_size_));

            // Store initial states. Slice (k, span, span) of a 3D
            // [seq+1, batch, hidden] array yields a (1, batch, hidden)
            // proxy — assigning a (batch, hidden) 2D `h` directly trips
            // af "Size mismatch between input and output" (Invalid input
            // size:203). Wrap with explicit moddims to add the leading
            // 1 dim and match the proxy's rank.
            h_states(0, af::span, af::span) = af::moddims(h, af::dim4(1, batch_size, hidden_size_));
            c_states(0, af::span, af::span) = af::moddims(c, af::dim4(1, batch_size, hidden_size_));

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

                // Store states. Same moddims-with-leading-1 pattern as
                // the initial-state assignments above to keep the slice
                // proxy and RHS rank consistent.
                h_states(t + 1, af::span, af::span) = af::moddims(h, af::dim4(1, batch_size, hidden_size_));
                c_states(t + 1, af::span, af::span) = af::moddims(c, af::dim4(1, batch_size, hidden_size_));

                // Store gates for backward pass (pre-activation for efficiency)
                all_gates(t, af::span, af::span) = af::moddims(gates, af::dim4(1, batch_size, 4 * hidden_size_));
            }

            // Cache for backward — same row-major treatment so all
            // four caches are consistent with what CPU BPTT reads.
            //   gates   : [seq_len,     batch, 4 * hidden_size]
            //   h_states: [seq_len + 1, batch, hidden_size]
            //   c_states: [seq_len + 1, batch, hidden_size]
            cached_gates_.push_back(AfToTensor3DRowMajor(all_gates));
            cached_hidden_states_.push_back(AfToTensor3DRowMajor(h_states));
            cached_cell_states_.push_back(AfToTensor3DRowMajor(c_states));

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

                h_states_r(seq_len, af::span, af::span) = af::moddims(h_r, af::dim4(1, batch_size, hidden_size_));
                c_states_r(seq_len, af::span, af::span) = af::moddims(c_r, af::dim4(1, batch_size, hidden_size_));

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

                    h_states_r(t, af::span, af::span) = af::moddims(h_r, af::dim4(1, batch_size, hidden_size_));
                    c_states_r(t, af::span, af::span) = af::moddims(c_r, af::dim4(1, batch_size, hidden_size_));
                }

                // Extract reverse output and concatenate
                af::array layer_output_r = h_states_r(af::seq(0, static_cast<double>(seq_len - 1)), af::span, af::span);
                layer_output = af::join(2, layer_output, layer_output_r);

                // Update final states. Same moddims-with-leading-1
                // pattern — slice (k, span, span) is rank-3 (1, b, H).
                h_full(num_layers_ + layer, af::span, af::span) = af::moddims(h_r, af::dim4(1, batch_size, hidden_size_));
                c_full(num_layers_ + layer, af::span, af::span) = af::moddims(c_r, af::dim4(1, batch_size, hidden_size_));
            }

            // Update final hidden and cell states for forward direction.
            h_full(layer, af::span, af::span) = af::moddims(h, af::dim4(1, batch_size, hidden_size_));
            c_full(layer, af::span, af::span) = af::moddims(c, af::dim4(1, batch_size, hidden_size_));

            // Update stored states. h_full / c_full are
            // [num_layers * num_directions, batch_size, hidden_size]
            // — 3D, needs the row-major helper. Otherwise stateful
            // LSTM (h_n fed back as initial state on next Forward)
            // sees scrambled axes.
            h_n_ = AfToTensor3DRowMajor(h_full);
            c_n_ = AfToTensor3DRowMajor(c_full);

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

        // Use the 3D-aware helper — bare AfToTensor on a semantic
        // [batch, seq, hidden] AF array would produce a row-major
        // Tensor with the axis order scrambled. Keep this in sync
        // with the TensorToAf3DRowMajor at the entry.
        return AfToTensor3DRowMajor(output);
    } catch (const af::exception& e) {
        BackendDebugHooks::EmitDebugEvent(
            "LSTMLayer::Forward",
            std::string("ArrayFire fallback: ") + e.what() +
            (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
        spdlog::warn("ArrayFire LSTMLayer::Forward failed: {}, falling back to CPU", e.what());
    }
#endif

    // CPU fallback implementation. This path ALSO populates cached_inputs_,
    // cached_gates_, cached_hidden_states_, cached_cell_states_ in row-major
    // CPU-friendly layout so LSTMLayer::Backward (CPU BPTT below) can read
    // them. These caches are distinct from the AF-shaped ones the disabled
    // AF Forward would have written — CPU backward operates on row-major
    // [seq_len, batch, ...] tensors throughout, no AF reshape involved.
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

    if (bidirectional_) {
        float* h_data = h_n_.Data<float>();
        float* c_data = c_n_.Data<float>();

        Tensor layer_input = input;
        cached_inputs_.clear();
        cached_gates_.clear();
        cached_hidden_states_.clear();
        cached_cell_states_.clear();
        cached_inputs_.reserve(static_cast<size_t>(num_layers_ * 2));
        cached_gates_.reserve(static_cast<size_t>(num_layers_ * 2));
        cached_hidden_states_.reserve(static_cast<size_t>(num_layers_ * 2));
        cached_cell_states_.reserve(static_cast<size_t>(num_layers_ * 2));

        for (int layer = 0; layer < num_layers_; ++layer) {
            Tensor init_h_fwd = Tensor::Zeros({batch_size, static_cast<size_t>(hidden_size_)});
            Tensor init_c_fwd = Tensor::Zeros({batch_size, static_cast<size_t>(hidden_size_)});
            Tensor init_h_rev = Tensor::Zeros({batch_size, static_cast<size_t>(hidden_size_)});
            Tensor init_c_rev = Tensor::Zeros({batch_size, static_cast<size_t>(hidden_size_)});
            const size_t fwd_state_idx = static_cast<size_t>(layer);
            const size_t rev_state_idx = static_cast<size_t>(num_layers_ + layer);
            for (size_t b = 0; b < batch_size; ++b) {
                for (int i = 0; i < hidden_size_; ++i) {
                    init_h_fwd.Data<float>()[b * hidden_size_ + i] =
                        h_data[fwd_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i];
                    init_c_fwd.Data<float>()[b * hidden_size_ + i] =
                        c_data[fwd_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i];
                    init_h_rev.Data<float>()[b * hidden_size_ + i] =
                        h_data[rev_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i];
                    init_c_rev.Data<float>()[b * hidden_size_ + i] =
                        c_data[rev_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i];
                }
            }

            auto fwd = RunLSTMCpuDirectionForward(
                layer_input,
                W_ih_[layer], W_hh_[layer],
                b_ih_[layer], b_hh_[layer],
                init_h_fwd, init_c_fwd,
                hidden_size_, batch_first_, false);
            auto rev = RunLSTMCpuDirectionForward(
                layer_input,
                W_ih_reverse_[layer], W_hh_reverse_[layer],
                b_ih_reverse_[layer], b_hh_reverse_[layer],
                init_h_rev, init_c_rev,
                hidden_size_, batch_first_, true);

            Tensor layer_output = batch_first_
                ? Tensor::Zeros({batch_size, seq_len, static_cast<size_t>(hidden_size_ * 2)})
                : Tensor::Zeros({seq_len, batch_size, static_cast<size_t>(hidden_size_ * 2)});
            float* out_data = layer_output.Data<float>();
            const float* fwd_data = fwd.output.Data<float>();
            const float* rev_data = rev.output.Data<float>();
            for (size_t t = 0; t < seq_len; ++t) {
                for (size_t b = 0; b < batch_size; ++b) {
                    for (int i = 0; i < hidden_size_; ++i) {
                        const float fwd_v = fwd_data[t * batch_size * hidden_size_ + b * hidden_size_ + i];
                        const float rev_v = rev_data[t * batch_size * hidden_size_ + b * hidden_size_ + i];
                        if (batch_first_) {
                            out_data[b * seq_len * hidden_size_ * 2 + t * hidden_size_ * 2 + i] = fwd_v;
                            out_data[b * seq_len * hidden_size_ * 2 + t * hidden_size_ * 2 + hidden_size_ + i] = rev_v;
                        } else {
                            out_data[t * batch_size * hidden_size_ * 2 + b * hidden_size_ * 2 + i] = fwd_v;
                            out_data[t * batch_size * hidden_size_ * 2 + b * hidden_size_ * 2 + hidden_size_ + i] = rev_v;
                        }
                    }
                }
            }

            cached_inputs_.push_back(std::move(fwd.input_cache));
            cached_gates_.push_back(std::move(fwd.gate_cache));
            cached_hidden_states_.push_back(std::move(fwd.hidden_cache));
            cached_cell_states_.push_back(std::move(fwd.cell_cache));
            cached_inputs_.push_back(std::move(rev.input_cache));
            cached_gates_.push_back(std::move(rev.gate_cache));
            cached_hidden_states_.push_back(std::move(rev.hidden_cache));
            cached_cell_states_.push_back(std::move(rev.cell_cache));

            for (size_t b = 0; b < batch_size; ++b) {
                for (int i = 0; i < hidden_size_; ++i) {
                    h_data[fwd_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i] =
                        fwd.final_hidden.Data<float>()[b * hidden_size_ + i];
                    c_data[fwd_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i] =
                        fwd.final_cell.Data<float>()[b * hidden_size_ + i];
                    h_data[rev_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i] =
                        rev.final_hidden.Data<float>()[b * hidden_size_ + i];
                    c_data[rev_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i] =
                        rev.final_cell.Data<float>()[b * hidden_size_ + i];
                }
            }

            layer_input = layer_output;

            if (layer < num_layers_ - 1 && dropout_ > 0.0f && training_) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                static thread_local std::mt19937 rng{std::random_device{}()};
                float* layer_out_ptr = layer_input.Data<float>();
                for (size_t i = 0; i < layer_output.NumElements(); ++i) {
                    const float keep = dist(rng) > dropout_ ? 1.0f : 0.0f;
                    layer_out_ptr[i] = keep > 0.0f ? layer_out_ptr[i] / (1.0f - dropout_) : 0.0f;
                }
            }
        }

        return layer_input;
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

    // Reset caches at the top of every forward pass so BPTT reads the
    // current run's state, not a stale one. CPU backward below branches
    // on emptiness to decide whether to run BPTT or fall through.
    cached_inputs_.clear();
    cached_gates_.clear();
    cached_hidden_states_.clear();
    cached_cell_states_.clear();
    cached_inputs_.reserve(num_layers_);
    cached_gates_.reserve(num_layers_);
    cached_hidden_states_.reserve(num_layers_);
    cached_cell_states_.reserve(num_layers_);

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

        // Per-layer caches. Row-major:
        //   input  [seq_len, batch, layer_input_size]
        //   gates  [seq_len, batch, 4 * hidden_size]        (pre-activations)
        //   h      [seq_len + 1, batch, hidden_size]        (index 0 = h_0)
        //   c      [seq_len + 1, batch, hidden_size]        (index 0 = c_0)
        Tensor layer_input_cache = Tensor::Zeros(
            {seq_len, batch_size, layer_input_size});
        Tensor layer_gates_cache = Tensor::Zeros(
            {seq_len, batch_size, static_cast<size_t>(gate_size)});
        Tensor layer_h_cache = Tensor::Zeros(
            {seq_len + 1, batch_size, static_cast<size_t>(hidden_size_)});
        Tensor layer_c_cache = Tensor::Zeros(
            {seq_len + 1, batch_size, static_cast<size_t>(hidden_size_)});
        float* in_cache_data = layer_input_cache.Data<float>();
        float* gate_cache_data = layer_gates_cache.Data<float>();
        float* h_cache_data = layer_h_cache.Data<float>();
        float* c_cache_data = layer_c_cache.Data<float>();

        // Seed h_0 / c_0 at t=0 from h_n_ / c_n_ for all batches.
        for (size_t b = 0; b < batch_size; b++) {
            for (int i = 0; i < hidden_size_; i++) {
                h_cache_data[0 * batch_size * hidden_size_ + b * hidden_size_ + i] =
                    h_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i];
                c_cache_data[0 * batch_size * hidden_size_ + b * hidden_size_ + i] =
                    c_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i];
            }
        }

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

                // Snapshot pre-activation gates + current input for BPTT.
                // gates_cache_data shape [seq_len, batch, 4H], index by
                // (t, b, g) = t*batch*4H + b*4H + g.
                for (int g = 0; g < gate_size; g++) {
                    gate_cache_data[t * batch_size * gate_size + b * gate_size + g] = gates[g];
                }
                for (size_t k = 0; k < layer_input_size; k++) {
                    in_cache_data[t * batch_size * layer_input_size + b * layer_input_size + k]
                        = x_ptr[k];
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

                // Snapshot h_t, c_t (post-step state) at cache index t+1.
                for (int i = 0; i < hidden_size_; i++) {
                    h_cache_data[(t + 1) * batch_size * hidden_size_ + b * hidden_size_ + i] = h[i];
                    c_cache_data[(t + 1) * batch_size * hidden_size_ + b * hidden_size_ + i] = c[i];
                }
            }

            for (int i = 0; i < hidden_size_; i++) {
                h_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i] = h[i];
                c_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i] = c[i];
            }
        }

        cached_inputs_.push_back(std::move(layer_input_cache));
        cached_gates_.push_back(std::move(layer_gates_cache));
        cached_hidden_states_.push_back(std::move(layer_h_cache));
        cached_cell_states_.push_back(std::move(layer_c_cache));

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
    // Guard: caches populated by the CPU Forward path above. The AF
    // Forward path is still gated off (column-major reorder bug) so we
    // never reach a state where caches exist but weren't built by CPU.
    // Empty caches mean Forward was never called; return zeros sized
    // by the input we saw (if any) so upstream grad flow is dimension-
    // safe rather than throwing.
    if (cached_inputs_.empty() || cached_gates_.empty() ||
        cached_hidden_states_.empty() || cached_cell_states_.empty()) {
        static std::atomic<bool> warned_once{false};
        if (!warned_once.exchange(true)) {
            spdlog::warn("LSTMLayer::Backward: caches empty (Forward not run?) "
                         "— returning zero gradients. This warning fires once.");
        }
        if (cached_input_.NumElements() > 0) {
            return Tensor::Zeros(cached_input_.Shape());
        }
        return Tensor::Zeros(grad_output.Shape());
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // AF Backward — uses row-major caches that AF Forward (or CPU
    // Forward) populated via AfToTensor3DRowMajor. Reads them back via
    // TensorToAf3DRowMajor so the AF-side semantic axes match the AF
    // Forward's working layout. Same slice-shape moddims pattern that
    // Forward needed: every (t, span, span) = X assignment to a 3D
    // proxy needs X wrapped in moddims with a leading 1 dim.
    //
    // On AF exception, falls through to the CPU BPTT below — same
    // dual-path pattern as Forward. Loss numerically agrees with the
    // CPU path within fp32 noise on the mini sentiment LSTM smoke test.
    //
    // Bidirectional Backward NOT implemented yet (forward-direction
    // only, like the legacy code). Bidirectional graphs will fall
    // through to CPU.
    constexpr bool kAfBackwardEnabled = true;
    if (kAfBackwardEnabled && !bidirectional_) try {
        af::array dout = TensorToAf3DRowMajor(grad_output);

        // Convert to seq-first if batch_first.
        if (batch_first_) {
            dout = af::reorder(dout, 1, 0, 2);
        }

        const dim_t seq_len = dout.dims(0);
        const dim_t batch_size = dout.dims(1);
        const int gate_size = 4 * hidden_size_;

        af::array layer_grad = dout;

        for (int layer = num_layers_ - 1; layer >= 0; layer--) {
            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);

            // Caches are row-major Tensors written by AF/CPU Forward —
            // bring them back to AF column-major with semantic axes
            // matching the Forward pass.
            af::array cached_input = TensorToAf3DRowMajor(cached_inputs_[layer]);
            af::array cached_gates = TensorToAf3DRowMajor(cached_gates_[layer]);
            af::array cached_h = TensorToAf3DRowMajor(cached_hidden_states_[layer]);
            af::array cached_c = TensorToAf3DRowMajor(cached_cell_states_[layer]);

            const dim_t layer_input_size = cached_input.dims(2);

            af::array dW_ih = af::constant(0.0f, W_ih.dims());
            af::array dW_hh = af::constant(0.0f, W_hh.dims());
            af::array db_ih = af::constant(0.0f, af::dim4(gate_size));
            af::array db_hh = af::constant(0.0f, af::dim4(gate_size));

            af::array dh_next = af::constant(0.0f, af::dim4(batch_size, hidden_size_));
            af::array dc_next = af::constant(0.0f, af::dim4(batch_size, hidden_size_));

            af::array d_layer_input = af::constant(0.0f, cached_input.dims());

            for (dim_t t = seq_len - 1; t >= 0; t--) {
                // h_prev / c_prev are at cache index t (state BEFORE step t).
                // c_t is at cache index t+1 (state AFTER step t).
                // gates is at cache index t (pre-activation gates from step t).
                af::array h_prev = af::moddims(cached_h(t, af::span, af::span),
                                                af::dim4(batch_size, hidden_size_));
                af::array c_prev = af::moddims(cached_c(t, af::span, af::span),
                                                af::dim4(batch_size, hidden_size_));
                af::array c_t    = af::moddims(cached_c(t + 1, af::span, af::span),
                                                af::dim4(batch_size, hidden_size_));
                af::array gates  = af::moddims(cached_gates(t, af::span, af::span),
                                                af::dim4(batch_size, gate_size));

                // Recompute gate activations from cached pre-activations.
                af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, hidden_size_ - 1)));
                af::array f_gate = af::sigmoid(gates(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1)));
                af::array g_gate = af::tanh   (gates(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1)));
                af::array o_gate = af::sigmoid(gates(af::span, af::seq(3 * hidden_size_, 4 * hidden_size_ - 1)));

                // Output gradient for this timestep + carry-over from t+1.
                af::array dh = af::moddims(layer_grad(t, af::span, af::span),
                                            af::dim4(batch_size, hidden_size_));
                dh = dh + dh_next;

                // h = o * tanh(c_t)
                af::array tanh_c = af::tanh(c_t);
                af::array do_pre = dh * tanh_c * o_gate * (1.0f - o_gate);
                af::array dc     = dh * o_gate * (1.0f - tanh_c * tanh_c) + dc_next;

                // c_t = f * c_prev + i * g
                af::array df_pre = dc * c_prev * f_gate * (1.0f - f_gate);
                af::array di_pre = dc * g_gate * i_gate * (1.0f - i_gate);
                af::array dg_pre = dc * i_gate * (1.0f - g_gate * g_gate);
                dc_next = dc * f_gate;

                // Pre-activation gradients in [i | f | g | o] order, matching
                // the cache layout from Forward.
                af::array dgates = af::join(1, di_pre, df_pre, dg_pre, do_pre);

                // Weight + bias grad accumulation.
                af::array x_t = af::moddims(cached_input(t, af::span, af::span),
                                             af::dim4(batch_size, layer_input_size));
                dW_ih = dW_ih + af::matmul(af::transpose(dgates), x_t);
                dW_hh = dW_hh + af::matmul(af::transpose(dgates), h_prev);

                af::array sum_g = af::sum(dgates, 0);  // [1, 4H]
                db_ih = db_ih + af::moddims(sum_g, af::dim4(gate_size));
                db_hh = db_hh + af::moddims(sum_g, af::dim4(gate_size));

                // dx_t = dgates @ W_ih    (shape [batch, layer_input_size])
                // Slice assignment to a rank-3 (1, batch, input) proxy needs
                // explicit moddims to add the leading 1 — same fix Forward
                // uses for h_states/c_states/all_gates writes.
                af::array dx_t = af::matmul(dgates, W_ih);
                d_layer_input(t, af::span, af::span) = af::moddims(
                    dx_t, af::dim4(1, batch_size, layer_input_size));

                // dh_prev = dgates @ W_hh
                dh_next = af::matmul(dgates, W_hh);
            }

            // Stash per-layer weight grads.
            grad_W_ih_[layer] = AfToTensor(dW_ih);
            grad_W_hh_[layer] = AfToTensor(dW_hh);
            grad_b_ih_[layer] = AfToTensor(db_ih);
            grad_b_hh_[layer] = AfToTensor(db_hh);

            // Pass dx down to the next layer (which becomes its layer_grad).
            layer_grad = d_layer_input;
        }

        // Convert back to batch_first if needed.
        if (batch_first_) {
            layer_grad = af::reorder(layer_grad, 1, 0, 2);
        }

        return AfToTensor3DRowMajor(layer_grad);
    } catch (const af::exception& e) {
        BackendDebugHooks::EmitDebugEvent(
            "LSTMLayer::Backward",
            std::string("ArrayFire fallback: ") + e.what() +
            (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
        spdlog::warn("ArrayFire LSTMLayer::Backward failed: {}, falling back to CPU", e.what());
    }
#endif

    // CPU BPTT. Reads the row-major caches populated by CPU Forward:
    //   cached_inputs_[L]          [seq_len, batch, layer_input_size]
    //   cached_gates_[L]           [seq_len, batch, 4 * hidden_size]   (pre-activations)
    //   cached_hidden_states_[L]   [seq_len + 1, batch, hidden_size]   (idx 0 = h_0)
    //   cached_cell_states_[L]     [seq_len + 1, batch, hidden_size]   (idx 0 = c_0)
    //
    // Walks layers in reverse (num_layers - 1 → 0), and within each
    // layer walks timesteps in reverse (seq_len - 1 → 0). Accumulates
    // dW_ih, dW_hh, db_ih, db_hh per layer and builds a dL/d(input)
    // tensor that feeds upstream (Embedding in the typical text graph).
    //
    // Gate layout matches Forward: [i | f | g | o] in contiguous slices
    // of 4*hidden_size.

    const auto& input_shape = cached_input_.Shape();
    size_t batch_size, seq_len, input_dim;
    if (batch_first_) {
        batch_size = input_shape[0];
        seq_len    = input_shape[1];
        input_dim  = input_shape[2];
    } else {
        seq_len    = input_shape[0];
        batch_size = input_shape[1];
        input_dim  = input_shape[2];
    }

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto tanh_f  = [](float x) { return std::tanh(x); };

    const int H = hidden_size_;
    const int G = 4 * H;

    // Ensure grad buffers exist and are sized correctly.
    if (static_cast<int>(grad_W_ih_.size()) < num_layers_) grad_W_ih_.resize(num_layers_);
    if (static_cast<int>(grad_W_hh_.size()) < num_layers_) grad_W_hh_.resize(num_layers_);
    if (static_cast<int>(grad_b_ih_.size()) < num_layers_) grad_b_ih_.resize(num_layers_);
    if (static_cast<int>(grad_b_hh_.size()) < num_layers_) grad_b_hh_.resize(num_layers_);
    if (static_cast<int>(grad_W_ih_reverse_.size()) < num_layers_) grad_W_ih_reverse_.resize(num_layers_);
    if (static_cast<int>(grad_W_hh_reverse_.size()) < num_layers_) grad_W_hh_reverse_.resize(num_layers_);
    if (static_cast<int>(grad_b_ih_reverse_.size()) < num_layers_) grad_b_ih_reverse_.resize(num_layers_);
    if (static_cast<int>(grad_b_hh_reverse_.size()) < num_layers_) grad_b_hh_reverse_.resize(num_layers_);

    if (bidirectional_) {
        Tensor layer_grad = Tensor::Zeros(cached_input_.Shape());

        for (int layer = num_layers_ - 1; layer >= 0; --layer) {
            const size_t fwd_idx = static_cast<size_t>(layer * 2);
            const size_t rev_idx = fwd_idx + 1;

            auto fwd = RunLSTMCpuDirectionBackward(
                grad_output,
                0,
                cached_inputs_[fwd_idx],
                cached_gates_[fwd_idx],
                cached_hidden_states_[fwd_idx],
                cached_cell_states_[fwd_idx],
                W_ih_[layer],
                W_hh_[layer],
                hidden_size_,
                batch_first_,
                false);
            auto rev = RunLSTMCpuDirectionBackward(
                grad_output,
                static_cast<size_t>(hidden_size_),
                cached_inputs_[rev_idx],
                cached_gates_[rev_idx],
                cached_hidden_states_[rev_idx],
                cached_cell_states_[rev_idx],
                W_ih_reverse_[layer],
                W_hh_reverse_[layer],
                hidden_size_,
                batch_first_,
                true);

            grad_W_ih_[layer] = std::move(fwd.grad_W_ih);
            grad_W_hh_[layer] = std::move(fwd.grad_W_hh);
            grad_b_ih_[layer] = std::move(fwd.grad_b_ih);
            grad_b_hh_[layer] = std::move(fwd.grad_b_hh);
            grad_W_ih_reverse_[layer] = std::move(rev.grad_W_ih);
            grad_W_hh_reverse_[layer] = std::move(rev.grad_W_hh);
            grad_b_ih_reverse_[layer] = std::move(rev.grad_b_ih);
            grad_b_hh_reverse_[layer] = std::move(rev.grad_b_hh);

            const float* fwd_dx = fwd.input_grad.Data<float>();
            const float* rev_dx = rev.input_grad.Data<float>();
            float* layer_dx = layer_grad.Data<float>();
            const size_t elem_count = layer_grad.NumElements();
            for (size_t i = 0; i < elem_count; ++i) {
                layer_dx[i] = fwd_dx[i] + rev_dx[i];
            }
        }

        return layer_grad;
    }

    // Upstream gradient entering the top layer. Shape is the OUTPUT
    // shape (batch_first convention matches the Forward output).
    // Reshape conceptually to [seq_len, batch, H] row-major regardless
    // of batch_first (we index with an explicit if inside the loop).
    const float* dout = grad_output.Data<float>();

    // `layer_grad` holds the gradient flowing into the CURRENT layer's
    // output (i.e. dL/d(layer_output)). For the topmost LSTM layer
    // it's dout; for inner layers it's the dx computed by the later
    // layer. Row-major [seq_len, batch, hidden_size].
    std::vector<float> layer_grad(seq_len * batch_size * H, 0.0f);
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (int i = 0; i < H; ++i) {
                float g = batch_first_
                    ? dout[b * seq_len * H + t * H + i]
                    : dout[t * batch_size * H + b * H + i];
                layer_grad[t * batch_size * H + b * H + i] = g;
            }
        }
    }

    // dL/d(input to this layer). Sized per-layer since input_dim varies.
    std::vector<float> d_layer_input;

    for (int layer = num_layers_ - 1; layer >= 0; --layer) {
        const size_t layer_input_size = (layer == 0)
            ? input_dim : static_cast<size_t>(H);

        const float* W_ih = W_ih_[layer].Data<float>();       // [4H, input_size]
        const float* W_hh = W_hh_[layer].Data<float>();       // [4H, H]
        const float* in_cache = cached_inputs_[layer].Data<float>();
        const float* gate_cache = cached_gates_[layer].Data<float>();
        const float* h_cache = cached_hidden_states_[layer].Data<float>();
        const float* c_cache = cached_cell_states_[layer].Data<float>();

        // Weight gradients for this layer, zero-initialized.
        std::vector<float> dW_ih(G * layer_input_size, 0.0f);
        std::vector<float> dW_hh(G * H, 0.0f);
        std::vector<float> db_ih(G, 0.0f);
        std::vector<float> db_hh(G, 0.0f);

        d_layer_input.assign(seq_len * batch_size * layer_input_size, 0.0f);

        for (size_t b = 0; b < batch_size; ++b) {
            std::vector<float> dh_next(H, 0.0f);
            std::vector<float> dc_next(H, 0.0f);

            for (int64_t t = static_cast<int64_t>(seq_len) - 1; t >= 0; --t) {
                const size_t gate_off = t * batch_size * G + b * G;
                const size_t h_prev_off = t * batch_size * H + b * H;      // cache idx t
                const size_t c_prev_off = t * batch_size * H + b * H;
                const size_t c_t_off = (t + 1) * batch_size * H + b * H;   // cache idx t+1
                const size_t in_off = t * batch_size * layer_input_size + b * layer_input_size;
                const size_t lg_off = t * batch_size * H + b * H;

                // dh combines upstream output gradient + carried-over
                // gradient from the next timestep's hidden dependency.
                std::vector<float> dh(H);
                for (int i = 0; i < H; ++i) {
                    dh[i] = layer_grad[lg_off + i] + dh_next[i];
                }

                // Recompute gate activations (cheaper than caching them).
                std::vector<float> i_g(H), f_g(H), g_g(H), o_g(H);
                for (int i = 0; i < H; ++i) {
                    i_g[i] = sigmoid(gate_cache[gate_off + i]);
                    f_g[i] = sigmoid(gate_cache[gate_off + H + i]);
                    g_g[i] = tanh_f (gate_cache[gate_off + 2 * H + i]);
                    o_g[i] = sigmoid(gate_cache[gate_off + 3 * H + i]);
                }

                // h = o * tanh(c_t)
                // c_t = f * c_prev + i * g
                std::vector<float> dgates(G, 0.0f);
                for (int i = 0; i < H; ++i) {
                    const float c_t = c_cache[c_t_off + i];
                    const float c_prev = c_cache[c_prev_off + i];
                    const float tanh_c = tanh_f(c_t);

                    // dh = dh, do_pre = dh * tanh(c) * sigmoid'(o_gate_pre)
                    const float do_pre = dh[i] * tanh_c * o_g[i] * (1.0f - o_g[i]);

                    // dc accumulates from this step's output and carry-over.
                    const float dc = dh[i] * o_g[i] * (1.0f - tanh_c * tanh_c)
                                    + dc_next[i];

                    const float df_pre = dc * c_prev * f_g[i] * (1.0f - f_g[i]);
                    const float di_pre = dc * g_g[i] * i_g[i] * (1.0f - i_g[i]);
                    const float dg_pre = dc * i_g[i] * (1.0f - g_g[i] * g_g[i]);

                    dc_next[i] = dc * f_g[i];

                    // Pre-activation gradients in [i | f | g | o] order.
                    dgates[i]           = di_pre;
                    dgates[H + i]       = df_pre;
                    dgates[2 * H + i]   = dg_pre;
                    dgates[3 * H + i]   = do_pre;
                }

                // Accumulate weight and bias grads.
                // dW_ih [G, layer_input_size] += outer(dgates, x_t)
                // dW_hh [G, H]                 += outer(dgates, h_prev)
                // db_ih [G]                   += dgates  (same for db_hh)
                for (int g = 0; g < G; ++g) {
                    const float dg = dgates[g];
                    db_ih[g] += dg;
                    db_hh[g] += dg;
                    for (size_t k = 0; k < layer_input_size; ++k) {
                        dW_ih[g * layer_input_size + k]
                            += dg * in_cache[in_off + k];
                    }
                    for (int k = 0; k < H; ++k) {
                        dW_hh[g * H + k] += dg * h_cache[h_prev_off + k];
                    }
                }

                // dx_t = dgates @ W_ih        (shape [layer_input_size])
                // dh_prev = dgates @ W_hh     (shape [H])
                for (size_t k = 0; k < layer_input_size; ++k) {
                    float s = 0.0f;
                    for (int g = 0; g < G; ++g) {
                        s += dgates[g] * W_ih[g * layer_input_size + k];
                    }
                    d_layer_input[in_off + k] = s;
                }
                for (int k = 0; k < H; ++k) {
                    float s = 0.0f;
                    for (int g = 0; g < G; ++g) {
                        s += dgates[g] * W_hh[g * H + k];
                    }
                    dh_next[k] = s;
                }
            }
        }

        // Stash per-layer weight grads.
        grad_W_ih_[layer] = Tensor({static_cast<size_t>(G), layer_input_size},
                                   dW_ih.data());
        grad_W_hh_[layer] = Tensor({static_cast<size_t>(G), static_cast<size_t>(H)},
                                   dW_hh.data());
        grad_b_ih_[layer] = Tensor({static_cast<size_t>(G)}, db_ih.data());
        grad_b_hh_[layer] = Tensor({static_cast<size_t>(G)}, db_hh.data());

        // This layer's dL/d(input) becomes the next iteration's
        // dL/d(output) for the layer below. Sizes must match: each
        // inner LSTM layer has input_size = H, so copying into
        // layer_grad is safe only when layer > 0. For layer == 0 we
        // exit the loop with d_layer_input sized [seq_len, batch, input_dim]
        // which we convert back to the user's layout below.
        if (layer > 0) {
            layer_grad = d_layer_input;  // size = seq_len * batch * H
        }
    }

    // Convert d_layer_input from row-major [seq_len, batch, input_dim]
    // to whatever batch_first says.
    Tensor dx = Tensor::Zeros(cached_input_.Shape());
    float* dx_data = dx.Data<float>();
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t k = 0; k < input_dim; ++k) {
                const float v = d_layer_input[t * batch_size * input_dim + b * input_dim + k];
                if (batch_first_) {
                    dx_data[b * seq_len * input_dim + t * input_dim + k] = v;
                } else {
                    dx_data[t * batch_size * input_dim + b * input_dim + k] = v;
                }
            }
        }
    }
    return dx;
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

    // Hoisted weight init guard — same pattern as LSTMLayer::Forward.
    // GRULayer's constructor calls InitializeWeights() which uses the AF
    // backend; if AF init silently failed the weight tensors carry null
    // data. CPU path needs valid weights before computing anything.
    {
        const auto& shape = input.Shape();
        size_t input_dim = shape.size() == 3
            ? (batch_first_ ? shape[2] : shape[2]) : 0;
        int num_directions = bidirectional_ ? 2 : 1;
        if (W_ih_.empty() || W_ih_[0].Data<float>() == nullptr) {
            for (int layer = 0; layer < num_layers_; layer++) {
                size_t layer_input_size = (layer == 0) ? input_dim
                    : static_cast<size_t>(hidden_size_ * num_directions);
                size_t gate_size = static_cast<size_t>(3 * hidden_size_);
                W_ih_[layer] = Tensor::Random({gate_size, layer_input_size});
                W_hh_[layer] = Tensor::Random({gate_size, static_cast<size_t>(hidden_size_)});
                b_ih_[layer] = Tensor::Zeros({gate_size});
                b_hh_[layer] = Tensor::Zeros({gate_size});
            }
        }
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!bidirectional_) try {
        af::array x = TensorToAf3DRowMajor(input);
        if (batch_first_) {
            x = af::reorder(x, 1, 0, 2);
        }

        const dim_t batch_size = x.dims(1);
        const dim_t seq_len = x.dims(0);

        const auto h_shape = h_n_.Shape();
        const bool h_needs_init = h_n_.NumElements() == 0 ||
                                  h_n_.Data<float>() == nullptr ||
                                  h_shape.size() != 3 ||
                                  h_shape[0] != static_cast<size_t>(num_layers_) ||
                                  h_shape[1] != static_cast<size_t>(batch_size) ||
                                  h_shape[2] != static_cast<size_t>(hidden_size_);
        if (h_needs_init) {
            h_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_),
                                  static_cast<size_t>(batch_size),
                                  static_cast<size_t>(hidden_size_)});
        }

        cached_inputs_.clear();
        cached_gates_.clear();
        cached_hidden_states_.clear();
        cached_inputs_.reserve(num_layers_);
        cached_gates_.reserve(num_layers_);
        cached_hidden_states_.reserve(num_layers_);

        af::array layer_input = x;

        for (int layer = 0; layer < num_layers_; ++layer) {
            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);
            af::array b_ih = TensorToAf(b_ih_[layer]);
            af::array b_hh = TensorToAf(b_hh_[layer]);

            const dim_t layer_input_size = layer_input.dims(2);
            af::array input_flat = af::moddims(layer_input,
                                               af::dim4(seq_len * batch_size, layer_input_size));
            af::array input_proj = af::matmul(input_flat, af::transpose(W_ih));
            input_proj = input_proj + af::tile(af::transpose(b_ih),
                                               static_cast<unsigned int>(seq_len * batch_size));
            input_proj = af::moddims(input_proj, af::dim4(seq_len, batch_size, 3 * hidden_size_));

            af::array h_full = TensorToAf3DRowMajor(h_n_);
            af::array h = af::moddims(h_full(layer, af::span, af::span),
                                      af::dim4(batch_size, hidden_size_));

            af::array layer_output = af::constant(0.0f, af::dim4(seq_len, batch_size, hidden_size_));
            af::array layer_gates = af::constant(0.0f, af::dim4(seq_len, batch_size, 4 * hidden_size_));
            af::array layer_h_states = af::constant(0.0f, af::dim4(seq_len + 1, batch_size, hidden_size_));
            layer_h_states(0, af::span, af::span) =
                af::moddims(h, af::dim4(1, batch_size, hidden_size_));

            for (dim_t t = 0; t < seq_len; ++t) {
                af::array x_t = af::moddims(input_proj(t, af::span, af::span),
                                            af::dim4(batch_size, 3 * hidden_size_));
                af::array h_proj = af::matmul(h, af::transpose(W_hh));
                h_proj = h_proj + af::tile(af::transpose(b_hh),
                                           static_cast<unsigned int>(batch_size));

                af::array x_r = x_t(af::span, af::seq(0, hidden_size_ - 1));
                af::array x_z = x_t(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1));
                af::array x_n = x_t(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1));

                af::array h_r = h_proj(af::span, af::seq(0, hidden_size_ - 1));
                af::array h_z = h_proj(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1));
                af::array h_n = h_proj(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1));

                af::array r = af::sigmoid(x_r + h_r);
                af::array z = af::sigmoid(x_z + h_z);
                af::array n = af::tanh(x_n + r * h_n);
                h = (1.0f - z) * n + z * h;

                af::array gates_t = af::join(1, r, z, n, h_n);
                layer_output(t, af::span, af::span) =
                    af::moddims(h, af::dim4(1, batch_size, hidden_size_));
                layer_gates(t, af::span, af::span) =
                    af::moddims(gates_t, af::dim4(1, batch_size, 4 * hidden_size_));
                layer_h_states(t + 1, af::span, af::span) =
                    af::moddims(h, af::dim4(1, batch_size, hidden_size_));
            }

            af::array h_full_out = TensorToAf3DRowMajor(h_n_);
            h_full_out(layer, af::span, af::span) =
                af::moddims(h, af::dim4(1, batch_size, hidden_size_));
            h_n_ = AfToTensor3DRowMajor(h_full_out);

            cached_inputs_.push_back(AfToTensor3DRowMajor(layer_input));
            cached_gates_.push_back(AfToTensor3DRowMajor(layer_gates));
            cached_hidden_states_.push_back(AfToTensor3DRowMajor(layer_h_states));

            layer_input = layer_output;
        }

        if (batch_first_) {
            layer_input = af::reorder(layer_input, 1, 0, 2);
        }

        return AfToTensor3DRowMajor(layer_input);
    } catch (const af::exception& e) {
        BackendDebugHooks::EmitDebugEvent(
            "GRULayer::Forward",
            std::string("ArrayFire fallback: ") + e.what() +
            (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
        spdlog::warn("ArrayFire GRULayer::Forward failed: {}, falling back to CPU", e.what());
    }
#endif

#if 0
    // ArrayFire GRU Forward — currently disabled. Has the same family of
    // bugs LSTM AF Forward had pre-2026-04-16: 3D column-major scrambling
    // at TensorToAf boundary, slice assignment shape mismatches at
    // h_states(t,...) = h writes, and r_gate/z_gate slot ordering bugs
    // that the buggy AF Backward also depended on. Fixing AF GRU is a
    // perf follow-up — needs the same TensorToAf3DRowMajor +
    // af::moddims(rhs, dim4(1,batch,hidden)) treatment LSTM got. The
    // original AF code is preserved for reference once the perf fix is
    // attempted.
    try {
        af::array x = TensorToAf(input);
        dim_t batch_size, seq_len, input_dim;
        if (batch_first_) {
            batch_size = x.dims(0); seq_len = x.dims(1); input_dim = x.dims(2);
            x = af::reorder(x, 1, 0, 2);
        } else {
            seq_len = x.dims(0); batch_size = x.dims(1); input_dim = x.dims(2);
        }
        // ... (legacy AF body removed for brevity — see git history at
        //      0dc11e1a / pre-fix for the original implementation.)
    } catch (const af::exception& e) {
        BackendDebugHooks::EmitDebugEvent(
            "GRULayer::Forward",
            std::string("ArrayFire fallback: ") + e.what() +
            (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
        spdlog::warn("ArrayFire GRULayer::Forward failed: {}, falling back to CPU", e.what());
    }
#endif

    // CPU fallback implementation. Populates cached_inputs_, cached_gates_,
    // cached_hidden_states_ in row-major [seq_len, batch, ...] layout so
    // GRULayer::Backward (CPU BPTT below) can read them. Mirror of the
    // LSTM CPU Forward + cache layout, adapted for GRU's 3-gate structure.
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

    // Reset caches at the top of every forward pass so BPTT reads the
    // current run's state, not a stale one.
    cached_inputs_.clear();
    cached_gates_.clear();
    cached_hidden_states_.clear();
    cached_inputs_.reserve(num_layers_);
    cached_gates_.reserve(num_layers_);
    cached_hidden_states_.reserve(num_layers_);

    Tensor layer_input = input;
    size_t layer_input_size = input_dim;

    for (int layer = 0; layer < num_layers_; layer++) {
        const float* W_ih = W_ih_[layer].Data<float>();
        const float* W_hh = W_hh_[layer].Data<float>();
        const float* b_ih = b_ih_[layer].Data<float>();
        const float* b_hh = b_hh_[layer].Data<float>();
        const int H = hidden_size_;
        const int G = 3 * H;

        Tensor layer_output = Tensor::Zeros({seq_len, batch_size, static_cast<size_t>(H)});
        float* layer_out = layer_output.Data<float>();
        const float* layer_in = layer_input.Data<float>();

        // Per-layer caches, row-major:
        //   input  [seq_len, batch, layer_input_size]
        //   gates  [seq_len, batch, 4 * H]   layout per (t,b):
        //                [0..H)   r post-sigmoid
        //                [H..2H)  z post-sigmoid
        //                [2H..3H) n post-tanh
        //                [3H..4H) hn_pre   = b_hh_n + W_hh_n @ h_prev
        //                                    (the unmodulated h-side
        //                                     projection feeding n; saved
        //                                     because BPTT needs both r and
        //                                     hn_pre to split d_n into
        //                                     x-side and h-side parts)
        //   h      [seq_len + 1, batch, H]   (idx 0 = h_0)
        Tensor layer_input_cache = Tensor::Zeros(
            {seq_len, batch_size, layer_input_size});
        Tensor layer_gates_cache = Tensor::Zeros(
            {seq_len, batch_size, static_cast<size_t>(4 * H)});
        Tensor layer_h_cache = Tensor::Zeros(
            {seq_len + 1, batch_size, static_cast<size_t>(H)});
        float* in_cache_data = layer_input_cache.Data<float>();
        float* gate_cache_data = layer_gates_cache.Data<float>();
        float* h_cache_data = layer_h_cache.Data<float>();

        // Seed h_0 at cache index 0 from h_n_ for all batches.
        for (size_t b = 0; b < batch_size; b++) {
            for (int i = 0; i < H; i++) {
                h_cache_data[0 * batch_size * H + b * H + i] =
                    h_data[layer * batch_size * H + b * H + i];
            }
        }

        for (size_t b = 0; b < batch_size; b++) {
            std::vector<float> h(H);
            for (int i = 0; i < H; i++) {
                h[i] = h_data[layer * batch_size * H + b * H + i];
            }

            std::vector<float> x_proj(G), h_proj(G);

            for (size_t t = 0; t < seq_len; t++) {
                const float* x_ptr;
                if (layer == 0) {
                    if (batch_first_) x_ptr = input_data + b * seq_len * input_dim + t * input_dim;
                    else x_ptr = input_data + t * batch_size * input_dim + b * input_dim;
                } else {
                    x_ptr = layer_in + t * batch_size * layer_input_size + b * layer_input_size;
                }

                // x_proj[g] = b_ih[g] + W_ih[g,:] · x
                // h_proj[g] = b_hh[g] + W_hh[g,:] · h
                for (int g = 0; g < G; g++) {
                    float xs = b_ih[g];
                    for (size_t k = 0; k < layer_input_size; k++)
                        xs += W_ih[g * layer_input_size + k] * x_ptr[k];
                    x_proj[g] = xs;

                    float hs = b_hh[g];
                    for (int k = 0; k < H; k++)
                        hs += W_hh[g * H + k] * h[k];
                    h_proj[g] = hs;
                }

                // Snapshot input for BPTT.
                for (size_t k = 0; k < layer_input_size; k++) {
                    in_cache_data[t * batch_size * layer_input_size + b * layer_input_size + k]
                        = x_ptr[k];
                }

                // Apply gate equations and snapshot post-activations.
                //   r = sigmoid(x_proj_r + h_proj_r)
                //   z = sigmoid(x_proj_z + h_proj_z)
                //   n = tanh(x_proj_n + r * h_proj_n)
                //   h_new = (1 - z) * n + z * h_prev
                for (int i = 0; i < H; i++) {
                    float r = sigmoid(x_proj[i] + h_proj[i]);
                    float z = sigmoid(x_proj[H + i] + h_proj[H + i]);
                    float hn_pre = h_proj[2 * H + i];           // unmodulated h-side
                    float n = tanh_f(x_proj[2 * H + i] + r * hn_pre);

                    const size_t off = t * batch_size * (4 * H) + b * (4 * H);
                    gate_cache_data[off + i]           = r;
                    gate_cache_data[off + H + i]       = z;
                    gate_cache_data[off + 2 * H + i]   = n;
                    gate_cache_data[off + 3 * H + i]   = hn_pre;

                    h[i] = (1.0f - z) * n + z * h[i];
                }

                for (int i = 0; i < H; i++)
                    layer_out[t * batch_size * H + b * H + i] = h[i];

                // Snapshot h_t at cache index t+1.
                for (int i = 0; i < H; i++) {
                    h_cache_data[(t + 1) * batch_size * H + b * H + i] = h[i];
                }
            }

            for (int i = 0; i < H; i++) {
                h_data[layer * batch_size * H + b * H + i] = h[i];
            }
        }

        cached_inputs_.push_back(std::move(layer_input_cache));
        cached_gates_.push_back(std::move(layer_gates_cache));
        cached_hidden_states_.push_back(std::move(layer_h_cache));

        layer_input = layer_output;
        layer_input_size = static_cast<size_t>(H);
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
    // Empty caches mean Forward was never called; return zeros sized by
    // the input we last saw so upstream grad flow is dimension-safe
    // rather than throwing. Mirror of LSTMLayer::Backward's guard.
    if (cached_inputs_.empty() || cached_gates_.empty() ||
        cached_hidden_states_.empty()) {
        static std::atomic<bool> warned_once{false};
        if (!warned_once.exchange(true)) {
            spdlog::warn("GRULayer::Backward: caches empty (Forward not run?) "
                         "— returning zero gradients. This warning fires once.");
        }
        if (cached_input_.NumElements() > 0) {
            return Tensor::Zeros(cached_input_.Shape());
        }
        return Tensor::Zeros(grad_output.Shape());
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!bidirectional_) try {
        af::array upstream = TensorToAf3DRowMajor(grad_output);
        if (batch_first_) {
            upstream = af::reorder(upstream, 1, 0, 2);
        }

        const auto& input_shape = cached_input_.Shape();
        size_t batch_size, seq_len, input_dim;
        if (batch_first_) {
            batch_size = input_shape[0];
            seq_len = input_shape[1];
            input_dim = input_shape[2];
        } else {
            seq_len = input_shape[0];
            batch_size = input_shape[1];
            input_dim = input_shape[2];
        }

        const int H = hidden_size_;
        const int G = 3 * H;

        if (static_cast<int>(grad_W_ih_.size()) < num_layers_) grad_W_ih_.resize(num_layers_);
        if (static_cast<int>(grad_W_hh_.size()) < num_layers_) grad_W_hh_.resize(num_layers_);
        if (static_cast<int>(grad_b_ih_.size()) < num_layers_) grad_b_ih_.resize(num_layers_);
        if (static_cast<int>(grad_b_hh_.size()) < num_layers_) grad_b_hh_.resize(num_layers_);

        af::array layer_grad = upstream;

        for (int layer = num_layers_ - 1; layer >= 0; --layer) {
            const size_t layer_input_size = (layer == 0)
                ? input_dim : static_cast<size_t>(H);

            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);
            af::array input_cache = TensorToAf3DRowMajor(cached_inputs_[layer]);
            af::array gate_cache = TensorToAf3DRowMajor(cached_gates_[layer]);
            af::array h_cache = TensorToAf3DRowMajor(cached_hidden_states_[layer]);

            af::array dW_ih = af::constant(0.0f, af::dim4(G, static_cast<dim_t>(layer_input_size)));
            af::array dW_hh = af::constant(0.0f, af::dim4(G, static_cast<dim_t>(H)));
            af::array db_ih = af::constant(0.0f, af::dim4(G));
            af::array db_hh = af::constant(0.0f, af::dim4(G));
            af::array d_layer_input = af::constant(
                0.0f, af::dim4(seq_len, batch_size, static_cast<dim_t>(layer_input_size)));
            af::array dh_next = af::constant(0.0f, af::dim4(batch_size, H));
            af::array ones = af::constant(1.0f, af::dim4(batch_size, H));

            for (int64_t t = static_cast<int64_t>(seq_len) - 1; t >= 0; --t) {
                af::array x_t = af::moddims(input_cache(t, af::span, af::span),
                                            af::dim4(batch_size, layer_input_size));
                af::array gates_t = af::moddims(gate_cache(t, af::span, af::span),
                                                af::dim4(batch_size, 4 * H));
                af::array h_prev = af::moddims(h_cache(t, af::span, af::span),
                                               af::dim4(batch_size, H));
                af::array dh = af::moddims(layer_grad(t, af::span, af::span),
                                           af::dim4(batch_size, H));
                dh = dh + dh_next;

                af::array r = gates_t(af::span, af::seq(0, H - 1));
                af::array z = gates_t(af::span, af::seq(H, 2 * H - 1));
                af::array n = gates_t(af::span, af::seq(2 * H, 3 * H - 1));
                af::array hn_pre = gates_t(af::span, af::seq(3 * H, 4 * H - 1));

                af::array dn = dh * (ones - z);
                af::array dz = dh * (h_prev - n);
                af::array dh_prev_direct = dh * z;

                af::array dn_pre = dn * (ones - n * n);
                af::array dr = dn_pre * hn_pre;
                af::array d_hn_pre = dn_pre * r;

                af::array d_r_pre = dr * r * (ones - r);
                af::array d_z_pre = dz * z * (ones - z);

                af::array dgates_x = af::join(1, d_r_pre, d_z_pre, dn_pre);
                af::array dgates_h = af::join(1, d_r_pre, d_z_pre, d_hn_pre);

                dW_ih = dW_ih + af::matmul(af::transpose(dgates_x), x_t);
                dW_hh = dW_hh + af::matmul(af::transpose(dgates_h), h_prev);
                db_ih = db_ih + af::moddims(af::sum(dgates_x, 0), af::dim4(G));
                db_hh = db_hh + af::moddims(af::sum(dgates_h, 0), af::dim4(G));

                af::array dx_t = af::matmul(dgates_x, W_ih);
                d_layer_input(t, af::span, af::span) =
                    af::moddims(dx_t, af::dim4(1, batch_size, static_cast<dim_t>(layer_input_size)));

                dh_next = dh_prev_direct + af::matmul(dgates_h, W_hh);
            }

            grad_W_ih_[layer] = AfToTensor(dW_ih);
            grad_W_hh_[layer] = AfToTensor(dW_hh);
            grad_b_ih_[layer] = AfToTensor(db_ih);
            grad_b_hh_[layer] = AfToTensor(db_hh);

            layer_grad = d_layer_input;
        }

        if (batch_first_) {
            layer_grad = af::reorder(layer_grad, 1, 0, 2);
        }

        return AfToTensor3DRowMajor(layer_grad);
    } catch (const af::exception& e) {
        BackendDebugHooks::EmitDebugEvent(
            "GRULayer::Backward",
            std::string("ArrayFire fallback: ") + e.what() +
            (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
        spdlog::warn("ArrayFire GRULayer::Backward failed: {}, falling back to CPU", e.what());
    }
#endif

    // CPU BPTT for GRU. Reads the row-major caches populated by CPU Forward:
    //   cached_inputs_[L]          [seq_len, batch, layer_input_size]
    //   cached_gates_[L]           [seq_len, batch, 4 * H]
    //                              layout per (t, b):
    //                                [0..H)   r post-sigmoid
    //                                [H..2H)  z post-sigmoid
    //                                [2H..3H) n post-tanh
    //                                [3H..4H) hn_pre   (= b_hh_n + W_hh_n @ h_prev)
    //   cached_hidden_states_[L]   [seq_len + 1, batch, H]   (idx 0 = h_0)
    //
    // GRU forward equations:
    //     r       = sigmoid(x_proj_r + h_proj_r)
    //     z       = sigmoid(x_proj_z + h_proj_z)
    //     n       = tanh(x_proj_n + r * hn_pre)        // hn_pre = h_proj_n
    //     h_new   = (1 - z) * n + z * h_prev
    //
    // BPTT — per timestep, given dh_total = dL/dh_new + dh carry from t+1:
    //     dn       = dh_total * (1 - z)
    //     dz       = dh_total * (h_prev - n)
    //     dh_prev_direct = dh_total * z
    //
    //     dn_pre   = dn * (1 - n*n)                    // tanh'
    //     d_x_proj_n = dn_pre
    //     dr       = dn_pre * hn_pre                   // through r * hn_pre
    //     d_hn_pre = dn_pre * r                        // through r * hn_pre
    //
    //     d_r_pre  = dr * r * (1 - r)                  // sigmoid'
    //     d_z_pre  = dz * z * (1 - z)
    //
    //     dgates_x = [d_r_pre | d_z_pre | d_x_proj_n]  // x-side projections
    //     dgates_h = [d_r_pre | d_z_pre | d_hn_pre  ]  // h-side projections
    //                                                  // (n-slot differs!)
    //
    //     dW_ih += outer(dgates_x, x);   db_ih += dgates_x
    //     dW_hh += outer(dgates_h, h_prev); db_hh += dgates_h
    //     dx     = dgates_x @ W_ih
    //     dh_prev = dh_prev_direct + dgates_h @ W_hh
    //
    // Note: dgates_x and dgates_h DIFFER in the n-slot. The legacy AF
    // backward used the same dgates for both sides which is wrong for GRU
    // (and additionally zeroed the r-slot, so reset gate weights never
    // updated). This implementation handles both correctly.

    const auto& input_shape = cached_input_.Shape();
    size_t batch_size, seq_len, input_dim;
    if (batch_first_) {
        batch_size = input_shape[0];
        seq_len    = input_shape[1];
        input_dim  = input_shape[2];
    } else {
        seq_len    = input_shape[0];
        batch_size = input_shape[1];
        input_dim  = input_shape[2];
    }

    const int H = hidden_size_;
    const int G = 3 * H;

    if (static_cast<int>(grad_W_ih_.size()) < num_layers_) grad_W_ih_.resize(num_layers_);
    if (static_cast<int>(grad_W_hh_.size()) < num_layers_) grad_W_hh_.resize(num_layers_);
    if (static_cast<int>(grad_b_ih_.size()) < num_layers_) grad_b_ih_.resize(num_layers_);
    if (static_cast<int>(grad_b_hh_.size()) < num_layers_) grad_b_hh_.resize(num_layers_);

    // Convert grad_output (in user's batch_first/seq_first layout) into a
    // canonical [seq_len, batch, H] row-major scratch buffer for the
    // top-layer gradient.
    const float* dout = grad_output.Data<float>();
    std::vector<float> layer_grad(seq_len * batch_size * H, 0.0f);
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (int i = 0; i < H; ++i) {
                float g = batch_first_
                    ? dout[b * seq_len * H + t * H + i]
                    : dout[t * batch_size * H + b * H + i];
                layer_grad[t * batch_size * H + b * H + i] = g;
            }
        }
    }

    std::vector<float> d_layer_input;

    for (int layer = num_layers_ - 1; layer >= 0; --layer) {
        const size_t layer_input_size = (layer == 0)
            ? input_dim : static_cast<size_t>(H);

        const float* W_ih = W_ih_[layer].Data<float>();       // [3H, input_size]
        const float* W_hh = W_hh_[layer].Data<float>();       // [3H, H]
        const float* in_cache = cached_inputs_[layer].Data<float>();
        const float* gate_cache = cached_gates_[layer].Data<float>();  // 4H per (t,b)
        const float* h_cache = cached_hidden_states_[layer].Data<float>();

        std::vector<float> dW_ih(G * layer_input_size, 0.0f);
        std::vector<float> dW_hh(G * H, 0.0f);
        std::vector<float> db_ih(G, 0.0f);
        std::vector<float> db_hh(G, 0.0f);

        d_layer_input.assign(seq_len * batch_size * layer_input_size, 0.0f);

        for (size_t b = 0; b < batch_size; ++b) {
            std::vector<float> dh_next(H, 0.0f);

            for (int64_t t = static_cast<int64_t>(seq_len) - 1; t >= 0; --t) {
                const size_t gate_off = t * batch_size * (4 * H) + b * (4 * H);
                const size_t h_prev_off = t * batch_size * H + b * H;        // cache idx t
                const size_t in_off = t * batch_size * layer_input_size + b * layer_input_size;
                const size_t lg_off = t * batch_size * H + b * H;

                std::vector<float> dgates_x(G, 0.0f);
                std::vector<float> dgates_h(G, 0.0f);

                for (int i = 0; i < H; ++i) {
                    const float r       = gate_cache[gate_off + i];
                    const float z       = gate_cache[gate_off + H + i];
                    const float n       = gate_cache[gate_off + 2 * H + i];
                    const float hn_pre  = gate_cache[gate_off + 3 * H + i];
                    const float h_prev  = h_cache[h_prev_off + i];

                    const float dh_total = layer_grad[lg_off + i] + dh_next[i];

                    const float dn = dh_total * (1.0f - z);
                    const float dz = dh_total * (h_prev - n);
                    const float dh_prev_direct = dh_total * z;

                    const float dn_pre = dn * (1.0f - n * n);
                    const float d_x_proj_n = dn_pre;
                    const float dr = dn_pre * hn_pre;
                    const float d_hn_pre = dn_pre * r;

                    const float d_r_pre = dr * r * (1.0f - r);
                    const float d_z_pre = dz * z * (1.0f - z);

                    dgates_x[i]            = d_r_pre;
                    dgates_x[H + i]        = d_z_pre;
                    dgates_x[2 * H + i]    = d_x_proj_n;

                    dgates_h[i]            = d_r_pre;
                    dgates_h[H + i]        = d_z_pre;
                    dgates_h[2 * H + i]    = d_hn_pre;

                    // Stash the direct (non-gate) carry; gate-side carry
                    // gets added below from dgates_h @ W_hh.
                    dh_next[i] = dh_prev_direct;
                }

                // Weight + bias accumulation.
                //   dW_ih [G, layer_input_size] += outer(dgates_x, x_t)
                //   dW_hh [G, H]                 += outer(dgates_h, h_prev)
                for (int g = 0; g < G; ++g) {
                    const float dgx = dgates_x[g];
                    const float dgh = dgates_h[g];
                    db_ih[g] += dgx;
                    db_hh[g] += dgh;
                    for (size_t k = 0; k < layer_input_size; ++k) {
                        dW_ih[g * layer_input_size + k] += dgx * in_cache[in_off + k];
                    }
                    for (int k = 0; k < H; ++k) {
                        dW_hh[g * H + k] += dgh * h_cache[h_prev_off + k];
                    }
                }

                // dx_t = dgates_x @ W_ih   (shape [layer_input_size])
                for (size_t k = 0; k < layer_input_size; ++k) {
                    float s = 0.0f;
                    for (int g = 0; g < G; ++g) {
                        s += dgates_x[g] * W_ih[g * layer_input_size + k];
                    }
                    d_layer_input[in_off + k] = s;
                }

                // dh_prev (carries to t-1) = dh_prev_direct + dgates_h @ W_hh
                for (int k = 0; k < H; ++k) {
                    float s = 0.0f;
                    for (int g = 0; g < G; ++g) {
                        s += dgates_h[g] * W_hh[g * H + k];
                    }
                    dh_next[k] += s;
                }
            }
        }

        grad_W_ih_[layer] = Tensor({static_cast<size_t>(G), layer_input_size},
                                   dW_ih.data());
        grad_W_hh_[layer] = Tensor({static_cast<size_t>(G), static_cast<size_t>(H)},
                                   dW_hh.data());
        grad_b_ih_[layer] = Tensor({static_cast<size_t>(G)}, db_ih.data());
        grad_b_hh_[layer] = Tensor({static_cast<size_t>(G)}, db_hh.data());

        if (layer > 0) {
            layer_grad = d_layer_input;
        }
    }

    Tensor dx = Tensor::Zeros(cached_input_.Shape());
    float* dx_data = dx.Data<float>();
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t k = 0; k < input_dim; ++k) {
                const float v = d_layer_input[t * batch_size * input_dim + b * input_dim + k];
                if (batch_first_) {
                    dx_data[b * seq_len * input_dim + t * input_dim + k] = v;
                } else {
                    dx_data[t * batch_size * input_dim + b * input_dim + k] = v;
                }
            }
        }
    }
    return dx;
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
        cached_attn_weights_ = Tensor(std::vector<size_t>{
            static_cast<size_t>(seq_len_q),
            static_cast<size_t>(seq_len_kv),
            static_cast<size_t>(batch_size),
            static_cast<size_t>(num_heads_)});
        attn_weights.host(cached_attn_weights_.Data());

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
                af::array grad_attn = af::matmul(af::transpose(gc_bh), v_bh);

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

namespace {

Tensor FlattenTransformerSequenceForDense(const Tensor& input) {
    const auto& shape = input.Shape();
    if (shape.size() != 3) {
        return input;
    }
    return input.Reshape({shape[0] * shape[1], shape[2]});
}

Tensor RestoreTransformerSequenceFromDense(const Tensor& input,
                                           size_t batch,
                                           size_t seq_len) {
    const auto& shape = input.Shape();
    if (shape.size() != 2) {
        return input;
    }
    return input.Reshape({batch, seq_len, shape[1]});
}

} // namespace

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
        Tensor ffn_out = linear1_->Forward(
            FlattenTransformerSequenceForDense(normed2));

        // ReLU activation
        float* ffn_data = ffn_out.Data<float>();
        size_t ffn_total = ffn_out.NumElements();
        for (size_t i = 0; i < ffn_total; i++) {
            ffn_data[i] = std::max(0.0f, ffn_data[i]);
        }
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = RestoreTransformerSequenceFromDense(ffn_out, shape[0], shape[1]);
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
        Tensor ffn_out = linear1_->Forward(
            FlattenTransformerSequenceForDense(x));

        // ReLU activation
        float* ffn_data = ffn_out.Data<float>();
        size_t ffn_total = ffn_out.NumElements();
        for (size_t i = 0; i < ffn_total; i++) {
            ffn_data[i] = std::max(0.0f, ffn_data[i]);
        }
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = RestoreTransformerSequenceFromDense(ffn_out, shape[0], shape[1]);
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
    const auto& grad_shape = grad_output.Shape();
    if (grad_shape.size() == 3) {
        grad_ffn = FlattenTransformerSequenceForDense(grad_ffn);
    }
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
    if (grad_shape.size() == 3) {
        grad_ffn = RestoreTransformerSequenceFromDense(
            grad_ffn, grad_shape[0], grad_shape[1]);
    }

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

// ============================================================================
// ConvTranspose2D Layer Implementation
// ============================================================================

ConvTranspose2DLayer::ConvTranspose2DLayer(int in_channels, int out_channels,
                                           int kernel_size, int stride, int padding,
                                           int output_padding, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      output_padding_(output_padding), use_bias_(use_bias) {

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Weights: [kernel_size, kernel_size, out_channels, in_channels]
    // Note: transposed conv weights are "flipped" relative to conv2d
    int fan_in = in_channels * kernel_size * kernel_size;
    af::dim4 weight_dims(kernel_size, kernel_size, out_channels, in_channels);
    af::array w = KaimingUniform(fan_in, weight_dims);
    weights_ = AfToTensor(w);

    if (use_bias_) {
        af::array b = af::constant(0.0f, af::dim4(out_channels));
        bias_ = AfToTensor(b);
    }

    grad_weights_ = Tensor::Zeros({static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(out_channels),
                                    static_cast<size_t>(in_channels)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
#else
    weights_ = Tensor::Random({static_cast<size_t>(kernel_size),
                                static_cast<size_t>(kernel_size),
                                static_cast<size_t>(out_channels),
                                static_cast<size_t>(in_channels)});
    if (use_bias_) {
        bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(kernel_size),
                                    static_cast<size_t>(out_channels),
                                    static_cast<size_t>(in_channels)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    }
#endif
}

Tensor ConvTranspose2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Input: [H, W, in_channels, batch] (ArrayFire column-major)
        af::array x = TensorToAf(input);
        af::array w = TensorToAf(weights_);

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        // Output size: (in - 1) * stride - 2 * padding + kernel + output_padding
        dim_t out_h = (in_h - 1) * stride_ - 2 * padding_ + kernel_size_ + output_padding_;
        dim_t out_w = (in_w - 1) * stride_ - 2 * padding_ + kernel_size_ + output_padding_;

        af::array output = af::constant(0.0f, af::dim4(out_h, out_w, out_channels_, batch_size));

        // Transposed convolution: scatter input values using flipped kernel
        for (int ic = 0; ic < in_channels_; ic++) {
            for (int oc = 0; oc < out_channels_; oc++) {
                af::array filter = w(af::span, af::span, oc, ic);

                for (dim_t b = 0; b < batch_size; b++) {
                    af::array input_slice = x(af::span, af::span, ic, b);

                    // For each input position, add filter * input_value to output
                    for (dim_t ih = 0; ih < in_h; ih++) {
                        for (dim_t iw = 0; iw < in_w; iw++) {
                            float val = input_slice(ih, iw).scalar<float>();
                            if (val == 0.0f) continue;

                            dim_t oh_start = ih * stride_ - padding_;
                            dim_t ow_start = iw * stride_ - padding_;

                            for (int kh = 0; kh < kernel_size_; kh++) {
                                for (int kw = 0; kw < kernel_size_; kw++) {
                                    dim_t oh = oh_start + kh;
                                    dim_t ow = ow_start + kw;
                                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                        output(oh, ow, oc, b) += val * filter(kh, kw).scalar<float>();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add bias
        if (use_bias_) {
            af::array b = TensorToAf(bias_);
            b = af::moddims(b, af::dim4(1, 1, out_channels_, 1));
            output = output + af::tile(b, af::dim4(out_h, out_w, 1, batch_size));
        }

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire ConvTranspose2DLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("ConvTranspose2D forward requires ArrayFire");
}

Tensor ConvTranspose2DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);
        af::array w = TensorToAf(weights_);

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        // Gradient w.r.t. bias
        if (use_bias_) {
            af::array db = af::sum(af::sum(af::sum(grad_out, 0), 1), 3);
            db = af::moddims(db, af::dim4(out_channels_));
            grad_bias_ = AfToTensor(db);
        }

        // Gradient w.r.t. input: accumulate on host to avoid AF per-element indexing
        dim_t out_h = grad_out.dims(0);
        dim_t out_w = grad_out.dims(1);

        // Copy data to host for efficient element-wise access
        std::vector<float> h_grad(grad_out.elements());
        grad_out.host(h_grad.data());
        std::vector<float> h_w(w.elements());
        w.host(h_w.data());
        std::vector<float> h_x(x.elements());
        x.host(h_x.data());

        // Host-side dx accumulation
        std::vector<float> h_dx(x.elements(), 0.0f);

        for (int ic = 0; ic < in_channels_; ic++) {
            for (int oc = 0; oc < out_channels_; oc++) {
                for (dim_t b = 0; b < batch_size; b++) {
                    for (dim_t ih = 0; ih < in_h; ih++) {
                        for (dim_t iw = 0; iw < in_w; iw++) {
                            dim_t oh_start = ih * stride_ - padding_;
                            dim_t ow_start = iw * stride_ - padding_;

                            float sum = 0.0f;
                            for (int kh = 0; kh < kernel_size_; kh++) {
                                for (int kw = 0; kw < kernel_size_; kw++) {
                                    dim_t oh = oh_start + kh;
                                    dim_t ow = ow_start + kw;
                                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                        // AF column-major: [H, W, C, N]
                                        size_t g_idx = oh + ow * out_h + oc * out_h * out_w + b * out_h * out_w * out_channels_;
                                        size_t w_idx = kh + kw * kernel_size_ + oc * kernel_size_ * kernel_size_ + ic * kernel_size_ * kernel_size_ * out_channels_;
                                        sum += h_grad[g_idx] * h_w[w_idx];
                                    }
                                }
                            }
                            size_t dx_idx = ih + iw * in_h + ic * in_h * in_w + b * in_h * in_w * in_channels_;
                            h_dx[dx_idx] += sum;
                        }
                    }
                }
            }
        }
        af::array dx = af::array(x.dims(), h_dx.data());

        // Host-side dW accumulation
        std::vector<float> h_dW(w.elements(), 0.0f);

        for (int ic = 0; ic < in_channels_; ic++) {
            for (int oc = 0; oc < out_channels_; oc++) {
                for (dim_t b = 0; b < batch_size; b++) {
                    for (int kh = 0; kh < kernel_size_; kh++) {
                        for (int kw = 0; kw < kernel_size_; kw++) {
                            float sum = 0.0f;
                            for (dim_t ih = 0; ih < in_h; ih++) {
                                for (dim_t iw = 0; iw < in_w; iw++) {
                                    dim_t oh = ih * stride_ - padding_ + kh;
                                    dim_t ow = iw * stride_ - padding_ + kw;
                                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                        size_t x_idx = ih + iw * in_h + ic * in_h * in_w + b * in_h * in_w * in_channels_;
                                        size_t g_idx = oh + ow * out_h + oc * out_h * out_w + b * out_h * out_w * out_channels_;
                                        sum += h_x[x_idx] * h_grad[g_idx];
                                    }
                                }
                            }
                            size_t w_idx = kh + kw * kernel_size_ + oc * kernel_size_ * kernel_size_ + ic * kernel_size_ * kernel_size_ * out_channels_;
                            h_dW[w_idx] += sum;
                        }
                    }
                }
            }
        }
        grad_weights_ = AfToTensor(af::array(w.dims(), h_dW.data()));

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire ConvTranspose2DLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("ConvTranspose2D backward requires ArrayFire");
}

std::map<std::string, Tensor> ConvTranspose2DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void ConvTranspose2DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("weights")) weights_ = params.at("weights");
    if (params.count("bias") && use_bias_) bias_ = params.at("bias");
}

// ============================================================================
// Upsample2D Layer Implementation
// ============================================================================

Upsample2DLayer::Upsample2DLayer(int scale_factor, UpsampleMode mode)
    : scale_factor_(scale_factor), mode_(mode) {
}

Tensor Upsample2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t channels = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        dim_t out_h = in_h * scale_factor_;
        dim_t out_w = in_w * scale_factor_;

        af::array output = af::constant(0.0f, af::dim4(out_h, out_w, channels, batch_size));

        if (mode_ == UpsampleMode::Nearest) {
            // Nearest neighbor: repeat each pixel scale_factor times
            for (dim_t c = 0; c < channels; c++) {
                for (dim_t b = 0; b < batch_size; b++) {
                    af::array slice = x(af::span, af::span, c, b);
                    // Use af::resize for nearest interpolation
                    af::array resized = af::resize(slice, out_h, out_w, AF_INTERP_NEAREST);
                    output(af::span, af::span, c, b) = resized;
                }
            }
        } else {
            // Bilinear interpolation
            for (dim_t c = 0; c < channels; c++) {
                for (dim_t b = 0; b < batch_size; b++) {
                    af::array slice = x(af::span, af::span, c, b);
                    af::array resized = af::resize(slice, out_h, out_w, AF_INTERP_BILINEAR_COSINE);
                    output(af::span, af::span, c, b) = resized;
                }
            }
        }

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire Upsample2DLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Upsample2D forward requires ArrayFire");
}

Tensor Upsample2DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t channels = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        af::array dx = af::constant(0.0f, x.dims());

        if (mode_ == UpsampleMode::Nearest) {
            // Backward of nearest: sum gradients in each scale_factor block
            for (dim_t c = 0; c < channels; c++) {
                for (dim_t b = 0; b < batch_size; b++) {
                    af::array grad_slice = grad_out(af::span, af::span, c, b);
                    for (dim_t ih = 0; ih < in_h; ih++) {
                        for (dim_t iw = 0; iw < in_w; iw++) {
                            af::array block = grad_slice(
                                af::seq(ih * scale_factor_, (ih + 1) * scale_factor_ - 1),
                                af::seq(iw * scale_factor_, (iw + 1) * scale_factor_ - 1));
                            dx(ih, iw, c, b) = af::sum<float>(af::flat(block));
                        }
                    }
                }
            }
        } else {
            // Bilinear backward: downsample gradient
            for (dim_t c = 0; c < channels; c++) {
                for (dim_t b = 0; b < batch_size; b++) {
                    af::array grad_slice = grad_out(af::span, af::span, c, b);
                    af::array resized = af::resize(grad_slice, in_h, in_w, AF_INTERP_BILINEAR_COSINE);
                    dx(af::span, af::span, c, b) = resized * static_cast<float>(scale_factor_ * scale_factor_);
                }
            }
        }

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire Upsample2DLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("Upsample2D backward requires ArrayFire");
}

// ============================================================================
// PixelShuffle Layer Implementation
// ============================================================================

PixelShuffleLayer::PixelShuffleLayer(int upscale_factor)
    : upscale_factor_(upscale_factor) {
}

Tensor PixelShuffleLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        dim_t in_c = x.dims(2);
        dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;

        int r = upscale_factor_;
        cached_in_channels_ = static_cast<int>(in_c);

        if (in_c % (r * r) != 0) {
            throw std::runtime_error("PixelShuffle: input channels (" + std::to_string(in_c) +
                                     ") must be divisible by r^2 (" + std::to_string(r * r) + ")");
        }

        dim_t out_c = in_c / (r * r);
        dim_t out_h = in_h * r;
        dim_t out_w = in_w * r;

        af::array output = af::constant(0.0f, af::dim4(out_h, out_w, out_c, batch_size));

        // Rearrange: (H, W, C*r^2, N) -> (H*r, W*r, C, N)
        for (dim_t b = 0; b < batch_size; b++) {
            for (dim_t oc = 0; oc < out_c; oc++) {
                for (int rh = 0; rh < r; rh++) {
                    for (int rw = 0; rw < r; rw++) {
                        dim_t ic = oc * r * r + rh * r + rw;
                        af::array channel = x(af::span, af::span, ic, b);

                        // Place each pixel from input channel into sub-pixel position
                        for (dim_t ih = 0; ih < in_h; ih++) {
                            for (dim_t iw = 0; iw < in_w; iw++) {
                                output(ih * r + rh, iw * r + rw, oc, b) = channel(ih, iw);
                            }
                        }
                    }
                }
            }
        }

        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire PixelShuffleLayer::Forward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("PixelShuffle forward requires ArrayFire");
}

Tensor PixelShuffleLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);

        dim_t out_h = grad_out.dims(0);
        dim_t out_w = grad_out.dims(1);
        dim_t out_c = grad_out.dims(2);
        dim_t batch_size = (grad_out.numdims() > 3) ? grad_out.dims(3) : 1;

        int r = upscale_factor_;
        dim_t in_h = out_h / r;
        dim_t in_w = out_w / r;
        dim_t in_c = out_c * r * r;

        af::array dx = af::constant(0.0f, af::dim4(in_h, in_w, in_c, batch_size));

        // Inverse rearrange: (H*r, W*r, C, N) -> (H, W, C*r^2, N)
        for (dim_t b = 0; b < batch_size; b++) {
            for (dim_t oc = 0; oc < out_c; oc++) {
                for (int rh = 0; rh < r; rh++) {
                    for (int rw = 0; rw < r; rw++) {
                        dim_t ic = oc * r * r + rh * r + rw;

                        for (dim_t ih = 0; ih < in_h; ih++) {
                            for (dim_t iw = 0; iw < in_w; iw++) {
                                dx(ih, iw, ic, b) = grad_out(ih * r + rh, iw * r + rw, oc, b);
                            }
                        }
                    }
                }
            }
        }

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire PixelShuffleLayer::Backward failed: {}", e.what());
    }
#endif

    throw std::runtime_error("PixelShuffle backward requires ArrayFire");
}

} // namespace cyxwiz
