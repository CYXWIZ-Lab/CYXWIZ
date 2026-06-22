#include "lstm_direction_helpers.h"
#include "layer_arrayfire_utils.h"
#include "layer_recurrent_utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cyxwiz::lstm_detail {

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
        std::vector<float> dh(hidden_size, 0.0f);
        std::vector<float> i_g(hidden_size, 0.0f);
        std::vector<float> f_g(hidden_size, 0.0f);
        std::vector<float> g_g(hidden_size, 0.0f);
        std::vector<float> o_g(hidden_size, 0.0f);
        std::vector<float> dgates(gate_size, 0.0f);

        for (int64_t step = static_cast<int64_t>(seq_len) - 1; step >= 0; --step) {
            const size_t src_t = reverse_time ? (seq_len - 1 - static_cast<size_t>(step))
                                              : static_cast<size_t>(step);
            const size_t gate_off = static_cast<size_t>(step) * batch_size * gate_size + b * gate_size;
            const size_t h_prev_off = static_cast<size_t>(step) * batch_size * hidden_size + b * hidden_size;
            const size_t c_prev_off = static_cast<size_t>(step) * batch_size * hidden_size + b * hidden_size;
            const size_t c_t_off = (static_cast<size_t>(step) + 1) * batch_size * hidden_size + b * hidden_size;
            const size_t in_off = static_cast<size_t>(step) * batch_size * input_size + b * input_size;
            const size_t lg_off = static_cast<size_t>(step) * batch_size * hidden_size + b * hidden_size;

            for (int i = 0; i < hidden_size; ++i) {
                dh[i] = layer_grad[lg_off + i] + dh_next[i];
            }

            for (int i = 0; i < hidden_size; ++i) {
                i_g[i] = sigmoid(g_cache[gate_off + i]);
                f_g[i] = sigmoid(g_cache[gate_off + hidden_size + i]);
                g_g[i] = tanh_f (g_cache[gate_off + 2 * hidden_size + i]);
                o_g[i] = sigmoid(g_cache[gate_off + 3 * hidden_size + i]);
            }

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

#ifdef CYXWIZ_HAS_ARRAYFIRE

LSTMAfDirectionResult RunLSTMAfDirectionForward(
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

    LSTMAfDirectionResult result;
    const int seq_dim = CheckedIntDim(seq_len, "seq_len");
    const int batch_dim = CheckedIntDim(batch_size, "batch_size");
    const int input_dim_af = CheckedIntDim(input_dim, "input_dim");
    const int batch_int = CheckedIntDim(batch_size, "batch_size");
    const int seq_batch_int = CheckedIntDim(seq_len * batch_size, "seq_len * batch_size");

    af::array h = af::moddims(init_h, af::dim4(batch_dim, hidden_size));
    af::array c = af::moddims(init_c, af::dim4(batch_dim, hidden_size));

    af::array input_flat = af::moddims(seq_input, af::dim4(seq_dim * batch_dim, input_dim_af));
    af::array input_proj = af::matmul(input_flat, af::transpose(W_ih));
    input_proj.eval();
    input_proj = input_proj + af::tile(af::transpose(b_ih), seq_batch_int);
    input_proj.eval();
    input_proj = af::moddims(input_proj, af::dim4(seq_dim, batch_dim, 4 * hidden_size));
    input_proj.eval();

    af::array h_states = af::constant(0.0f, af::dim4(seq_dim + 1, batch_dim, hidden_size));
    af::array c_states = af::constant(0.0f, af::dim4(seq_dim + 1, batch_dim, hidden_size));
    af::array all_gates = af::constant(0.0f, af::dim4(seq_dim, batch_dim, 4 * hidden_size));

    h_states(0, af::span, af::span) = af::moddims(h, af::dim4(1, batch_dim, hidden_size));
    c_states(0, af::span, af::span) = af::moddims(c, af::dim4(1, batch_dim, hidden_size));

    for (size_t t = 0; t < seq_len; ++t) {
        af::array x_t = input_proj(CheckedIntDim(t, "t"), af::span, af::span);
        x_t = af::moddims(x_t, af::dim4(batch_dim, 4 * hidden_size));

        af::array h_proj = af::matmul(h, af::transpose(W_hh));
        h_proj.eval();
        h_proj = h_proj + af::tile(af::transpose(b_hh), batch_int);
        h_proj.eval();

        af::array gates = x_t + h_proj;
        gates.eval();
        af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, hidden_size - 1)));
        af::array f_gate = af::sigmoid(gates(af::span, af::seq(hidden_size, 2 * hidden_size - 1)));
        af::array g_gate = af::tanh(gates(af::span, af::seq(2 * hidden_size, 3 * hidden_size - 1)));
        af::array o_gate = af::sigmoid(gates(af::span, af::seq(3 * hidden_size, 4 * hidden_size - 1)));
        i_gate.eval();
        f_gate.eval();
        g_gate.eval();
        o_gate.eval();

        c = f_gate * c + i_gate * g_gate;
        c.eval();
        h = o_gate * af::tanh(c);
        h.eval();

        h_states(CheckedIntDim(t + 1, "t + 1"), af::span, af::span) =
            af::moddims(h, af::dim4(1, batch_dim, hidden_size));
        c_states(CheckedIntDim(t + 1, "t + 1"), af::span, af::span) =
            af::moddims(c, af::dim4(1, batch_dim, hidden_size));
        all_gates(CheckedIntDim(t, "t"), af::span, af::span) =
            af::moddims(gates, af::dim4(1, batch_dim, 4 * hidden_size));
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

#endif // CYXWIZ_HAS_ARRAYFIRE

} // namespace cyxwiz::lstm_detail
