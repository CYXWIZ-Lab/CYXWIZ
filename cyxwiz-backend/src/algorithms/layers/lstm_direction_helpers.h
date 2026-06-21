#pragma once

#include "cyxwiz/layers/recurrent.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include "layer_arrayfire_utils.h"
#endif

namespace cyxwiz::lstm_detail {

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
    bool reverse_time);

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
    bool reverse_time);

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
    int hidden_size);
#endif

} // namespace cyxwiz::lstm_detail