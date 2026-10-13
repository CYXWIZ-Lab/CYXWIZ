#include "cyxwiz/layers/recurrent.h"
#include "layer_recurrent_utils.h"

#include <map>
#include <string>

namespace cyxwiz {

void LSTMLayer::ResetState() {
    h_n_ = Tensor();
    c_n_ = Tensor();
    initial_state_pending_ = false;
    for (auto* levels : {&forward_levels_, &reverse_levels_}) {
        for (auto& child : *levels) child->ResetState();
    }
}

void LSTMLayer::SetHiddenState(const Tensor& h0) {
    h_n_ = h0.Clone();
    initial_state_pending_ = true;
}

void LSTMLayer::SetCellState(const Tensor& c0) {
    c_n_ = c0.Clone();
    initial_state_pending_ = true;
}

std::map<std::string, Tensor> LSTMLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    if (bidirectional_) {
        for (size_t level = 0; level < forward_levels_.size(); ++level) {
            recurrent_detail::AppendBidirectionalParameters(params, *forward_levels_[level], level, false);
            recurrent_detail::AppendBidirectionalParameters(params, *reverse_levels_[level], level, true);
        }
        return params;
    }
    for (int layer = 0; layer < num_layers_; layer++) {
        const std::string prefix = "layer" + std::to_string(layer) + "_";
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

void LSTMLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (bidirectional_) {
        for (size_t level = 0; level < forward_levels_.size(); ++level) {
            recurrent_detail::SetBidirectionalParameters(params, *forward_levels_[level], level, false);
            recurrent_detail::SetBidirectionalParameters(params, *reverse_levels_[level], level, true);
        }
        return;
    }
    for (int layer = 0; layer < num_layers_; layer++) {
        const std::string prefix = "layer" + std::to_string(layer) + "_";
        if (params.count(prefix + "W_ih")) W_ih_[layer] = params.at(prefix + "W_ih");
        if (params.count(prefix + "W_hh")) W_hh_[layer] = params.at(prefix + "W_hh");
        if (params.count(prefix + "b_ih")) b_ih_[layer] = params.at(prefix + "b_ih");
        if (params.count(prefix + "b_hh")) b_hh_[layer] = params.at(prefix + "b_hh");
    }
}

} // namespace cyxwiz
