#include "cyxwiz/layers/recurrent.h"
#include "layer_recurrent_utils.h"

#include <cmath>
#include <stdexcept>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

LSTMLayer::LSTMLayer(int input_size, int hidden_size, int num_layers,
                     bool batch_first, bool bidirectional, float dropout)
    : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
      batch_first_(batch_first), bidirectional_(bidirectional), dropout_(dropout) {
    if (input_size <= 0 || hidden_size <= 0 || num_layers <= 0) {
        throw std::invalid_argument("LSTMLayer needs positive input_size, hidden_size and num_layers");
    }
    if (!(dropout >= 0.0f && dropout < 1.0f)) {
        throw std::invalid_argument("LSTMLayer dropout must be in [0, 1)");
    }
    if (bidirectional_) {
        for (int level = 0; level < num_layers_; ++level) {
            const int level_input = level == 0 ? input_size_ : 2 * hidden_size_;
            forward_levels_.push_back(std::make_unique<LSTMLayer>(
                level_input, hidden_size_, 1, batch_first_, false, 0.0f));
            reverse_levels_.push_back(std::make_unique<LSTMLayer>(
                level_input, hidden_size_, 1, batch_first_, false, 0.0f));
        }
        return;
    }
    InitializeWeights();
}

LSTMLayer::~LSTMLayer() = default;

void LSTMLayer::InitializeWeights() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    W_ih_.resize(num_layers_);
    W_hh_.resize(num_layers_);
    b_ih_.resize(num_layers_);
    b_hh_.resize(num_layers_);
    grad_W_ih_.resize(num_layers_);
    grad_W_hh_.resize(num_layers_);
    grad_b_ih_.resize(num_layers_);
    grad_b_hh_.resize(num_layers_);
    try {
        const size_t gates = static_cast<size_t>(4 * hidden_size_);
        const size_t hidden = static_cast<size_t>(hidden_size_);
        for (int layer = 0; layer < num_layers_; ++layer) {
            const size_t in = static_cast<size_t>(layer == 0 ? input_size_ : hidden_size_);
            // Xavier-uniform weights; zero biases with forget-gate bias 1.
            const float limit_ih = std::sqrt(6.0f / static_cast<float>(in + hidden));
            const float limit_hh = std::sqrt(6.0f / static_cast<float>(2 * hidden));
            const af::array w_ih =
                af::randu(af::dim4(static_cast<dim_t>(gates), static_cast<dim_t>(in)), f32) *
                    (2.0f * limit_ih) - limit_ih;
            const af::array w_hh =
                af::randu(af::dim4(static_cast<dim_t>(gates), static_cast<dim_t>(hidden)), f32) *
                    (2.0f * limit_hh) - limit_hh;
            af::array b_ih = af::constant(0.0f, af::dim4(static_cast<dim_t>(gates)));
            b_ih(af::seq(hidden_size_, 2 * hidden_size_ - 1)) = 1.0f;
            W_ih_[layer] = Tensor::FromSemanticArray(w_ih, {gates, in});
            W_hh_[layer] = Tensor::FromSemanticArray(w_hh, {gates, hidden});
            b_ih_[layer] = Tensor::FromSemanticArray(b_ih, {gates});
            b_hh_[layer] = Tensor::FromSemanticArray(
                af::constant(0.0f, af::dim4(static_cast<dim_t>(gates))), {gates});
            grad_W_ih_[layer] = Tensor::Zeros({gates, in});
            grad_W_hh_[layer] = Tensor::Zeros({gates, hidden});
            grad_b_ih_[layer] = Tensor::Zeros({gates});
            grad_b_hh_[layer] = Tensor::Zeros({gates});
        }
    } catch (const af::exception& e) {
        throw std::runtime_error(
            std::string("LSTMLayer weight initialization failed on the ArrayFire device: ") + e.what());
    }
#else
    recurrent_detail::ThrowWithoutArrayFire("LSTMLayer");
#endif
}

} // namespace cyxwiz
