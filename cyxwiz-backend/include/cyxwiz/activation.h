#pragma once

#include "api_export.h"

namespace cyxwiz {
    class Tensor;
}

namespace cyxwiz {

enum class ActivationType {
    ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, GELU
};

class CYXWIZ_API Activation {
public:
    virtual ~Activation() = default;
    virtual Tensor Forward(const Tensor& input) = 0;
    virtual Tensor Backward(const Tensor& grad_output, const Tensor& input) = 0;
};

} // namespace cyxwiz
