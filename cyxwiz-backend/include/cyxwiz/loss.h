#pragma once

#include "api_export.h"

namespace cyxwiz {
    class Tensor;
}

namespace cyxwiz {

enum class LossType {
    MSE, CrossEntropy, BinaryCrossEntropy, Huber
};

class CYXWIZ_API Loss {
public:
    virtual ~Loss() = default;
    virtual Tensor Forward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor Backward(const Tensor& predictions, const Tensor& targets) = 0;
};

} // namespace cyxwiz
