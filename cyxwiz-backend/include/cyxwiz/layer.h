#pragma once

#include "api_export.h"
#include <string>
#include <map>

namespace cyxwiz {
    class Tensor;
}

namespace cyxwiz {

class CYXWIZ_API Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor Forward(const Tensor& input) = 0;
    virtual Tensor Backward(const Tensor& grad_output) = 0;
    virtual std::map<std::string, Tensor> GetParameters() = 0;
    virtual void SetParameters(const std::map<std::string, Tensor>& params) = 0;
};

} // namespace cyxwiz
