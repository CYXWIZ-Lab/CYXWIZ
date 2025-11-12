#pragma once

#include "api_export.h"
#include <vector>
#include <memory>

namespace cyxwiz {
    class Tensor;
    class Layer;
}

namespace cyxwiz {

class CYXWIZ_API Model {
public:
    Model() = default;
    virtual ~Model() = default;

    virtual Tensor Forward(const Tensor& input) = 0;
    virtual void Train() { training_mode_ = true; }
    virtual void Eval() { training_mode_ = false; }

    void AddLayer(std::shared_ptr<Layer> layer) { layers_.push_back(layer); }

protected:
    std::vector<std::shared_ptr<Layer>> layers_;
    bool training_mode_ = false;
};

} // namespace cyxwiz
