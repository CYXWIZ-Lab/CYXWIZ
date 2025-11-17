#include "cyxwiz/optimizer.h"
#include "cyxwiz/tensor.h"

namespace cyxwiz {

// SGD Implementation
SGDOptimizer::SGDOptimizer(double learning_rate, double momentum)
    : momentum_(momentum) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
}

void SGDOptimizer::Step(std::map<std::string, Tensor>& parameters,
                        const std::map<std::string, Tensor>& gradients) {
    // Simple SGD: param = param - lr * grad
    // For now, implement simple SGD without momentum using manual data updates

    for (auto& param_pair : parameters) {
        const std::string& name = param_pair.first;
        Tensor& param = param_pair.second;

        // Find corresponding gradient
        auto grad_it = gradients.find(name);
        if (grad_it == gradients.end()) {
            continue; // No gradient for this parameter
        }

        const Tensor& grad = grad_it->second;

        // Manual SGD update: param -= lr * grad
        if (param.GetDataType() == DataType::Float32) {
            float* param_data = static_cast<float*>(param.Data());
            const float* grad_data = static_cast<const float*>(grad.Data());
            size_t num_elements = param.NumElements();

            for (size_t i = 0; i < num_elements; i++) {
                param_data[i] -= static_cast<float>(learning_rate_) * grad_data[i];
            }
        }
    }

    step_count_++;
}

void SGDOptimizer::ZeroGrad() {
    velocity_.clear();
}

// Adam Implementation
AdamOptimizer::AdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    learning_rate_ = learning_rate;
    step_count_ = 0;
}

void AdamOptimizer::Step(std::map<std::string, Tensor>& parameters,
                         const std::map<std::string, Tensor>& gradients) {
    // TODO: Implement Adam optimizer
    step_count_++;
}

void AdamOptimizer::ZeroGrad() {
    m_.clear();
    v_.clear();
}

// AdamW Implementation
AdamWOptimizer::AdamWOptimizer(double learning_rate, double beta1, double beta2,
                               double epsilon, double weight_decay)
    : AdamOptimizer(learning_rate, beta1, beta2, epsilon), weight_decay_(weight_decay) {
}

void AdamWOptimizer::Step(std::map<std::string, Tensor>& parameters,
                          const std::map<std::string, Tensor>& gradients) {
    // TODO: Implement AdamW (Adam + weight decay)
    AdamOptimizer::Step(parameters, gradients);
}

// Factory
std::unique_ptr<Optimizer> CreateOptimizer(OptimizerType type, double learning_rate) {
    switch (type) {
        case OptimizerType::SGD:
            return std::make_unique<SGDOptimizer>(learning_rate);
        case OptimizerType::Adam:
            return std::make_unique<AdamOptimizer>(learning_rate);
        case OptimizerType::AdamW:
            return std::make_unique<AdamWOptimizer>(learning_rate);
        default:
            return nullptr;
    }
}

} // namespace cyxwiz
