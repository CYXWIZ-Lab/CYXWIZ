#include "executable_model.h"

#include <chrono>
#include <stdexcept>

namespace cyxwiz {

SequentialExecutableModel::SequentialExecutableModel(std::shared_ptr<SequentialModel> model)
    : model_(std::move(model)) {
    if (!model_) {
        throw std::invalid_argument("SequentialExecutableModel requires a model");
    }
}

Tensor SequentialExecutableModel::Forward(const Tensor& input) {
    if (!layer_forward_observer_) return model_->Forward(input);
    // Module by module, so the observer sees each layer (Backward reads only
    // what each module kept from its own forward).
    Tensor current = input.Clone();
    for (size_t i = 0; i < model_->Size(); ++i) {
        const auto start = std::chrono::steady_clock::now();
        Tensor output = model_->GetModule(i)->Forward(current);
        layer_forward_observer_(
            i, current, output,
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count());
        current = std::move(output);
    }
    return current;
}

Tensor SequentialExecutableModel::Backward(const Tensor& grad_output) {
    return model_->Backward(grad_output);
}

bool SequentialExecutableModel::SupportsSparseCsrInput() const {
    return model_->SupportsSparseCsrInput();
}

Tensor SequentialExecutableModel::ForwardSparseCsr(
    const LinearSparseCsrBatchView& input) {
    return model_->ForwardSparseCsr(input);
}

void SequentialExecutableModel::BackwardSparseCsr(
    const LinearSparseCsrBatchView& input,
    const Tensor& grad_output) {
    model_->BackwardSparseCsr(input, grad_output);
}

void SequentialExecutableModel::SetTraining(bool training) {
    model_->SetTraining(training);
}

std::map<std::string, Tensor> SequentialExecutableModel::GetParameters() {
    return model_->GetParameters();
}

void SequentialExecutableModel::SetParameters(const std::map<std::string, Tensor>& params) {
    model_->SetParameters(params);
}

std::map<std::string, Tensor> SequentialExecutableModel::GetGradients() {
    return model_->GetGradients();
}

void SequentialExecutableModel::UpdateParameters(Optimizer* optimizer) {
    model_->UpdateParameters(optimizer);
}

} // namespace cyxwiz
