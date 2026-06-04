#include "executable_model.h"

#include <stdexcept>

namespace cyxwiz {

SequentialExecutableModel::SequentialExecutableModel(std::unique_ptr<SequentialModel> model)
    : model_(std::move(model)) {
    if (!model_) {
        throw std::invalid_argument("SequentialExecutableModel requires a model");
    }
}

Tensor SequentialExecutableModel::Forward(const Tensor& input) {
    return model_->Forward(input);
}

Tensor SequentialExecutableModel::Backward(const Tensor& grad_output) {
    return model_->Backward(grad_output);
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
