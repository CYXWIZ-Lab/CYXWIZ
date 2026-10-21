#include "guarded_extension_module.h"

#include <set>
#include <stdexcept>
#include <utility>

namespace cyxwiz {

std::string FormatTensorShape(const std::vector<size_t>& shape) {
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) text += ", ";
        text += std::to_string(shape[i]);
    }
    return text + "]";
}

GuardedExtensionModule::GuardedExtensionModule(std::unique_ptr<Module> inner, int node_id,
                                               std::string node_name, std::string type_id,
                                               std::vector<size_t> expected_sample_shape,
                                               std::shared_ptr<void> lease)
    : inner_(std::move(inner)),
      node_id_(node_id),
      node_name_(std::move(node_name)),
      type_id_(std::move(type_id)),
      expected_sample_shape_(std::move(expected_sample_shape)),
      lease_(std::move(lease)) {
    if (!inner_) {
        throw std::invalid_argument("extension node '" + node_name_ + "' (" + type_id_ +
                                    ") has no module");
    }
}

void GuardedExtensionModule::Fail(const std::string& what) const {
    throw std::runtime_error("Extension node '" + node_name_ + "' (" + type_id_ + "): " + what);
}

void GuardedExtensionModule::CheckOutput(const Tensor& input, const Tensor& output) const {
    const auto& in = input.Shape();
    const auto& out = output.Shape();
    // The batch dimension, when the input has one beyond the sample shape,
    // must survive the node.
    if (out.empty()) Fail("returned an empty tensor");
    if (expected_sample_shape_.empty()) return;
    const bool batched = out.size() == expected_sample_shape_.size() + 1;
    const std::vector<size_t> sample(batched ? out.begin() + 1 : out.begin(), out.end());
    if (sample != expected_sample_shape_) {
        Fail("expected output shape " + FormatTensorShape(expected_sample_shape_) + " per sample, got " +
             FormatTensorShape(out) + " for input " + FormatTensorShape(in));
    }
    if (batched && !in.empty() && in[0] != out[0]) {
        Fail("changed the batch size from " + std::to_string(in[0]) + " to " + std::to_string(out[0]));
    }
}

Tensor GuardedExtensionModule::Forward(const Tensor& input) {
    Tensor output;
    try {
        output = inner_->Forward(input);
    } catch (const std::exception& e) {
        Fail(std::string("forward failed: ") + e.what());
    }
    CheckOutput(input, output);
    last_input_shape_ = input.Shape();
    return output;
}

Tensor GuardedExtensionModule::Backward(const Tensor& grad_output) {
    Tensor grad_input;
    try {
        grad_input = inner_->Backward(grad_output);
    } catch (const std::exception& e) {
        Fail(std::string("backward failed: ") + e.what());
    }
    if (!last_input_shape_.empty() && grad_input.Shape() != last_input_shape_) {
        Fail("backward returned gradient shape " + FormatTensorShape(grad_input.Shape()) +
             ", expected the input shape " + FormatTensorShape(last_input_shape_));
    }
    return grad_input;
}

std::map<std::string, Tensor> GuardedExtensionModule::GetParameters() {
    return inner_->GetParameters();
}

void GuardedExtensionModule::SetParameters(const std::map<std::string, Tensor>& params) {
    inner_->SetParameters(params);
}

std::map<std::string, Tensor> GuardedExtensionModule::GetGradients() {
    auto gradients = inner_->GetGradients();
    if (!gradients_checked_) {
        const auto parameters = inner_->GetParameters();
        std::set<std::string> parameter_keys;
        std::set<std::string> gradient_keys;
        for (const auto& [key, value] : parameters) parameter_keys.insert(key);
        for (const auto& [key, value] : gradients) gradient_keys.insert(key);
        if (parameter_keys != gradient_keys) {
            Fail("its gradients do not match its parameters (" + std::to_string(parameter_keys.size()) +
                 " parameters, " + std::to_string(gradient_keys.size()) + " gradients)");
        }
        for (const auto& [key, value] : parameters) {
            if (gradients.at(key).Shape() != value.Shape()) {
                Fail("gradient of '" + key + "' has shape " + FormatTensorShape(gradients.at(key).Shape()) +
                     ", the parameter has " + FormatTensorShape(value.Shape()));
            }
        }
        gradients_checked_ = true;
    }
    return gradients;
}

bool GuardedExtensionModule::HasParameters() const { return inner_->HasParameters(); }

std::string GuardedExtensionModule::GetName() const { return inner_->GetName(); }

void GuardedExtensionModule::SetTraining(bool training) {
    Module::SetTraining(training);
    inner_->SetTraining(training);
}

Module::IncrementalDecoding GuardedExtensionModule::GetIncrementalDecoding() const {
    return inner_->GetIncrementalDecoding();
}

Tensor GuardedExtensionModule::ForwardIncremental(const Tensor& input, size_t position_offset) {
    try {
        return inner_->ForwardIncremental(input, position_offset);
    } catch (const std::exception& e) {
        Fail(std::string("incremental forward failed: ") + e.what());
    }
}

void GuardedExtensionModule::ResetIncrementalState() { inner_->ResetIncrementalState(); }

}  // namespace cyxwiz
