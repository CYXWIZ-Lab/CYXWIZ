#include <cyxwiz/sequential.h>

namespace cyxwiz {

// ============================================================================
// ReLUModule Implementation
// ============================================================================

ReLUModule::ReLUModule() {
    activation_ = std::make_unique<ReLU>();
}

Tensor ReLUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor ReLUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// PReLUModule Implementation
// ============================================================================

PReLUModule::PReLUModule(int num_parameters, float init)
    : activation_(
          std::make_unique<PReLUActivation>(num_parameters, init))
    , num_parameters_(num_parameters) {}

Tensor PReLUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor PReLUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

std::map<std::string, Tensor> PReLUModule::GetParameters() {
    return {{"alpha", activation_->GetAlpha()}};
}

void PReLUModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    const auto alpha = params.find("alpha");
    if (alpha != params.end()) {
        activation_->SetAlpha(alpha->second);
    }
}

std::map<std::string, Tensor> PReLUModule::GetGradients() {
    return {{"alpha", activation_->GetAlphaGradient()}};
}

std::string PReLUModule::GetName() const {
    return "PReLU(num_parameters=" + std::to_string(num_parameters_) + ")";
}

// ============================================================================
// SigmoidModule Implementation
// ============================================================================

SigmoidModule::SigmoidModule() {
    activation_ = std::make_unique<Sigmoid>();
}

Tensor SigmoidModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor SigmoidModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// TanhModule Implementation
// ============================================================================

TanhModule::TanhModule() {
    activation_ = std::make_unique<Tanh>();
}

Tensor TanhModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor TanhModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// LeakyReLUModule Implementation
// ============================================================================

LeakyReLUModule::LeakyReLUModule(float negative_slope)
    : negative_slope_(negative_slope)
{
    activation_ = std::make_unique<LeakyReLUActivation>(negative_slope);
}

Tensor LeakyReLUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor LeakyReLUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

std::string LeakyReLUModule::GetName() const {
    return "LeakyReLU(slope=" + std::to_string(negative_slope_) + ")";
}

// ============================================================================
// ELUModule Implementation
// ============================================================================

ELUModule::ELUModule(float alpha)
    : alpha_(alpha)
{
    activation_ = std::make_unique<ELUActivation>(alpha);
}

Tensor ELUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor ELUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

std::string ELUModule::GetName() const {
    return "ELU(alpha=" + std::to_string(alpha_) + ")";
}

// ============================================================================
// GELUModule Implementation
// ============================================================================

GELUModule::GELUModule() {
    activation_ = std::make_unique<GELUActivation>();
}

Tensor GELUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor GELUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// SwishModule Implementation
// ============================================================================

SwishModule::SwishModule() {
    activation_ = std::make_unique<SwishActivation>();
}

Tensor SwishModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor SwishModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// MishModule Implementation
// ============================================================================

MishModule::MishModule() {
    activation_ = std::make_unique<MishActivation>();
}

Tensor MishModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor MishModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

} // namespace cyxwiz
