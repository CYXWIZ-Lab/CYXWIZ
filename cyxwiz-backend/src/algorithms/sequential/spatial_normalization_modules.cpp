#include <cyxwiz/sequential.h>

#include <map>
#include <string>

namespace cyxwiz {
namespace {

enum class AffineStateKind {
    Parameters,
    Gradients,
};

std::map<std::string, Tensor> SelectAffineState(
    const std::map<std::string, Tensor>& legacy,
    AffineStateKind kind) {
    std::map<std::string, Tensor> result;
    const auto copy = [&](const char* source_key, const char* target_key) {
        const auto it = legacy.find(source_key);
        if (it != legacy.end()) {
            result.emplace(target_key, it->second);
        }
    };

    if (kind == AffineStateKind::Gradients) {
        copy("grad_gamma", "gamma");
        copy("grad_beta", "beta");
    } else {
        copy("gamma", "gamma");
        copy("beta", "beta");
    }
    return result;
}

} // namespace

GroupNormModule::GroupNormModule(int num_groups, int num_channels,
                                 float eps, bool affine)
    : layer_(std::make_unique<GroupNormLayer>(
          num_groups, num_channels, eps, affine))
    , num_groups_(num_groups)
    , num_channels_(num_channels)
    , eps_(eps)
    , affine_(affine) {}

Tensor GroupNormModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor GroupNormModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> GroupNormModule::GetParameters() {
    return SelectAffineState(
        layer_->GetParameters(), AffineStateKind::Parameters);
}

void GroupNormModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> GroupNormModule::GetGradients() {
    return SelectAffineState(
        layer_->GetParameters(), AffineStateKind::Gradients);
}

std::string GroupNormModule::GetName() const {
    return "GroupNorm(groups=" + std::to_string(num_groups_) +
        ", channels=" + std::to_string(num_channels_) +
        ", eps=" + std::to_string(eps_) +
        ", affine=" + (affine_ ? "true" : "false") + ")";
}

InstanceNorm2DModule::InstanceNorm2DModule(int num_features, float eps,
                                           bool affine)
    : layer_(std::make_unique<InstanceNorm2DLayer>(
          num_features, eps, affine))
    , num_features_(num_features)
    , eps_(eps)
    , affine_(affine) {}

Tensor InstanceNorm2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor InstanceNorm2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> InstanceNorm2DModule::GetParameters() {
    return SelectAffineState(
        layer_->GetParameters(), AffineStateKind::Parameters);
}

void InstanceNorm2DModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> InstanceNorm2DModule::GetGradients() {
    return SelectAffineState(
        layer_->GetParameters(), AffineStateKind::Gradients);
}

std::string InstanceNorm2DModule::GetName() const {
    return "InstanceNorm2D(features=" + std::to_string(num_features_) +
        ", eps=" + std::to_string(eps_) +
        ", affine=" + (affine_ ? "true" : "false") + ")";
}

} // namespace cyxwiz
