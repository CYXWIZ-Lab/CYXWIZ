#include <cyxwiz/sequential.h>

#include <map>
#include <string>

namespace cyxwiz {
namespace {

void CopyIfPresent(std::map<std::string, Tensor>& destination,
                   const std::map<std::string, Tensor>& source,
                   const char* source_key,
                   const char* destination_key) {
    const auto it = source.find(source_key);
    if (it != source.end()) {
        destination.emplace(destination_key, it->second);
    }
}

} // namespace

Conv2DModule::Conv2DModule(int in_channels, int out_channels, int kernel_size,
                           int stride, int padding, bool use_bias)
    : layer_(std::make_unique<Conv2DLayer>(
          in_channels, out_channels, kernel_size, stride, padding, use_bias))
    , in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , use_bias_(use_bias) {}

Tensor Conv2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor Conv2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> Conv2DModule::GetParameters() {
    const auto legacy = layer_->GetParameters();
    std::map<std::string, Tensor> parameters;
    CopyIfPresent(parameters, legacy, "weights", "weights");
    CopyIfPresent(parameters, legacy, "bias", "bias");
    return parameters;
}

void Conv2DModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> Conv2DModule::GetGradients() {
    const auto legacy = layer_->GetParameters();
    std::map<std::string, Tensor> gradients;
    CopyIfPresent(gradients, legacy, "grad_weights", "weights");
    CopyIfPresent(gradients, legacy, "grad_bias", "bias");
    return gradients;
}

std::string Conv2DModule::GetName() const {
    return "Conv2D(" + std::to_string(in_channels_) + " -> " +
        std::to_string(out_channels_) + ", kernel=" +
        std::to_string(kernel_size_) + ", stride=" +
        std::to_string(stride_) + ", padding=" +
        std::to_string(padding_) + ", bias=" +
        (use_bias_ ? "true" : "false") + ")";
}

MaxPool2DModule::MaxPool2DModule(int pool_size, int stride, int padding)
    : layer_(std::make_unique<MaxPool2DLayer>(pool_size, stride, padding))
    , pool_size_(pool_size)
    , stride_(stride > 0 ? stride : pool_size)
    , padding_(padding) {}

Tensor MaxPool2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor MaxPool2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::string MaxPool2DModule::GetName() const {
    return "MaxPool2D(pool=" + std::to_string(pool_size_) +
        ", stride=" + std::to_string(stride_) +
        ", padding=" + std::to_string(padding_) + ")";
}

AvgPool2DModule::AvgPool2DModule(int pool_size, int stride, int padding)
    : layer_(std::make_unique<AvgPool2DLayer>(pool_size, stride, padding))
    , pool_size_(pool_size)
    , stride_(stride > 0 ? stride : pool_size)
    , padding_(padding) {}

Tensor AvgPool2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor AvgPool2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::string AvgPool2DModule::GetName() const {
    return "AvgPool2D(pool=" + std::to_string(pool_size_) +
        ", stride=" + std::to_string(stride_) +
        ", padding=" + std::to_string(padding_) + ")";
}

GlobalAvgPool2DModule::GlobalAvgPool2DModule()
    : layer_(std::make_unique<GlobalAvgPool2DLayer>()) {}

Tensor GlobalAvgPool2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor GlobalAvgPool2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::string GlobalAvgPool2DModule::GetName() const {
    return "GlobalAvgPool2D";
}

} // namespace cyxwiz
