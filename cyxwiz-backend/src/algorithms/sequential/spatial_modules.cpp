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

Conv1DModule::Conv1DModule(int in_channels, int out_channels, int kernel_size,
                           int stride, int padding, int dilation,
                           bool use_bias)
    : layer_(std::make_unique<Conv1DLayer>(
          in_channels, out_channels, kernel_size, stride, padding, dilation,
          use_bias))
    , in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , dilation_(dilation)
    , use_bias_(use_bias) {}

Tensor Conv1DModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor Conv1DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> Conv1DModule::GetParameters() {
    const auto legacy = layer_->GetParameters();
    std::map<std::string, Tensor> parameters;
    CopyIfPresent(parameters, legacy, "weights", "weights");
    CopyIfPresent(parameters, legacy, "bias", "bias");
    return parameters;
}

void Conv1DModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> Conv1DModule::GetGradients() {
    const auto legacy = layer_->GetParameters();
    std::map<std::string, Tensor> gradients;
    CopyIfPresent(gradients, legacy, "grad_weights", "weights");
    CopyIfPresent(gradients, legacy, "grad_bias", "bias");
    return gradients;
}

std::string Conv1DModule::GetName() const {
    return "Conv1D(" + std::to_string(in_channels_) + " -> " +
        std::to_string(out_channels_) + ", kernel=" +
        std::to_string(kernel_size_) + ", stride=" +
        std::to_string(stride_) + ", padding=" +
        std::to_string(padding_) + ", dilation=" +
        std::to_string(dilation_) + ", bias=" +
        (use_bias_ ? "true" : "false") + ")";
}

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

ConvTranspose2DModule::ConvTranspose2DModule(
    int in_channels, int out_channels, int kernel_size, int stride,
    int padding, int output_padding, bool use_bias)
    : layer_(std::make_unique<ConvTranspose2DLayer>(
          in_channels, out_channels, kernel_size, stride, padding,
          output_padding, use_bias))
    , in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , output_padding_(output_padding)
    , use_bias_(use_bias) {}

Tensor ConvTranspose2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor ConvTranspose2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> ConvTranspose2DModule::GetParameters() {
    const auto legacy = layer_->GetParameters();
    std::map<std::string, Tensor> parameters;
    CopyIfPresent(parameters, legacy, "weights", "weights");
    CopyIfPresent(parameters, legacy, "bias", "bias");
    return parameters;
}

void ConvTranspose2DModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> ConvTranspose2DModule::GetGradients() {
    const auto legacy = layer_->GetParameters();
    std::map<std::string, Tensor> gradients;
    CopyIfPresent(gradients, legacy, "grad_weights", "weights");
    CopyIfPresent(gradients, legacy, "grad_bias", "bias");
    return gradients;
}

std::string ConvTranspose2DModule::GetName() const {
    return "ConvTranspose2D(" + std::to_string(in_channels_) + " -> " +
        std::to_string(out_channels_) + ", kernel=" +
        std::to_string(kernel_size_) + ", stride=" +
        std::to_string(stride_) + ", padding=" +
        std::to_string(padding_) + ", output_padding=" +
        std::to_string(output_padding_) + ", bias=" +
        (use_bias_ ? "true" : "false") + ")";
}

Upsample2DModule::Upsample2DModule(int scale_factor, UpsampleMode mode)
    : layer_(std::make_unique<Upsample2DLayer>(scale_factor, mode))
    , scale_factor_(scale_factor)
    , mode_(mode) {}

Tensor Upsample2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor Upsample2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::string Upsample2DModule::GetName() const {
    return "Upsample2D(scale=" + std::to_string(scale_factor_) +
        ", mode=" +
        (mode_ == UpsampleMode::Nearest ? "nearest" : "bilinear") + ")";
}

PixelShuffleModule::PixelShuffleModule(int upscale_factor)
    : layer_(std::make_unique<PixelShuffleLayer>(upscale_factor))
    , upscale_factor_(upscale_factor) {}

Tensor PixelShuffleModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor PixelShuffleModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::string PixelShuffleModule::GetName() const {
    return "PixelShuffle(upscale=" + std::to_string(upscale_factor_) + ")";
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
