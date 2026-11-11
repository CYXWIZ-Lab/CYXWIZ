#include <cyxwiz/sequential.h>

#include <map>
#include <stdexcept>
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

DepthwiseConv2DModule::DepthwiseConv2DModule(int channels, int depth_multiplier, int kernel_size,
                                             int stride, int padding, bool use_bias)
    : layer_(std::make_unique<DepthwiseConv2DLayer>(channels, depth_multiplier, kernel_size,
                                                    stride, padding, use_bias)),
      channels_(channels),
      multiplier_(depth_multiplier),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {}

Tensor DepthwiseConv2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor DepthwiseConv2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> DepthwiseConv2DModule::GetParameters() {
    const auto legacy = layer_->GetParameters();
    std::map<std::string, Tensor> parameters;
    CopyIfPresent(parameters, legacy, "weights", "weights");
    CopyIfPresent(parameters, legacy, "bias", "bias");
    return parameters;
}

void DepthwiseConv2DModule::SetParameters(const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> DepthwiseConv2DModule::GetGradients() {
    const auto legacy = layer_->GetParameters();
    std::map<std::string, Tensor> gradients;
    CopyIfPresent(gradients, legacy, "grad_weights", "weights");
    CopyIfPresent(gradients, legacy, "grad_bias", "bias");
    return gradients;
}

std::string DepthwiseConv2DModule::GetName() const {
    return "DepthwiseConv2D(" + std::to_string(channels_) + " x " + std::to_string(multiplier_) +
        ", kernel=" + std::to_string(kernel_size_) + ", stride=" + std::to_string(stride_) +
        ", padding=" + std::to_string(padding_) + ")";
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

// The layer gives [C,N]; the model's next layer (Dense) takes [N,C] rows, as
// torch's adaptive_avg_pool2d(x, 1).flatten(1).
Tensor GlobalAvgPool2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input).Transpose();
}

Tensor GlobalAvgPool2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output.Transpose());
}

std::string GlobalAvgPool2DModule::GetName() const {
    return "GlobalAvgPool2D";
}

SqueezeExcitationModule::SqueezeExcitationModule(size_t channels, size_t squeeze_channels)
    : channels_(channels), squeeze_channels_(squeeze_channels) {
    if (channels_ == 0 || squeeze_channels_ == 0) {
        throw std::invalid_argument("SqueezeExcitation requires at least one channel and one squeeze channel");
    }
    pool_ = std::make_unique<GlobalAvgPool2DLayer>();
    reduce_ = std::make_unique<LinearModule>(channels_, squeeze_channels_);
    expand_ = std::make_unique<LinearModule>(squeeze_channels_, channels_);
}

Tensor SqueezeExcitationModule::Over(const std::vector<size_t>& shape) const {
    return gate_.Transpose().Reshape({1, 1, shape[2], shape[3]}).BroadcastTo(shape);
}

Tensor SqueezeExcitationModule::Forward(const Tensor& input) {
    const auto& shape = input.Shape();
    if (shape.size() != 4 || shape[2] != channels_) {
        throw std::invalid_argument("SqueezeExcitation expects [H,W,C,N] with C = " + std::to_string(channels_));
    }
    input_cache_ = input;
    // Squeeze: each channel's mean [C,N] as rows [N,C]; excite: one value per channel in (0, 1).
    reduce_out_ = reduce_->Forward(pool_->Forward(input).Transpose());
    expand_out_ = expand_->Forward(relu_.Forward(reduce_out_));
    gate_ = sigmoid_.Forward(expand_out_);
    return input * Over(shape);
}

Tensor SqueezeExcitationModule::Backward(const Tensor& grad_output) {
    if (grad_output.Shape() != input_cache_.Shape() || gate_.Shape().size() != 2) {
        throw std::invalid_argument("SqueezeExcitation: backward requires a matching forward pass");
    }
    // The map reaches the output directly (times the gate) and through the gate.
    const Tensor direct = grad_output * Over(grad_output.Shape());
    const Tensor to_gate = (grad_output * input_cache_).Sum(0, false).Sum(0, false).Transpose();  // [N,C]
    const Tensor to_reduced = expand_->Backward(sigmoid_.Backward(to_gate, expand_out_));
    const Tensor to_pooled = reduce_->Backward(relu_.Backward(to_reduced, reduce_out_));
    return direct + pool_->Backward(to_pooled.Transpose());
}

std::map<std::string, Tensor> SqueezeExcitationModule::GetParameters() {
    std::map<std::string, Tensor> parameters;
    for (auto& [name, tensor] : reduce_->GetParameters()) parameters.emplace("reduce_" + name, tensor);
    for (auto& [name, tensor] : expand_->GetParameters()) parameters.emplace("expand_" + name, tensor);
    return parameters;
}

void SqueezeExcitationModule::SetParameters(const std::map<std::string, Tensor>& params) {
    std::map<std::string, Tensor> reduce, expand;
    for (const auto& [name, tensor] : params) {
        if (name.rfind("reduce_", 0) == 0) reduce.emplace(name.substr(7), tensor);
        if (name.rfind("expand_", 0) == 0) expand.emplace(name.substr(7), tensor);
    }
    if (!reduce.empty()) reduce_->SetParameters(reduce);
    if (!expand.empty()) expand_->SetParameters(expand);
}

std::map<std::string, Tensor> SqueezeExcitationModule::GetGradients() {
    std::map<std::string, Tensor> gradients;
    for (auto& [name, tensor] : reduce_->GetGradients()) gradients.emplace("reduce_" + name, tensor);
    for (auto& [name, tensor] : expand_->GetGradients()) gradients.emplace("expand_" + name, tensor);
    return gradients;
}

size_t SqueezeExcitationModule::BackwardWorkingElementsPerSample(const std::vector<size_t>& sample_input_shape) const {
    size_t elements = 1;
    for (const size_t dim : sample_input_shape) elements *= dim;
    return 3 * elements;
}

std::string SqueezeExcitationModule::GetName() const {
    return "SqueezeExcitation(channels=" + std::to_string(channels_) +
        ", squeeze=" + std::to_string(squeeze_channels_) + ")";
}

GlobalMaxPool2DModule::GlobalMaxPool2DModule()
    : layer_(std::make_unique<GlobalMaxPool2DLayer>()) {}

// As GlobalAvgPool2DModule: [C,N] from the layer, [N,C] rows for Dense.
Tensor GlobalMaxPool2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input).Transpose();
}

Tensor GlobalMaxPool2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output.Transpose());
}

std::string GlobalMaxPool2DModule::GetName() const {
    return "GlobalMaxPool2D";
}

AdaptiveAvgPool2DModule::AdaptiveAvgPool2DModule(int out_h, int out_w)
    : layer_(std::make_unique<AdaptiveAvgPool2DLayer>(out_h, out_w)),
      out_h_(out_h),
      out_w_(out_w) {}

Tensor AdaptiveAvgPool2DModule::Forward(const Tensor& input) {
    return layer_->Forward(input);
}

Tensor AdaptiveAvgPool2DModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::string AdaptiveAvgPool2DModule::GetName() const {
    return "AdaptiveAvgPool2D(" + std::to_string(out_h_) + "x" + std::to_string(out_w_) + ")";
}

} // namespace cyxwiz
