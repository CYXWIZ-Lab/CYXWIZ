#include "cyxwiz/fused_mlp.h"

#include "cyxwiz/neural_provider.h"
#include "cyxwiz/sequential.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace cyxwiz {

struct FusedMlp::Impl {
    SequentialModel* model = nullptr;
    std::shared_ptr<INeuralNetworkProvider> provider;
    NeuralOpRequest request;                 // the chain; batch set per call
    std::vector<std::string> weight_names;   // "layer<i>.weight", per Dense layer
    std::vector<std::string> bias_names;
    std::vector<std::vector<size_t>> weight_shapes, bias_shapes;
    size_t z_width = 0;
    Tensor x, z;                             // kept by Forward for Backward
    bool kept = false;
};

namespace {

// The activation module's code, or false when the fused kernels do not have it
bool ActivationOf(Module& module, NeuralActivation& act, NeuralOpRequest& request) {
    if (dynamic_cast<ReLUModule*>(&module)) act = NeuralActivation::Relu;
    else if (dynamic_cast<TanhModule*>(&module)) act = NeuralActivation::Tanh;
    else if (dynamic_cast<SigmoidModule*>(&module)) act = NeuralActivation::Sigmoid;
    else if (dynamic_cast<GELUModule*>(&module)) act = NeuralActivation::Gelu;
    else if (dynamic_cast<SwishModule*>(&module)) act = NeuralActivation::Swish;
    else if (dynamic_cast<SELUModule*>(&module)) act = NeuralActivation::Selu;
    else if (auto* leaky = dynamic_cast<LeakyReLUModule*>(&module)) {
        act = NeuralActivation::LeakyRelu;
        request.mlp_negative_slope = leaky->NegativeSlope();
    } else if (auto* elu = dynamic_cast<ELUModule*>(&module)) {
        act = NeuralActivation::Elu;
        request.mlp_elu_alpha = elu->Alpha();
    } else {
        return false;
    }
    return true;
}

}  // namespace

FusedMlp::FusedMlp(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FusedMlp::~FusedMlp() = default;

std::unique_ptr<FusedMlp> FusedMlp::For(SequentialModel& model, std::string& why) {
    auto impl = std::make_unique<Impl>();
    impl->model = &model;
    NeuralOpRequest& r = impl->request;
    r.op = NeuralOp::MlpForward;
    r.target = CaptureCurrentNeuralDeviceTarget();
    r.device_resident = true;
    r.dtype = DataType::Float32;
    r.batch = 1;
    float slope = -1.0f, alpha = -1.0f;  // one value per chain (the request holds one)
    size_t layers = 0;
    for (size_t i = 0; i < model.Size(); ++i) {
        Module* module = model.GetModule(i);
        if (auto* linear = dynamic_cast<LinearModule*>(module)) {
            if (layers == NeuralOpRequest::kMaxMlpLayers) {
                why = "more than " + std::to_string(NeuralOpRequest::kMaxMlpLayers) + " Dense layers";
                return nullptr;
            }
            const auto params = linear->GetParameters();
            const auto w = params.find("weight");
            const auto b = params.find("bias");
            if (w == params.end() || b == params.end() || w->second.Shape().size() != 2) {
                why = "a Dense layer without a bias";
                return nullptr;
            }
            const auto& shape = w->second.Shape();  // [out, in]
            if (layers > 0 && shape[1] != r.mlp_widths[layers]) {
                why = "the Dense layers do not chain";
                return nullptr;
            }
            r.mlp_widths[layers] = shape[1];
            r.mlp_widths[layers + 1] = shape[0];
            r.mlp_activations[layers] = NeuralActivation::None;
            impl->weight_names.push_back("layer" + std::to_string(i) + ".weight");
            impl->bias_names.push_back("layer" + std::to_string(i) + ".bias");
            impl->weight_shapes.push_back(shape);
            impl->bias_shapes.push_back(b->second.Shape());
            ++layers;
            continue;
        }
        NeuralActivation act = NeuralActivation::None;
        NeuralOpRequest probe;
        if (!module || !ActivationOf(*module, act, probe)) {
            why = (module ? module->GetName() : std::string("an empty slot")) + " is not a Dense layer or a fused activation";
            return nullptr;
        }
        if (layers == 0 || r.mlp_activations[layers - 1] != NeuralActivation::None) {
            why = "an activation without a Dense layer before it";
            return nullptr;
        }
        if (act == NeuralActivation::LeakyRelu) {
            if (slope >= 0.0f && slope != probe.mlp_negative_slope) {
                why = "LeakyReLU layers with different slopes";
                return nullptr;
            }
            slope = probe.mlp_negative_slope;
            r.mlp_negative_slope = slope;
        }
        if (act == NeuralActivation::Elu) {
            if (alpha >= 0.0f && alpha != probe.mlp_elu_alpha) {
                why = "ELU layers with different alphas";
                return nullptr;
            }
            alpha = probe.mlp_elu_alpha;
            r.mlp_elu_alpha = alpha;
        }
        r.mlp_activations[layers - 1] = act;
    }
    if (layers == 0) {
        why = "no Dense layer";
        return nullptr;
    }
    r.mlp_layers = layers;
    for (size_t l = 0; l < layers; ++l) impl->z_width += r.mlp_widths[l + 1];
    impl->provider = NeuralProviderRegistry::Instance().FindSupporting(r);
    if (!impl->provider) {
        why = std::string("no neural provider runs this network on ") + NeuralDevicePlatformName(r.target.platform);
        return nullptr;
    }
    NeuralOpRequest backward = r;
    backward.op = NeuralOp::MlpBackward;
    if (!impl->provider->QueryCapability(backward).supported) {
        why = std::string(impl->provider->ProviderId()) + " has no mlp_backward for this network";
        return nullptr;
    }
    return std::unique_ptr<FusedMlp>(new FusedMlp(std::move(impl)));
}

Tensor FusedMlp::Forward(const Tensor& x, bool keep) {
    Impl& m = *impl_;
    const auto& shape = x.Shape();
    if (shape.size() != 2 || shape[1] != m.request.mlp_widths[0])
        throw std::runtime_error("FusedMlp: the input must be [rows, " + std::to_string(m.request.mlp_widths[0]) + "]");
    NeuralOpRequest r = m.request;
    r.op = NeuralOp::MlpForward;
    r.batch = shape[0];
    r.training = keep;
    const auto params = m.model->GetParameters();
    std::vector<const Tensor*> inputs{&x};
    for (size_t l = 0; l < r.mlp_layers; ++l) {
        inputs.push_back(&params.at(m.weight_names[l]));
        inputs.push_back(&params.at(m.bias_names[l]));
    }
    std::vector<std::vector<size_t>> shapes{{r.batch, r.mlp_widths[r.mlp_layers]}};
    if (keep) shapes.push_back({r.batch, m.z_width});
    std::vector<Tensor> outputs;
    const NeuralOpStatus status = ExecuteNeuralOpOnDevice(*m.provider, r, inputs, shapes, outputs);
    if (!status.ok) throw std::runtime_error("FusedMlp forward: " + status.detail);
    m.kept = keep;
    if (keep) {
        m.x = x;
        m.z = std::move(outputs[1]);
    }
    return std::move(outputs[0]);
}

std::map<std::string, Tensor> FusedMlp::Backward(const Tensor& dy) {
    Impl& m = *impl_;
    if (!m.kept) throw std::runtime_error("FusedMlp: Backward needs a Forward with keep");
    NeuralOpRequest r = m.request;
    r.op = NeuralOp::MlpBackward;
    r.batch = m.x.Shape()[0];
    r.training = true;
    if (dy.Shape() != std::vector<size_t>{r.batch, r.mlp_widths[r.mlp_layers]})
        throw std::runtime_error("FusedMlp: dY must be [rows, outputs] of the kept Forward");
    const auto params = m.model->GetParameters();
    std::vector<const Tensor*> inputs{&m.x};
    for (size_t l = 0; l < r.mlp_layers; ++l) inputs.push_back(&params.at(m.weight_names[l]));
    inputs.push_back(&m.z);
    inputs.push_back(&dy);
    std::vector<std::vector<size_t>> shapes;
    for (size_t l = 0; l < r.mlp_layers; ++l) {
        shapes.push_back(m.weight_shapes[l]);
        shapes.push_back(m.bias_shapes[l]);
    }
    std::vector<Tensor> outputs;
    const NeuralOpStatus status = ExecuteNeuralOpOnDevice(*m.provider, r, inputs, shapes, outputs);
    if (!status.ok) throw std::runtime_error("FusedMlp backward: " + status.detail);
    std::map<std::string, Tensor> grads;
    for (size_t l = 0; l < r.mlp_layers; ++l) {
        grads[m.weight_names[l]] = std::move(outputs[2 * l]);
        grads[m.bias_names[l]] = std::move(outputs[2 * l + 1]);
    }
    return grads;
}

std::string FusedMlp::ProviderId() const { return impl_->provider->ProviderId(); }

}  // namespace cyxwiz
