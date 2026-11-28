#include "cyxwiz/fused_mlp.h"

#include "cyxwiz/neural_provider.h"
#include "cyxwiz/neural_provider_dispatch.h"
#include "cyxwiz/sequential.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyxwiz {

struct FusedMlp::Impl {
    SequentialModel* model = nullptr;
    std::string provider_id;                 // the provider the probe found, for the compute path
    NeuralOpRequest request;                 // the chain; batch set per call
    std::vector<std::string> weight_names;   // "layer<i>.weight", per Dense layer
    std::vector<std::string> bias_names;
    std::vector<std::vector<size_t>> weight_shapes, bias_shapes;
    size_t z_width = 0;
    Tensor x, z;                             // kept by Forward for Backward
    bool kept = false;
    const Impl* kept_with = nullptr;         // the other network of a kept ForwardPair
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

// A two-network request: `second`'s chain as the request's second network
void AddSecondNetwork(NeuralOpRequest& r, const NeuralOpRequest& second) {
    r.mlp_networks = 2;
    r.mlp2_layers = second.mlp_layers;
    std::copy(std::begin(second.mlp_widths), std::end(second.mlp_widths), std::begin(r.mlp2_widths));
    std::copy(std::begin(second.mlp_activations), std::end(second.mlp_activations), std::begin(r.mlp2_activations));
    r.mlp2_negative_slope = second.mlp_negative_slope;
    r.mlp2_elu_alpha = second.mlp_elu_alpha;
}

// One fused call through the provider dispatch (TOFIX233 B), so the run's
// observer sees it and its failure policy applies (a Strict run stops with
// NeuralProviderRouteError). Throws runtime_error when it did not run: the
// trainer then runs the network's layers.
std::vector<Tensor> RunFused(const NeuralOpRequest& r, const std::vector<const Tensor*>& inputs,
                             const std::vector<std::vector<size_t>>& shapes, const char* what) {
    std::vector<Tensor> outputs;
    NeuralOpStatus status;
    bool asked = false;
    const bool ran = DispatchNeuralProviderOp(r, "fused MLP", [&](INeuralNetworkProvider& provider) {
        asked = true;
        status = ExecuteNeuralOpOnDevice(provider, r, inputs, shapes, outputs);
        return status;
    });
    if (ran) return outputs;
    std::string why = status.detail;
    if (!asked) {  // declined: the serving provider's own reason (a row limit, say)
        const auto serving = NeuralProviderRegistry::Instance().ListServing(r.target);
        why = serving.empty() ? WhyNoNeuralProviderServes(r.target.platform, r.op).Sentence()
                              : std::string(serving.front()->ProviderId()) + " declined it: " +
                                    serving.front()->QueryCapability(r).detail;
    }
    throw std::runtime_error(std::string("FusedMlp ") + what + ": " + why);
}

}  // namespace

FusedMlp::FusedMlp(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FusedMlp::~FusedMlp() = default;

std::unique_ptr<FusedMlp> FusedMlp::For(SequentialModel& model, std::string& why) {
    Refusal refusal;
    auto fused = For(model, refusal);
    why = refusal.text;
    return fused;
}

std::unique_ptr<FusedMlp> FusedMlp::For(SequentialModel& model, Refusal& refusal) {
    refusal = Refusal{};
    std::string& why = refusal.text;
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
    // A probe, not a run: the calls themselves go through the dispatch
    if (!CheckNeuralRunTarget(r, "fused MLP")) {  // reported as target_mismatch
        refusal.reason = BackendFallbackReasonName(BackendFallbackReason::TargetMismatch);
        why = DescribeNeuralProviderAbsence(BackendFallbackReason::TargetMismatch, r.target.platform).Sentence();
        return nullptr;
    }
    auto& registry = NeuralProviderRegistry::Instance();
    const auto provider = registry.FindSupporting(r);
    if (!provider) {
        // An absent provider and a provider that declines this network are
        // different causes with different remedies: say which
        const auto serving = registry.ListServing(r.target);
        if (serving.empty()) {
            refusal.no_provider = true;
            refusal.absence = WhyNoNeuralProviderServes(r.target.platform, r.op);
            refusal.reason = BackendFallbackReasonName(refusal.absence.reason);
            why = refusal.absence.fixable ? refusal.absence.Sentence() : refusal.absence.name;
        } else {
            const NeuralCapability capability = serving.front()->QueryCapability(r);
            refusal.reason = BackendFallbackReasonName(capability.reason);
            why = std::string(serving.front()->ProviderId()) + " declined this network: " + capability.detail;
        }
        ReportNeuralProviderProbe(r, "fused MLP");
        return nullptr;
    }
    NeuralOpRequest backward = r;
    backward.op = NeuralOp::MlpBackward;
    if (const NeuralCapability capability = provider->QueryCapability(backward); !capability.supported) {
        refusal.reason = BackendFallbackReasonName(capability.reason);
        why = std::string(provider->ProviderId()) + " has no mlp_backward for this network";
        ReportNeuralProviderProbe(backward, "fused MLP");
        return nullptr;
    }
    impl->provider_id = provider->ProviderId();
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
    std::vector<Tensor> outputs = RunFused(r, inputs, shapes, "forward");
    m.kept = keep;
    m.kept_with = nullptr;
    if (keep) {
        m.x = x;
        m.z = std::move(outputs[1]);
    }
    return std::move(outputs[0]);
}

std::pair<Tensor, Tensor> FusedMlp::ForwardPair(FusedMlp& first, FusedMlp& second, const Tensor& x, bool keep) {
    Impl& a = *first.impl_;
    Impl& b = *second.impl_;
    const auto& shape = x.Shape();
    if (shape.size() != 2 || shape[1] != a.request.mlp_widths[0] || shape[1] != b.request.mlp_widths[0])
        throw std::runtime_error("FusedMlp: the input must be [rows, " + std::to_string(a.request.mlp_widths[0]) +
                                 "] for both networks");
    if (a.provider_id != b.provider_id) throw std::runtime_error("FusedMlp: the two networks run on different providers");
    NeuralOpRequest r = a.request;
    r.op = NeuralOp::MlpForward;
    r.batch = shape[0];
    r.training = keep;
    AddSecondNetwork(r, b.request);
    const auto params_a = a.model->GetParameters();
    const auto params_b = b.model->GetParameters();
    std::vector<const Tensor*> inputs{&x};
    for (size_t l = 0; l < a.request.mlp_layers; ++l) {
        inputs.push_back(&params_a.at(a.weight_names[l]));
        inputs.push_back(&params_a.at(a.bias_names[l]));
    }
    for (size_t l = 0; l < b.request.mlp_layers; ++l) {
        inputs.push_back(&params_b.at(b.weight_names[l]));
        inputs.push_back(&params_b.at(b.bias_names[l]));
    }
    std::vector<std::vector<size_t>> shapes{{r.batch, a.request.mlp_widths[a.request.mlp_layers]},
                                            {r.batch, b.request.mlp_widths[b.request.mlp_layers]}};
    if (keep) {
        shapes.push_back({r.batch, a.z_width});
        shapes.push_back({r.batch, b.z_width});
    }
    std::vector<Tensor> outputs = RunFused(r, inputs, shapes, "forward (two networks)");
    a.kept = b.kept = keep;
    a.kept_with = keep ? &b : nullptr;
    b.kept_with = keep ? &a : nullptr;
    if (keep) {
        a.x = b.x = x;
        a.z = std::move(outputs[2]);
        b.z = std::move(outputs[3]);
    }
    return {std::move(outputs[0]), std::move(outputs[1])};
}

std::vector<Tensor> FusedMlp::RunBackward(const Tensor& dy, bool parameters) {
    Impl& m = *impl_;
    if (!m.kept) throw std::runtime_error("FusedMlp: Backward needs a Forward with keep");
    NeuralOpRequest r = m.request;
    r.op = NeuralOp::MlpBackward;
    r.batch = m.x.Shape()[0];
    r.training = true;
    r.mlp_parameter_gradients = parameters;
    r.mlp_input_gradient = !parameters;
    if (dy.Shape() != std::vector<size_t>{r.batch, r.mlp_widths[r.mlp_layers]})
        throw std::runtime_error("FusedMlp: dY must be [rows, outputs] of the kept Forward");
    const auto params = m.model->GetParameters();
    std::vector<const Tensor*> inputs{&m.x};
    for (size_t l = 0; l < r.mlp_layers; ++l) inputs.push_back(&params.at(m.weight_names[l]));
    inputs.push_back(&m.z);
    inputs.push_back(&dy);
    std::vector<std::vector<size_t>> shapes;
    if (parameters) {
        for (size_t l = 0; l < r.mlp_layers; ++l) {
            shapes.push_back(m.weight_shapes[l]);
            shapes.push_back(m.bias_shapes[l]);
        }
    } else {
        shapes.push_back({r.batch, r.mlp_widths[0]});
    }
    return RunFused(r, inputs, shapes, "backward");
}

std::map<std::string, Tensor> FusedMlp::Backward(const Tensor& dy) {
    auto outputs = RunBackward(dy, true);
    const Impl& m = *impl_;
    std::map<std::string, Tensor> grads;
    for (size_t l = 0; l < m.request.mlp_layers; ++l) {
        grads[m.weight_names[l]] = std::move(outputs[2 * l]);
        grads[m.bias_names[l]] = std::move(outputs[2 * l + 1]);
    }
    return grads;
}

Tensor FusedMlp::InputGradient(const Tensor& dy) { return std::move(RunBackward(dy, false).front()); }

std::vector<Tensor> FusedMlp::RunBackwardPair(FusedMlp& first, const Tensor& dy_first, FusedMlp& second,
                                              const Tensor& dy_second, bool parameters) {
    Impl& a = *first.impl_;
    Impl& b = *second.impl_;
    if (!a.kept || !b.kept || a.kept_with != &b || b.kept_with != &a)
        throw std::runtime_error("FusedMlp: BackwardPair needs a ForwardPair of the same two with keep");
    NeuralOpRequest r = a.request;
    r.op = NeuralOp::MlpBackward;
    r.batch = a.x.Shape()[0];
    r.training = true;
    r.mlp_parameter_gradients = parameters;
    r.mlp_input_gradient = !parameters;
    AddSecondNetwork(r, b.request);
    if (dy_first.Shape() != std::vector<size_t>{r.batch, a.request.mlp_widths[a.request.mlp_layers]} ||
        dy_second.Shape() != std::vector<size_t>{r.batch, b.request.mlp_widths[b.request.mlp_layers]})
        throw std::runtime_error("FusedMlp: each dY must be [rows, outputs] of the kept ForwardPair");
    const auto params_a = a.model->GetParameters();
    const auto params_b = b.model->GetParameters();
    std::vector<const Tensor*> inputs{&a.x};
    for (size_t l = 0; l < a.request.mlp_layers; ++l) inputs.push_back(&params_a.at(a.weight_names[l]));
    inputs.push_back(&a.z);
    inputs.push_back(&dy_first);
    for (size_t l = 0; l < b.request.mlp_layers; ++l) inputs.push_back(&params_b.at(b.weight_names[l]));
    inputs.push_back(&b.z);
    inputs.push_back(&dy_second);
    std::vector<std::vector<size_t>> shapes;
    if (parameters) {
        for (const Impl* m : {&a, &b})
            for (size_t l = 0; l < m->request.mlp_layers; ++l) {
                shapes.push_back(m->weight_shapes[l]);
                shapes.push_back(m->bias_shapes[l]);
            }
    } else {
        shapes.assign(2, {r.batch, r.mlp_widths[0]});
    }
    return RunFused(r, inputs, shapes, "backward (two networks)");
}

std::pair<FusedMlp::Gradients, FusedMlp::Gradients> FusedMlp::BackwardPair(FusedMlp& first, const Tensor& dy_first,
                                                                            FusedMlp& second, const Tensor& dy_second) {
    auto outputs = RunBackwardPair(first, dy_first, second, dy_second, true);
    const Impl& a = *first.impl_;
    const Impl& b = *second.impl_;
    std::pair<Gradients, Gradients> grads;
    size_t o = 0;
    for (size_t l = 0; l < a.request.mlp_layers; ++l, o += 2) {
        grads.first[a.weight_names[l]] = std::move(outputs[o]);
        grads.first[a.bias_names[l]] = std::move(outputs[o + 1]);
    }
    for (size_t l = 0; l < b.request.mlp_layers; ++l, o += 2) {
        grads.second[b.weight_names[l]] = std::move(outputs[o]);
        grads.second[b.bias_names[l]] = std::move(outputs[o + 1]);
    }
    return grads;
}

std::pair<Tensor, Tensor> FusedMlp::InputGradientPair(FusedMlp& first, const Tensor& dy_first, FusedMlp& second,
                                                      const Tensor& dy_second) {
    auto outputs = RunBackwardPair(first, dy_first, second, dy_second, false);
    return {std::move(outputs[0]), std::move(outputs[1])};
}

std::string FusedMlp::ProviderId() const { return impl_->provider_id; }

}  // namespace cyxwiz
