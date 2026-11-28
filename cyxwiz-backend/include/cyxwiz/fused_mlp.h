// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// A SequentialModel that is a small chain of Dense layers and activations,
// run as whole-network kernels on the active device's neural provider
// (NeuralOp::MlpForward / MlpBackward): one launch forward, two backward,
// instead of several per layer. The model keeps its
// parameters; each call reads them, so optimizer steps on the model apply.
// Gradients are named as the model names them ("layer<i>.weight").

#include "api_export.h"
#include "neural_provider.h"
#include "tensor.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

class SequentialModel;

class CYXWIZ_API FusedMlp {
public:
    ~FusedMlp();
    FusedMlp(const FusedMlp&) = delete;
    FusedMlp& operator=(const FusedMlp&) = delete;

    // Why For gave no fused form. The three causes are told apart:
    //  - the model is not a fusable chain: `text` says what is in the way;
    //  - a provider serves the device and declines this network: `text` names
    //    it and gives its reason, `reason` is its reason code;
    //  - no provider serves the device: `no_provider` is set, `absence` is
    //    WhyNoNeuralProviderServes' answer (not built, not installed, another
    //    ABI, turned off, or none offered), `reason` its code, and `text` its
    //    name with what to do.
    struct Refusal {
        std::string text;
        std::string reason;  // BackendFallbackReasonName; empty for a chain that is not fusable
        bool no_provider = false;
        NeuralProviderAbsence absence;
    };

    // The fused form of `model` on the active device, or null with the
    // reason: not a Dense/activation chain (another layer, a layer without
    // bias, two activations in a row), outside the provider's limits (at most
    // 8 layers, inputs up to 1024, layers up to 512 units), or no provider
    // serves it on this device. When a provider was asked and none takes the
    // network, the thread's dispatch observer is told (layer "fused MLP").
    static std::unique_ptr<FusedMlp> For(SequentialModel& model, Refusal& refusal);
    // The same, with the reason in words only.
    static std::unique_ptr<FusedMlp> For(SequentialModel& model, std::string& why);

    // Y [rows, outputs] for X [rows, inputs]. `keep` keeps what Backward
    // needs. Each call goes through DispatchNeuralProviderOp (the run's
    // dispatch observer sees it, layer "fused MLP"). Throws
    // std::runtime_error when the call did not run (the provider failed or
    // declined it): the caller then runs the model's layers. Under the Fail
    // provider policy (a Strict run) a failure throws NeuralProviderRouteError
    // instead, which the caller lets through.
    Tensor Forward(const Tensor& x, bool keep);
    // The parameters' gradients for dY [rows, outputs] after a kept
    // Forward, each summed over the rows. Throws as Forward does.
    std::map<std::string, Tensor> Backward(const Tensor& dy);
    // The input's gradient dX [rows, inputs] for dY after a kept Forward,
    // without the parameters' (SAC's actor loss reaches the action through
    // the critics' input). Throws as Forward does.
    Tensor InputGradient(const Tensor& dy);

    // Two networks over the same X in one launch forward and two backward
    // for both (PPO's policy and value networks): the same results as each
    // one's Forward and Backward. Both on one provider; throws as Forward.
    using Gradients = std::map<std::string, Tensor>;
    static std::pair<Tensor, Tensor> ForwardPair(FusedMlp& first, FusedMlp& second, const Tensor& x, bool keep);
    // After a kept ForwardPair of the same two
    static std::pair<Gradients, Gradients> BackwardPair(FusedMlp& first, const Tensor& dy_first, FusedMlp& second,
                                                        const Tensor& dy_second);
    // Each one's InputGradient, in one launch (SAC's twin critics)
    static std::pair<Tensor, Tensor> InputGradientPair(FusedMlp& first, const Tensor& dy_first, FusedMlp& second,
                                                       const Tensor& dy_second);

    // The provider For found to run it ("cyxwiz.nvidia-cublas-cell"). Each
    // call asks the dispatch again; with two providers on one platform a
    // forward and its backward could run on different ones (the kept forward
    // follows the request contract, so either can read it).
    std::string ProviderId() const;

private:
    struct Impl;
    explicit FusedMlp(std::unique_ptr<Impl> impl);
    // The backward's outputs: the parameters' gradients or the input's
    std::vector<Tensor> RunBackward(const Tensor& dy, bool parameters);
    static std::vector<Tensor> RunBackwardPair(FusedMlp& first, const Tensor& dy_first, FusedMlp& second,
                                               const Tensor& dy_second, bool parameters);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cyxwiz
