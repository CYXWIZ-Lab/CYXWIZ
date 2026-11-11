// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// A SequentialModel that is a small chain of Dense layers and activations,
// run as whole-network kernels on the active device's neural provider
// (NeuralOp::MlpForward / MlpBackward): one launch forward, two backward,
// instead of several per layer (TOFIX206 3.2). The model keeps its
// parameters; each call reads them, so optimizer steps on the model apply.
// Gradients are named as the model names them ("layer<i>.weight").

#include "api_export.h"
#include "tensor.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace cyxwiz {

class SequentialModel;

class CYXWIZ_API FusedMlp {
public:
    ~FusedMlp();
    FusedMlp(const FusedMlp&) = delete;
    FusedMlp& operator=(const FusedMlp&) = delete;

    // The fused form of `model` on the active device, or null with the
    // reason: not a Dense/activation chain (another layer, a layer without
    // bias, two activations in a row), outside the provider's limits (at most
    // 8 layers, inputs up to 1024, layers up to 256 units), or no provider
    // serves it on this device.
    static std::unique_ptr<FusedMlp> For(SequentialModel& model, std::string& why);

    // Y [rows, outputs] for X [rows, inputs]. `keep` keeps what Backward
    // needs. Throws std::runtime_error when the provider fails.
    Tensor Forward(const Tensor& x, bool keep);
    // The parameters' gradients for dY [rows, outputs] after a kept
    // Forward, each summed over the rows. Throws as Forward does.
    std::map<std::string, Tensor> Backward(const Tensor& dy);

    // Two networks over the same X in one launch forward and two backward
    // for both (PPO's policy and value networks): the same results as each
    // one's Forward and Backward. Both on one provider; throws as Forward.
    using Gradients = std::map<std::string, Tensor>;
    static std::pair<Tensor, Tensor> ForwardPair(FusedMlp& first, FusedMlp& second, const Tensor& x, bool keep);
    // After a kept ForwardPair of the same two
    static std::pair<Gradients, Gradients> BackwardPair(FusedMlp& first, const Tensor& dy_first, FusedMlp& second,
                                                        const Tensor& dy_second);

    // The provider that runs it ("cyxwiz.nvidia-cublas-cell")
    std::string ProviderId() const;

private:
    struct Impl;
    explicit FusedMlp(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace cyxwiz
