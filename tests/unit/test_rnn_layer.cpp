// Vanilla RNN CPU reference (tofix68 phase 3): shape contracts, a
// hand-checkable fixture, and numerical gradient verification — this layer
// is the parity oracle for the native provider's pilot kernel.

#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/layers/recurrent.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

cyxwiz::Tensor FilledTensor(const std::vector<size_t>& shape, float scale,
                            float phase) {
    cyxwiz::Tensor tensor(shape);
    float* data = tensor.Data<float>();
    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        data[i] = scale * std::sin(0.37f * static_cast<float>(i) + phase);
    }
    return tensor;
}

std::map<std::string, cyxwiz::Tensor> DeterministicWeights(
    cyxwiz::RNNLayer& layer) {
    auto params = layer.GetParameters();
    float value = -0.15f;
    for (auto& [name, tensor] : params) {
        if (name.find("grad_") != std::string::npos) continue;
        float* data = tensor.Data<float>();
        for (size_t i = 0; i < tensor.NumElements(); ++i) {
            value = 0.31f - value * 0.9f;  // deterministic, non-repeating-ish
            data[i] = value * 0.4f;
        }
    }
    layer.SetParameters(params);
    return params;
}

double LossOf(cyxwiz::RNNLayer& layer, const cyxwiz::Tensor& input) {
    cyxwiz::Tensor output = layer.Forward(input);
    const float* y = output.ReadData<float>();
    double loss = 0.0;
    for (size_t i = 0; i < output.NumElements(); ++i) {
        loss += 0.5 * static_cast<double>(y[i]) * static_cast<double>(y[i]);
    }
    return loss;
}

} // namespace

TEST_CASE("RNNLayer validates its contract and shapes",
          "[rnn][recurrent]") {
    CHECK_THROWS_AS(cyxwiz::RNNLayer(0, 4), std::invalid_argument);
    CHECK_THROWS_AS(cyxwiz::RNNLayer(4, 4, 1, true, true),
                    std::invalid_argument);
    CHECK_THROWS_AS(cyxwiz::RNNLayer(4, 4, 1, true, false, "gelu"),
                    std::invalid_argument);

    cyxwiz::RNNLayer layer(3, 5);
    const auto input = FilledTensor({2, 4, 3}, 0.5f, 0.0f);
    const auto output = layer.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{2, 4, 5});
    CHECK(layer.GetHiddenState().Shape() == std::vector<size_t>{2, 5});

    const auto grad = layer.Backward(FilledTensor({2, 4, 5}, 0.1f, 1.0f));
    CHECK(grad.Shape() == std::vector<size_t>{2, 4, 3});

    CHECK_THROWS_AS(layer.Forward(FilledTensor({2, 4, 4}, 0.1f, 0.0f)),
                    std::invalid_argument);
}

TEST_CASE("RNNLayer matches a hand-computed single-step fixture",
          "[rnn][recurrent]") {
    // 1 batch, 1 timestep, 1 input, 1 hidden: h = tanh(w*x + b_ih + b_hh).
    cyxwiz::RNNLayer layer(1, 1);
    auto params = layer.GetParameters();
    params.at("layer0_W_ih").Data<float>()[0] = 0.5f;
    params.at("layer0_W_hh").Data<float>()[0] = 0.25f;
    params.at("layer0_b_ih").Data<float>()[0] = 0.1f;
    params.at("layer0_b_hh").Data<float>()[0] = -0.05f;
    layer.SetParameters(params);

    cyxwiz::Tensor input(std::vector<size_t>{1, 1, 1});
    input.Data<float>()[0] = 0.8f;
    const auto output = layer.Forward(input);
    const float expected = std::tanh(0.5f * 0.8f + 0.1f - 0.05f);
    CHECK(std::fabs(output.ReadData<float>()[0] - expected) < 1e-6f);
}

TEST_CASE("RNNLayer backward matches numerical gradients",
          "[rnn][recurrent]") {
    for (const char* nonlinearity : {"tanh", "relu"}) {
        cyxwiz::RNNLayer layer(3, 4, 2, true, false, nonlinearity);
        DeterministicWeights(layer);
        const auto input = FilledTensor({2, 5, 3}, 0.6f, 0.3f);

        // Analytic: loss = 0.5*sum(y^2) -> upstream gradient is y itself.
        cyxwiz::Tensor output = layer.Forward(input);
        const auto analytic_input_grad = layer.Backward(output);
        const auto params = layer.GetParameters();

        constexpr float kEpsilon = 1e-3f;
        constexpr float kTolerance = 2e-2f;

        // Input gradient, spot-checked across positions.
        {
            auto probe = input.Clone();
            float* data = probe.Data<float>();
            const float* analytic = analytic_input_grad.ReadData<float>();
            for (size_t i = 0; i < probe.NumElements(); i += 7) {
                const float original = data[i];
                data[i] = original + kEpsilon;
                const double loss_plus = LossOf(layer, probe);
                data[i] = original - kEpsilon;
                const double loss_minus = LossOf(layer, probe);
                data[i] = original;
                const float numeric = static_cast<float>(
                    (loss_plus - loss_minus) / (2.0 * kEpsilon));
                INFO(nonlinearity << " input grad index " << i);
                CHECK(std::fabs(numeric - analytic[i]) < kTolerance);
            }
        }

        // Weight gradients, spot-checked for every parameter tensor.
        for (const auto& [name, tensor] : params) {
            if (name.find("grad_") != std::string::npos) continue;
            // "layer0_W_ih" -> "layer0_grad_W_ih" (single-digit layers).
            const auto grad_it = params.find(
                name.substr(0, 7) + "grad_" + name.substr(7));
            REQUIRE(grad_it != params.end());
            const float* analytic = grad_it->second.ReadData<float>();
            auto mutable_params = params;
            float* data = mutable_params.at(name).Data<float>();
            for (size_t i = 0; i < grad_it->second.NumElements(); i += 5) {
                const float original = data[i];
                data[i] = original + kEpsilon;
                layer.SetParameters(mutable_params);
                const double loss_plus = LossOf(layer, input);
                data[i] = original - kEpsilon;
                layer.SetParameters(mutable_params);
                const double loss_minus = LossOf(layer, input);
                data[i] = original;
                layer.SetParameters(mutable_params);
                const float numeric = static_cast<float>(
                    (loss_plus - loss_minus) / (2.0 * kEpsilon));
                INFO(nonlinearity << " " << name << " index " << i);
                CHECK(std::fabs(numeric - analytic[i]) < kTolerance);
            }
        }
    }
}

TEST_CASE("RNNModule reduces to the last step and re-expands its gradient",
          "[rnn][sequential]") {
    // Studio RNN wiring: the SequentialModel adapter mirrors LSTMModule.
    const auto input = FilledTensor({3, 5, 4}, 0.4f, 0.3f);

    cyxwiz::RNNModule last_step(4, 6, 1, false, "tanh");
    const auto out = last_step.Forward(input);
    REQUIRE(out.Shape() == std::vector<size_t>{3, 6});
    const auto dx = last_step.Backward(FilledTensor({3, 6}, 0.2f, 0.9f));
    REQUIRE(dx.Shape() == std::vector<size_t>{3, 5, 4});
    CHECK(last_step.GetName() == "RNN(4 -> 6, tanh, last)");

    // The last-step output equals the last timestep of the full sequence.
    cyxwiz::RNNModule full(4, 6, 1, true, "relu");
    full.SetParameters(last_step.GetParameters());
    cyxwiz::RNNModule full_tanh(4, 6, 1, true, "tanh");
    full_tanh.SetParameters(last_step.GetParameters());
    const auto seq = full_tanh.Forward(input);
    REQUIRE(seq.Shape() == std::vector<size_t>{3, 5, 6});
    const float* o = out.ReadData<float>();
    const float* q = seq.ReadData<float>();
    for (size_t b = 0; b < 3; ++b) {
        for (size_t j = 0; j < 6; ++j) {
            CHECK(o[b * 6 + j] == q[(b * 5 + 4) * 6 + j]);
        }
    }
    CHECK(full.GetName() == "RNN(4 -> 6, relu, seq)");

    // Gradients are exposed through the recurrent "grad_*" convention.
    const auto grads = last_step.GetGradients();
    CHECK(grads.count("layer0_grad_W_ih") == 1);
    CHECK(grads.count("layer0_grad_W_hh") == 1);
    CHECK(grads.count("layer0_grad_b_ih") == 1);
    CHECK(grads.count("layer0_grad_b_hh") == 1);
    CHECK(last_step.HasParameters());
}
