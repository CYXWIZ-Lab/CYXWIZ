#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/layers/attention.h>
#include <cyxwiz/layers/dense.h>
#include <cyxwiz/layers/recurrent.h>
#include <cyxwiz/losses/classification.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/tensor.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

size_t g_cross_entropy_host_sync_count = 0;
size_t g_recurrent_host_sync_count = 0;

void CountCrossEntropyHostSync(
    const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_cross_entropy_host_sync_count;
}

void CountRecurrentHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_recurrent_host_sync_count;
}

bool ShapeEquals(const cyxwiz::Tensor& tensor, std::vector<size_t> expected) {
    return tensor.Shape() == expected;
}


} // namespace

TEST_CASE("ArrayFire-capable Dense path returns clean forward and backward tensors",
          "[arrayfire][backend_smoke][dense]") {
    cyxwiz::DenseLayer dense(3, 2, true);

    float input_values[] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 2.0f,
    };
    cyxwiz::Tensor input({2, 3}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = dense.Forward(input);
    REQUIRE(ShapeEquals(output, {2, 2}));

    float grad_values[] = {
        1.0f, 0.5f,
        -0.25f, 2.0f,
    };
    cyxwiz::Tensor grad_output({2, 2}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = dense.Backward(grad_output);
    REQUIRE(ShapeEquals(grad_input, {2, 3}));
}

TEST_CASE("ArrayFire recurrent paths return clean LSTM and GRU tensors",
          "[arrayfire][backend_smoke][recurrent]") {
    std::vector<float> input_values(2 * 3 * 4, 0.25f);
    cyxwiz::Tensor input({2, 3, 4}, input_values.data(), cyxwiz::DataType::Float32);

    cyxwiz::LSTMLayer lstm(4, 5, 1, true, false, 0.0f);
    cyxwiz::Tensor lstm_output = lstm.Forward(input);
    REQUIRE(ShapeEquals(lstm_output, {2, 3, 5}));

    cyxwiz::GRULayer gru(4, 5, 1, true, false, 0.0f);
    cyxwiz::Tensor gru_output = gru.Forward(input);
    REQUIRE(ShapeEquals(gru_output, {2, 3, 5}));
}

TEST_CASE("Attention path returns clean self-attention tensor",
          "[arrayfire][backend_smoke][attention]") {
    cyxwiz::MultiHeadAttentionLayer attention(4, 2, 0.0f, true);

    float input_values[] = {
        1.0f, 0.0f, 0.5f, -0.5f,
        0.0f, 1.0f, 0.25f, 0.75f,
        0.5f, 0.5f, 1.0f, 0.0f,
    };
    cyxwiz::Tensor input({1, 3, 4}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = attention.Forward(input);
    REQUIRE(ShapeEquals(output, {1, 3, 4}));

    cyxwiz::Tensor grad_output = cyxwiz::Tensor::Ones({1, 3, 4});
    cyxwiz::Tensor grad_input = attention.Backward(grad_output);
    REQUIRE(ShapeEquals(grad_input, {1, 3, 4}));
}

TEST_CASE("Loss path returns clean forward value and gradient",
          "[arrayfire][backend_smoke][loss]") {
    float logit_values[] = {
        1.0f, 2.0f, -1.0f,
        0.5f, -0.5f, 1.0f,
    };
    int32_t target_values[] = {1, 2};
    cyxwiz::Tensor logits({2, 3}, logit_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({2}, target_values, cyxwiz::DataType::Int32);

    cyxwiz::CrossEntropyLoss loss(cyxwiz::Reduction::Mean, -100);
    cyxwiz::Tensor loss_value = loss.Forward(logits, targets);
    REQUIRE(ShapeEquals(loss_value, {1}));
    REQUIRE(std::isfinite(loss_value.Data<float>()[0]));

    cyxwiz::Tensor grad = loss.Backward(logits, targets);
    REQUIRE(ShapeEquals(grad, {2, 3}));
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Weighted smoothed CrossEntropy keeps device logits resident",
          "[arrayfire][backend_smoke][loss][residency]") {
    float logit_values[] = {
        2.0f, 0.0f, -1.0f,
        0.0f, 1.0f, 2.0f,
    };
    float target_values[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    cyxwiz::Tensor host_logits(
        {2, 3}, logit_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor logits = cyxwiz::Tensor::FromSemanticArray(
        host_logits.GetSemanticArray(), host_logits.Shape());
    cyxwiz::Tensor targets(
        {2, 3}, target_values, cyxwiz::DataType::Float32);
    cyxwiz::CrossEntropyLoss loss(
        cyxwiz::Reduction::Mean,
        -100,
        {1.0f, 2.0f, 4.0f},
        0.1f);

    cyxwiz::Tensor loss_value;
    cyxwiz::Tensor grad;
    g_cross_entropy_host_sync_count = 0;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CountCrossEntropyHostSync);
        loss_value = loss.Forward(logits, targets);
        grad = loss.Backward(logits, targets);
    }

    REQUIRE(g_cross_entropy_host_sync_count == 0);
    REQUIRE(ShapeEquals(loss_value, {1}));
    REQUIRE(ShapeEquals(grad, {2, 3}));
    REQUIRE(std::isfinite(loss_value.ReadData<float>()[0]));
}

TEST_CASE("Recurrent layers keep forward and backward on the device",
          "[arrayfire][backend_smoke][recurrent][residency]") {
    // The ArrayFire recurrence (providers hidden), uni- and bidirectional,
    // stacked: no host read between the device input and the device dx.
    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    struct RestoreProviders {
        ~RestoreProviders() { cyxwiz::SetNeuralProvidersDisabledForTesting(false); }
    } restore_providers;
    std::vector<float> values(2 * 3 * 4);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.1f * static_cast<float>(i % 7) - 0.3f;
    }
    const cyxwiz::Tensor host_input({2, 3, 4}, values.data(), cyxwiz::DataType::Float32);
    const cyxwiz::Tensor input = cyxwiz::Tensor::FromSemanticArray(
        host_input.GetSemanticArray(), host_input.Shape());
    for (const bool bidirectional : {false, true}) {
        const size_t features = bidirectional ? 10 : 5;
        const cyxwiz::Tensor grad = cyxwiz::Tensor::FromSemanticArray(
            af::constant(0.5f, af::dim4(2, 3, static_cast<dim_t>(features))), {2, 3, features});
        cyxwiz::LSTMLayer lstm(4, 5, 2, true, bidirectional, 0.0f);
        cyxwiz::GRULayer gru(4, 5, 2, true, bidirectional, 0.0f);
        cyxwiz::Tensor lstm_dx;
        cyxwiz::Tensor gru_dx;
        g_recurrent_host_sync_count = 0;
        {
            const cyxwiz::ScopedArrayFireHostSyncObserver observer(&CountRecurrentHostSync);
            lstm.Forward(input);
            lstm_dx = lstm.Backward(grad);
            gru.Forward(input);
            gru_dx = gru.Backward(grad);
        }
        INFO("bidirectional=" << bidirectional);
        REQUIRE(g_recurrent_host_sync_count == 0);
        REQUIRE(ShapeEquals(lstm_dx, {2, 3, 4}));
        REQUIRE(ShapeEquals(gru_dx, {2, 3, 4}));
    }
}
#endif
