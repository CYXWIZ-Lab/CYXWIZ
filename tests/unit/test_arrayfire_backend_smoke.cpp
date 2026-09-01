#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/layers/attention.h>
#include <cyxwiz/layers/dense.h>
#include <cyxwiz/layers/recurrent.h>
#include <cyxwiz/losses/classification.h>
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

void CountCrossEntropyHostSync(
    const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_cross_entropy_host_sync_count;
}

bool ShapeEquals(const cyxwiz::Tensor& tensor, std::vector<size_t> expected) {
    return tensor.Shape() == expected;
}

#ifndef NDEBUG
constexpr const char* kForceFallbackEnv =
    "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";

void SetEnvVar(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void ClearEnvVar(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value)
        : name_(name) {
        const char* previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        SetEnvVar(name_, value);
    }

    ~ScopedEnvVar() {
        if (had_previous_) {
            SetEnvVar(name_, previous_.c_str());
        } else {
            ClearEnvVar(name_);
        }
    }

private:
    const char* name_;
    bool had_previous_ = false;
    std::string previous_;
};
#endif

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

#ifndef NDEBUG
TEST_CASE("Forced ArrayFire Dense fallback returns CPU-equivalent tensors",
          "[arrayfire][fallback][dense]") {
    ScopedEnvVar env(
        kForceFallbackEnv, "DenseLayer::Forward,DenseLayer::Backward");

    cyxwiz::DenseLayer dense(3, 2, true);
    float weight_values[] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 1.0f,
    };
    float bias_values[] = {0.5f, -0.5f};
    dense.SetParameters({
        {"weights",
         cyxwiz::Tensor({2, 3}, weight_values, cyxwiz::DataType::Float32)},
        {"bias",
         cyxwiz::Tensor({2}, bias_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 2.0f,
    };
    cyxwiz::Tensor input({2, 3}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = dense.Forward(input);
    REQUIRE(ShapeEquals(output, {2, 2}));
    const float* output_data = output.Data<float>();
    REQUIRE(output_data[0] == Catch::Approx(14.5f));
    REQUIRE(output_data[1] == Catch::Approx(2.5f));
    REQUIRE(output_data[2] == Catch::Approx(5.5f));
    REQUIRE(output_data[3] == Catch::Approx(2.5f));

    float grad_values[] = {
        1.0f, 2.0f,
        -1.0f, 0.5f,
    };
    cyxwiz::Tensor grad_output({2, 2}, grad_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor grad_input = dense.Backward(grad_output);
    REQUIRE(ShapeEquals(grad_input, {2, 3}));
    const float* grad_input_data = grad_input.Data<float>();
    REQUIRE(grad_input_data[0] == Catch::Approx(-1.0f));
    REQUIRE(grad_input_data[1] == Catch::Approx(3.0f));
    REQUIRE(grad_input_data[2] == Catch::Approx(5.0f));
    REQUIRE(grad_input_data[3] == Catch::Approx(-1.5f));
    REQUIRE(grad_input_data[4] == Catch::Approx(-1.75f));
    REQUIRE(grad_input_data[5] == Catch::Approx(-2.5f));

    const std::map<std::string, cyxwiz::Tensor> params =
        dense.GetParameters();
    const float* grad_weight_data = params.at("grad_weights").Data<float>();
    REQUIRE(grad_weight_data[0] == Catch::Approx(2.0f));
    REQUIRE(grad_weight_data[1] == Catch::Approx(2.0f));
    REQUIRE(grad_weight_data[2] == Catch::Approx(1.0f));
    REQUIRE(grad_weight_data[3] == Catch::Approx(1.5f));
    REQUIRE(grad_weight_data[4] == Catch::Approx(4.0f));
    REQUIRE(grad_weight_data[5] == Catch::Approx(7.0f));

    const float* grad_bias_data = params.at("grad_bias").Data<float>();
    REQUIRE(grad_bias_data[0] == Catch::Approx(0.0f));
    REQUIRE(grad_bias_data[1] == Catch::Approx(2.5f));
}

TEST_CASE("Strict ArrayFire policy rejects forced Dense native CPU fallback",
          "[arrayfire][fallback][dense][policy]") {
    ScopedEnvVar env(kForceFallbackEnv, "DenseLayer::Forward");
    const cyxwiz::ScopedArrayFireFallbackPolicy strict(
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);

    cyxwiz::DenseLayer dense(3, 2, true);
    float input_values[] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 2.0f,
    };
    cyxwiz::Tensor input({2, 3}, input_values, cyxwiz::DataType::Float32);

    bool threw = false;
    try {
        (void)dense.Forward(input);
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string message = e.what();
        REQUIRE(message.find("DenseLayer::Forward") != std::string::npos);
        REQUIRE(message.find("native CPU fallback is forbidden") !=
                std::string::npos);
    }
    REQUIRE(threw);
}
#endif

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
#endif
