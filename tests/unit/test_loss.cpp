#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "algorithms/arrayfire_backend_utils.h"
#include <cyxwiz/device.h>
#include <cyxwiz/loss.h>
#include <cyxwiz/tensor.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace {

constexpr const char* kForceFallbackEnv =
    "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";
std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent>* g_loss_fallback_events =
    nullptr;
std::vector<cyxwiz::ArrayFireHostSyncEvent>* g_loss_host_sync_events = nullptr;

void CaptureLossFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent& event) {
    if (g_loss_fallback_events != nullptr) {
        g_loss_fallback_events->push_back(event);
    }
}

void CaptureLossHostSync(const cyxwiz::ArrayFireHostSyncEvent& event) {
    if (g_loss_host_sync_events != nullptr) {
        g_loss_host_sync_events->push_back(event);
    }
}

void SetLossFallbackEnv(const char* value) {
#ifdef _WIN32
    _putenv_s(kForceFallbackEnv, value);
#else
    setenv(kForceFallbackEnv, value, 1);
#endif
}

class ScopedLossFallbackEnv {
public:
    explicit ScopedLossFallbackEnv(const char* value) {
        const char* previous = std::getenv(kForceFallbackEnv);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        SetLossFallbackEnv(value);
    }

    ~ScopedLossFallbackEnv() {
        if (had_previous_) {
            SetLossFallbackEnv(previous_.c_str());
        } else {
            SetLossFallbackEnv("");
        }
    }

private:
    bool had_previous_ = false;
    std::string previous_;
};

class ScopedLossEventCapture {
public:
    ScopedLossEventCapture(
        std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent>& fallback_events,
        std::vector<cyxwiz::ArrayFireHostSyncEvent>& host_sync_events) {
        g_loss_fallback_events = &fallback_events;
        g_loss_host_sync_events = &host_sync_events;
    }

    ~ScopedLossEventCapture() {
        g_loss_fallback_events = nullptr;
        g_loss_host_sync_events = nullptr;
    }
};

#if defined(CYXWIZ_HAS_ARRAYFIRE) && !defined(NDEBUG)
using LossFactory = std::function<std::unique_ptr<cyxwiz::Loss>()>;

void RequireRegressionLossFallbackContract(
    const LossFactory& factory,
    const std::string& operation_name,
    bool forward) {
    const float prediction_values[] = {-2.5f, -0.5f, 0.0f, 0.75f, 3.0f, 1.0f};
    const float target_values[] = {0.0f, -1.0f, 0.0f, 0.25f, 0.0f, -2.0f};
    const auto make_predictions = [&]() {
        return cyxwiz::Tensor(af::array(2, 3, prediction_values));
    };
    const auto make_targets = [&]() {
        return cyxwiz::Tensor(af::array(2, 3, target_values));
    };
    const auto invoke = [forward](cyxwiz::Loss& loss,
                                  const cyxwiz::Tensor& predictions,
                                  const cyxwiz::Tensor& targets) {
        return forward ? loss.Forward(predictions, targets)
                       : loss.Backward(predictions, targets);
    };

    std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent> strict_fallback_events;
    std::vector<cyxwiz::ArrayFireHostSyncEvent> strict_host_sync_events;
    {
        const ScopedLossEventCapture capture(
            strict_fallback_events, strict_host_sync_events);
        const ScopedLossFallbackEnv forced(operation_name.c_str());
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CaptureLossFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &CaptureLossHostSync);
        auto loss = factory();
        const auto predictions = make_predictions();
        const auto targets = make_targets();
        REQUIRE_THROWS_AS(invoke(*loss, predictions, targets), std::runtime_error);
    }
    REQUIRE(strict_fallback_events.size() == 1);
    REQUIRE(strict_fallback_events.front().fallback_forbidden);
    REQUIRE(strict_host_sync_events.empty());

    auto arrayfire_loss = factory();
    const auto arrayfire_result = invoke(
        *arrayfire_loss, make_predictions(), make_targets());

    std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent> fallback_events;
    std::vector<cyxwiz::ArrayFireHostSyncEvent> host_sync_events;
    cyxwiz::Tensor native_result;
    {
        const ScopedLossEventCapture capture(fallback_events, host_sync_events);
        const ScopedLossFallbackEnv forced(operation_name.c_str());
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CaptureLossFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &CaptureLossHostSync);
        auto loss = factory();
        native_result = invoke(*loss, make_predictions(), make_targets());
    }

    REQUIRE(fallback_events.size() == 1);
    REQUIRE(fallback_events.front().operation_name == operation_name);
    REQUIRE(fallback_events.front().reason_code == "gpu_backend_exception");
    REQUIRE_FALSE(fallback_events.front().fallback_forbidden);
    REQUIRE_FALSE(host_sync_events.empty());
    for (const auto& event : host_sync_events) {
        REQUIRE(event.attribution_category == "loss_cpu_path");
        REQUIRE(event.attribution_operation == operation_name);
    }

    REQUIRE(native_result.Shape() == arrayfire_result.Shape());
    const float* expected = arrayfire_result.ReadData<float>();
    const float* actual = native_result.ReadData<float>();
    for (size_t index = 0; index < native_result.NumElements(); ++index) {
        REQUIRE(actual[index] == Catch::Approx(expected[index]).margin(1.0e-6f));
    }
}
#endif

} // namespace

TEST_CASE("Core losses compute CPU forward reductions", "[loss]") {
    float pred_values[] = {1.0f, 3.0f, -2.0f};
    float target_values[] = {0.0f, 1.0f, 1.0f};
    cyxwiz::Tensor predictions({3}, pred_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({3}, target_values, cyxwiz::DataType::Float32);

    auto mse = cyxwiz::CreateLoss(cyxwiz::LossType::MSE, cyxwiz::Reduction::Mean);
    auto l1 = cyxwiz::CreateLoss(cyxwiz::LossType::L1, cyxwiz::Reduction::Sum);
    auto smooth_l1 = cyxwiz::CreateLoss(cyxwiz::LossType::SmoothL1, cyxwiz::Reduction::None, 1.0f);

    cyxwiz::Tensor mse_loss = mse->Forward(predictions, targets);
    cyxwiz::Tensor l1_loss = l1->Forward(predictions, targets);
    cyxwiz::Tensor smooth_l1_loss = smooth_l1->Forward(predictions, targets);

    REQUIRE(mse_loss.Data<float>()[0] == Catch::Approx((1.0f + 4.0f + 9.0f) / 3.0f));
    REQUIRE(l1_loss.Data<float>()[0] == Catch::Approx(6.0f));

    REQUIRE(smooth_l1_loss.Shape() == std::vector<size_t>{3});
    REQUIRE(smooth_l1_loss.Data<float>()[0] == Catch::Approx(0.5f));
    REQUIRE(smooth_l1_loss.Data<float>()[1] == Catch::Approx(1.5f));
    REQUIRE(smooth_l1_loss.Data<float>()[2] == Catch::Approx(2.5f));
}

TEST_CASE("Mean loss globally reduces multi-output tensors", "[loss]") {
    float pred_values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float target_values[] = {
        0.0f, 2.0f, 4.0f,
        2.0f, 5.0f, 8.0f
    };
    cyxwiz::Tensor predictions({2, 3}, pred_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({2, 3}, target_values, cyxwiz::DataType::Float32);

    auto mse = cyxwiz::CreateLoss(cyxwiz::LossType::MSE, cyxwiz::Reduction::Mean);
    cyxwiz::Tensor loss = mse->Forward(predictions, targets);
    cyxwiz::Tensor grad = mse->Backward(predictions, targets);

    REQUIRE(loss.NumElements() == 1);
    REQUIRE(loss.Data<float>()[0] == Catch::Approx(10.0f / 6.0f));
    REQUIRE(grad.Shape() == predictions.Shape());
}

TEST_CASE("Core losses compute CPU backward values", "[loss]") {
    float pred_values[] = {1.0f, 3.0f, -2.0f};
    float target_values[] = {0.0f, 1.0f, 1.0f};
    cyxwiz::Tensor predictions({3}, pred_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({3}, target_values, cyxwiz::DataType::Float32);

    auto mse = cyxwiz::CreateLoss(cyxwiz::LossType::MSE, cyxwiz::Reduction::Mean);
    auto l1 = cyxwiz::CreateLoss(cyxwiz::LossType::L1, cyxwiz::Reduction::Mean);
    auto huber = cyxwiz::CreateLoss(cyxwiz::LossType::Huber, cyxwiz::Reduction::Mean, 2.0f);

    cyxwiz::Tensor mse_grad = mse->Backward(predictions, targets);
    cyxwiz::Tensor l1_grad = l1->Backward(predictions, targets);
    cyxwiz::Tensor huber_grad = huber->Backward(predictions, targets);

    REQUIRE(mse_grad.Data<float>()[0] == Catch::Approx(2.0f / 3.0f));
    REQUIRE(mse_grad.Data<float>()[1] == Catch::Approx(4.0f / 3.0f));
    REQUIRE(mse_grad.Data<float>()[2] == Catch::Approx(-2.0f));

    REQUIRE(l1_grad.Data<float>()[0] == Catch::Approx(1.0f / 3.0f));
    REQUIRE(l1_grad.Data<float>()[1] == Catch::Approx(1.0f / 3.0f));
    REQUIRE(l1_grad.Data<float>()[2] == Catch::Approx(-1.0f / 3.0f));

    REQUIRE(huber_grad.Data<float>()[0] == Catch::Approx(1.0f / 3.0f));
    REQUIRE(huber_grad.Data<float>()[1] == Catch::Approx(2.0f / 3.0f));
    REQUIRE(huber_grad.Data<float>()[2] == Catch::Approx(-2.0f / 3.0f));
}

TEST_CASE("Regression loss parameters follow PyTorch domains", "[loss][regression]") {
    REQUIRE_NOTHROW(cyxwiz::SmoothL1Loss(0.0f, cyxwiz::Reduction::Mean));
    REQUIRE_THROWS_AS(
        cyxwiz::SmoothL1Loss(-0.1f, cyxwiz::Reduction::Mean),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::SmoothL1Loss(
            std::numeric_limits<float>::infinity(), cyxwiz::Reduction::Mean),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::HuberLoss(0.0f, cyxwiz::Reduction::Mean),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::HuberLoss(-1.0f, cyxwiz::Reduction::Mean),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::HuberLoss(
            std::numeric_limits<float>::quiet_NaN(), cyxwiz::Reduction::Mean),
        std::invalid_argument);
}

TEST_CASE("SmoothL1 beta zero is exactly L1", "[loss][regression]") {
    const float prediction_values[] = {-2.0f, 0.0f, 3.0f};
    const float target_values[] = {-1.0f, 0.0f, 1.0f};
    const cyxwiz::Tensor predictions(
        {3}, prediction_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor targets(
        {3}, target_values, cyxwiz::DataType::Float32);
    cyxwiz::SmoothL1Loss smooth(0.0f, cyxwiz::Reduction::None);
    cyxwiz::L1Loss l1(cyxwiz::Reduction::None);
    const auto smooth_forward = smooth.Forward(predictions, targets);
    const auto l1_forward = l1.Forward(predictions, targets);
    const auto smooth_backward = smooth.Backward(predictions, targets);
    const auto l1_backward = l1.Backward(predictions, targets);
    for (size_t index = 0; index < predictions.NumElements(); ++index) {
        REQUIRE(smooth_forward.ReadData<float>()[index] ==
                Catch::Approx(l1_forward.ReadData<float>()[index]));
        REQUIRE(smooth_backward.ReadData<float>()[index] ==
                Catch::Approx(l1_backward.ReadData<float>()[index]));
    }
}

#if defined(CYXWIZ_HAS_ARRAYFIRE) && !defined(NDEBUG)
TEST_CASE("Regression losses declare strict and compatible fallback truth",
          "[loss][regression][arrayfire][fallback]") {
    const std::vector<std::pair<LossFactory, std::string>> losses = {
        {[] { return std::make_unique<cyxwiz::MSELoss>(cyxwiz::Reduction::None); },
         "MSELoss"},
        {[] { return std::make_unique<cyxwiz::L1Loss>(cyxwiz::Reduction::None); },
         "L1Loss"},
        {[] { return std::make_unique<cyxwiz::SmoothL1Loss>(
                   0.5f, cyxwiz::Reduction::None); },
         "SmoothL1Loss"},
        {[] { return std::make_unique<cyxwiz::HuberLoss>(
                   2.0f, cyxwiz::Reduction::None); },
         "HuberLoss"},
    };
    for (const auto& [factory, name] : losses) {
        DYNAMIC_SECTION(name << " forward") {
            RequireRegressionLossFallbackContract(
                factory, name + "::Forward", true);
        }
        DYNAMIC_SECTION(name << " backward") {
            RequireRegressionLossFallbackContract(
                factory, name + "::Backward", false);
        }
    }
}
#endif

TEST_CASE("Binary losses compute forward reductions", "[loss]") {
    float probability_values[] = {0.8f, 0.2f};
    float label_values[] = {1.0f, 0.0f};
    cyxwiz::Tensor probabilities({2}, probability_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor labels({2}, label_values, cyxwiz::DataType::Float32);

    auto bce = cyxwiz::CreateLoss(cyxwiz::LossType::BinaryCrossEntropy, cyxwiz::Reduction::Mean);
    cyxwiz::Tensor bce_loss = bce->Forward(probabilities, labels);
    REQUIRE(bce_loss.Data<float>()[0] == Catch::Approx(-std::log(0.8f)));

    float logit_values[] = {0.0f, 2.0f};
    cyxwiz::Tensor logits({2}, logit_values, cyxwiz::DataType::Float32);
    auto bce_logits = cyxwiz::CreateLoss(cyxwiz::LossType::BCEWithLogits, cyxwiz::Reduction::Sum);
    cyxwiz::Tensor logits_loss = bce_logits->Forward(logits, labels);
    REQUIRE(logits_loss.Data<float>()[0] ==
            Catch::Approx(std::log(2.0f) + 2.0f + std::log1p(std::exp(-2.0f))));
}

TEST_CASE("Binary losses compute backward values", "[loss]") {
    float probability_values[] = {0.8f, 0.2f};
    float label_values[] = {1.0f, 0.0f};
    cyxwiz::Tensor probabilities({2}, probability_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor labels({2}, label_values, cyxwiz::DataType::Float32);

    auto bce = cyxwiz::CreateLoss(cyxwiz::LossType::BinaryCrossEntropy, cyxwiz::Reduction::Mean);
    cyxwiz::Tensor bce_grad = bce->Backward(probabilities, labels);
    REQUIRE(bce_grad.Data<float>()[0] == Catch::Approx(-0.625f));
    REQUIRE(bce_grad.Data<float>()[1] == Catch::Approx(0.625f));

    float logit_values[] = {0.0f, 2.0f};
    cyxwiz::Tensor logits({2}, logit_values, cyxwiz::DataType::Float32);
    auto bce_logits = cyxwiz::CreateLoss(cyxwiz::LossType::BCEWithLogits, cyxwiz::Reduction::None);
    cyxwiz::Tensor logits_grad = bce_logits->Backward(logits, labels);
    REQUIRE(logits_grad.Data<float>()[0] == Catch::Approx(0.5f - 1.0f));
    REQUIRE(logits_grad.Data<float>()[1] ==
            Catch::Approx(1.0f / (1.0f + std::exp(-2.0f))));
}

TEST_CASE("Weighted BCEWithLogits matches the reference on every available backend",
          "[loss][arrayfire][device_switch][weighted_bce]") {
    const cyxwiz::Device* current = cyxwiz::Device::GetCurrentDevice();
    REQUIRE(current != nullptr);
    const cyxwiz::DeviceType original_type = current->GetType();
    const int original_id = current->GetDeviceId();

    struct RestoreDevice {
        cyxwiz::DeviceType type;
        int id;
        ~RestoreDevice() {
            cyxwiz::Device(type, id).SetActive();
        }
    } restore{original_type, original_id};

    const float logit_values[] = {0.0f, 0.0f};
    const float target_values[] = {1.0f, 0.0f};
    const float expected_loss = 2.5f * std::log(2.0f);

    for (const auto& info : cyxwiz::Device::GetAvailableDevices()) {
        if (info.type == cyxwiz::DeviceType::ONEAPI &&
            std::getenv("CYXWIZ_TEST_ONEAPI_LOSS") == nullptr) {
            continue;
        }
        DYNAMIC_SECTION("backend type " << static_cast<int>(info.type)
                        << " device " << info.device_id) {
            cyxwiz::Device device(info.type, info.device_id);
            device.SetActive();
            REQUIRE(device.GetType() == info.type);
            REQUIRE(device.IsActive());

            cyxwiz::Tensor logits(
                {2}, logit_values, cyxwiz::DataType::Float32);
            cyxwiz::Tensor targets(
                {2}, target_values, cyxwiz::DataType::Float32);
            cyxwiz::BCEWithLogitsLoss loss(cyxwiz::Reduction::Mean, 4.0f);

            const cyxwiz::Tensor value = loss.Forward(logits, targets);
            const cyxwiz::Tensor grad = loss.Backward(logits, targets);

            REQUIRE(value.Data<float>()[0] ==
                    Catch::Approx(expected_loss).margin(1e-5f));
            REQUIRE(grad.Shape() == std::vector<size_t>{2});
            REQUIRE(grad.Data<float>()[0] ==
                    Catch::Approx(-1.0f).margin(1e-6f));
            REQUIRE(grad.Data<float>()[1] ==
                    Catch::Approx(0.25f).margin(1e-6f));
        }
    }
}

TEST_CASE("Class-index losses compute forward reductions", "[loss]") {
    float logit_values[] = {1.0f, 2.0f, 0.0f, 2.0f, -1.0f, 0.0f};
    int32_t target_values[] = {1, 0};
    cyxwiz::Tensor logits({2, 3}, logit_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({2}, target_values, cyxwiz::DataType::Int32);

    auto cross_entropy = cyxwiz::CreateLoss(cyxwiz::LossType::CrossEntropy, cyxwiz::Reduction::Mean);
    cyxwiz::Tensor ce_loss = cross_entropy->Forward(logits, targets);

    const float row0_loss = std::log(std::exp(-1.0f) + 1.0f + std::exp(-2.0f));
    const float row1_loss = std::log(1.0f + std::exp(-3.0f) + std::exp(-2.0f));
    REQUIRE(ce_loss.Data<float>()[0] == Catch::Approx((row0_loss + row1_loss) / 2.0f));

    float log_prob_values[] = {
        std::log(0.1f), std::log(0.7f), std::log(0.2f),
        std::log(0.6f), std::log(0.3f), std::log(0.1f)
    };
    cyxwiz::Tensor log_probs({2, 3}, log_prob_values, cyxwiz::DataType::Float32);

    auto nll = cyxwiz::CreateLoss(cyxwiz::LossType::NLLLoss, cyxwiz::Reduction::Sum);
    cyxwiz::Tensor nll_loss = nll->Forward(log_probs, targets);
    REQUIRE(nll_loss.Data<float>()[0] == Catch::Approx(-std::log(0.7f) - std::log(0.6f)));
}

TEST_CASE("Class-index losses compute backward values", "[loss]") {
    float logit_values[] = {1.0f, 2.0f, 0.0f, 2.0f, -1.0f, 0.0f};
    int32_t target_values[] = {1, 0};
    cyxwiz::Tensor logits({2, 3}, logit_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({2}, target_values, cyxwiz::DataType::Int32);

    auto cross_entropy = cyxwiz::CreateLoss(cyxwiz::LossType::CrossEntropy, cyxwiz::Reduction::Mean);
    cyxwiz::Tensor ce_grad = cross_entropy->Backward(logits, targets);
    const float* grad = ce_grad.Data<float>();

    const float row0_denom = std::exp(-1.0f) + 1.0f + std::exp(-2.0f);
    REQUIRE(grad[0] == Catch::Approx((std::exp(-1.0f) / row0_denom) / 2.0f));
    REQUIRE(grad[1] == Catch::Approx((1.0f / row0_denom - 1.0f) / 2.0f));
    REQUIRE(grad[2] == Catch::Approx((std::exp(-2.0f) / row0_denom) / 2.0f));

    const float row1_denom = 1.0f + std::exp(-3.0f) + std::exp(-2.0f);
    REQUIRE(grad[3] == Catch::Approx((1.0f / row1_denom - 1.0f) / 2.0f));
    REQUIRE(grad[4] == Catch::Approx((std::exp(-3.0f) / row1_denom) / 2.0f));
    REQUIRE(grad[5] == Catch::Approx((std::exp(-2.0f) / row1_denom) / 2.0f));

    float log_prob_values[] = {
        std::log(0.1f), std::log(0.7f), std::log(0.2f),
        std::log(0.6f), std::log(0.3f), std::log(0.1f)
    };
    cyxwiz::Tensor log_probs({2, 3}, log_prob_values, cyxwiz::DataType::Float32);

    auto nll = cyxwiz::CreateLoss(cyxwiz::LossType::NLLLoss, cyxwiz::Reduction::Mean);
    cyxwiz::Tensor nll_grad = nll->Backward(log_probs, targets);
    const float* nll_data = nll_grad.Data<float>();
    REQUIRE(nll_data[0] == Catch::Approx(0.0f));
    REQUIRE(nll_data[1] == Catch::Approx(-0.5f));
    REQUIRE(nll_data[2] == Catch::Approx(0.0f));
    REQUIRE(nll_data[3] == Catch::Approx(-0.5f));
    REQUIRE(nll_data[4] == Catch::Approx(0.0f));
    REQUIRE(nll_data[5] == Catch::Approx(0.0f));
}

TEST_CASE("CrossEntropyLoss supports token-level logits and ignored targets", "[loss][language_model]") {
    float logit_values[] = {
        2.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 2.0f
    };
    int64_t target_values[] = {0, -100};
    cyxwiz::Tensor logits({1, 2, 3}, logit_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({1, 2}, target_values, cyxwiz::DataType::Int64);

    cyxwiz::CrossEntropyLoss cross_entropy(cyxwiz::Reduction::Mean, -100);
    cyxwiz::Tensor loss = cross_entropy.Forward(logits, targets);
    REQUIRE(loss.Data<float>()[0] == Catch::Approx(0.40760595f).margin(1e-6f));

    cyxwiz::Tensor grad = cross_entropy.Backward(logits, targets);
    REQUIRE(grad.Shape() == std::vector<size_t>{1, 2, 3});
    const float* grad_data = grad.Data<float>();
    REQUIRE(grad_data[0] == Catch::Approx(-0.3347590f).margin(1e-5f));
    REQUIRE(grad_data[1] == Catch::Approx(0.2447285f).margin(1e-5f));
    REQUIRE(grad_data[2] == Catch::Approx(0.0900306f).margin(1e-5f));
    REQUIRE(grad_data[3] == Catch::Approx(0.0f));
    REQUIRE(grad_data[4] == Catch::Approx(0.0f));
    REQUIRE(grad_data[5] == Catch::Approx(0.0f));
}

TEST_CASE("CrossEntropyLoss supports label smoothing", "[loss]") {
    float logit_values[] = {2.0f, 0.0f, -1.0f};
    int32_t target_values[] = {0};
    cyxwiz::Tensor logits({1, 3}, logit_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({1}, target_values, cyxwiz::DataType::Int32);

    cyxwiz::CrossEntropyLoss cross_entropy(
        cyxwiz::Reduction::Mean, -100, {}, 0.2f);
    cyxwiz::Tensor loss = cross_entropy.Forward(logits, targets);
    cyxwiz::Tensor grad = cross_entropy.Backward(logits, targets);

    const float denom = 1.0f + std::exp(-2.0f) + std::exp(-3.0f);
    const float probs[] = {
        1.0f / denom,
        std::exp(-2.0f) / denom,
        std::exp(-3.0f) / denom,
    };
    const float targets_smoothed[] = {
        1.0f - 0.2f + 0.2f / 3.0f,
        0.2f / 3.0f,
        0.2f / 3.0f,
    };
    const float expected_loss =
        -(targets_smoothed[0] * std::log(probs[0]) +
          targets_smoothed[1] * std::log(probs[1]) +
          targets_smoothed[2] * std::log(probs[2]));

    REQUIRE(cross_entropy.GetLabelSmoothing() == Catch::Approx(0.2f));
    REQUIRE(loss.Data<float>()[0] == Catch::Approx(expected_loss).margin(1e-6f));
    REQUIRE(grad.Shape() == std::vector<size_t>{1, 3});
    REQUIRE(grad.Data<float>()[0] ==
            Catch::Approx(probs[0] - targets_smoothed[0]).margin(1e-6f));
    REQUIRE(grad.Data<float>()[1] ==
            Catch::Approx(probs[1] - targets_smoothed[1]).margin(1e-6f));
    REQUIRE(grad.Data<float>()[2] ==
            Catch::Approx(probs[2] - targets_smoothed[2]).margin(1e-6f));
}

TEST_CASE("CrossEntropyLoss weights and smooths one-hot targets", "[loss]") {
    float logit_values[] = {
        2.0f, 0.0f, -1.0f,
        0.0f, 1.0f, 2.0f,
    };
    float target_values[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    const float weights[] = {1.0f, 2.0f, 4.0f};
    constexpr float smoothing = 0.1f;
    cyxwiz::Tensor logits(
        {2, 3}, logit_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets(
        {2, 3}, target_values, cyxwiz::DataType::Float32);
    cyxwiz::CrossEntropyLoss cross_entropy(
        cyxwiz::Reduction::Mean,
        -100,
        {weights[0], weights[1], weights[2]},
        smoothing);

    cyxwiz::Tensor loss = cross_entropy.Forward(logits, targets);
    cyxwiz::Tensor grad = cross_entropy.Backward(logits, targets);

    float expected_loss = 0.0f;
    float expected_grad[6] = {};
    constexpr float mean_denominator = 2.0f;
    for (size_t row = 0; row < 2; ++row) {
        const size_t base = row * 3;
        float max_logit = logit_values[base];
        for (size_t column = 1; column < 3; ++column) {
            if (logit_values[base + column] > max_logit) {
                max_logit = logit_values[base + column];
            }
        }
        float exp_values[3];
        float exp_sum = 0.0f;
        for (size_t column = 0; column < 3; ++column) {
            exp_values[column] =
                std::exp(logit_values[base + column] - max_logit);
            exp_sum += exp_values[column];
        }

        float row_weight = 0.0f;
        float weighted_targets[3];
        for (size_t column = 0; column < 3; ++column) {
            const float smoothed_target =
                target_values[base + column] * (1.0f - smoothing) +
                smoothing / 3.0f;
            weighted_targets[column] =
                weights[column] * smoothed_target;
            row_weight += weighted_targets[column];
            expected_loss -= weighted_targets[column] *
                std::log(exp_values[column] / exp_sum);
        }
        for (size_t column = 0; column < 3; ++column) {
            const float probability = exp_values[column] / exp_sum;
            expected_grad[base + column] =
                probability * row_weight - weighted_targets[column];
        }
    }
    expected_loss /= mean_denominator;
    for (float& value : expected_grad) {
        value /= mean_denominator;
    }

    REQUIRE(loss.ReadData<float>()[0] ==
            Catch::Approx(expected_loss).margin(1e-6f));
    const float* actual_grad = grad.ReadData<float>();
    for (size_t index = 0; index < 6; ++index) {
        REQUIRE(actual_grad[index] ==
                Catch::Approx(expected_grad[index]).margin(1e-6f));
    }
}

TEST_CASE("KL divergence computes forward reductions", "[loss]") {
    float log_pred_values[] = {std::log(0.2f), std::log(0.5f), std::log(0.3f)};
    float target_values[] = {0.1f, 0.7f, 0.2f};
    cyxwiz::Tensor log_predictions({3}, log_pred_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({3}, target_values, cyxwiz::DataType::Float32);

    auto kl_sum = cyxwiz::CreateLoss(cyxwiz::LossType::KLDivergence, cyxwiz::Reduction::Sum);
    cyxwiz::Tensor loss = kl_sum->Forward(log_predictions, targets);

    const float expected =
        0.1f * (std::log(0.1f) - std::log(0.2f)) +
        0.7f * (std::log(0.7f) - std::log(0.5f)) +
        0.2f * (std::log(0.2f) - std::log(0.3f));
    REQUIRE(loss.Data<float>()[0] == Catch::Approx(expected));
}

TEST_CASE("KL divergence computes backward values", "[loss]") {
    float log_pred_values[] = {std::log(0.2f), std::log(0.5f), std::log(0.3f)};
    float target_values[] = {0.1f, 0.7f, 0.0f};
    cyxwiz::Tensor log_predictions({3}, log_pred_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({3}, target_values, cyxwiz::DataType::Float32);

    auto kl_mean = cyxwiz::CreateLoss(cyxwiz::LossType::KLDivergence, cyxwiz::Reduction::Mean);
    cyxwiz::Tensor grad = kl_mean->Backward(log_predictions, targets);

    REQUIRE(grad.Data<float>()[0] == Catch::Approx(-0.1f / 3.0f));
    REQUIRE(grad.Data<float>()[1] == Catch::Approx(-0.7f / 3.0f));
    REQUIRE(grad.Data<float>()[2] == Catch::Approx(0.0f));
}

TEST_CASE("SoftDiceLoss computes forward and backward values", "[loss]") {
    float pred_values[] = {0.8f, 0.2f, 0.4f, 0.9f};
    float target_values[] = {1.0f, 0.0f, 0.0f, 1.0f};
    cyxwiz::Tensor predictions({1, 4}, pred_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({1, 4}, target_values, cyxwiz::DataType::Float32);

    cyxwiz::SoftDiceLoss dice(cyxwiz::Reduction::Mean, 1.0f);
    cyxwiz::Tensor loss = dice.Forward(predictions, targets);
    cyxwiz::Tensor grad = dice.Backward(predictions, targets);

    const float intersection = 0.8f + 0.9f;
    const float pred_sum = 0.8f + 0.2f + 0.4f + 0.9f;
    const float target_sum = 2.0f;
    const float numerator = 2.0f * intersection + 1.0f;
    const float denominator = pred_sum + target_sum + 1.0f;
    const float expected_loss = 1.0f - numerator / denominator;

    REQUIRE(dice.GetSmooth() == Catch::Approx(1.0f));
    REQUIRE(loss.Data<float>()[0] == Catch::Approx(expected_loss));
    REQUIRE(grad.Shape() == std::vector<size_t>{1, 4});

    const float denom_sq = denominator * denominator;
    REQUIRE(grad.Data<float>()[0] ==
            Catch::Approx(-((2.0f * denominator) - numerator) / denom_sq));
    REQUIRE(grad.Data<float>()[1] ==
            Catch::Approx(-((0.0f * denominator) - numerator) / denom_sq));
    REQUIRE(grad.Data<float>()[3] ==
            Catch::Approx(-((2.0f * denominator) - numerator) / denom_sq));
}

TEST_CASE("TverskyLoss computes forward and backward values", "[loss]") {
    float pred_values[] = {0.8f, 0.2f, 0.4f, 0.9f};
    float target_values[] = {1.0f, 0.0f, 0.0f, 1.0f};
    cyxwiz::Tensor predictions({1, 4}, pred_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({1, 4}, target_values, cyxwiz::DataType::Float32);

    cyxwiz::TverskyLoss tversky(cyxwiz::Reduction::Mean, 0.3f, 0.7f, 1.0f);
    cyxwiz::Tensor loss = tversky.Forward(predictions, targets);
    cyxwiz::Tensor grad = tversky.Backward(predictions, targets);

    const float tp = 0.8f + 0.9f;
    const float fp = 0.2f + 0.4f;
    const float fn = 0.2f + 0.1f;
    const float numerator = tp + 1.0f;
    const float denominator = tp + 0.3f * fp + 0.7f * fn + 1.0f;
    const float expected_loss = 1.0f - numerator / denominator;

    REQUIRE(tversky.GetAlpha() == Catch::Approx(0.3f));
    REQUIRE(tversky.GetBeta() == Catch::Approx(0.7f));
    REQUIRE(tversky.GetSmooth() == Catch::Approx(1.0f));
    REQUIRE(loss.Data<float>()[0] == Catch::Approx(expected_loss));
    REQUIRE(grad.Shape() == std::vector<size_t>{1, 4});

    const float denom_sq = denominator * denominator;
    const float target_denominator_derivative = 1.0f - 0.7f;
    const float background_denominator_derivative = 0.3f;
    REQUIRE(grad.Data<float>()[0] ==
            Catch::Approx(-((1.0f * denominator) -
                            (numerator * target_denominator_derivative)) /
                          denom_sq));
    REQUIRE(grad.Data<float>()[1] ==
            Catch::Approx(-((0.0f * denominator) -
                            (numerator * background_denominator_derivative)) /
                          denom_sq));
    REQUIRE(grad.Data<float>()[3] ==
            Catch::Approx(-((1.0f * denominator) -
                            (numerator * target_denominator_derivative)) /
                          denom_sq));
}

TEST_CASE("JaccardLoss computes forward and backward values", "[loss]") {
    float pred_values[] = {0.8f, 0.2f, 0.4f, 0.9f};
    float target_values[] = {1.0f, 0.0f, 0.0f, 1.0f};
    cyxwiz::Tensor predictions({1, 4}, pred_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({1, 4}, target_values, cyxwiz::DataType::Float32);

    cyxwiz::JaccardLoss jaccard(cyxwiz::Reduction::Mean, 1.0f);
    cyxwiz::Tensor loss = jaccard.Forward(predictions, targets);
    cyxwiz::Tensor grad = jaccard.Backward(predictions, targets);

    const float intersection = 0.8f + 0.9f;
    const float pred_sum = 0.8f + 0.2f + 0.4f + 0.9f;
    const float target_sum = 2.0f;
    const float numerator = intersection + 1.0f;
    const float denominator = pred_sum + target_sum - intersection + 1.0f;
    const float expected_loss = 1.0f - numerator / denominator;

    REQUIRE(jaccard.GetSmooth() == Catch::Approx(1.0f));
    REQUIRE(loss.Data<float>()[0] == Catch::Approx(expected_loss));
    REQUIRE(grad.Shape() == std::vector<size_t>{1, 4});

    const float denom_sq = denominator * denominator;
    REQUIRE(grad.Data<float>()[0] ==
            Catch::Approx(-((1.0f * denominator) -
                            (numerator * 0.0f)) /
                          denom_sq));
    REQUIRE(grad.Data<float>()[1] ==
            Catch::Approx(-((0.0f * denominator) -
                            (numerator * 1.0f)) /
                          denom_sq));
    REQUIRE(grad.Data<float>()[3] ==
            Catch::Approx(-((1.0f * denominator) -
                            (numerator * 0.0f)) /
                          denom_sq));
}

TEST_CASE("Focal loss computes class-index forward reduction", "[loss]") {
    float logit_values[] = {1.0f, 2.0f, 0.0f, 2.0f, -1.0f, 0.0f};
    int32_t target_values[] = {1, 0};
    cyxwiz::Tensor logits({2, 3}, logit_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({2}, target_values, cyxwiz::DataType::Int32);

    cyxwiz::FocalLoss focal(0.25f, 2.0f, cyxwiz::Reduction::Mean);
    cyxwiz::Tensor loss = focal.Forward(logits, targets);

    const float row0_denom = std::exp(-1.0f) + 1.0f + std::exp(-2.0f);
    const float row0_pt = 1.0f / row0_denom;
    const float row1_denom = 1.0f + std::exp(-3.0f) + std::exp(-2.0f);
    const float row1_pt = 1.0f / row1_denom;
    const float expected = (
        -0.25f * std::pow(1.0f - row0_pt, 2.0f) * std::log(row0_pt) +
        -0.25f * std::pow(1.0f - row1_pt, 2.0f) * std::log(row1_pt)
    ) / 2.0f;
    REQUIRE(loss.Data<float>()[0] == Catch::Approx(expected));
}

TEST_CASE("Focal loss computes class-index backward values", "[loss]") {
    float logit_values[] = {1.0f, 2.0f, 0.0f, 2.0f, -1.0f, 0.0f};
    int32_t target_values[] = {1, 0};
    cyxwiz::Tensor logits({2, 3}, logit_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({2}, target_values, cyxwiz::DataType::Int32);

    cyxwiz::FocalLoss focal(0.25f, 2.0f, cyxwiz::Reduction::Mean);
    cyxwiz::Tensor grad = focal.Backward(logits, targets);
    const float* data = grad.Data<float>();

    const float row0_denom = std::exp(-1.0f) + 1.0f + std::exp(-2.0f);
    const float row0_probs[] = {std::exp(-1.0f) / row0_denom, 1.0f / row0_denom,
                                std::exp(-2.0f) / row0_denom};
    const float row0_pt = row0_probs[1];
    const float row0_scale = 0.25f *
        (std::pow(1.0f - row0_pt, 2.0f) -
         2.0f * row0_pt * std::pow(1.0f - row0_pt, 1.0f) * std::log(row0_pt)) / 2.0f;
    REQUIRE(data[0] == Catch::Approx(row0_scale * row0_probs[0]));
    REQUIRE(data[1] == Catch::Approx(row0_scale * (row0_probs[1] - 1.0f)));
    REQUIRE(data[2] == Catch::Approx(row0_scale * row0_probs[2]));

    const float row1_denom = 1.0f + std::exp(-3.0f) + std::exp(-2.0f);
    const float row1_probs[] = {1.0f / row1_denom, std::exp(-3.0f) / row1_denom,
                                std::exp(-2.0f) / row1_denom};
    const float row1_pt = row1_probs[0];
    const float row1_scale = 0.25f *
        (std::pow(1.0f - row1_pt, 2.0f) -
         2.0f * row1_pt * std::pow(1.0f - row1_pt, 1.0f) * std::log(row1_pt)) / 2.0f;
    REQUIRE(data[3] == Catch::Approx(row1_scale * (row1_probs[0] - 1.0f)));
    REQUIRE(data[4] == Catch::Approx(row1_scale * row1_probs[1]));
    REQUIRE(data[5] == Catch::Approx(row1_scale * row1_probs[2]));
}

TEST_CASE("Cosine embedding loss computes forward reduction", "[loss]") {
    float x1_values[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float x2_values[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float label_values[] = {1.0f, -1.0f};
    cyxwiz::Tensor x1({2, 2}, x1_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor x2({2, 2}, x2_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor labels({2}, label_values, cyxwiz::DataType::Float32);

    cyxwiz::CosineEmbeddingLoss loss(0.25f, cyxwiz::Reduction::Mean);
    loss.SetLabels(labels);

    cyxwiz::Tensor output = loss.Forward(x1, x2);
    REQUIRE(output.Data<float>()[0] == Catch::Approx((0.0f + 0.75f) / 2.0f));
}

TEST_CASE("Cosine embedding loss computes backward values", "[loss]") {
    float x1_values[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float x2_values[] = {0.0f, 1.0f, 1.0f, 0.0f};
    float label_values[] = {1.0f, -1.0f};
    cyxwiz::Tensor x1({2, 2}, x1_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor x2({2, 2}, x2_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor labels({2}, label_values, cyxwiz::DataType::Float32);

    cyxwiz::CosineEmbeddingLoss loss(0.25f, cyxwiz::Reduction::Mean);
    loss.SetLabels(labels);

    cyxwiz::Tensor grad = loss.Backward(x1, x2);
    const float* data = grad.Data<float>();

    REQUIRE(data[0] == Catch::Approx(0.0f));
    REQUIRE(data[1] == Catch::Approx(-0.5f));
    REQUIRE(data[2] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(data[3] == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("Triplet loss computes Euclidean forward reduction", "[loss]") {
    float anchor_values[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float positive_values[] = {1.0f, 0.0f, 2.0f, 0.0f};
    float negative_values[] = {0.0f, 2.0f, 0.0f, 2.0f};
    cyxwiz::Tensor anchor({2, 2}, anchor_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor positive({2, 2}, positive_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor negative({2, 2}, negative_values, cyxwiz::DataType::Float32);

    cyxwiz::TripletLoss loss(1.0f, cyxwiz::TripletLoss::DistanceType::Euclidean,
                             cyxwiz::Reduction::Mean);
    loss.SetNegative(negative);

    cyxwiz::Tensor output = loss.Forward(anchor, positive);
    REQUIRE(output.Data<float>()[0] == Catch::Approx((0.0f + 1.0f) / 2.0f));
}

TEST_CASE("Triplet loss computes Euclidean anchor gradients", "[loss]") {
    float anchor_values[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float positive_values[] = {1.0f, 0.0f, 2.0f, 0.0f};
    float negative_values[] = {0.0f, 2.0f, 0.0f, 2.0f};
    cyxwiz::Tensor anchor({2, 2}, anchor_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor positive({2, 2}, positive_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor negative({2, 2}, negative_values, cyxwiz::DataType::Float32);

    cyxwiz::TripletLoss loss(1.0f, cyxwiz::TripletLoss::DistanceType::Euclidean,
                             cyxwiz::Reduction::Mean);
    loss.SetNegative(negative);

    cyxwiz::Tensor grad = loss.Backward(anchor, positive);
    const float* data = grad.Data<float>();

    REQUIRE(data[0] == Catch::Approx(0.0f));
    REQUIRE(data[1] == Catch::Approx(0.0f));
    REQUIRE(data[2] == Catch::Approx(-0.5f));
    REQUIRE(data[3] == Catch::Approx(0.5f));
}

TEST_CASE("Contrastive loss computes forward reduction", "[loss]") {
    float x1_values[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x2_values[] = {1.0f, 0.0f, 0.0f, 0.5f};
    float label_values[] = {0.0f, 1.0f};
    cyxwiz::Tensor x1({2, 2}, x1_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor x2({2, 2}, x2_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor labels({2}, label_values, cyxwiz::DataType::Float32);

    cyxwiz::ContrastiveLoss loss(1.0f, cyxwiz::Reduction::Mean);
    loss.SetLabels(labels);

    cyxwiz::Tensor output = loss.Forward(x1, x2);
    REQUIRE(output.Data<float>()[0] == Catch::Approx((1.0f + 0.25f) / 2.0f));
}

TEST_CASE("Contrastive loss computes x1 gradients", "[loss]") {
    float x1_values[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float x2_values[] = {1.0f, 0.0f, 0.0f, 0.5f};
    float label_values[] = {0.0f, 1.0f};
    cyxwiz::Tensor x1({2, 2}, x1_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor x2({2, 2}, x2_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor labels({2}, label_values, cyxwiz::DataType::Float32);

    cyxwiz::ContrastiveLoss loss(1.0f, cyxwiz::Reduction::Mean);
    loss.SetLabels(labels);

    cyxwiz::Tensor grad = loss.Backward(x1, x2);
    const float* data = grad.Data<float>();

    REQUIRE(data[0] == Catch::Approx(-1.0f));
    REQUIRE(data[1] == Catch::Approx(0.0f));
    REQUIRE(data[2] == Catch::Approx(0.0f));
    REQUIRE(data[3] == Catch::Approx(0.5f));
}
