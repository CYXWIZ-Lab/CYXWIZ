#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/loss.h>
#include <cyxwiz/tensor.h>
#include <cstdint>
#include <cmath>
#include <vector>

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

TEST_CASE("Core losses compute CPU backward values", "[loss]") {
    float pred_values[] = {1.0f, 3.0f, -2.0f};
    float target_values[] = {0.0f, 1.0f, 1.0f};
    cyxwiz::Tensor predictions({3}, pred_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({3}, target_values, cyxwiz::DataType::Float32);

    auto mse = cyxwiz::CreateLoss(cyxwiz::LossType::MSE, cyxwiz::Reduction::Mean);
    auto l1 = cyxwiz::CreateLoss(cyxwiz::LossType::L1, cyxwiz::Reduction::Mean);
    auto smooth_l1 = cyxwiz::CreateLoss(cyxwiz::LossType::Huber, cyxwiz::Reduction::Mean, 2.0f);

    cyxwiz::Tensor mse_grad = mse->Backward(predictions, targets);
    cyxwiz::Tensor l1_grad = l1->Backward(predictions, targets);
    cyxwiz::Tensor smooth_l1_grad = smooth_l1->Backward(predictions, targets);

    REQUIRE(mse_grad.Data<float>()[0] == Catch::Approx(2.0f / 3.0f));
    REQUIRE(mse_grad.Data<float>()[1] == Catch::Approx(4.0f / 3.0f));
    REQUIRE(mse_grad.Data<float>()[2] == Catch::Approx(-2.0f));

    REQUIRE(l1_grad.Data<float>()[0] == Catch::Approx(1.0f / 3.0f));
    REQUIRE(l1_grad.Data<float>()[1] == Catch::Approx(1.0f / 3.0f));
    REQUIRE(l1_grad.Data<float>()[2] == Catch::Approx(-1.0f / 3.0f));

    REQUIRE(smooth_l1_grad.Data<float>()[0] == Catch::Approx(1.0f / 6.0f));
    REQUIRE(smooth_l1_grad.Data<float>()[1] == Catch::Approx(1.0f / 3.0f));
    REQUIRE(smooth_l1_grad.Data<float>()[2] == Catch::Approx(-1.0f / 3.0f));
}

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
