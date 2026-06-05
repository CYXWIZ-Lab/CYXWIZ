#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/loss.h>
#include <cyxwiz/tensor.h>
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
