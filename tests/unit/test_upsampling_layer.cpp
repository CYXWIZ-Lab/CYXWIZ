#include "convolution_test_support.h"
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/layers/upsampling.h>
#include <cyxwiz/tensor.h>
#include <stdexcept>
#include <vector>

namespace {
double Dot(const cyxwiz::Tensor &tensor, const std::vector<float> &weights) {
  REQUIRE(tensor.NumElements() == weights.size());
  const float *values = tensor.ReadData<float>();
  double result = 0.0;
  for (size_t i = 0; i < weights.size(); ++i) {
    result += static_cast<double>(values[i]) * weights[i];
  }
  return result;
}

// A scalar objective gives an independent gradient check without reproducing
// the production backward formula or adding a second numerical-oracle system.
void CheckInputGradient(cyxwiz::Layer &layer,
                        const std::vector<size_t> &shape) {
  size_t count = 1;
  for (size_t extent : shape)
    count *= extent;
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 8.0f;
  }
  const cyxwiz::Tensor input(shape, values.data(), cyxwiz::DataType::Float32);
  const auto output = layer.Forward(input);
  std::vector<float> weights(output.NumElements());
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 16.0f;
  }
  const auto gradient = layer.Backward(cyxwiz::Tensor(
      output.Shape(), weights.data(), cyxwiz::DataType::Float32));
  REQUIRE(gradient.Shape() == shape);
  REQUIRE(gradient.GetDataType() == cyxwiz::DataType::Float32);
  const float *analytic = gradient.ReadData<float>();
  CHECK(Dot(output, weights) ==
        Catch::Approx(Dot(gradient, values)).margin(2e-5));
  constexpr float epsilon = 1.0f / 256.0f;
  for (size_t i = 0; i < values.size(); ++i) {
    CAPTURE(i, shape);
    const float original = values[i];
    values[i] = original + epsilon;
    const double plus =
        Dot(layer.Forward(cyxwiz::Tensor(shape, values.data(),
                                         cyxwiz::DataType::Float32)),
            weights);
    values[i] = original - epsilon;
    const double minus =
        Dot(layer.Forward(cyxwiz::Tensor(shape, values.data(),
                                         cyxwiz::DataType::Float32)),
            weights);
    values[i] = original;
    CHECK(analytic[i] ==
          Catch::Approx((plus - minus) / (2 * epsilon)).margin(2e-4));
  }
}
} // namespace

TEST_CASE("Upsample2DLayer nearest computes forward and backward values",
          "[upsample][layer]") {
  float input_values[] = {
      1.0f,
      2.0f,
      3.0f,
      4.0f,
  };
  cyxwiz::Tensor input({2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);
  cyxwiz::Upsample2DLayer upsample(2, cyxwiz::UpsampleMode::Nearest);

  cyxwiz::Tensor output = upsample.Forward(input);
  REQUIRE(output.NumElements() == 16);
  const float *output_data = output.Data<float>();
  const float expected_output[] = {
      1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f,
      3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f,
  };
  for (size_t i = 0; i < 16; ++i) {
    REQUIRE(output_data[i] == Catch::Approx(expected_output[i]));
  }

  float grad_values[16];
  for (float &value : grad_values) {
    value = 1.0f;
  }
  cyxwiz::Tensor grad_output({4, 4, 1, 1}, grad_values,
                             cyxwiz::DataType::Float32);
  cyxwiz::Tensor grad_input = upsample.Backward(grad_output);
  REQUIRE(grad_input.NumElements() == 4);
  const float *grad_input_data = grad_input.Data<float>();
  for (size_t i = 0; i < 4; ++i) {
    REQUIRE(grad_input_data[i] == Catch::Approx(4.0f));
  }
}

TEST_CASE("Upsample2DLayer bilinear computes forward and backward values",
          "[upsample][layer]") {
  float input_values[] = {
      1.0f,
      2.0f,
      3.0f,
      4.0f,
  };
  cyxwiz::Tensor input({2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);
  cyxwiz::Upsample2DLayer upsample(2, cyxwiz::UpsampleMode::Bilinear);

  cyxwiz::Tensor output = upsample.Forward(input);
  REQUIRE(output.NumElements() == 16);
  const float *output_data = output.Data<float>();
  const float expected_output[] = {
      1.0f, 1.25f, 1.75f, 2.0f, 1.5f, 1.75f, 2.25f, 2.5f,
      2.5f, 2.75f, 3.25f, 3.5f, 3.0f, 3.25f, 3.75f, 4.0f,
  };
  for (size_t i = 0; i < 16; ++i) {
    REQUIRE(output_data[i] == Catch::Approx(expected_output[i]));
  }

  float grad_values[16];
  for (float &value : grad_values) {
    value = 1.0f;
  }
  cyxwiz::Tensor grad_output({4, 4, 1, 1}, grad_values,
                             cyxwiz::DataType::Float32);
  cyxwiz::Tensor grad_input = upsample.Backward(grad_output);
  REQUIRE(grad_input.NumElements() == 4);
  const float *grad_input_data = grad_input.Data<float>();
  for (size_t i = 0; i < 4; ++i) {
    REQUIRE(grad_input_data[i] == Catch::Approx(4.0f));
  }
}

TEST_CASE("PixelShuffleLayer computes forward and backward values",
          "[pixelshuffle][layer]") {
  cyxwiz::test::convolution::BackendLane lane;
  float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  cyxwiz::Tensor input({1, 1, 4, 1}, input_values, cyxwiz::DataType::Float32);
  cyxwiz::PixelShuffleLayer shuffle(2);

  cyxwiz::Tensor output = shuffle.Forward(input);
  REQUIRE(output.NumElements() == 4);
  const float *output_data = output.Data<float>();
  REQUIRE(output_data[0] == Catch::Approx(1.0f));
  REQUIRE(output_data[1] == Catch::Approx(2.0f));
  REQUIRE(output_data[2] == Catch::Approx(3.0f));
  REQUIRE(output_data[3] == Catch::Approx(4.0f));

  float grad_values[] = {10.0f, 20.0f, 30.0f, 40.0f};
  cyxwiz::Tensor grad_output({2, 2, 1, 1}, grad_values,
                             cyxwiz::DataType::Float32);
  cyxwiz::Tensor grad_input = shuffle.Backward(grad_output);
  REQUIRE(grad_input.NumElements() == 4);
  const float *grad_input_data = grad_input.Data<float>();
  REQUIRE(grad_input_data[0] == Catch::Approx(10.0f));
  REQUIRE(grad_input_data[1] == Catch::Approx(20.0f));
  REQUIRE(grad_input_data[2] == Catch::Approx(30.0f));
  REQUIRE(grad_input_data[3] == Catch::Approx(40.0f));
}

TEST_CASE(
    "Upsample preserves affine ramps across spatial edges channels and batches",
    "[upsample][layer][contract]") {
  for (const auto mode :
       {cyxwiz::UpsampleMode::Nearest, cyxwiz::UpsampleMode::Bilinear}) {
    for (const int factor : {1, 2, 3}) {
      for (const auto &shape :
           {std::vector<size_t>{2, 3, 2, 2}, std::vector<size_t>{1, 3, 2, 2},
            std::vector<size_t>{3, 1, 2, 2}}) {
        CAPTURE(factor, shape, static_cast<int>(mode));
        std::vector<float> values(shape[0] * shape[1] * 4);
        size_t index = 0;
        for (size_t h = 0; h < shape[0]; ++h)
          for (size_t w = 0; w < shape[1]; ++w)
            for (size_t c = 0; c < 2; ++c)
              for (size_t n = 0; n < 2; ++n)
                values[index++] = static_cast<float>(8 * h + 4 * w + 2 * c + n);
        cyxwiz::Upsample2DLayer layer(factor, mode);
        const auto output = layer.Forward(
            cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32));
        const size_t out_h = shape[0] * factor, out_w = shape[1] * factor;
        REQUIRE(output.Shape() == std::vector<size_t>{out_h, out_w, 2, 2});
        REQUIRE(output.GetDataType() == cyxwiz::DataType::Float32);
        const float *actual = output.ReadData<float>();
        index = 0;
        for (size_t h = 0; h < out_h; ++h) {
          const double source_h = mode == cyxwiz::UpsampleMode::Nearest
                                      ? static_cast<double>(h / factor)
                                      : std::clamp((h + 0.5) / factor - 0.5,
                                                   0.0, double(shape[0] - 1));
          for (size_t w = 0; w < out_w; ++w) {
            const double source_w = mode == cyxwiz::UpsampleMode::Nearest
                                        ? static_cast<double>(w / factor)
                                        : std::clamp((w + 0.5) / factor - 0.5,
                                                     0.0, double(shape[1] - 1));
            for (size_t c = 0; c < 2; ++c)
              for (size_t n = 0; n < 2; ++n)
                CHECK(actual[index++] ==
                      Catch::Approx(8 * source_h + 4 * source_w + 2 * c + n)
                          .margin(5e-6));
          }
        }
      }
    }
  }
}

TEST_CASE("Upsample nonuniform input gradients match finite differences and "
          "the adjoint",
          "[upsample][layer][gradient]") {
  for (const auto mode :
       {cyxwiz::UpsampleMode::Nearest, cyxwiz::UpsampleMode::Bilinear}) {
    for (const int factor : {1, 2, 3}) {
      CAPTURE(factor, static_cast<int>(mode));
      cyxwiz::Upsample2DLayer layer(factor, mode);
      CheckInputGradient(layer, {2, 3, 2, 2});
      CheckInputGradient(layer, {1, 3, 1, 1});
      CheckInputGradient(layer, {3, 1, 1, 1});
    }
  }
}

TEST_CASE("PixelShuffle preserves channel order and its standalone inverse",
          "[pixelshuffle][layer][contract]") {
  cyxwiz::test::convolution::BackendLane lane;
  for (const int factor : {1, 2, 3}) {
    CAPTURE(factor);
    const size_t r = static_cast<size_t>(factor), channels = 2 * r * r;
    const std::vector<size_t> shape{2, 3, channels, 2};
    std::vector<float> values(2 * 3 * channels * 2);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(i);
    cyxwiz::PixelShuffleLayer layer(factor);
    const auto output = layer.Forward(
        cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32));
    REQUIRE(output.Shape() == std::vector<size_t>{2 * r, 3 * r, 2, 2});
    REQUIRE(output.GetDataType() == cyxwiz::DataType::Float32);
    const float *actual = output.ReadData<float>();
    size_t index = 0;
    for (size_t h = 0; h < 2 * r; ++h)
      for (size_t w = 0; w < 3 * r; ++w)
        for (size_t c = 0; c < 2; ++c)
          for (size_t n = 0; n < 2; ++n) {
            const size_t source_channel = c * r * r + (h % r) * r + w % r;
            const size_t source =
                (((h / r) * 3 + w / r) * channels + source_channel) * 2 + n;
            CHECK(actual[index++] == values[source]);
          }
    // Existing public Backward can be used without Forward; do not silently
    // introduce a cached-forward dependency in the ArrayFire implementation.
    cyxwiz::PixelShuffleLayer inverse(factor);
    const auto restored = inverse.Backward(output);
    REQUIRE(restored.Shape() == shape);
    const float *restored_values = restored.ReadData<float>();
    for (size_t i = 0; i < values.size(); ++i)
      CHECK(restored_values[i] == values[i]);
    CheckInputGradient(layer, {1, 2, channels, 2});
  }
}

TEST_CASE("Upsampling rejects existing invalid constructor input and gradient "
          "contracts",
          "[upsample][pixelshuffle][layer][validation]") {
  cyxwiz::test::convolution::BackendLane lane;
  using cyxwiz::DataType;
  using cyxwiz::Tensor;
  for (const int factor : {0, -1}) {
    REQUIRE_THROWS_AS(cyxwiz::Upsample2DLayer(factor), std::invalid_argument);
    REQUIRE_THROWS_AS(cyxwiz::PixelShuffleLayer(factor), std::invalid_argument);
  }
  cyxwiz::Upsample2DLayer upsample(2);
  cyxwiz::PixelShuffleLayer shuffle(2);
  const Tensor wrong_rank({2, 3, 4}, DataType::Float32);
  const Tensor wrong_dtype({2, 3, 4, 1}, DataType::Float64);
  REQUIRE_THROWS_AS(upsample.Forward(wrong_rank), std::runtime_error);
  REQUIRE_THROWS_AS(upsample.Forward(wrong_dtype), std::runtime_error);
  REQUIRE_THROWS_AS(shuffle.Forward(wrong_rank), std::runtime_error);
  REQUIRE_THROWS_AS(shuffle.Forward(wrong_dtype), std::runtime_error);
  REQUIRE_THROWS_AS(shuffle.Forward(Tensor({2, 3, 5, 1}, DataType::Float32)),
                    std::runtime_error);
  REQUIRE_THROWS_AS(shuffle.Backward(wrong_rank), std::runtime_error);
  REQUIRE_THROWS_AS(shuffle.Backward(wrong_dtype), std::runtime_error);
  REQUIRE_THROWS_AS(shuffle.Backward(Tensor({3, 4, 1, 1}, DataType::Float32)),
                    std::runtime_error);
  REQUIRE_THROWS_AS(shuffle.Backward(Tensor({4, 3, 1, 1}, DataType::Float32)),
                    std::runtime_error);
  cyxwiz::Upsample2DLayer fresh(2);
  REQUIRE_THROWS_AS(fresh.Backward(Tensor({4, 6, 1, 1}, DataType::Float32)),
                    std::runtime_error);
  REQUIRE_NOTHROW(fresh.Forward(Tensor({2, 3, 1, 1}, DataType::Float32)));
  REQUIRE_THROWS_AS(fresh.Backward(Tensor({4, 6, 1, 1}, DataType::Float64)),
                    std::runtime_error);
  // Same element count is insufficient: full logical shape must match.
  REQUIRE_THROWS_AS(fresh.Backward(Tensor({6, 4, 1, 1}, DataType::Float32)),
                    std::runtime_error);
  REQUIRE_THROWS_AS(fresh.Backward(Tensor({4, 3, 2, 1}, DataType::Float32)),
                    std::runtime_error);
  REQUIRE_THROWS_AS(fresh.Backward(Tensor({4, 3, 1, 2}, DataType::Float32)),
                    std::runtime_error);
}
