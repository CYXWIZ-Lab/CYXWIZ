#include "convolution_test_support.h"
#include <cyxwiz/layers/convolution.h>
#include <cyxwiz/optimizers/sgd.h>
#include <cyxwiz/sequential.h>
#include <limits>
#include <memory>
#include <numeric>

using namespace cyxwiz;
using namespace cyxwiz::test::convolution;

namespace {

Tensor Values(const std::vector<size_t> &shape,
              const std::vector<float> &values) {
  return Tensor(shape, values.data(), DataType::Float32);
}
std::vector<float> Read(const Tensor &tensor) {
  const float *data = tensor.ReadData<float>();
  return {data, data + tensor.NumElements()};
}
double Dot(const Tensor &tensor, const std::vector<float> &vector) {
  const auto values = Read(tensor);
  return std::inner_product(values.begin(), values.end(), vector.begin(), 0.0);
}
} // namespace

TEST_CASE(
    "ConvTranspose2DLayer computes deterministic forward and backward values",
    "[conv][conv_transpose][correctness]") {
  BackendLane lane;
  cyxwiz::ConvTranspose2DLayer layer(1, 1, 2, 2, 0, 0, true);

  float weight_values[] = {
      1.0f,
      2.0f,
      3.0f,
      4.0f,
  };
  float bias_values[] = {0.5f};
  layer.SetParameters({
      {"weights",
       cyxwiz::Tensor({2, 2, 1, 1}, weight_values, cyxwiz::DataType::Float32)},
      {"bias", cyxwiz::Tensor({1}, bias_values, cyxwiz::DataType::Float32)},
  });

  float input_values[] = {
      1.0f,
      2.0f,
      3.0f,
      4.0f,
  };
  cyxwiz::Tensor input({2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);

  cyxwiz::Tensor output = layer.Forward(input);
  REQUIRE(output.Shape() == std::vector<size_t>{4, 4, 1, 1});
  const float *output_data = output.ReadData<float>();
  const float expected_output[] = {
      1.5f, 2.5f, 2.5f, 4.5f, 3.5f, 4.5f,  6.5f,  8.5f,
      3.5f, 6.5f, 4.5f, 8.5f, 9.5f, 12.5f, 12.5f, 16.5f,
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
  cyxwiz::Tensor grad_input = layer.Backward(grad_output);
  REQUIRE(grad_input.Shape() == std::vector<size_t>{2, 2, 1, 1});
  const float *grad_input_data = grad_input.ReadData<float>();
  for (size_t i = 0; i < 4; ++i) {
    REQUIRE(grad_input_data[i] == Catch::Approx(10.0f));
  }

  const std::map<std::string, cyxwiz::Tensor> params = layer.GetParameters();
  const float *grad_weight_data = params.at("grad_weights").ReadData<float>();
  for (size_t i = 0; i < 4; ++i) {
    REQUIRE(grad_weight_data[i] == Catch::Approx(10.0f));
  }
  REQUIRE(params.at("grad_bias").ReadData<float>()[0] == Catch::Approx(16.0f));
}

TEST_CASE("ConvTranspose2D initialization preserves singleton parameter shapes",
          "[conv_transpose][initialization]") {
  BackendLane lane;
  ConvTranspose2DLayer layer(1, 1, 1);
  REQUIRE(layer.GetParameters().at("weights").Shape() ==
          std::vector<size_t>{1, 1, 1, 1});
  REQUIRE(layer.Forward(Values({1, 1, 1, 1}, {2})).Shape() ==
          std::vector<size_t>{1, 1, 1, 1});
}

TEST_CASE("ConvTranspose2D output padding extends biased output and gradient",
          "[conv_transpose][correctness]") {
  BackendLane lane;
  ConvTranspose2DLayer layer(1, 1, 1, 2, 0, 1, true);
  layer.SetParameters(
      {{"weights", Values({1, 1, 1, 1}, {2})}, {"bias", Values({1}, {0.5f})}});
  CheckValues(layer.Forward(Values({2, 2, 1, 1}, {1, 2, 3, 4})), {4, 4, 1, 1},
              {2.5f, .5f, 4.5f, .5f, .5f, .5f, .5f, .5f, 6.5f, .5f, 8.5f, .5f,
               .5f, .5f, .5f, .5f});
  CheckValues(layer.Backward(Values({4, 4, 1, 1}, std::vector<float>(16, 1))),
              {2, 2, 1, 1}, {2, 2, 2, 2});
  CheckValues(layer.GetParameters().at("grad_weights"), {1, 1, 1, 1}, {10});
  CheckValues(layer.GetParameters().at("grad_bias"), {1}, {16});
}

TEST_CASE("ConvTranspose2D multi-channel batched gradients satisfy finite "
          "differences",
          "[conv_transpose][correctness][gradient][layout]") {
  BackendLane lane;
  const std::vector<size_t> xs{2, 3, 2, 2}, ws{3, 3, 3, 2}, ys{4, 6, 3, 2};
  std::vector<float> x(24), w(54), b{.1f, -.2f, .3f}, dy(144);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * .07f;
  for (size_t i = 0; i < w.size(); ++i)
    w[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * .03f;
  for (size_t i = 0; i < dy.size(); ++i)
    dy[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * .05f;
  ConvTranspose2DLayer layer(2, 3, 3, 2, 1, 1, true);
  const auto set = [&] {
    layer.SetParameters({{"weights", Values(ws, w)}, {"bias", Values({3}, b)}});
  };
  set();
  REQUIRE(layer.Forward(Values(xs, x)).Shape() == ys);
  const auto dx = Read(layer.Backward(Values(ys, dy)));
  const auto dw = Read(layer.GetParameters().at("grad_weights"));
  const auto db = Read(layer.GetParameters().at("grad_bias"));
  const float epsilon = .001f;
  const auto check = [&](std::vector<float> &values,
                         const std::vector<float> &analytic) {
    for (size_t i = 0; i < values.size(); ++i) {
      const float saved = values[i];
      values[i] = saved + epsilon;
      set();
      const double plus = Dot(layer.Forward(Values(xs, x)), dy);
      values[i] = saved - epsilon;
      set();
      const double minus = Dot(layer.Forward(Values(xs, x)), dy);
      values[i] = saved;
      CAPTURE(i);
      CHECK(analytic[i] ==
            Catch::Approx((plus - minus) / (2 * epsilon)).margin(.0001));
    }
  };
  check(x, dx);
  check(w, dw);
  check(b, db);
  // Independent adjoint identity against the already audited convolution.
  Conv2DLayer convolution(3, 2, 3, 2, 1, false);
  convolution.SetParameters({{"weights", Values(ws, w)}});
  b.assign(3, 0);
  set();
  CHECK(Dot(layer.Forward(Values(xs, x)), dy) ==
        Catch::Approx(Dot(convolution.Forward(Values(ys, dy)), x))
            .margin(.00001));
}

TEST_CASE(
    "ConvTranspose2D validates lifecycle dimensions and parameter contracts",
    "[conv_transpose][validation]") {
  BackendLane lane;
  ConvTranspose2DLayer layer(1, 1, 1);
  auto x = Values({2, 2, 1, 1}, {1, 2, 3, 4});
  CHECK_THROWS_AS(layer.Backward(x), std::logic_error);
  (void)layer.Forward(x);
  CHECK_THROWS_AS(layer.Backward(Tensor({2, 2, 1, 1}, DataType::Float64)),
                  std::runtime_error);
  CHECK_THROWS_AS(layer.Backward(Values({4}, {1, 2, 3, 4})),
                  std::runtime_error);
  CHECK_THROWS_AS(layer.Forward(Tensor({2, 2, 2, 1}, DataType::Float32)),
                  std::runtime_error);
  CHECK_THROWS_AS(layer.Backward(x), std::logic_error);
  (void)layer.Forward(x);
  layer.SetParameters(layer.GetParameters());
  CHECK_THROWS_AS(layer.Backward(x), std::logic_error);
  CHECK_THROWS_AS(layer.Forward(Tensor({0, 2, 1, 1}, DataType::Float32)),
                  std::runtime_error);
  CHECK_THROWS_AS(layer.Forward(Tensor({2, 2, 1, 1}, DataType::Float64)),
                  std::runtime_error);
  layer.SetParameters({{"weights", Values({1}, {1})}});
  CHECK_THROWS_AS(layer.Forward(x), std::runtime_error);
  layer.SetParameters(
      {{"weights", Values({1, 1, 1, 1}, {1})}, {"bias", Values({2}, {1, 2})}});
  CHECK_THROWS_AS(layer.Forward(x), std::runtime_error);
  CHECK_THROWS_AS(ConvTranspose2DLayer(0, 1, 1), std::invalid_argument);
  CHECK_THROWS_AS(ConvTranspose2DLayer(1, 0, 1), std::invalid_argument);
  CHECK_THROWS_AS(ConvTranspose2DLayer(1, 1, 0), std::invalid_argument);
  CHECK_THROWS_AS(ConvTranspose2DLayer(1, 1, 1, 0), std::invalid_argument);
  CHECK_THROWS_AS(ConvTranspose2DLayer(1, 1, 1, 1, -1), std::invalid_argument);
  CHECK_THROWS_AS(ConvTranspose2DLayer(1, 1, 1, 2, 0, 2),
                  std::invalid_argument);
  CHECK_THROWS_AS(ConvTranspose2DLayer(1, 1, 1, 2, 0, -1),
                  std::invalid_argument);
  CHECK_THROWS_AS(ConvTranspose2DLayer(1, 1, 50000), std::overflow_error);
  ConvTranspose2DLayer cropped(1, 1, 1, 1, 2);
  CHECK_THROWS_AS(cropped.Forward(x), std::runtime_error);
  ConvTranspose2DLayer huge(1, 1, 1, (std::numeric_limits<int>::max)());
  CHECK_THROWS_AS(huge.Forward(x), std::exception);
}

TEST_CASE("ConvTranspose2D updates through successive training batches",
          "[conv_transpose][optimizer][multi_batch]") {
  BackendLane lane;
  auto module =
      std::make_unique<ConvTranspose2DModule>(1, 1, 1, 1, 0, 0, false);
  module->SetParameters({{"weights", Values({1, 1, 1, 1}, {2})}});
  SequentialModel model;
  model.AddModule(std::move(module));
  SGDOptimizer optimizer(.1);
  for (int step = 0; step < 2; ++step) {
    const float weight = 2.0f - 1.1f * static_cast<float>(step);
    CheckValues(model.Forward(Values({1, 1, 1, 2}, {3, 4})), {1, 1, 1, 2},
                {3 * weight, 4 * weight});
    CheckValues(model.Backward(Values({1, 1, 1, 2}, {1, 2})), {1, 1, 1, 2},
                {weight, 2 * weight});
    model.UpdateParameters(&optimizer);
    CheckValues(model.GetParameters().at("layer0.weights"), {1, 1, 1, 1},
                {weight - 1.1f});
  }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE(
    "ConvTranspose2D strict device execution has zero host synchronization",
    "[conv_transpose][residency]") {
  BackendLane lane;
  ConvTranspose2DLayer layer(2, 3, 3, 2, 1, 1, true);
  layer.SetParameters(
      {{"weights", DeviceOnlyTensor({3, 3, 3, 2}, std::vector<float>(54, .1f))},
       {"bias", DeviceOnlyTensor({3}, {0, 0, 0})}});
  const auto x = DeviceOnlyOnes({2, 3, 2, 2});
  const auto dy = DeviceOnlyOnes({4, 6, 3, 2});
  ResetConvObservations();
  Tensor output, gradient;
  {
    ScopedArrayFireFallbackPolicy strict(
        ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
    ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
    ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
    output = layer.Forward(x);
    gradient = layer.Backward(dy);
    output.GetSemanticArray().eval();
    gradient.GetSemanticArray().eval();
    layer.GetParameters().at("grad_weights").GetSemanticArray().eval();
    layer.GetParameters().at("grad_bias").GetSemanticArray().eval();
    af::sync();
  }
  CHECK(conv_host_sync_count == 0);
  CHECK(conv_host_sync_bytes == 0);
  CHECK(conv_fallback_count == 0);
  REQUIRE(output.Shape() == std::vector<size_t>{4, 6, 3, 2});
  REQUIRE(gradient.Shape() == x.Shape());
  CheckValues(layer.GetParameters().at("grad_bias"), {3}, {48, 48, 48});
}

void CheckFallback(bool forced, ArrayFireFallbackPolicy policy) {
  BackendLane lane;
  const bool strict =
      policy == ArrayFireFallbackPolicy::ForbidNativeCpuFallback;
  const int padding = forced ? 0 : 1;
  ConvTranspose2DLayer layer(1, 1, 1, 1, padding, 0, true);
  layer.SetParameters({{"weights", DeviceOnlyTensor({1, 1, 1, 1}, {2})},
                       {"bias", DeviceOnlyTensor({1}, {.5f})}});
  auto x =
      DeviceOnlyTensor(forced ? std::vector<size_t>{1, 1, 1, 1}
                              : std::vector<size_t>{3, 3, 1, 1},
                       forced ? std::vector<float>{5}
                              : std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto dy = DeviceOnlyOnes({1, 1, 1, 1});
  for (bool backward : {false, true}) {
    if (backward) {
      ScopedArrayFireFallbackPolicy compatible(
          ArrayFireFallbackPolicy::AllowNativeCpuFallback);
      (void)layer.Forward(x);
    }
    ResetConvObservations();
    Tensor output;
    {
      ScopedEnvVar hook("CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK",
                        forced ? (backward ? "ConvTranspose2DLayer::Backward"
                                           : "ConvTranspose2DLayer::Forward")
                               : "");
      ScopedArrayFireFallbackPolicy scoped_policy(policy);
      ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
      ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
      const auto run = [&] {
        return backward ? layer.Backward(dy) : layer.Forward(x);
      };
      if (strict)
        CHECK_THROWS_AS(run(), std::runtime_error);
      else
        output = run();
    }
    CHECK(conv_fallback_count == 1);
    CHECK(last_conv_fallback.fallback_forbidden == strict);
    CHECK(last_conv_fallback.operation_name ==
          (backward ? "ConvTranspose2DLayer::Backward"
                    : "ConvTranspose2DLayer::Forward"));
    if (!forced)
      CHECK(last_conv_fallback.reason_code == "unsupported_shape");
    if (strict) {
      CHECK(conv_host_sync_count == 0);
      CHECK(conv_host_sync_bytes == 0);
    } else {
      CHECK(conv_host_sync_count > 0);
      CHECK(conv_host_sync_bytes > 0);
      CHECK(saw_conv_cpu_path);
      if (backward) {
        CheckValues(layer.GetParameters().at("grad_weights"), {1, 1, 1, 1},
                    {5});
        CheckValues(layer.GetParameters().at("grad_bias"), {1}, {1});
        CheckValues(output, x.Shape(),
                    forced ? std::vector<float>{2}
                           : std::vector<float>{0, 0, 0, 0, 2, 0, 0, 0, 0});
      } else
        CheckValues(output, {1, 1, 1, 1}, {10.5f});
    }
  }
}
TEST_CASE("ConvTranspose2D declared padding fallback honors compatibility and "
          "strict policy",
          "[conv_transpose][fallback]") {
  SECTION("compatible") {
    CheckFallback(false, ArrayFireFallbackPolicy::AllowNativeCpuFallback);
  }
  SECTION("strict") {
    CheckFallback(false, ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
  }
}
#ifndef NDEBUG
TEST_CASE(
    "ConvTranspose2D native fallback matches device overlap and batch layout",
    "[conv_transpose][fallback][layout]") {
  BackendLane lane;
  ConvTranspose2DLayer layer(2, 3, 3, 2, 1, 1, true);
  std::vector<float> x(24), weights(54), dy(144);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<float>(i % 5) * .25f;
  for (size_t i = 0; i < weights.size(); ++i)
    weights[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * .125f;
  for (size_t i = 0; i < dy.size(); ++i)
    dy[i] = static_cast<float>(i % 3) * .25f;
  layer.SetParameters({{"weights", Values({3, 3, 3, 2}, weights)},
                       {"bias", Values({3}, {.25f, -.5f, .75f})}});
  const auto input = Values({2, 3, 2, 2}, x);
  const auto upstream = Values({4, 6, 3, 2}, dy);
  std::vector<float> output, dx, dw, db;
  {
    ScopedArrayFireFallbackPolicy strict(
        ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
    output = Read(layer.Forward(input));
    dx = Read(layer.Backward(upstream));
    dw = Read(layer.GetParameters().at("grad_weights"));
    db = Read(layer.GetParameters().at("grad_bias"));
  }
  ScopedArrayFireFallbackPolicy compatible(
      ArrayFireFallbackPolicy::AllowNativeCpuFallback);
  ResetConvObservations();
  ScopedArrayFireNativeCpuFallbackObserver observer(&CountConvFallback);
  {
    ScopedEnvVar hook("CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK",
                      "ConvTranspose2DLayer::Forward");
    CheckValues(layer.Forward(input), {4, 6, 3, 2}, output);
  }
  {
    ScopedEnvVar hook("CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK",
                      "ConvTranspose2DLayer::Backward");
    CheckValues(layer.Backward(upstream), input.Shape(), dx);
    CheckValues(layer.GetParameters().at("grad_weights"), {3, 3, 3, 2}, dw);
    CheckValues(layer.GetParameters().at("grad_bias"), {3}, db);
  }
  CHECK(conv_fallback_count == 2);
}

TEST_CASE(
    "ConvTranspose2D forced fallback honors compatibility and strict policy",
    "[conv_transpose][fallback]") {
  SECTION("compatible") {
    CheckFallback(true, ArrayFireFallbackPolicy::AllowNativeCpuFallback);
  }
  SECTION("strict") {
    CheckFallback(true, ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
  }
}
#endif
#endif
