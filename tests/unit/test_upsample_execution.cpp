#include "convolution_test_support.h"
#include "model_test_path.h"
#include <algorithm>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <cyxwiz/layers/upsampling.h>
#include <cyxwiz/sequential.h>
#include <filesystem>
#include <limits>
#include <memory>

using namespace cyxwiz;
using namespace cyxwiz::test::convolution;

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Nearest Upsample forward and adjoint stay device resident",
          "[upsample][residency]") {
  BackendLane lane;
  for (int factor : {1, 2, 3}) {
    for (const auto &shape :
         {std::vector<size_t>{2, 3, 2, 2}, std::vector<size_t>{1, 3, 1, 1},
          std::vector<size_t>{3, 1, 1, 1}}) {
      CAPTURE(factor, shape);
      std::vector<float> values(shape[0] * shape[1] * shape[2] * shape[3]);
      for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(static_cast<int>(i) - 7) / 8.0f;
      const auto input = DeviceOnlyTensor(shape, values);
      Upsample2DLayer layer(factor);
      Tensor output, gradient;
      ResetConvObservations();
      {
        ScopedArrayFireFallbackPolicy strict(
            ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
        ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
        output = layer.Forward(input);
        // A nonuniform cotangent: U* U x = factor^2 x for nearest replication.
        gradient = layer.Backward(output);
        gradient.GetSemanticArray().eval();
        af::sync();
      }
      CHECK(conv_host_sync_count == 0);
      CHECK(conv_host_sync_bytes == 0);
      CHECK(conv_fallback_count == 0);
      REQUIRE(output.Shape() == std::vector<size_t>{shape[0] * factor,
                                                    shape[1] * factor, shape[2],
                                                    shape[3]});
      for (float &value : values)
        value *= factor * factor;
      CheckValues(gradient, shape, values);
    }
  }
}

TEST_CASE("Upsample backward retains geometry without input storage",
          "[upsample][ownership]") {
  BackendLane lane;
  struct InspectableUpsample : Upsample2DLayer {
    using Upsample2DLayer::Upsample2DLayer;
    const Tensor &Context() const { return cached_input_; }
  };
  for (auto mode : {UpsampleMode::Nearest, UpsampleMode::Bilinear}) {
    InspectableUpsample layer(2, mode);
    {
      const auto input = DeviceOnlyOnes({2, 3, 1, 1});
      layer.Forward(input);
    }
    REQUIRE(layer.Context().Shape() == std::vector<size_t>{2, 3, 1, 1});
    REQUIRE(layer.Context().ReadData<float>() == nullptr);
    CheckValues(layer.Backward(DeviceOnlyOnes({4, 6, 1, 1})), {2, 3, 1, 1},
                std::vector<float>(6, 4.0f));
  }
}

TEST_CASE("Nearest Upsample rejects provider dimensions before device access",
          "[upsample][validation]") {
  BackendLane lane;
  if (sizeof(size_t) <= sizeof(int))
    SKIP("Oversized metadata fixture requires size_t wider than int");
  const size_t limit = static_cast<size_t>((std::numeric_limits<int>::max)());
  for (const auto &shape : {std::vector<size_t>{limit / 2 + 1, 1, 1, 1},
                            std::vector<size_t>{1, limit / 2 + 1, 1, 1},
                            std::vector<size_t>{1, 1, limit / 2 + 1, 2}}) {
    for (auto mode : {UpsampleMode::Nearest, UpsampleMode::Bilinear}) {
      Upsample2DLayer layer(2, mode);
      const Tensor input(shape, nullptr, DataType::Float32);
      ResetConvObservations();
      {
        ScopedArrayFireFallbackPolicy strict(
            ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
        ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
        REQUIRE_THROWS_WITH(
            layer.Forward(input),
            "Upsample2D dimension exceeds ArrayFire int dimension limit");
      }
      CHECK(conv_host_sync_count == 0);
      CHECK(conv_fallback_count == 0);
    }
  }
}

#ifndef NDEBUG
TEST_CASE(
    "Bilinear device and native formulas agree on wider nonuniform batches",
    "[upsample][bilinear][fallback][parity]") {
  BackendLane lane;
  for (const int factor : {2, 3, 5}) {
    const std::vector<size_t> low{2, 257, 2, 2};
    const std::vector<size_t> high{2u * factor, 257u * factor, 2, 2};
    std::vector<float> x(2 * 257 * 4), dy(x.size() * factor * factor);
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8;
    for (size_t i = 0; i < dy.size(); ++i)
      dy[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 16;
    const auto input = DeviceOnlyTensor(low, x);
    const auto cotangent = DeviceOnlyTensor(high, dy);
    Upsample2DLayer layer(factor, UpsampleMode::Bilinear);
    Tensor device_output, device_gradient;
    {
      ScopedArrayFireFallbackPolicy strict(
          ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
      device_output = layer.Forward(input);
      device_gradient = layer.Backward(cotangent);
    }
    for (bool backward : {false, true}) {
      ScopedEnvVar hook("CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK",
                        backward ? "Upsample2DLayer::Backward"
                                 : "Upsample2DLayer::Forward");
      ScopedArrayFireFallbackPolicy compat(
          ArrayFireFallbackPolicy::AllowNativeCpuFallback);
      const auto native =
          backward ? layer.Backward(cotangent) : layer.Forward(input);
      const auto &device = backward ? device_gradient : device_output;
      REQUIRE(native.Shape() == device.Shape());
      const float *actual = device.ReadData<float>();
      const float *expected = native.ReadData<float>();
      double maximum_error = 0;
      for (size_t i = 0; i < native.NumElements(); ++i) {
        const double error =
            std::abs(static_cast<double>(actual[i]) - expected[i]);
        maximum_error = std::isfinite(error)
                            ? (std::max)(maximum_error, error)
                            : std::numeric_limits<double>::infinity();
      }
      CAPTURE(factor, backward, maximum_error);
      // Both paths use factor-local weights. Float32 weight algebra and
      // accumulation order differ for the separable device and native paths.
      CHECK(maximum_error < 2e-5);
    }
  }
}

TEST_CASE("Upsample native routes record compatibility and strict rejection",
          "[upsample][fallback]") {
  BackendLane lane;
  for (const auto mode : {UpsampleMode::Nearest, UpsampleMode::Bilinear}) {
    for (const bool backward : {false, true}) {
      for (const bool strict : {false, true}) {
        CAPTURE(static_cast<int>(mode), backward, strict);
        Upsample2DLayer layer(2, mode);
        const auto low = DeviceOnlyTensor({1, 2, 1, 2}, {1, -2, 3, 4});
        const auto high = layer.Forward(low);
        const auto expected = backward ? layer.Backward(high) : high;
        const auto &source = backward ? high : low;
        const auto input = Tensor::FromSemanticArray(source.GetSemanticArray(),
                                                     source.Shape());
        const char *operation =
            backward ? "Upsample2DLayer::Backward" : "Upsample2DLayer::Forward";
        Tensor result;
        ResetConvObservations();
        {
          ScopedEnvVar hook("CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK", operation);
          ScopedArrayFireFallbackPolicy policy(
              strict ? ArrayFireFallbackPolicy::ForbidNativeCpuFallback
                     : ArrayFireFallbackPolicy::AllowNativeCpuFallback);
          ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
          ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
          const auto run = [&] {
            return backward ? layer.Backward(input) : layer.Forward(input);
          };
          if (strict)
            REQUIRE_THROWS_AS(run(), std::runtime_error);
          else
            result = run();
        }
        CHECK(conv_fallback_count == 1);
        CHECK(last_conv_fallback.operation_name == operation);
        CHECK(last_conv_fallback.selected_backend ==
              CurrentArrayFireBackendName());
        CHECK(last_conv_fallback.reason_code == "backend_internal_error");
        CHECK(last_conv_fallback.fallback_forbidden == strict);
        CHECK(conv_host_sync_count == (strict ? 0 : 1));
        CHECK(conv_host_sync_bytes == (strict ? 0 : input.NumBytes()));
        if (!strict) {
          CHECK(saw_conv_cpu_path);
          const auto data = expected.ReadData<float>();
          CheckValues(result, expected.Shape(),
                      {data, data + expected.NumElements()});
        } else if (!backward) {
          REQUIRE_THROWS_WITH(
              layer.Backward(high),
              "Upsample2D::Backward requires a successful Forward call");
        }
      }
    }
  }
}
#endif

TEST_CASE("Upsample supports two optimizer updates and checkpoint reload",
          "[upsample][integration]") {
  BackendLane lane;
  for (auto mode : {UpsampleMode::Nearest, UpsampleMode::Bilinear}) {
    SequentialModel model;
    auto conv = std::make_unique<Conv2DModule>(1, 1, 1, 1, 0, false);
    conv->SetParameters({{"weights", DeviceOnlyTensor({1, 1, 1, 1}, {2})}});
    model.AddModule(std::move(conv));
    model.Add<Upsample2DModule>(2, mode);
    SGDOptimizer optimizer(.125);
    const auto input = DeviceOnlyTensor({1, 1, 1, 2}, {1, 2});
    const auto dy = DeviceOnlyOnes({2, 2, 1, 2});
    for (int step = 0; step < 2; ++step) {
      Tensor y, dx;
      ResetConvObservations();
      {
        ScopedArrayFireFallbackPolicy strict(
            ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
        ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
        y = model.Forward(input);
        dx = model.Backward(dy);
        model.UpdateParameters(&optimizer);
        for (const auto &item : model.GetParameters())
          item.second.GetSemanticArray().eval();
        af::sync();
      }
      CHECK(conv_host_sync_count == 0);
      CHECK(conv_host_sync_bytes == 0);
      CHECK(conv_fallback_count == 0);
      const float weight = 2.0f - 1.5f * step;
      CheckValues(y, {2, 2, 1, 2},
                  {weight, 2 * weight, weight, 2 * weight, weight, 2 * weight,
                   weight, 2 * weight});
      CheckValues(dx, input.Shape(), {4 * weight, 4 * weight});
      REQUIRE(model.GetParameters().size() == 1);
    }
    const auto path = cyxwiz::test::UniqueModelPath("cyxwiz_upsample_");
    REQUIRE(model.Save(path.string()));
    SequentialModel restored;
    restored.Add<Conv2DModule>(1, 1, 1, 1, 0, false);
    restored.Add<Upsample2DModule>(2, mode);
    REQUIRE(restored.Load(path.string()));
    CheckValues(restored.Forward(input), {2, 2, 1, 2},
                {-1, -2, -1, -2, -1, -2, -1, -2});
    std::error_code error;
    std::filesystem::remove(path, error);
    REQUIRE_FALSE(error);
  }
}

TEST_CASE("Bilinear Upsample and its nonuniform adjoint stay device resident",
          "[upsample][bilinear][residency]") {
  BackendLane lane;
  for (int factor : {1, 2, 3, 4, 5}) {
    for (const auto &shape :
         {std::vector<size_t>{2, 3, 2, 2}, std::vector<size_t>{1, 3, 1, 1},
          std::vector<size_t>{3, 1, 1, 1}, std::vector<size_t>{1, 1, 2, 2}}) {
      CAPTURE(factor, shape);
      const size_t count = shape[0] * shape[1] * shape[2] * shape[3];
      std::vector<float> x(count), dy(count * factor * factor);
      for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 8;
      for (size_t i = 0; i < dy.size(); ++i)
        dy[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 16;
      const std::vector<size_t> high{shape[0] * factor, shape[1] * factor,
                                     shape[2], shape[3]};
      const auto input = DeviceOnlyTensor(shape, x);
      const auto cotangent = DeviceOnlyTensor(high, dy);
      Upsample2DLayer layer(factor, UpsampleMode::Bilinear);
      Tensor y, dx;
      ResetConvObservations();
      {
        ScopedArrayFireFallbackPolicy strict(
            ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
        ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
        y = layer.Forward(input);
        dx = layer.Backward(cotangent);
        dx.GetSemanticArray().eval();
        af::sync();
      }
      CHECK(conv_host_sync_count == 0);
      CHECK(conv_host_sync_bytes == 0);
      CHECK(conv_fallback_count == 0);
      REQUIRE(y.Shape() == high);
      REQUIRE(dx.Shape() == shape);
      double forward_dot = 0, backward_dot = 0;
      const auto y_values = y.ReadData<float>();
      const auto dx_values = dx.ReadData<float>();
      for (size_t i = 0; i < dy.size(); ++i)
        forward_dot += static_cast<double>(y_values[i]) * dy[i];
      for (size_t i = 0; i < x.size(); ++i)
        backward_dot += static_cast<double>(dx_values[i]) * x[i];
      CHECK(forward_dot == Catch::Approx(backward_dot).margin(2e-5));
    }
  }
}
#endif
