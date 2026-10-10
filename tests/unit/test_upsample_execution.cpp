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
