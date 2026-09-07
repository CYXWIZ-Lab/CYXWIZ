#include "convolution_test_support.h"
#include "model_test_path.h"
#include <catch2/matchers/catch_matchers.hpp>
#include <algorithm>
#include <chrono>
#include <cyxwiz/layers/upsampling.h>
#include <cyxwiz/sequential.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>

using namespace cyxwiz;
using namespace cyxwiz::test::convolution;

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("PixelShuffle rejects empty and unrepresentable geometry before "
          "device access",
          "[pixelshuffle][validation]") {
  BackendLane lane;
  PixelShuffleLayer layer(2);
  for (size_t axis = 0; axis < 4; ++axis) {
    std::vector<size_t> shape{2, 2, 4, 1};
    shape[axis] = 0;
    const Tensor empty(shape, DataType::Float32);
    REQUIRE_THROWS_AS(layer.Forward(empty), std::runtime_error);
    REQUIRE_THROWS_AS(layer.Backward(empty), std::runtime_error);
  }
  if (sizeof(size_t) > sizeof(int)) {
    // Metadata-only Tensor avoids a multi-GB allocation. The provider
    // dimension guard must reject before any attempt to read its storage.
    const size_t extent =
        static_cast<size_t>((std::numeric_limits<int>::max)()) + 1;
    const Tensor oversized({extent, 1, 4, 1}, nullptr, DataType::Float32);
    ResetConvObservations();
    {
      ScopedArrayFireFallbackPolicy strict(
          ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
      ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
      ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
      REQUIRE_THROWS_WITH(
          layer.Forward(oversized),
          "PixelShuffle dimension exceeds ArrayFire int dimension limit");
    }
    CHECK(conv_host_sync_count == 0);
    CHECK(conv_fallback_count == 0);
  }
}

TEST_CASE("PixelShuffle strict permutation and inverse stay device resident",
          "[pixelshuffle][residency]") {
  BackendLane lane;
  for (const int factor : {1, 2, 3}) {
    CAPTURE(factor);
    const size_t r = static_cast<size_t>(factor);
    const std::vector<size_t> shape{2, 3, 2 * r * r, 2};
    std::vector<float> values(24 * r * r);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(i) / 8.0f;
    const auto x = DeviceOnlyTensor(shape, values);
    PixelShuffleLayer layer(factor), inverse(factor);
    Tensor output, restored;
    ResetConvObservations();
    {
      ScopedArrayFireFallbackPolicy strict(
          ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
      ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
      ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
      output = layer.Forward(x);
      restored = inverse.Backward(output);
      restored.GetSemanticArray().eval();
      af::sync();
    }
    CHECK(conv_host_sync_count == 0);
    CHECK(conv_host_sync_bytes == 0);
    CHECK(conv_fallback_count == 0);
    REQUIRE(output.Shape() == std::vector<size_t>{2 * r, 3 * r, 2, 2});
    CheckValues(restored, shape, values);
  }
}

TEST_CASE("PixelShuffle composes with trainable modules and checkpoint state",
          "[pixelshuffle][integration]") {
  BackendLane lane;
  auto conv = std::make_unique<Conv2DModule>(1, 4, 1, 1, 0, false);
  conv->SetParameters(
      {{"weights", DeviceOnlyTensor({1, 1, 1, 4}, {1, 2, 3, 4})}});
  SequentialModel model;
  model.AddModule(std::move(conv));
  model.Add<PixelShuffleModule>(2);
  SGDOptimizer optimizer(.1);
  const auto x = DeviceOnlyTensor({1, 1, 1, 2}, {2, 3});
  const auto dy = DeviceOnlyOnes({2, 2, 1, 2});
  for (int step = 0; step < 2; ++step) {
    Tensor y, dx;
    ResetConvObservations();
    {
      ScopedArrayFireFallbackPolicy strict(
          ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
      ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
      ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
      y = model.Forward(x);
      dx = model.Backward(dy);
      model.UpdateParameters(&optimizer);
      for (const auto &item : model.GetParameters())
        item.second.GetSemanticArray().eval();
      af::sync();
    }
    CHECK(conv_host_sync_count == 0);
    CHECK(conv_host_sync_bytes == 0);
    CHECK(conv_fallback_count == 0);
    const float offset = .5f * step;
    CheckValues(y, {2, 2, 1, 2},
                {2 * (1 - offset), 3 * (1 - offset), 2 * (2 - offset),
                 3 * (2 - offset), 2 * (3 - offset), 3 * (3 - offset),
                 2 * (4 - offset), 3 * (4 - offset)});
    CheckValues(dx, x.Shape(), {10.0f - 2 * step, 10.0f - 2 * step});
    REQUIRE(model.GetParameters().size() == 1);
  }
  const auto path = cyxwiz::test::UniqueModelPath("cyxwiz_pixel_shuffle_");
  REQUIRE(model.Save(path.string()));
  SequentialModel restored;
  restored.Add<Conv2DModule>(1, 4, 1, 1, 0, false);
  restored.Add<PixelShuffleModule>(2);
  REQUIRE(restored.Load(path.string()));
  const auto loaded_output = restored.Forward(x);
  REQUIRE(loaded_output.Shape() == std::vector<size_t>{2, 2, 1, 2});
  const std::vector<float> expected{0, 0, 2, 3, 4, 6, 6, 9};
  const float *loaded_values = loaded_output.ReadData<float>();
  // SGD uses Float32 arithmetic: cancellation to zero may leave ~2e-8 on
  // CUDA. This tolerance is local to optimizer/checkpoint integration, not
  // the exact permutation tests or the shared assertion helper.
  for (size_t i = 0; i < expected.size(); ++i) {
    CHECK(loaded_values[i] == Catch::Approx(expected[i]).margin(1e-6));
  }
  std::error_code error;
  std::filesystem::remove(path, error);
  REQUIRE_FALSE(error);
}

#ifndef NDEBUG
TEST_CASE(
    "PixelShuffle observed fallback honors strict and compatibility policies",
    "[pixelshuffle][fallback]") {
  BackendLane lane;
  for (const bool inverse : {false, true}) {
    for (const bool strict : {false, true}) {
      CAPTURE(inverse, strict);
      const char *operation = inverse ? "PixelShuffleLayer::Backward"
                                      : "PixelShuffleLayer::Forward";
      std::vector<float> values(48);
      for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(i) - 20.0f;
      const auto low = DeviceOnlyTensor({2, 3, 4, 2}, values);
      PixelShuffleLayer layer(2);
      const auto high = layer.Forward(low);
      const auto expected = inverse ? low : high;
      // Fresh input wrappers ensure compatibility readback remains observable.
      const auto input = Tensor::FromSemanticArray(
          inverse ? high.GetSemanticArray() : low.GetSemanticArray(),
          inverse ? high.Shape() : low.Shape());
      Tensor result;
      ResetConvObservations();
      {
        ScopedEnvVar hook("CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK", operation);
        ScopedArrayFireFallbackPolicy policy(
            strict ? ArrayFireFallbackPolicy::ForbidNativeCpuFallback
                   : ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
        ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
        const auto run = [&] {
          return inverse ? layer.Backward(input) : layer.Forward(input);
        };
        if (strict)
          CHECK_THROWS_AS(run(), std::runtime_error);
        else
          result = run();
      }
      CHECK(conv_fallback_count == 1);
      CHECK(last_conv_fallback.operation_name == operation);
      CHECK(last_conv_fallback.fallback_forbidden == strict);
      if (strict) {
        CHECK(conv_host_sync_count == 0);
        CHECK(conv_host_sync_bytes == 0);
      } else {
        CHECK(conv_host_sync_count == 1);
        CHECK(conv_host_sync_bytes == 48 * sizeof(float));
        CHECK(saw_conv_cpu_path);
        const float *data = expected.ReadData<float>();
        CheckValues(result, expected.Shape(),
                    {data, data + expected.NumElements()});
      }
    }
  }
}
#endif

TEST_CASE("PixelShuffle fixed device-input baseline",
          "[.][pixelshuffle][benchmark]") {
  BackendLane lane;
  PixelShuffleLayer layer(2);
  constexpr int iterations = 20;
  const auto x = DeviceOnlyOnes({32, 48, 32, 4});
  const auto dy = DeviceOnlyOnes({64, 96, 8, 4});
  // Fresh device-owned handles prevent native readback caching from hiding
  // the transfers made for a new batch. Ingress is outside observation.
  const af::array x_array = x.GetSemanticArray(),
                  dy_array = dy.GetSemanticArray();
  const auto run = [&] {
    auto fresh_x = Tensor::FromSemanticArray(x_array, x.Shape());
    auto fresh_dy = Tensor::FromSemanticArray(dy_array, dy.Shape());
    auto y = layer.Forward(fresh_x);
    auto dx = layer.Backward(fresh_dy);
    y.GetSemanticArray().eval();
    dx.GetSemanticArray().eval();
  };
  for (int i = 0; i < 5; ++i)
    run();
  af::sync();
  std::vector<double> samples;
  size_t syncs = 0, fallbacks = 0;
  uint64_t bytes = 0;
  for (int sample = 0; sample < 5; ++sample) {
    ResetConvObservations();
    const auto start = std::chrono::steady_clock::now();
    {
      ScopedArrayFireFallbackPolicy policy(
          ArrayFireFallbackPolicy::AllowNativeCpuFallback);
      ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
      ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
      for (int i = 0; i < iterations; ++i)
        run();
      af::sync();
    }
    samples.push_back(std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count() /
                      iterations);
    syncs += conv_host_sync_count;
    bytes += conv_host_sync_bytes;
    fallbacks += conv_fallback_count;
  }
  std::sort(samples.begin(), samples.end());
  std::cout << "PixelShuffle baseline backend=" << CurrentArrayFireBackendName()
            << " device=" << af::getDevice() << " input=[32,48,32,4] factor=2"
            << " policy=allow_native samples=5 iterations=20 median_pair_ms="
            << samples[2] << " host_syncs=" << syncs << " host_bytes=" << bytes
            << " fallbacks=" << fallbacks << '\n';
  REQUIRE(samples[2] > 0.0);
}
#endif
