#include "convolution_test_support.h"
#include "core/model_builder.h"
#include "core/spatial_batch_layout.h"
#include "core/synthetic_batch.h"
#include <limits>

using namespace cyxwiz;
using namespace cyxwiz::test::convolution;

TEST_CASE("Spatial sample shape excludes the runtime batch axis",
          "[spatial_layout][geometry]") {
  CHECK(SpatialSampleElements({2, 3, 4}) == 24);
  CHECK(SpatialRuntimeShape({2, 3, 4}, 5) == std::vector<size_t>{2, 3, 4, 5});
  for (const auto &shape :
       {std::vector<size_t>{}, {2, 3}, {2, 3, 4, 5}, {0, 3, 4}})
    CHECK_THROWS_AS(SpatialSampleElements(shape), std::invalid_argument);
  CHECK_THROWS_AS(SpatialRuntimeShape({2, 3, 4}, 0), std::invalid_argument);
  const auto limit = static_cast<size_t>((std::numeric_limits<int>::max)());
  if (sizeof(size_t) > sizeof(int)) {
    CHECK_THROWS_AS(SpatialSampleElements({limit + 1, 1, 1}),
                    std::invalid_argument);
    CHECK_THROWS_AS(SpatialSampleElements({limit, limit, limit}),
                    std::overflow_error);
    CHECK_THROWS_AS(SpatialRuntimeShape({limit, limit, 1}, 2),
                    std::overflow_error);
  }
  const Tensor bad_rows({2, 5}, nullptr, DataType::Float32);
  CHECK_THROWS_AS(SpatialBatchFromRows(bad_rows, {1, 2, 3}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      SpatialBatchFromRows(Tensor({2, 6}, nullptr, DataType::Int32), {1, 2, 3}),
      std::invalid_argument);
  CHECK_THROWS_AS(SpatialBatchFromRows(
                      Tensor({0, 6}, nullptr, DataType::Float32), {1, 2, 3}),
                  std::invalid_argument);
  CHECK_THROWS_AS(SpatialBatchToRows(bad_rows), std::invalid_argument);
  CHECK_THROWS_AS(
      SpatialBatchToRows(Tensor({1, 2, 3, 1}, nullptr, DataType::Int32)),
      std::invalid_argument);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Spatial row boundaries preserve sample channel order on device",
          "[spatial_layout][residency]") {
  BackendLane lane;
  for (const auto &sample :
       {std::vector<size_t>{2, 3, 2}, {1, 3, 1}, {2, 1, 3}}) {
    const size_t features = SpatialSampleElements(sample);
    for (size_t batch : {size_t{3}, size_t{1}}) {
      std::vector<float> rows(batch * features), expected;
      for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < features; ++f)
          rows[b * features + f] = static_cast<float>(100 * b + f);
      for (size_t f = 0; f < features; ++f)
        for (size_t b = 0; b < batch; ++b)
          expected.push_back(rows[b * features + f]);
      const auto input = DeviceOnlyTensor({batch, features}, rows);
      Tensor spatial, restored;
      ResetConvObservations();
      {
        ScopedArrayFireFallbackPolicy strict(
            ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
        ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
        spatial = SpatialBatchFromRows(input, sample);
        restored = SpatialBatchToRows(spatial);
        restored.GetSemanticArray().eval();
        af::sync();
      }
      CHECK(conv_host_sync_count == 0);
      CHECK(conv_host_sync_bytes == 0);
      CHECK(conv_fallback_count == 0);
      CheckValues(spatial, SpatialRuntimeShape(sample, batch), expected);
      CheckValues(restored, {batch, features}, rows);
    }
  }
}

TEST_CASE("Spatial boundaries compose with ModelBuilder forward and gradient",
          "[spatial_layout][integration]") {
  BackendLane lane;
  for (int mode : {0, 1}) {
    TrainingConfiguration config;
    config.loss_type = gui::NodeType::MSELoss;
    CompiledLayer layer;
    layer.type = gui::NodeType::Upsample;
    layer.parameters = {{"scale_factor", "2"}, {"mode", std::to_string(mode)}};
    config.layers.push_back(layer);
    auto built = BuildSequentialFromConfig(config);
    REQUIRE(built.ok());
    for (size_t batch : {size_t{2}, size_t{1}}) {
      std::vector<float> input_values;
      for (size_t b = 0; b < batch; ++b) {
        input_values.push_back(static_cast<float>(b + 1));
        input_values.push_back(static_cast<float>(b + 1));
      }
      const auto input = DeviceOnlyTensor({batch, 2}, input_values);
      const auto dy = DeviceOnlyOnes({batch, 8});
      Tensor y, dx;
      ResetConvObservations();
      {
        ScopedArrayFireFallbackPolicy strict(
            ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
        ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
        y = SpatialBatchToRows(
            built.model->Forward(SpatialBatchFromRows(input, {1, 2, 1})));
        dx = SpatialBatchToRows(
            built.model->Backward(SpatialBatchFromRows(dy, {2, 4, 1})));
        dx.GetSemanticArray().eval();
        af::sync();
      }
      CHECK(conv_host_sync_count == 0);
      CHECK(conv_host_sync_bytes == 0);
      CHECK(conv_fallback_count == 0);
      std::vector<float> expected;
      for (size_t b = 0; b < batch; ++b)
        expected.insert(expected.end(), 8, static_cast<float>(b + 1));
      CheckValues(y, {batch, 8}, expected);
      CheckValues(dx, {batch, 2}, std::vector<float>(batch * 2, 4));
    }
  }
}

TEST_CASE(
    "Synthetic spatial ingress is explicit and leaves Dense inputs unchanged",
    "[spatial_layout][synthetic]") {
  BackendLane lane;
  TrainingConfiguration config;
  config.input_shape = {2, 3, 2};
  config.input_size = 12;
  config.output_size = 2;
  config.loss_type = gui::NodeType::MSELoss;
  config.preprocessing_domain = PreprocessingDomain::Image;
  CompiledLayer first;
  first.type = gui::NodeType::Dense;
  config.layers.push_back(first);
  const auto dense = MakeSyntheticBatch(config, 100);
  REQUIRE(dense.features.Shape() == std::vector<size_t>{1, 12});
  for (auto type : {gui::NodeType::Upsample, gui::NodeType::PixelShuffle}) {
    config.layers[0].type = type;
    const auto spatial = MakeSyntheticBatch(config, 100);
    REQUIRE(spatial.features.Shape() == std::vector<size_t>{2, 3, 2, 1});
    const auto expected = dense.features.ReadData<float>();
    CheckValues(SpatialBatchToRows(spatial.features), {1, 12},
                {expected, expected + 12});
    CHECK(spatial.labels.Shape() ==
          dense.labels.Shape()); // Label/head integration is separate.
  }
  config.input_size = 13;
  CHECK_THROWS_AS(MakeSyntheticBatch(config, 100), std::invalid_argument);
  config.input_size = 0;
  config.input_shape = {2, 3};
  CHECK_THROWS_AS(MakeSyntheticBatch(config, 100), std::invalid_argument);
}
#endif
