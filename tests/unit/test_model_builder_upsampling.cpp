#include "convolution_test_support.h"
#include "core/model_builder.h"
#include "core/upsampling_configuration_policy.h"
#include "model_test_path.h"
#include <filesystem>

using namespace cyxwiz;
using namespace cyxwiz::test::convolution;

namespace {
TrainingConfiguration Configuration(gui::NodeType type, int mode = 0) {
  TrainingConfiguration config;
  config.loss_type = gui::NodeType::MSELoss;
  config.optimizer_type = gui::NodeType::SGD;
  CompiledLayer layer;
  layer.type = type;
  layer.node_id = 100;
  layer.name = "Spatial output";
  layer.parameters =
      type == gui::NodeType::Upsample
          ? std::map<std::string, std::string>{{"scale_factor", "2"},
                                               {"mode", std::to_string(mode)}}
          : std::map<std::string, std::string>{{"upscale_factor", "2"}};
  config.layers.push_back(layer);
  return config;
}
} // namespace

TEST_CASE("Upsampling configuration rejects malformed persisted values",
          "[upsampling_builder][configuration]") {
  for (auto type : {gui::NodeType::Upsample, gui::NodeType::PixelShuffle}) {
    const char *key =
        type == gui::NodeType::Upsample ? "scale_factor" : "upscale_factor";
    for (const std::string value : {"", "0", "-1", "2junk", "2.5", "1048577",
                                    "999999999999999999999", " 2", "2 "}) {
      auto config = Configuration(type);
      config.layers[0].parameters[key] = value;
      const auto built = BuildSequentialFromConfig(config);
      CAPTURE(type, key, value);
      REQUIRE_FALSE(built.ok());
      CHECK(built.error_message.find(key) != std::string::npos);
      CHECK(built.error_message.find("index 0") != std::string::npos);
    }
  }
  for (const std::string value : {"", "-1", "2", "nearest", "1junk"}) {
    auto config = Configuration(gui::NodeType::Upsample);
    config.layers[0].parameters["mode"] = value;
    const auto built = BuildSequentialFromConfig(config);
    REQUIRE_FALSE(built.ok());
    CHECK(built.error_message.find("mode") != std::string::npos);
  }
}

TEST_CASE("Upsampling config uses persisted keys or exact legacy fields",
          "[upsampling_builder][configuration]") {
  UpsamplingConfiguration resolved;
  REQUIRE_FALSE(
      ResolveUpsamplingConfiguration(gui::NodeType::Upsample, {}, resolved));
  CHECK(resolved.factor == 2);
  CHECK(resolved.mode == 0);
  REQUIRE_FALSE(ResolveUpsamplingConfiguration(gui::NodeType::Upsample, {},
                                               resolved, 3, 1));
  CHECK(resolved.factor == 3);
  CHECK(resolved.mode == 1);
  REQUIRE_FALSE(ResolveUpsamplingConfiguration(
      gui::NodeType::Upsample, {{"scale_factor", "1"}, {"mode", "0"}}, resolved,
      -1, 9));
  CHECK(resolved.factor == 1);
  CHECK(resolved.mode == 0);
  REQUIRE(
      ResolveUpsamplingConfiguration(gui::NodeType::Upsample, {}, resolved, 0));
  REQUIRE(ResolveUpsamplingConfiguration(gui::NodeType::Upsample, {}, resolved,
                                         2, 9));
  REQUIRE_FALSE(ResolveUpsamplingConfiguration(
      gui::NodeType::PixelShuffle, {{"upscale_factor", "3"}}, resolved, 1));
  CHECK(resolved.factor == 3);
}

TEST_CASE("Upsampling sample inference validates metadata without allocation",
          "[upsampling_builder][geometry]") {
  const auto up = gui::NodeType::Upsample;
  const auto shuffle = gui::NodeType::PixelShuffle;
  for (int mode : {0, 1}) {
    CHECK(InferUpsamplingSampleShape(up, {3, mode}, {2, 5, 4}) ==
          std::vector<size_t>{6, 15, 4});
  }
  CHECK(InferUpsamplingSampleShape(shuffle, {3, 0}, {2, 5, 18}) ==
        std::vector<size_t>{6, 15, 2});
  for (auto type : {up, shuffle}) {
    CHECK(InferUpsamplingSampleShape(type, {1, 0}, {1, 3, 1}) ==
          std::vector<size_t>{1, 3, 1});
    for (const auto &shape :
         {std::vector<size_t>{}, {2, 3}, {2, 3, 4, 1}, {0, 3, 4}})
      CHECK_THROWS_AS(InferUpsamplingSampleShape(type, {2, 0}, shape),
                      std::invalid_argument);
    for (int factor : {0, -1, 1048577})
      CHECK_THROWS_AS(InferUpsamplingSampleShape(type, {factor, 0}, {2, 3, 4}),
                      std::invalid_argument);
    const auto limit = static_cast<size_t>((std::numeric_limits<int>::max)());
    CHECK_THROWS_AS(InferUpsamplingSampleShape(type, {2, 0}, {limit, 1, 4}),
                    std::overflow_error);
  }
  for (size_t channels : {size_t{1}, size_t{2}, size_t{6}})
    CHECK_THROWS_AS(
        InferUpsamplingSampleShape(shuffle, {2, 0}, {2, 3, channels}),
        std::invalid_argument);
  CHECK_THROWS_AS(InferUpsamplingSampleShape(up, {2, 2}, {2, 3, 4}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      InferUpsamplingSampleShape(gui::NodeType::Dense, {2, 0}, {2, 3, 4}),
      std::invalid_argument);
  CHECK_THROWS_AS(InferUpsamplingSampleShape(shuffle, {1048576, 0}, {1, 1, 4}),
                  std::invalid_argument);
  if (sizeof(size_t) == 8) {
    // Each axis fits int, input bytes fit size_t, but output bytes do not.
    CHECK_THROWS_AS(
        InferUpsamplingSampleShape(up, {2, 0}, {1000000000, 1000000000, 2}),
        std::overflow_error);
  }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("ModelBuilder preserves upsampling modes layout and node provenance",
          "[upsampling_builder][residency]") {
  BackendLane lane;
  for (auto type : {gui::NodeType::Upsample, gui::NodeType::PixelShuffle}) {
    for (int mode = 0; mode < (type == gui::NodeType::Upsample ? 2 : 1);
         ++mode) {
      auto config = Configuration(type, mode);
      // Raw persisted graph keys take priority over legacy struct defaults.
      config.layers[0].scale_factor = 3;
      config.layers[0].upsample_mode = 1 - mode;
      auto built = BuildSequentialFromConfig(config);
      REQUIRE(built.ok());
      REQUIRE(built.model->Size() == 1);
      REQUIRE(built.module_provenance.size() == 1);
      const auto &origin = built.module_provenance[0];
      CHECK(origin.created());
      CHECK(origin.module_index == 0);
      CHECK(origin.node_id == 100);
      CHECK(origin.node_type == type);
      CHECK(origin.configured_parameters == config.layers[0].parameters);
      if (type == gui::NodeType::Upsample)
        REQUIRE(dynamic_cast<Upsample2DModule *>(built.model->GetModule(0)) !=
                nullptr);
      else
        REQUIRE(dynamic_cast<PixelShuffleModule *>(built.model->GetModule(0)) !=
                nullptr);

      // Two successive batches, including a partial final batch. This
      // tests direct construction, not Studio batching/graph compilation.
      for (size_t batch : {size_t{2}, size_t{1}}) {
        const bool shuffle = type == gui::NodeType::PixelShuffle;
        const std::vector<size_t> low =
            shuffle ? std::vector<size_t>{1, 1, 4, batch}
                    : std::vector<size_t>{1, 2, 1, batch};
        std::vector<float> x, expected;
        for (int pixel = 0; pixel < (shuffle ? 4 : 2); ++pixel)
          for (size_t b = 0; b < batch; ++b)
            x.push_back(static_cast<float>(4 * pixel + 8 * b));
        constexpr float bilinear_row[] = {0, 1, 3, 4};
        for (int row = 0; row < 2; ++row)
          for (int col = 0; col < (shuffle ? 2 : 4); ++col)
            for (size_t b = 0; b < batch; ++b) {
              const float value =
                  shuffle ? static_cast<float>(4 * (row * 2 + col))
                          : (mode == 0 ? static_cast<float>(4 * (col / 2))
                                       : bilinear_row[col]);
              expected.push_back(value + 8 * static_cast<float>(b));
            }
        const std::vector<size_t> high{2, shuffle ? size_t{2} : size_t{4}, 1,
                                       batch};
        const auto input = DeviceOnlyTensor(low, x);
        const auto dy = DeviceOnlyOnes(high);
        Tensor y, dx;
        ResetConvObservations();
        {
          ScopedArrayFireFallbackPolicy strict(
              ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
          ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
          ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
          y = built.model->Forward(input);
          dx = built.model->Backward(dy);
          dx.GetSemanticArray().eval();
          af::sync();
        }
        CHECK(conv_host_sync_count == 0);
        CHECK(conv_host_sync_bytes == 0);
        CHECK(conv_fallback_count == 0);
        CheckValues(y, high, expected);
        UpsamplingConfiguration resolved;
        REQUIRE_FALSE(ResolveUpsamplingConfiguration(
            type, config.layers[0].parameters, resolved));
        CHECK(y.Shape() ==
              SpatialRuntimeShape(InferUpsamplingSampleShape(
                                      type, resolved, {low[0], low[1], low[2]}),
                                  batch));
        CheckValues(dx, low,
                    std::vector<float>(x.size(), shuffle ? 1.0f : 4.0f));
        const auto path = cyxwiz::test::UniqueModelPath("builder_upsampling_");
        REQUIRE(built.model->Save(path.string()));
        auto restored = BuildSequentialFromConfig(config);
        REQUIRE(restored.ok());
        REQUIRE(restored.model->Load(path.string()));
        CheckValues(restored.model->Forward(input), high, expected);
        REQUIRE(std::filesystem::remove(path));
      }
    }
  }
}
#endif
