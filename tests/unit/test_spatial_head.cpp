#include "convolution_test_support.h"
#include "core/model_builder.h"
#include "core/spatial_batch_layout.h"
#include "core/spatial_head_module.h"
#include "core/spatial_sequential_head.h"
#include "model_test_path.h"
#include <filesystem>

using namespace cyxwiz;
using namespace cyxwiz::test::convolution;

namespace {
TrainingConfiguration HeadConfiguration(gui::NodeType type, int mode = 0) {
  TrainingConfiguration config;
  config.input_shape = type == gui::NodeType::Upsample
                           ? std::vector<size_t>{1, 2, 1}
                           : std::vector<size_t>{1, 1, 4};
  config.input_size = SpatialSampleElements(config.input_shape);
  config.output_size = 1;
  config.loss_type = gui::NodeType::MSELoss;
  config.optimizer_type = gui::NodeType::SGD;
  config.learning_rate = .01f;
  for (auto node_type : {type, gui::NodeType::ReLU, gui::NodeType::Flatten,
                         gui::NodeType::Dense}) {
    CompiledLayer layer;
    layer.type = node_type;
    layer.node_id = static_cast<int>(100 + config.layers.size());
    layer.units = 1;
    config.layers.push_back(layer);
  }
  config.layers.front().parameters =
      type == gui::NodeType::Upsample
          ? std::map<std::string, std::string>{{"scale_factor", "2"},
                                               {"mode", std::to_string(mode)}}
          : std::map<std::string, std::string>{{"upscale_factor", "2"}};
  return config;
}
} // namespace

TEST_CASE("Spatial head construction rejects ambiguous layouts",
          "[spatial_head][configuration]") {
  auto config = HeadConfiguration(gui::NodeType::Upsample);
  const auto head = ResolveSpatialSequentialHead(config);
  REQUIRE(head);
  CHECK(head->flatten_index == 2);
  CHECK(head->sample_shape == std::vector<size_t>{2, 4, 1});
  CHECK(head->features == 8);
  for (int bad : {0, 1, 2, 3, 4}) {
    auto invalid = config;
    if (bad == 0)
      invalid.input_shape.clear();
    if (bad == 1)
      invalid.input_size = 99;
    if (bad == 2)
      invalid.layers.erase(invalid.layers.begin() + 2);
    if (bad == 3)
      invalid.layers[1].type = gui::NodeType::Reshape;
    if (bad == 4)
      invalid.layers.push_back(invalid.layers.front());
    const auto built = BuildSequentialFromConfig(invalid);
    CHECK_FALSE(built.ok());
    CHECK_FALSE(built.error_message.empty());
  }
  config.layers.front().type = gui::NodeType::Dense;
  CHECK_FALSE(ResolveSpatialSequentialHead(config));
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Spatial Flatten preserves row ordering and backward lifecycle",
          "[spatial_head][residency]") {
  BackendLane lane;
  SpatialFlattenModule module({2, 3, 2});
  CHECK_THROWS_AS(module.Backward(DeviceOnlyOnes({1, 12})), std::logic_error);
  for (size_t batch : {size_t{3}, size_t{1}}) {
    std::vector<float> values(batch * 12);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(i);
    const auto rows = DeviceOnlyTensor({batch, 12}, values);
    const auto spatial = SpatialBatchFromRows(rows, {2, 3, 2});
    Tensor result, gradient;
    ResetConvObservations();
    {
      ScopedArrayFireFallbackPolicy strict(
          ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
      ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
      ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
      result = module.Forward(spatial);
      gradient = SpatialBatchToRows(module.Backward(rows));
      gradient.GetSemanticArray().eval();
      af::sync();
    }
    CHECK(conv_host_sync_count == 0);
    CHECK(conv_host_sync_bytes == 0);
    CHECK(conv_fallback_count == 0);
    CheckValues(result, {batch, 12}, values);
    CheckValues(gradient, {batch, 12}, values);
    CHECK_THROWS_AS(module.Backward(DeviceOnlyOnes({batch + 1, 12})),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        module.Backward(Tensor({batch, 12}, nullptr, DataType::Int32)),
        std::invalid_argument);
    CHECK_THROWS_AS(module.Forward(DeviceOnlyOnes({1, 3, 2, batch})),
                    std::invalid_argument);
    CHECK_THROWS_AS(module.Backward(rows), std::logic_error);
  }
}

TEST_CASE("Built spatial heads train two batch sizes and reload weights",
          "[spatial_head][integration]") {
  BackendLane lane;
  for (auto type : {gui::NodeType::Upsample, gui::NodeType::PixelShuffle}) {
    for (int mode = 0; mode < (type == gui::NodeType::Upsample ? 2 : 1);
         ++mode) {
      auto config = HeadConfiguration(type, mode);
      auto built = BuildSequentialFromConfig(config);
      REQUIRE(built.ok());
      REQUIRE(built.model->Size() == 4);
      REQUIRE(dynamic_cast<SpatialFlattenModule *>(built.model->GetModule(2)));
      REQUIRE(built.module_provenance.size() == 4);
      CHECK(built.module_provenance[2].node_id == 102);
      CHECK(built.module_provenance[2].module_name == "SpatialFlatten");
      const size_t features = type == gui::NodeType::Upsample ? 8 : 4;
      auto *dense = built.model->GetModule(3);
      CHECK(dense->GetParameters().at("weight").Shape() ==
            std::vector<size_t>{1, features});
      dense->SetParameters(
          {{"weight",
            DeviceOnlyTensor({1, features},
                             std::vector<float>(features, 1.0f / features))},
           {"bias", DeviceOnlyTensor({1}, {0})}});
      for (size_t batch : {size_t{2}, size_t{1}}) {
        std::vector<float> values;
        for (size_t b = 0; b < batch; ++b)
          values.insert(values.end(), config.input_size,
                        static_cast<float>(b + 1));
        const auto rows = DeviceOnlyTensor({batch, config.input_size}, values);
        const auto targets =
            DeviceOnlyTensor({batch, 1}, std::vector<float>(batch, 0));
        Tensor predictions, loss, dx;
        ResetConvObservations();
        {
          ScopedArrayFireFallbackPolicy strict(
              ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
          ScopedArrayFireHostSyncObserver host(&CountConvHostSync);
          ScopedArrayFireNativeCpuFallbackObserver fallback(&CountConvFallback);
          predictions = built.model->Forward(
              SpatialBatchFromRows(rows, config.input_shape));
          loss = built.loss->Forward(predictions, targets);
          auto loss_gradient = built.loss->Backward(predictions, targets);
          // Match TrainingExecutor::Backward's established boundary: ArrayFire
          // loss output may drop singleton axes. Restore the prediction shape
          // on device before entering the model, including the N=1 batch.
          REQUIRE(loss_gradient.NumElements() == predictions.NumElements());
          loss_gradient = loss_gradient.Reshape(predictions.Shape());
          dx = SpatialBatchToRows(built.model->Backward(loss_gradient));
          built.model->UpdateParameters(built.optimizer.get());
          for (const auto &item : built.model->GetParameters())
            item.second.GetSemanticArray().eval();
          loss.GetSemanticArray().eval();
          dx.GetSemanticArray().eval();
          af::sync();
        }
        CHECK(conv_host_sync_count == 0);
        CHECK(conv_host_sync_bytes == 0);
        CHECK(conv_fallback_count == 0);
        const float second_prediction = 1.0f - .05f * features - .03f;
        CheckValues(predictions, {batch, 1},
                    batch == 2 ? std::vector<float>{1, 2}
                               : std::vector<float>{second_prediction});
        CHECK(loss.ReadData<float>()[0] ==
              Catch::Approx(
                  batch == 2 ? 2.5f : second_prediction * second_prediction));
        const float weight = 1.0f / features - (batch == 1 ? .05f : 0.0f);
        const float copies = type == gui::NodeType::Upsample ? 4.0f : 1.0f;
        std::vector<float> expected_dx;
        for (size_t b = 0; b < batch; ++b)
          expected_dx.insert(expected_dx.end(), config.input_size,
                             copies * weight *
                                 (batch == 2 ? static_cast<float>(b + 1)
                                             : 2 * second_prediction));
        CheckValues(dx, {batch, config.input_size}, expected_dx);
      }
      const auto path = cyxwiz::test::UniqueModelPath("spatial_head_");
      REQUIRE(built.model->Save(path.string()));
      auto restored = BuildSequentialFromConfig(config);
      REQUIRE(restored.ok());
      REQUIRE(restored.model->Load(path.string()));
      const auto input = SpatialBatchFromRows(
          DeviceOnlyOnes({1, config.input_size}), config.input_shape);
      const auto expected = built.model->Forward(input);
      CheckValues(restored.model->Forward(input), {1, 1},
                  {expected.ReadData<float>()[0]});
      REQUIRE(std::filesystem::remove(path));
    }
  }
}
#endif
