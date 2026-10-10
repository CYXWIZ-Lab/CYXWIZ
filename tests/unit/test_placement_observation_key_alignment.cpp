// tofix67 slice 6: runtime observation writers and compile-time lookups
// must form the SAME key, or evidence is recorded but never consumed. The
// compiler's shapes are batchless; runtime tensor shapes carry a leading
// batch dimension, normalized via StripBatchDimensionForPlacementSignature.

#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"
#include "algorithms/layers/layer_utils.h"

#include <cyxwiz/backend_placement_observation.h>
#include <cyxwiz/tensor.h>

#include <string>
#include <vector>

TEST_CASE("Runtime writer shape signatures align with compile-time lookups",
          "[gpu_execution][taxonomy][placement]") {
    using namespace cyxwiz;

    // Activation: runtime sees [batch, features]; the compiler looks up
    // [features]. The normalized runtime signature must equal the
    // compiler-side signature.
    const std::vector<size_t> runtime_activation_shape = {32, 128};
    const std::vector<size_t> compiler_activation_shape = {128};
    CHECK(BuildActivationPlacementShapeSignature(
              StripBatchDimensionForPlacementSignature(
                  runtime_activation_shape),
              "float32") ==
          BuildActivationPlacementShapeSignature(compiler_activation_shape,
                                                 "float32"));

    // Embedding: runtime sees [batch, seq]; the compiler looks up [seq].
    const std::vector<size_t> runtime_embedding_shape = {16, 64};
    const std::vector<size_t> compiler_embedding_shape = {64};
    CHECK(BuildEmbeddingPlacementShapeSignature(
              1000, 32,
              StripBatchDimensionForPlacementSignature(
                  runtime_embedding_shape),
              "int32") ==
          BuildEmbeddingPlacementShapeSignature(
              1000, 32, compiler_embedding_shape, "int32"));

    // Dense was already batch-insensitive (features-only signature); the
    // helper must not disturb that property.
    CHECK(BuildDensePlacementShapeSignature(
              StripBatchDimensionForPlacementSignature({32, 128}), 10) ==
          BuildDensePlacementShapeSignature({128}, 10));

    // Rank-1 shapes have no batch dimension to strip.
    CHECK(StripBatchDimensionForPlacementSignature({7}) ==
          std::vector<size_t>{7});
    CHECK(StripBatchDimensionForPlacementSignature({}).empty());

    // Tensor-layer signatures are input-shape-only (tofix67 slice 7): a
    // runtime Forward failure happens before an output exists, so an
    // output-bearing key could never be formed by the writer. Runtime
    // [batch, H, W, C] normalizes to the compiler's batchless [H, W, C].
    const std::vector<size_t> runtime_conv_shape = {8, 28, 28, 3};
    const std::vector<size_t> compiler_conv_shape = {28, 28, 3};
    CHECK(BuildTensorLayerPlacementShapeSignature(
              StripBatchDimensionForPlacementSignature(runtime_conv_shape)) ==
          BuildTensorLayerPlacementShapeSignature(compiler_conv_shape));
}

TEST_CASE("Layer fallback observations land under the compiler's lookup key",
          "[gpu_execution][taxonomy][placement]") {
    using namespace cyxwiz;

    ClearBackendPlacementObservationCacheForTesting();

    // Simulate a Conv2D runtime Forward failure through the shared layer
    // funnel with a batched input tensor, then look it up exactly the way
    // BuildArrayFireTensorPlacement does at compile time.
    Tensor input(std::vector<size_t>{8, 28, 28, 3});
    float* data = input.Data<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        data[i] = 0.0f;
    }
    RecordLayerArrayFireFallbackObservation(
        "Conv2DLayer::Forward",
        "Conv2D",
        BackendFallbackReason::UnsupportedShape,
        "simulated unsupported conv shape",
        input,
        "input");

    BackendPlacementObservation observation;
    REQUIRE(TryGetBackendPlacementObservationForActiveDevice(
        "Conv2D",
        CurrentArrayFireBackendName(),
        "float32",
        BuildTensorLayerPlacementShapeSignature({28, 28, 3}),
        observation));
    CHECK(observation.reason_code ==
          BackendPlacementObservationReason::UnsupportedShape);
    CHECK(observation.source ==
          BackendPlacementObservationSource::RuntimeFallback);

    ClearBackendPlacementObservationCacheForTesting();
}
