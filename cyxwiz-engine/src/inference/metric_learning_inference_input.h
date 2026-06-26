#pragma once

#include "core/metric_learning_inference_outputs.h"

#include <cyxwiz/tensor.h>
#include <nlohmann/json_fwd.hpp>

namespace cyxwiz {

struct MetricEmbeddingInferenceInput {
    Tensor input;
    Tensor sample_ids;
    Tensor class_ids;
    bool has_sample_ids = false;
    bool has_class_ids = false;
};

struct MetricPairScoreInferenceInput {
    Tensor input_a;
    Tensor input_b;
    Tensor sample_id_a;
    Tensor sample_id_b;
    Tensor class_id_a;
    Tensor class_id_b;
    bool has_sample_ids = false;
    bool has_class_ids = false;
    PairScoreMode score_mode = PairScoreMode::EuclideanDistance;
};

MetricEmbeddingInferenceInput ParseMetricEmbeddingInferenceInput(
    const nlohmann::json& request_body);

MetricPairScoreInferenceInput ParseMetricPairScoreInferenceInput(
    const nlohmann::json& request_body);

}  // namespace cyxwiz
