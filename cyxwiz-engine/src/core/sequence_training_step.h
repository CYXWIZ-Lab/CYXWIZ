#pragma once

#include "dataset_batcher.h"
#include "graph_compiler.h"
#include "model_builder.h"
#include "sequence_tag_metrics.h"

#include <cmath>
#include <exception>
#include <string>
#include <vector>

namespace cyxwiz {

struct SequenceTrainingEpochResult {
    bool success = false;
    std::string error;
    float mean_loss = 0.0f;
    size_t batches = 0;
    size_t samples = 0;
    SequenceTagMetrics metrics;
};

inline void FinalizeSequenceTagMetricRates(SequenceTagMetrics& metrics) {
    if (metrics.total_tokens > 0) {
        metrics.token_accuracy =
            static_cast<double>(metrics.correct_tokens) /
            static_cast<double>(metrics.total_tokens);
    }
    if (metrics.predicted_entities > 0) {
        metrics.entity_precision =
            static_cast<double>(metrics.matched_entities) /
            static_cast<double>(metrics.predicted_entities);
    }
    if (metrics.gold_entities > 0) {
        metrics.entity_recall =
            static_cast<double>(metrics.matched_entities) /
            static_cast<double>(metrics.gold_entities);
    }
    const double denom = metrics.entity_precision + metrics.entity_recall;
    if (denom > 0.0) {
        metrics.entity_f1 =
            2.0 * metrics.entity_precision * metrics.entity_recall / denom;
    }
}

inline void AccumulateSequenceTagMetrics(SequenceTagMetrics& total,
                                         const SequenceTagMetrics& batch) {
    total.correct_tokens += batch.correct_tokens;
    total.total_tokens += batch.total_tokens;
    total.predicted_entities += batch.predicted_entities;
    total.gold_entities += batch.gold_entities;
    total.matched_entities += batch.matched_entities;
}

inline SequenceTrainingEpochResult TrainSequenceTaggerEpoch(
    const TrainingConfiguration& config,
    ISequenceBatcher& batcher,
    const std::vector<std::string>& id_to_label) {
    SequenceTrainingEpochResult result;

    try {
        if (!config.sequence_batch.enabled) {
            result.error = "sequence batch config is not enabled";
            return result;
        }
        if (!config.sequence_batch.create_causal_lm_targets &&
            id_to_label.empty()) {
            result.error = "sequence tag label vocabulary is empty";
            return result;
        }

        auto built = BuildExecutableFromConfig(config);
        if (!built.ok() || !built.model || !built.loss || !built.optimizer) {
            result.error = "failed to build sequence tagger model";
            return result;
        }

        built.model->SetTraining(true);
        batcher.Reset();

        double loss_sum = 0.0;
        while (!batcher.IsEpochComplete()) {
            SequenceBatch batch = batcher.GetNextSequenceBatch();
            if (!batch.IsValid()) {
                break;
            }
            if (!batch.IsSupervised()) {
                result.error = "sequence batch is missing tag_ids or target_ids";
                return result;
            }
            const bool is_language_modeling = batch.HasTargetIds();
            const Tensor& targets =
                is_language_modeling ? batch.target_ids : batch.tag_ids;

            Tensor predictions = built.model->Forward(batch.word_ids);
            Tensor loss_tensor = built.loss->Forward(predictions, targets);
            const float batch_loss = loss_tensor.Data<float>()[0];
            if (!std::isfinite(batch_loss)) {
                result.error = "sequence tagger loss is not finite";
                return result;
            }
            loss_sum += static_cast<double>(batch_loss);

            if (!is_language_modeling) {
                const auto batch_metrics = ComputeSequenceTagMetricsFromLogits(
                    predictions,
                    targets,
                    id_to_label,
                    config.sequence_batch.ignore_index);
                AccumulateSequenceTagMetrics(result.metrics, batch_metrics);
            }

            Tensor grad = built.loss->Backward(predictions, targets);
            built.model->Backward(grad);
            built.model->UpdateParameters(built.optimizer.get());

            ++result.batches;
            result.samples += batch.size;
        }

        if (result.batches == 0) {
            result.error = "sequence batcher produced no batches";
            return result;
        }

        result.mean_loss = static_cast<float>(
            loss_sum / static_cast<double>(result.batches));
        FinalizeSequenceTagMetricRates(result.metrics);
        result.success = true;
        return result;
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        return result;
    }
}

} // namespace cyxwiz
