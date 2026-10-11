#pragma once

#include <cyxwiz/tensor.h>
#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <af/array.h>
#include <af/arith.h>
#include <af/algorithm.h>
#include <af/data.h>
#endif
#include "algorithms/arrayfire_backend_utils.h"
#include <cmath>
#include <limits>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

struct BioEntitySpan {
    size_t sequence = 0;
    size_t start = 0;
    size_t end = 0;  // exclusive
    std::string type;

    bool operator<(const BioEntitySpan& other) const {
        if (sequence != other.sequence) return sequence < other.sequence;
        if (start != other.start) return start < other.start;
        if (end != other.end) return end < other.end;
        return type < other.type;
    }

    bool operator==(const BioEntitySpan& other) const {
        return sequence == other.sequence &&
               start == other.start &&
               end == other.end &&
               type == other.type;
    }
};

struct SequenceTagMetrics {
    size_t correct_tokens = 0;
    size_t total_tokens = 0;
    double token_accuracy = 0.0;

    size_t predicted_entities = 0;
    size_t gold_entities = 0;
    size_t matched_entities = 0;
    double entity_precision = 0.0;
    double entity_recall = 0.0;
    double entity_f1 = 0.0;
};

inline int64_t SequenceTagIdAt(const Tensor& tensor, size_t index) {
    if (tensor.GetDataType() == DataType::Int64) {
        return tensor.Data<int64_t>()[index];
    }
    if (tensor.GetDataType() == DataType::Int32) {
        return tensor.Data<int32_t>()[index];
    }
    throw std::runtime_error("sequence tag ids must be Int32 or Int64");
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
// Argmax over the last axis of semantic [batch, seq, classes] logits. Ties go
// to the first class; a row without a maximum (all NaN) gives class 0, as a
// strict ">" scan from class 0 does.
inline af::array SequenceTagArgmaxDevice(const af::array& values, size_t classes) {
    const af::array maximum = af::tile((af::max)(values, 2), af::dim4(1, 1, static_cast<unsigned>(classes)));
    const af::array indices = af::range(values.dims(), 2, u32);
    const af::array first = (af::min)(af::select(values == maximum, indices, static_cast<double>(classes)), 2).as(s64);
    return af::select(first >= static_cast<double>(classes), 0.0, first);
}
#endif

inline Tensor ArgmaxSequenceTagLogits(const Tensor& logits) {
    if (logits.GetDataType() != DataType::Float32) {
        throw std::runtime_error("sequence tag logits must be Float32");
    }
    const auto& shape = logits.Shape();
    if (shape.size() != 3 || shape[2] == 0 || shape[2] > (std::numeric_limits<unsigned>::max)()) {
        throw std::runtime_error(
            "sequence tag logits must be shaped [batch, seq, tags]");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array ids = af::moddims(SequenceTagArgmaxDevice(logits.GetSemanticArray(), shape[2]),
                                          static_cast<dim_t>(shape[0]), static_cast<dim_t>(shape[1]));
        ids.eval();
        return Tensor::FromSemanticArray(ids, {shape[0], shape[1]});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("ArgmaxSequenceTagLogits failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("ArgmaxSequenceTagLogits runs on ArrayFire, and this build has no ArrayFire");
#endif
}

inline void ValidateSequenceTagIdShapes(const Tensor& predicted_ids,
                                        const Tensor& gold_ids) {
    const auto& shape = gold_ids.Shape();
    if (shape.size() != 2) {
        throw std::runtime_error("gold sequence tags must be shaped [batch, seq]");
    }
    if (predicted_ids.Shape() != shape) {
        throw std::runtime_error(
            "predicted sequence tags must match gold [batch, seq] shape");
    }
    if ((predicted_ids.GetDataType() != DataType::Int64 &&
         predicted_ids.GetDataType() != DataType::Int32) ||
        (gold_ids.GetDataType() != DataType::Int64 &&
         gold_ids.GetDataType() != DataType::Int32)) {
        throw std::runtime_error("sequence tag ids must be Int32 or Int64");
    }
}

inline std::string LabelType(const std::string& label) {
    const size_t dash = label.find('-');
    if (dash == std::string::npos || dash + 1 >= label.size()) {
        return {};
    }
    return label.substr(dash + 1);
}

inline bool IsBioBegin(const std::string& label) {
    return label.size() > 2 && label[0] == 'B' && label[1] == '-';
}

inline bool IsBioInside(const std::string& label) {
    return label.size() > 2 && label[0] == 'I' && label[1] == '-';
}

inline std::vector<BioEntitySpan> ExtractBioEntities(
    const Tensor& tag_ids,
    const std::vector<std::string>& id_to_label,
    int64_t ignore_index = -100) {
    const auto& shape = tag_ids.Shape();
    if (shape.size() != 2) {
        throw std::runtime_error("BIO tags must be shaped [batch, seq]");
    }
    if (tag_ids.GetDataType() != DataType::Int64 &&
        tag_ids.GetDataType() != DataType::Int32) {
        throw std::runtime_error("BIO tag ids must be Int32 or Int64");
    }

    std::vector<BioEntitySpan> entities;
    const size_t batch = shape[0];
    const size_t seq = shape[1];

    for (size_t row = 0; row < batch; ++row) {
        bool open = false;
        BioEntitySpan current;

        const auto close_current = [&](size_t end) {
            if (open) {
                current.end = end;
                entities.push_back(current);
                open = false;
            }
        };

        for (size_t col = 0; col < seq; ++col) {
            const size_t offset = row * seq + col;
            const int64_t id = SequenceTagIdAt(tag_ids, offset);
            if (id == ignore_index) {
                close_current(col);
                continue;
            }
            if (id < 0 || static_cast<size_t>(id) >= id_to_label.size()) {
                throw std::runtime_error("BIO tag id is outside label vocabulary");
            }

            const std::string& label = id_to_label[static_cast<size_t>(id)];
            if (label == "O" || label.empty()) {
                close_current(col);
                continue;
            }

            const std::string type = LabelType(label);
            if (type.empty() || IsBioBegin(label) ||
                (IsBioInside(label) && (!open || current.type != type))) {
                close_current(col);
                if (IsBioBegin(label) || IsBioInside(label)) {
                    current = {row, col, col + 1, type};
                    open = true;
                }
                continue;
            }

            if (IsBioInside(label) && open && current.type == type) {
                continue;
            }

            close_current(col);
        }
        close_current(seq);
    }

    std::sort(entities.begin(), entities.end());
    return entities;
}

inline SequenceTagMetrics ComputeSequenceTagMetrics(
    const Tensor& predicted_ids,
    const Tensor& gold_ids,
    const std::vector<std::string>& id_to_label,
    int64_t ignore_index = -100) {
    ValidateSequenceTagIdShapes(predicted_ids, gold_ids);

    SequenceTagMetrics metrics;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    Tensor masked_predicted_ids;
    try {
        const af::array gold = gold_ids.GetSemanticArray().as(s64);
        const af::array predicted = predicted_ids.GetSemanticArray().as(s64);
        const af::array valid = gold != af::constant(static_cast<long long>(ignore_index), gold.dims(), s64);
        // Ignored positions break predicted spans exactly where they break gold ones.
        const af::array masked = af::select(valid, predicted, static_cast<double>(ignore_index)).as(s64);
        const auto count = [](const af::array& mask) { return af::sum(af::flat(mask).as(s64), 0).as(s64); };
        const af::array counts = af::join(0, count(valid), count(valid && (predicted == gold)));
        masked.eval();
        masked_predicted_ids = Tensor::FromSemanticArray(masked, predicted_ids.Shape());
        const Tensor counts_tensor = Tensor::FromSemanticArray(counts, {2});
        const ScopedArrayFireHostSyncAttribution attribution(
            ArrayFireHostSyncCategory::MetricScalarReadback, "ComputeSequenceTagMetrics");
        const auto* host = counts_tensor.ReadData<int64_t>();
        metrics.total_tokens = static_cast<size_t>(host[0]);
        metrics.correct_tokens = static_cast<size_t>(host[1]);
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("ComputeSequenceTagMetrics failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("ComputeSequenceTagMetrics runs on ArrayFire, and this build has no ArrayFire");
#endif
    if (metrics.total_tokens > 0) {
        metrics.token_accuracy =
            static_cast<double>(metrics.correct_tokens) /
            static_cast<double>(metrics.total_tokens);
    }

    // BIO span matching is the declared host boundary: it reads the [batch, seq]
    // id matrices, never the logits.
    auto predicted_entities =
        ExtractBioEntities(masked_predicted_ids, id_to_label, ignore_index);
    auto gold_entities = ExtractBioEntities(gold_ids, id_to_label, ignore_index);

    metrics.predicted_entities = predicted_entities.size();
    metrics.gold_entities = gold_entities.size();

    size_t pred_index = 0;
    size_t gold_index = 0;
    while (pred_index < predicted_entities.size() &&
           gold_index < gold_entities.size()) {
        if (predicted_entities[pred_index] == gold_entities[gold_index]) {
            ++metrics.matched_entities;
            ++pred_index;
            ++gold_index;
        } else if (predicted_entities[pred_index] < gold_entities[gold_index]) {
            ++pred_index;
        } else {
            ++gold_index;
        }
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
    const double f1_denom = metrics.entity_precision + metrics.entity_recall;
    if (f1_denom > 0.0) {
        metrics.entity_f1 =
            2.0 * metrics.entity_precision * metrics.entity_recall / f1_denom;
    }

    return metrics;
}

inline SequenceTagMetrics ComputeSequenceTagMetricsFromLogits(
    const Tensor& logits,
    const Tensor& gold_ids,
    const std::vector<std::string>& id_to_label,
    int64_t ignore_index = -100) {
    return ComputeSequenceTagMetrics(
        ArgmaxSequenceTagLogits(logits),
        gold_ids,
        id_to_label,
        ignore_index);
}

struct NextTokenAccuracyCount {
    size_t correct = 0;
    size_t valid = 0;
};

inline NextTokenAccuracyCount CountNextTokenAccuracyFromLogits(
    const Tensor& logits,
    const Tensor& target_ids,
    int64_t ignore_index) {

    const auto& logit_shape = logits.Shape();
    const auto& target_shape = target_ids.Shape();
    if (logits.GetDataType() != DataType::Float32 ||
        logit_shape.size() != 3 || logit_shape[0] == 0 || logit_shape[1] == 0 ||
        logit_shape[2] == 0 || logit_shape[2] > (std::numeric_limits<unsigned>::max)() ||
        (target_ids.GetDataType() != DataType::Int64 && target_ids.GetDataType() != DataType::Int32) ||
        target_shape.size() != 2 ||
        target_shape[0] != logit_shape[0] ||
        target_shape[1] != logit_shape[1]) {
        throw std::runtime_error(
            "language-model accuracy expects logits [batch, seq, vocab] "
            "and target_ids [batch, seq]");
    }

    const size_t vocab_size = logit_shape[2];
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array values = logits.GetSemanticArray();
        const af::array targets = target_ids.GetSemanticArray().as(s64);
        const af::array ignored = af::constant(static_cast<long long>(ignore_index), targets.dims(), s64);
        const af::array valid = targets != ignored;
        const af::array invalid = valid && ((targets < 0) || (targets >= static_cast<double>(vocab_size)));
        const af::array predicted = SequenceTagArgmaxDevice(values, vocab_size);
        const auto count = [](const af::array& mask) {
            return af::sum(af::flat(mask).as(s64), 0).as(s64);
        };
        const af::array counts = af::join(0, count(valid), count(valid && (predicted == targets)),
            count(invalid), count(af::isNaN(values) || af::isInf(values)));
        const Tensor output = Tensor::FromSemanticArray(counts, {4});
        const ScopedArrayFireHostSyncAttribution attribution(
            ArrayFireHostSyncCategory::MetricScalarReadback, "CountNextTokenAccuracyFromLogits");
        const auto* host = output.ReadData<int64_t>();
        if (host[2] != 0) throw std::runtime_error("language-model target id is outside the vocabulary range");
        if (host[3] != 0) throw std::runtime_error("language-model accuracy requires finite logits");
        return {static_cast<size_t>(host[1]), static_cast<size_t>(host[0])};
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("CountNextTokenAccuracyFromLogits failed on the ArrayFire device: ") + e.what());
    }
#else
    (void)vocab_size;
    (void)ignore_index;
    throw std::runtime_error("CountNextTokenAccuracyFromLogits runs on ArrayFire, and this build has no ArrayFire");
#endif
}

} // namespace cyxwiz
