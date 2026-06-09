#pragma once

#include <cyxwiz/tensor.h>

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

inline Tensor ArgmaxSequenceTagLogits(const Tensor& logits) {
    if (logits.GetDataType() != DataType::Float32) {
        throw std::runtime_error("sequence tag logits must be Float32");
    }
    const auto& shape = logits.Shape();
    if (shape.size() != 3 || shape[2] == 0) {
        throw std::runtime_error(
            "sequence tag logits must be shaped [batch, seq, tags]");
    }

    const size_t batch = shape[0];
    const size_t seq = shape[1];
    const size_t classes = shape[2];
    const float* data = logits.Data<float>();
    std::vector<int64_t> ids(batch * seq, 0);

    for (size_t i = 0; i < batch * seq; ++i) {
        const size_t base = i * classes;
        size_t best = 0;
        float best_value = data[base];
        for (size_t c = 1; c < classes; ++c) {
            if (data[base + c] > best_value) {
                best = c;
                best_value = data[base + c];
            }
        }
        ids[i] = static_cast<int64_t>(best);
    }

    return Tensor({batch, seq}, ids.data(), DataType::Int64);
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
    for (size_t i = 0; i < gold_ids.NumElements(); ++i) {
        const int64_t gold = SequenceTagIdAt(gold_ids, i);
        if (gold == ignore_index) {
            continue;
        }
        const int64_t pred = SequenceTagIdAt(predicted_ids, i);
        ++metrics.total_tokens;
        if (pred == gold) {
            ++metrics.correct_tokens;
        }
    }
    if (metrics.total_tokens > 0) {
        metrics.token_accuracy =
            static_cast<double>(metrics.correct_tokens) /
            static_cast<double>(metrics.total_tokens);
    }

    std::vector<int64_t> masked_predicted_data(predicted_ids.NumElements(), 0);
    for (size_t i = 0; i < gold_ids.NumElements(); ++i) {
        const int64_t gold = SequenceTagIdAt(gold_ids, i);
        masked_predicted_data[i] =
            gold == ignore_index ? ignore_index : SequenceTagIdAt(predicted_ids, i);
    }
    Tensor masked_predicted_ids(predicted_ids.Shape(),
                                masked_predicted_data.data(),
                                DataType::Int64);

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

} // namespace cyxwiz
