#pragma once

#include "sequence_tag_metrics.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

struct SequenceTagDecodeResponse {
    std::vector<std::vector<int64_t>> tag_ids;
    std::vector<std::vector<std::string>> tag_labels;
    std::vector<size_t> effective_lengths;
};

inline SequenceTagDecodeResponse DecodeSequenceTagIdsForInference(
    const Tensor& predicted_ids,
    const std::vector<std::string>& id_to_label,
    const std::vector<int64_t>& sequence_lengths = {}) {
    const auto& shape = predicted_ids.Shape();
    if (shape.size() != 2) {
        throw std::runtime_error(
            "predicted sequence tag ids must be shaped [batch, seq]");
    }
    if (predicted_ids.GetDataType() != DataType::Int64 &&
        predicted_ids.GetDataType() != DataType::Int32) {
        throw std::runtime_error("predicted sequence tag ids must be Int64 or Int32");
    }

    const size_t batch = shape[0];
    const size_t seq = shape[1];
    if (!sequence_lengths.empty() && sequence_lengths.size() != batch) {
        throw std::runtime_error(
            "sequence_lengths length must match decoded tag batch size");
    }

    SequenceTagDecodeResponse response;
    response.tag_ids.reserve(batch);
    response.tag_labels.reserve(batch);
    response.effective_lengths.reserve(batch);

    for (size_t row = 0; row < batch; ++row) {
        size_t effective_length = seq;
        if (!sequence_lengths.empty()) {
            effective_length = static_cast<size_t>(
                std::max<int64_t>(0, sequence_lengths[row]));
            effective_length = std::min(effective_length, seq);
        }
        response.effective_lengths.push_back(effective_length);

        std::vector<int64_t> row_ids;
        std::vector<std::string> row_labels;
        row_ids.reserve(effective_length);
        row_labels.reserve(effective_length);

        for (size_t col = 0; col < effective_length; ++col) {
            const size_t offset = row * seq + col;
            const int64_t id = SequenceTagIdAt(predicted_ids, offset);
            row_ids.push_back(id);
            if (id >= 0 && static_cast<size_t>(id) < id_to_label.size()) {
                row_labels.push_back(id_to_label[static_cast<size_t>(id)]);
            } else {
                row_labels.emplace_back();
            }
        }

        response.tag_ids.push_back(std::move(row_ids));
        response.tag_labels.push_back(std::move(row_labels));
    }

    return response;
}

} // namespace cyxwiz
