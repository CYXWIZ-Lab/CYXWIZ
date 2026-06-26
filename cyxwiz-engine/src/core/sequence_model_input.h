#pragma once

#include "dataset_batcher.h"
#include "graph_compiler.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

inline bool IsSequenceFeatureFusionLayer(const CompiledLayer& layer) {
    if (layer.type != gui::NodeType::Concatenate) {
        return false;
    }
    const auto it = layer.parameters.find("sequence_feature_fusion");
    return it != layer.parameters.end() &&
           (it->second == "true" || it->second == "1");
}

inline int FindSequenceFeatureFusionLayerIndex(
    const TrainingConfiguration& config) {
    for (size_t i = 0; i < config.layers.size(); ++i) {
        if (IsSequenceFeatureFusionLayer(config.layers[i])) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

inline bool UsesSequenceFeatureFusion(const TrainingConfiguration& config) {
    return FindSequenceFeatureFusionLayerIndex(config) >= 0;
}

inline int64_t ReadSequenceIdAt(const Tensor& tensor, size_t index) {
    switch (tensor.GetDataType()) {
        case DataType::Float32:
            return static_cast<int64_t>(tensor.Data<float>()[index]);
        case DataType::Int32:
            return static_cast<int64_t>(tensor.Data<int32_t>()[index]);
        case DataType::Int64:
            return tensor.Data<int64_t>()[index];
        default:
            throw std::runtime_error(
                "sequence model ids must be Float32, Int32, or Int64");
    }
}

inline Tensor BuildSequenceModelInput(const SequenceBatch& batch,
                                      const TrainingConfiguration& config) {
    const int fusion_index = FindSequenceFeatureFusionLayerIndex(config);
    if (fusion_index < 0) {
        return batch.word_ids.Clone();
    }
    if (fusion_index != 0) {
        throw std::runtime_error(
            "sequence feature fusion must be the first model layer");
    }
    if (!batch.HasWordIds()) {
        throw std::runtime_error(
            "sequence feature fusion requires word ids");
    }
    if (!batch.HasPosIds()) {
        throw std::runtime_error(
            "sequence feature fusion requires POS ids");
    }

    const auto& word_shape = batch.word_ids.Shape();
    const auto& pos_shape = batch.pos_ids.Shape();
    if (word_shape.size() != 2) {
        throw std::runtime_error(
            "sequence feature fusion expects word ids with shape [batch, seq]");
    }
    if (pos_shape != word_shape) {
        throw std::runtime_error(
            "sequence feature fusion requires POS ids shape to match word ids");
    }

    const size_t token_count = batch.word_ids.NumElements();
    std::vector<int64_t> packed(token_count * 2, 0);
    for (size_t i = 0; i < token_count; ++i) {
        packed[i * 2] = ReadSequenceIdAt(batch.word_ids, i);
        packed[i * 2 + 1] = ReadSequenceIdAt(batch.pos_ids, i);
    }
    return Tensor({word_shape[0], word_shape[1], 2},
                  packed.data(),
                  DataType::Int64);
}

} // namespace cyxwiz
