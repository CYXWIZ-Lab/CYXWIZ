#pragma once

#include "dataset_batcher.h"
#include "graph_compiler.h"

#include <cyxwiz/sequential.h>

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
            return static_cast<int64_t>(tensor.ReadData<float>()[index]);
        case DataType::Int32:
            return static_cast<int64_t>(tensor.ReadData<int32_t>()[index]);
        case DataType::Int64:
            return tensor.ReadData<int64_t>()[index];
        default:
            throw std::runtime_error(
                "sequence model ids must be Float32, Int32, or Int64");
    }
}

inline bool ReadSequenceMaskAt(const Tensor& tensor, size_t index) {
    switch (tensor.GetDataType()) {
        case DataType::Float32:
            return tensor.ReadData<float>()[index] != 0.0f;
        case DataType::Float64:
            return tensor.ReadData<double>()[index] != 0.0;
        case DataType::Int32:
            return tensor.ReadData<int32_t>()[index] != 0;
        case DataType::Int64:
            return tensor.ReadData<int64_t>()[index] != 0;
        case DataType::UInt8:
            return tensor.ReadData<uint8_t>()[index] != 0;
        default:
            throw std::runtime_error(
                "sequence attention mask must be numeric");
    }
}

inline void WriteSequenceIdAt(Tensor& tensor, size_t index, int64_t value) {
    switch (tensor.GetDataType()) {
        case DataType::Float32:
            tensor.MutableData<float>()[index] = static_cast<float>(value);
            return;
        case DataType::Int32:
            tensor.MutableData<int32_t>()[index] = static_cast<int32_t>(value);
            return;
        case DataType::Int64:
            tensor.MutableData<int64_t>()[index] = value;
            return;
        default:
            throw std::runtime_error(
                "sequence model ids must be Float32, Int32, or Int64");
    }
}

inline bool ModelUsesSequenceFeatureFusion(const SequentialModel& model) {
    return dynamic_cast<const SequenceFeatureFusionModule*>(
               model.GetModule(0)) != nullptr;
}

inline void ValidateSequenceAttentionMaskShape(const Tensor& ids,
                                               const Tensor& mask,
                                               const std::string& name) {
    const auto& id_shape = ids.Shape();
    if (id_shape.size() != 2) {
        throw std::runtime_error(
            name + " ids must have shape [batch, seq] before masking");
    }
    if (mask.Shape() != id_shape) {
        throw std::runtime_error(
            "sequence attention mask shape must match " + name +
            " ids shape");
    }
}

inline Tensor ApplySequenceAttentionMask(const Tensor& ids,
                                         const Tensor& mask,
                                         int64_t pad_id,
                                         const std::string& name) {
    ValidateSequenceAttentionMaskShape(ids, mask, name);

    Tensor masked = ids.Clone();
    for (size_t i = 0; i < masked.NumElements(); ++i) {
        if (!ReadSequenceMaskAt(mask, i)) {
            WriteSequenceIdAt(masked, i, pad_id);
        }
    }
    return masked;
}

inline Tensor BuildPackedWordPosSequenceInput(const Tensor& word_ids,
                                              const Tensor& pos_ids) {
    const auto& word_shape = word_ids.Shape();
    const auto& pos_shape = pos_ids.Shape();
    if (word_shape.size() != 2) {
        throw std::runtime_error(
            "sequence feature fusion expects word ids with shape [batch, seq]");
    }
    if (pos_shape != word_shape) {
        throw std::runtime_error(
            "sequence feature fusion requires POS ids shape to match word ids");
    }

    const size_t token_count = word_ids.NumElements();
    std::vector<int64_t> packed(token_count * 2, 0);
    for (size_t i = 0; i < token_count; ++i) {
        packed[i * 2] = ReadSequenceIdAt(word_ids, i);
        packed[i * 2 + 1] = ReadSequenceIdAt(pos_ids, i);
    }
    return Tensor({word_shape[0], word_shape[1], 2},
                  packed.data(),
                  DataType::Int64);
}

inline Tensor BuildSequenceModelInput(const SequenceBatch& batch,
                                      const TrainingConfiguration& config) {
    const int fusion_index = FindSequenceFeatureFusionLayerIndex(config);
    if (fusion_index < 0) {
        if (batch.HasAttentionMask()) {
            return ApplySequenceAttentionMask(
                batch.word_ids,
                batch.attention_mask,
                config.sequence_batch.word_pad_id,
                "word");
        }
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

    if (batch.HasAttentionMask()) {
        Tensor masked_words = ApplySequenceAttentionMask(
            batch.word_ids,
            batch.attention_mask,
            config.sequence_batch.word_pad_id,
            "word");
        Tensor masked_pos = ApplySequenceAttentionMask(
            batch.pos_ids,
            batch.attention_mask,
            config.sequence_batch.pos_pad_id,
            "POS");
        return BuildPackedWordPosSequenceInput(masked_words, masked_pos);
    }

    return BuildPackedWordPosSequenceInput(batch.word_ids, batch.pos_ids);
}

} // namespace cyxwiz
