#pragma once

#include "metric_learning_batch.h"
#include "metric_learning_metrics.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

enum class PairScoreMode {
    EuclideanDistance,
    NegativeEuclideanDistance,
    CosineSimilarity,
};

inline const char* PairScoreModeName(PairScoreMode mode) {
    switch (mode) {
        case PairScoreMode::EuclideanDistance:
            return "euclidean_distance";
        case PairScoreMode::NegativeEuclideanDistance:
            return "negative_euclidean_distance";
        case PairScoreMode::CosineSimilarity:
            return "cosine_similarity";
    }
    return "euclidean_distance";
}

inline PairScoreMode ParsePairScoreMode(const std::string& mode) {
    if (mode.empty() ||
        mode == "distance" ||
        mode == "euclidean" ||
        mode == "euclidean_distance") {
        return PairScoreMode::EuclideanDistance;
    }
    if (mode == "negative_distance" ||
        mode == "negative_euclidean" ||
        mode == "negative_euclidean_distance" ||
        mode == "similarity_from_distance") {
        return PairScoreMode::NegativeEuclideanDistance;
    }
    if (mode == "cosine" ||
        mode == "cosine_similarity") {
        return PairScoreMode::CosineSimilarity;
    }
    throw std::invalid_argument("unknown PairScoreOutput score_mode: " + mode);
}

struct EmbeddingOutputRecord {
    std::vector<float> embedding;
    bool has_sample_id = false;
    int64_t sample_id = 0;
    bool has_class_id = false;
    int64_t class_id = 0;
};

struct EmbeddingOutputResponse {
    std::vector<size_t> embedding_shape;
    std::vector<EmbeddingOutputRecord> records;
};

struct PairScoreRecord {
    double score = 0.0;
    double distance = 0.0;
    bool has_sample_ids = false;
    int64_t sample_id_a = 0;
    int64_t sample_id_b = 0;
    bool has_class_ids = false;
    int64_t class_id_a = 0;
    int64_t class_id_b = 0;
};

struct PairScoreOutputResponse {
    PairScoreMode mode = PairScoreMode::EuclideanDistance;
    std::vector<PairScoreRecord> records;
};

inline bool HasBatchVector(const Tensor& tensor, size_t batch_size) {
    return !TensorIsEmpty(tensor) && TensorIsBatchVector(tensor, batch_size);
}

inline std::vector<size_t> PerSampleEmbeddingShape(const Tensor& embeddings) {
    const auto& shape = embeddings.Shape();
    if (shape.size() < 2 || shape[0] == 0) {
        throw std::invalid_argument(
            "EmbeddingOutput requires embeddings shaped [batch, ...features]");
    }
    return std::vector<size_t>(shape.begin() + 1, shape.end());
}

inline int64_t MetadataIdAt(const Tensor& ids, size_t row) {
    return ClassIdAt(ids, row);
}

inline EmbeddingOutputResponse BuildEmbeddingOutputResponse(
    const Tensor& embeddings,
    const Tensor& sample_ids = Tensor(),
    const Tensor& class_ids = Tensor()) {
    const auto& shape = embeddings.Shape();
    const auto sample_shape = PerSampleEmbeddingShape(embeddings);
    const size_t batch_size = shape[0];
    const size_t dim = FlattenedEmbeddingDim(embeddings);

    const bool has_sample_ids = HasBatchVector(sample_ids, batch_size);
    const bool has_class_ids = HasBatchVector(class_ids, batch_size);
    if (!TensorIsEmpty(sample_ids) && !has_sample_ids) {
        throw std::invalid_argument(
            "EmbeddingOutput sample IDs must be [batch] or [batch, 1]");
    }
    if (!TensorIsEmpty(class_ids) && !has_class_ids) {
        throw std::invalid_argument(
            "EmbeddingOutput class IDs must be [batch] or [batch, 1]");
    }

    EmbeddingOutputResponse response;
    response.embedding_shape = sample_shape;
    response.records.reserve(batch_size);

    const float* data = embeddings.ReadData<float>();
    for (size_t row = 0; row < batch_size; ++row) {
        EmbeddingOutputRecord record;
        record.embedding.assign(data + row * dim, data + (row + 1) * dim);
        if (has_sample_ids) {
            record.has_sample_id = true;
            record.sample_id = MetadataIdAt(sample_ids, row);
        }
        if (has_class_ids) {
            record.has_class_id = true;
            record.class_id = MetadataIdAt(class_ids, row);
        }
        response.records.push_back(std::move(record));
    }

    return response;
}

inline double CosineSimilarityRow(const Tensor& left,
                                  const Tensor& right,
                                  size_t row,
                                  size_t dim) {
    const float* left_data = left.ReadData<float>();
    const float* right_data = right.ReadData<float>();
    const size_t offset = row * dim;
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double left_value = static_cast<double>(left_data[offset + i]);
        const double right_value = static_cast<double>(right_data[offset + i]);
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    if (left_norm == 0.0 || right_norm == 0.0) {
        throw std::invalid_argument(
            "PairScoreOutput cosine similarity requires non-zero embeddings");
    }
    return dot / (std::sqrt(left_norm) * std::sqrt(right_norm));
}

inline double ScoreFromDistance(PairScoreMode mode,
                                double distance,
                                const Tensor& embedding_a,
                                const Tensor& embedding_b,
                                size_t row,
                                size_t dim) {
    switch (mode) {
        case PairScoreMode::EuclideanDistance:
            return distance;
        case PairScoreMode::NegativeEuclideanDistance:
            return -distance;
        case PairScoreMode::CosineSimilarity:
            return CosineSimilarityRow(embedding_a, embedding_b, row, dim);
    }
    return distance;
}

inline PairScoreOutputResponse BuildPairScoreOutputResponse(
    const Tensor& embedding_a,
    const Tensor& embedding_b,
    PairScoreMode mode = PairScoreMode::EuclideanDistance,
    const Tensor& sample_id_a = Tensor(),
    const Tensor& sample_id_b = Tensor(),
    const Tensor& class_id_a = Tensor(),
    const Tensor& class_id_b = Tensor()) {
    if (embedding_a.Shape() != embedding_b.Shape()) {
        throw std::invalid_argument(
            "PairScoreOutput embeddings must have identical shapes");
    }
    const auto& shape = embedding_a.Shape();
    const size_t pair_count = shape.empty() ? 0 : shape[0];
    if (pair_count == 0) {
        throw std::invalid_argument("PairScoreOutput embeddings cannot be empty");
    }

    const bool has_sample_ids =
        HasBatchVector(sample_id_a, pair_count) &&
        HasBatchVector(sample_id_b, pair_count);
    const bool has_class_ids =
        HasBatchVector(class_id_a, pair_count) &&
        HasBatchVector(class_id_b, pair_count);
    if ((!TensorIsEmpty(sample_id_a) || !TensorIsEmpty(sample_id_b)) &&
        !has_sample_ids) {
        throw std::invalid_argument(
            "PairScoreOutput sample IDs must both be [batch] or [batch, 1]");
    }
    if ((!TensorIsEmpty(class_id_a) || !TensorIsEmpty(class_id_b)) &&
        !has_class_ids) {
        throw std::invalid_argument(
            "PairScoreOutput class IDs must both be [batch] or [batch, 1]");
    }

    const size_t dim = FlattenedEmbeddingDim(embedding_a);
    PairScoreOutputResponse response;
    response.mode = mode;
    response.records.reserve(pair_count);

    for (size_t row = 0; row < pair_count; ++row) {
        PairScoreRecord record;
        record.distance =
            EuclideanDistanceRow(embedding_a, embedding_b, row, dim);
        record.score = ScoreFromDistance(
            mode, record.distance, embedding_a, embedding_b, row, dim);
        if (has_sample_ids) {
            record.has_sample_ids = true;
            record.sample_id_a = MetadataIdAt(sample_id_a, row);
            record.sample_id_b = MetadataIdAt(sample_id_b, row);
        }
        if (has_class_ids) {
            record.has_class_ids = true;
            record.class_id_a = MetadataIdAt(class_id_a, row);
            record.class_id_b = MetadataIdAt(class_id_b, row);
        }
        response.records.push_back(record);
    }

    return response;
}

inline nlohmann::json EmbeddingOutputResponseToJson(
    const EmbeddingOutputResponse& response) {
    nlohmann::json records = nlohmann::json::array();
    for (const auto& record : response.records) {
        nlohmann::json item = {
            {"embedding", record.embedding},
        };
        if (record.has_sample_id) {
            item["sample_id"] = record.sample_id;
        }
        if (record.has_class_id) {
            item["class_id"] = record.class_id;
        }
        records.push_back(std::move(item));
    }

    return {
        {"output_type", "embedding"},
        {"embedding_shape", response.embedding_shape},
        {"records", std::move(records)},
    };
}

inline nlohmann::json PairScoreOutputResponseToJson(
    const PairScoreOutputResponse& response) {
    nlohmann::json records = nlohmann::json::array();
    for (const auto& record : response.records) {
        nlohmann::json item = {
            {"score", record.score},
            {"distance", record.distance},
        };
        if (record.has_sample_ids) {
            item["sample_id_a"] = record.sample_id_a;
            item["sample_id_b"] = record.sample_id_b;
        }
        if (record.has_class_ids) {
            item["class_id_a"] = record.class_id_a;
            item["class_id_b"] = record.class_id_b;
        }
        records.push_back(std::move(item));
    }

    return {
        {"output_type", "pair_score"},
        {"score_mode", PairScoreModeName(response.mode)},
        {"records", std::move(records)},
    };
}

}  // namespace cyxwiz
