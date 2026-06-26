#include "metric_learning_inference_input.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {
namespace {

using json = nlohmann::json;

Tensor ParseFloatInputTensor(const json& value, const std::string& path) {
    if (!value.is_array()) {
        throw std::invalid_argument(path + " must be a numeric array");
    }
    if (value.empty()) {
        throw std::invalid_argument(path + " array cannot be empty");
    }

    std::vector<float> data;
    std::vector<size_t> shape;

    if (!value[0].is_array()) {
        data.reserve(value.size());
        for (const auto& item : value) {
            if (!item.is_number()) {
                throw std::invalid_argument(path + " values must be numeric");
            }
            data.push_back(item.get<float>());
        }
        shape = {1, data.size()};
    } else {
        const size_t rows = value.size();
        size_t cols = 0;
        for (size_t row_index = 0; row_index < rows; ++row_index) {
            const auto& row = value[row_index];
            if (!row.is_array()) {
                throw std::invalid_argument(
                    path + " must be consistently 1D or 2D");
            }
            if (row.empty()) {
                throw std::invalid_argument(path + " rows cannot be empty");
            }
            if (cols == 0) {
                cols = row.size();
            } else if (row.size() != cols) {
                throw std::invalid_argument(
                    path + " rows must have consistent feature dimensions");
            }
            for (const auto& item : row) {
                if (!item.is_number()) {
                    throw std::invalid_argument(
                        path + " values must be numeric");
                }
                data.push_back(item.get<float>());
            }
        }
        shape = {rows, cols};
    }

    return Tensor(shape, data.data(), DataType::Float32);
}

Tensor ParseIdVector(const json& value,
                     const std::string& path,
                     size_t expected_batch) {
    if (!value.is_array()) {
        throw std::invalid_argument(path + " must be an integer array");
    }
    if (value.size() != expected_batch) {
        throw std::invalid_argument(path + " length must match batch size");
    }

    std::vector<int64_t> ids;
    ids.reserve(value.size());
    for (const auto& item : value) {
        if (!item.is_number_integer()) {
            throw std::invalid_argument(path + " values must be integers");
        }
        ids.push_back(item.get<int64_t>());
    }
    return Tensor({expected_batch}, ids.data(), DataType::Int64);
}

Tensor ParseOptionalIdVector(const json& request_body,
                             const char* key,
                             size_t expected_batch,
                             bool& present) {
    present = request_body.contains(key);
    if (!present) {
        return Tensor();
    }
    return ParseIdVector(request_body.at(key), key, expected_batch);
}

void ParsePairedOptionalIds(const json& request_body,
                            const char* left_key,
                            const char* right_key,
                            size_t expected_batch,
                            Tensor& left_out,
                            Tensor& right_out,
                            bool& present) {
    const bool has_left = request_body.contains(left_key);
    const bool has_right = request_body.contains(right_key);
    if (has_left != has_right) {
        throw std::invalid_argument(
            std::string(left_key) + " and " + right_key +
            " must be provided together");
    }
    present = has_left;
    if (!present) {
        left_out = Tensor();
        right_out = Tensor();
        return;
    }

    left_out = ParseIdVector(request_body.at(left_key),
                             left_key,
                             expected_batch);
    right_out = ParseIdVector(request_body.at(right_key),
                              right_key,
                              expected_batch);
}

const json& ResolvePairBranchInput(const json& request_body,
                                   const char* root_key,
                                   const char* nested_key,
                                   const char* alias_key) {
    if (request_body.contains(root_key)) {
        return request_body.at(root_key);
    }
    if (request_body.contains("input") && request_body.at("input").is_object()) {
        const auto& input = request_body.at("input");
        if (input.contains(nested_key)) {
            return input.at(nested_key);
        }
        if (input.contains(alias_key)) {
            return input.at(alias_key);
        }
    }
    throw std::invalid_argument(std::string("Missing required field: ") +
                                root_key);
}

}  // namespace

MetricEmbeddingInferenceInput ParseMetricEmbeddingInferenceInput(
    const json& request_body) {
    if (!request_body.contains("input")) {
        throw std::invalid_argument("Missing required field: input");
    }

    MetricEmbeddingInferenceInput parsed;
    parsed.input = ParseFloatInputTensor(request_body.at("input"), "input");
    const size_t batch_size = parsed.input.Shape().at(0);

    parsed.sample_ids = ParseOptionalIdVector(
        request_body, "sample_ids", batch_size, parsed.has_sample_ids);
    parsed.class_ids = ParseOptionalIdVector(
        request_body, "class_ids", batch_size, parsed.has_class_ids);
    return parsed;
}

MetricPairScoreInferenceInput ParseMetricPairScoreInferenceInput(
    const json& request_body) {
    MetricPairScoreInferenceInput parsed;
    parsed.input_a = ParseFloatInputTensor(
        ResolvePairBranchInput(request_body, "input_a", "a", "input_a"),
        "input_a");
    parsed.input_b = ParseFloatInputTensor(
        ResolvePairBranchInput(request_body, "input_b", "b", "input_b"),
        "input_b");

    if (parsed.input_a.Shape() != parsed.input_b.Shape()) {
        throw std::invalid_argument(
            "input_a and input_b shapes must match for pair scoring");
    }
    const size_t batch_size = parsed.input_a.Shape().at(0);

    if (request_body.contains("score_mode")) {
        if (!request_body.at("score_mode").is_string()) {
            throw std::invalid_argument("score_mode must be a string");
        }
        parsed.score_mode =
            ParsePairScoreMode(request_body.at("score_mode").get<std::string>());
    }

    ParsePairedOptionalIds(request_body,
                           "sample_id_a",
                           "sample_id_b",
                           batch_size,
                           parsed.sample_id_a,
                           parsed.sample_id_b,
                           parsed.has_sample_ids);
    ParsePairedOptionalIds(request_body,
                           "class_id_a",
                           "class_id_b",
                           batch_size,
                           parsed.class_id_a,
                           parsed.class_id_b,
                           parsed.has_class_ids);
    return parsed;
}

}  // namespace cyxwiz
