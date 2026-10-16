#include "metric_learning_evaluation.h"

#include "arrow_dataset.h"
#include "graph_compiler.h"

#include <arrow/api.h>

#include "metric_learning_mining.h"
#include "metric_learning_sampling.h"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cyxwiz {

namespace {

void NormalizeRows(std::vector<float>& embeddings, size_t d) {
    for (size_t start = 0; start + d <= embeddings.size(); start += d) {
        double norm = 0.0;
        for (size_t k = 0; k < d; ++k) norm += static_cast<double>(embeddings[start + k]) * embeddings[start + k];
        norm = std::sqrt(norm);
        if (norm == 0.0) continue;
        for (size_t k = 0; k < d; ++k) embeddings[start + k] = static_cast<float>(embeddings[start + k] / norm);
    }
}

// The class of each row's nearest other row (squared Euclidean; ties to the
// lowest index).
std::vector<int64_t> NearestClasses(const std::vector<float>& embeddings, size_t d,
                                    const std::vector<int64_t>& class_ids) {
    const size_t n = class_ids.size();
    std::vector<int64_t> nearest(n, -1);
    for (size_t i = 0; i < n; ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            double distance = 0.0;
            for (size_t k = 0; k < d; ++k) {
                const double diff = static_cast<double>(embeddings[i * d + k]) - embeddings[j * d + k];
                distance += diff * diff;
            }
            if (distance < best) {
                best = distance;
                nearest[i] = class_ids[j];
            }
        }
    }
    return nearest;
}

}  // namespace

MetricEvaluationSpec MetricEvaluationSpecFor(const TrainingConfiguration& config) {
    MetricEvaluationSpec spec;
    spec.pair_metrics = config.pair_metrics;
    spec.pair_threshold = config.pair_metric_threshold;
    spec.retrieval_metrics = config.retrieval_metrics;
    spec.retrieval_k = config.retrieval_k;
    spec.normalize = config.loss_type == gui::NodeType::CosineEmbeddingLoss;
    spec.pair_seed = static_cast<uint64_t>(std::max(config.dataloader_seed, 0));
    return spec;
}

std::string DescribeMetricEvaluation(const MetricEvaluation& evaluation) {
    std::string text;
    if (evaluation.pair) {
        text += fmt::format("pair accuracy {:.4f} (threshold {:g}, {} pairs; mean distance same {:.4f} / other {:.4f})",
                            evaluation.pair->accuracy, evaluation.pair->threshold, evaluation.pair->pair_count,
                            evaluation.pair->positive_distance_mean, evaluation.pair->negative_distance_mean);
    }
    if (evaluation.retrieval) {
        if (!text.empty()) text += ", ";
        text += fmt::format("Recall@{} {:.4f}, MRR {:.4f}, 1-NN {:.4f}", evaluation.retrieval->k,
                            evaluation.retrieval->recall_at_k, evaluation.retrieval->mean_reciprocal_rank,
                            evaluation.retrieval->nearest_neighbor_class_agreement);
    }
    text += fmt::format(" over {} rows{}", evaluation.rows, evaluation.truncated ? " (capped)" : "");
    return text;
}

MetricEvaluation EvaluateMetricEmbeddings(std::vector<float> embeddings, size_t d,
                                          std::vector<int64_t> class_ids,
                                          const std::vector<std::array<int, 3>>& pairs,
                                          const MetricEvaluationSpec& spec) {
    MetricEvaluation result;
    result.rows = class_ids.size();
    if (spec.normalize) NormalizeRows(embeddings, d);
    const size_t n = result.rows;

    if (spec.pair_metrics && !pairs.empty()) {
        std::vector<float> first(pairs.size() * d), second(pairs.size() * d), labels(pairs.size());
        for (size_t i = 0; i < pairs.size(); ++i) {
            std::copy_n(embeddings.begin() + static_cast<std::ptrdiff_t>(pairs[i][0] * d), d,
                        first.begin() + static_cast<std::ptrdiff_t>(i * d));
            std::copy_n(embeddings.begin() + static_cast<std::ptrdiff_t>(pairs[i][1] * d), d,
                        second.begin() + static_cast<std::ptrdiff_t>(i * d));
            // Contrastive convention: 0 = similar.
            labels[i] = pairs[i][2] == 1 ? 0.0f : 1.0f;
        }
        result.pair = ComputePairDistanceMetrics(
            Tensor({pairs.size(), d}, first.data(), DataType::Float32),
            Tensor({pairs.size(), d}, second.data(), DataType::Float32),
            Tensor({pairs.size()}, labels.data(), DataType::Float32),
            MetricLearningLabelConvention::ContrastiveZeroSimilarOneDissimilar, spec.pair_threshold);
    }

    if (spec.retrieval_metrics && n >= 2) {
        std::vector<float> ids(n);
        for (size_t i = 0; i < n; ++i) ids[i] = static_cast<float>(class_ids[i]);
        result.retrieval = ComputeRetrievalMetrics(Tensor({n, d}, embeddings.data(), DataType::Float32),
                                                   Tensor({n}, ids.data(), DataType::Float32), spec.retrieval_k);
    }

    if (spec.nearest_neighbours && n >= 2) result.nearest_classes = NearestClasses(embeddings, d, class_ids);
    result.class_ids = std::move(class_ids);
    return result;
}

size_t WriteEmbeddingsParquet(const std::vector<EmbeddingExportPart>& parts,
                              const std::function<Tensor(const Batch&)>& forward,
                              bool include_metadata,
                              const std::string& path) {
    std::vector<float> embeddings;
    std::vector<int64_t> class_ids, positions;
    std::vector<std::string> partitions;
    size_t d = 0;
    for (const auto& part : parts) {
        if (!part.rows) continue;
        int64_t position = 0;
        part.rows->Reset();
        while (!part.rows->IsEpochComplete()) {
            Batch batch = part.rows->GetNextBatch();
            if (!batch.IsValid()) break;
            const auto ids = ReadBatchClassIds(batch.labels, batch.size);
            const Tensor output = forward(batch);
            const auto& shape = output.Shape();
            if (shape.size() != 2 || shape[0] != batch.size || (d != 0 && shape[1] != d)) {
                throw std::runtime_error("Embedding Output needs [N, D] embeddings from the encoder");
            }
            d = shape[1];
            const float* values = output.ReadData<float>();
            embeddings.insert(embeddings.end(), values, values + batch.size * d);
            class_ids.insert(class_ids.end(), ids.begin(), ids.end());
            for (size_t i = 0; i < batch.size; ++i) {
                positions.push_back(position++);
                partitions.push_back(part.partition);
            }
        }
        part.rows->Reset();
    }
    const size_t n = class_ids.size();
    if (n == 0) throw std::runtime_error("Embedding Output found no rows in the chosen partition");

    const auto ok = [](const arrow::Status& status) {
        if (!status.ok()) throw std::runtime_error("Embedding Output: " + status.ToString());
    };
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (size_t k = 0; k < d; ++k) {
        arrow::FloatBuilder builder;
        ok(builder.Reserve(static_cast<int64_t>(n)));
        for (size_t i = 0; i < n; ++i) builder.UnsafeAppend(embeddings[i * d + k]);
        std::shared_ptr<arrow::Array> array;
        ok(builder.Finish(&array));
        fields.push_back(arrow::field("e" + std::to_string(k), arrow::float32()));
        columns.push_back(array);
    }
    if (include_metadata) {
        arrow::Int64Builder classes, rows;
        arrow::StringBuilder names;
        ok(classes.AppendValues(class_ids));
        ok(rows.AppendValues(positions));
        ok(names.AppendValues(partitions));
        std::shared_ptr<arrow::Array> class_array, name_array, row_array;
        ok(classes.Finish(&class_array));
        ok(names.Finish(&name_array));
        ok(rows.Finish(&row_array));
        fields.push_back(arrow::field("class", arrow::int64()));
        fields.push_back(arrow::field("partition", arrow::utf8()));
        fields.push_back(arrow::field("row", arrow::int64()));
        columns.push_back(class_array);
        columns.push_back(name_array);
        columns.push_back(row_array);
    }
    const auto parent = std::filesystem::path(path).parent_path();
    std::error_code ec;
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    const ArrowDataset table(arrow::Table::Make(arrow::schema(fields), columns, static_cast<int64_t>(n)),
                             "embeddings");
    if (!table.ExportParquet(path)) {
        throw std::runtime_error("Embedding Output could not write '" + path + "'");
    }
    return n;
}

MetricEvaluation EvaluateMetricLearning(IBatcher& rows,
                                        const std::function<Tensor(const Batch&)>& forward,
                                        const MetricEvaluationSpec& spec) {
    std::vector<float> embeddings;
    std::vector<int64_t> class_ids;
    std::vector<std::array<int, 3>> pairs;
    size_t d = 0;
    bool truncated = false;
    uint64_t batch_index = 0;

    rows.Reset();
    while (!rows.IsEpochComplete() && !truncated) {
        Batch batch = rows.GetNextBatch();
        if (!batch.IsValid()) break;
        const auto ids = ReadBatchClassIds(batch.labels, batch.size);
        const Tensor output = forward(batch);
        const auto& shape = output.Shape();
        if (shape.size() != 2 || shape[0] != batch.size) {
            throw std::runtime_error("metric evaluation needs [N, D] embeddings from the encoder");
        }
        if (d == 0) d = shape[1];
        const float* values = output.ReadData<float>();

        const size_t offset = class_ids.size();
        const size_t take = std::min(batch.size, spec.max_rows - offset);
        truncated = take < batch.size;
        embeddings.insert(embeddings.end(), values, values + take * d);
        class_ids.insert(class_ids.end(), ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(take));
        if (spec.pair_metrics) {
            for (const auto& p : SelectBatchPairs(ids, spec.pair_seed, batch_index)) {
                if (static_cast<size_t>(p[0]) < take && static_cast<size_t>(p[1]) < take) {
                    pairs.push_back({static_cast<int>(offset) + p[0], static_cast<int>(offset) + p[1], p[2]});
                }
            }
        }
        ++batch_index;
        if (class_ids.size() >= spec.max_rows && !rows.IsEpochComplete()) truncated = true;
    }
    rows.Reset();
    if (truncated) {
        spdlog::warn("Metric evaluation used the first {} rows of the partition (the evaluation cap)", spec.max_rows);
    }

    MetricEvaluation result = EvaluateMetricEmbeddings(std::move(embeddings), d, std::move(class_ids), pairs, spec);
    result.truncated = truncated;
    return result;
}

}  // namespace cyxwiz
