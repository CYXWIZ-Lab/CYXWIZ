#pragma once

// Pair / Retrieval Metrics (TOFIX140 A5): one evaluation pass of a metric
// model over the plain rows of a partition. Each row goes through the encoder
// once (the caller's forward, in eval mode); the embeddings and class ids are
// collected and scored with metric_learning_metrics:
//   pair metrics      over the seeded pairs of each batch (SelectBatchPairs,
//                     keyed by the DataLoader seed), at a distance threshold;
//   retrieval metrics leave-one-out over all collected rows (recall@k, MRR,
//                     1-NN class agreement).
// Embeddings of a Cosine Embedding model are L2-normalised first, so Euclidean
// ranking is cosine ranking.

#include "dataset_batcher.h"
#include "metric_learning_metrics.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct TrainingConfiguration;

struct MetricEvaluationSpec {
    bool pair_metrics = false;
    double pair_threshold = 0.5;
    bool retrieval_metrics = false;
    size_t retrieval_k = 10;
    bool normalize = false;
    // Also give every row the class of its nearest other row (Test step).
    bool nearest_neighbours = false;
    uint64_t pair_seed = 0;
    size_t max_rows = 4096;
};

struct MetricEvaluation {
    size_t rows = 0;
    bool truncated = false;  // more rows than max_rows; the first max_rows were used
    std::optional<PairMetricResult> pair;
    std::optional<RetrievalMetricResult> retrieval;
    std::vector<int64_t> class_ids;
    // nearest_neighbours: the class of each row's nearest other row.
    std::vector<int64_t> nearest_classes;
};

// The spec a compiled graph asks for: its Pair / Retrieval Metrics nodes, the
// Cosine Embedding normalisation, the DataLoader seed for the pairs.
MetricEvaluationSpec MetricEvaluationSpecFor(const TrainingConfiguration& config);

// One line for the Console: "pair accuracy 0.91 (threshold 0.5, 240 pairs),
// Recall@10 0.97, MRR 0.88, 1-NN 0.85 over 240 rows".
std::string DescribeMetricEvaluation(const MetricEvaluation& evaluation);

// Rows from `rows` (reset first; read to the end of its current phase).
MetricEvaluation EvaluateMetricLearning(IBatcher& rows,
                                        const std::function<Tensor(const Batch&)>& forward,
                                        const MetricEvaluationSpec& spec);

// Embedding Output: the embeddings of each part's rows (read to the end of
// its current phase), written to one Parquet file: e0 .. e(D-1) and, with
// metadata, class (int64), partition (string) and row (position in the
// partition). Returns the number of rows written; throws on failure.
struct EmbeddingExportPart {
    std::string partition;
    IBatcher* rows = nullptr;
};
size_t WriteEmbeddingsParquet(const std::vector<EmbeddingExportPart>& parts,
                              const std::function<Tensor(const Batch&)>& forward,
                              bool include_metadata,
                              const std::string& path);

// The same over embeddings already collected (row-major [n, d]) and their
// class ids; pairs as {first row, second row, similar}.
MetricEvaluation EvaluateMetricEmbeddings(std::vector<float> embeddings, size_t d,
                                          std::vector<int64_t> class_ids,
                                          const std::vector<std::array<int, 3>>& pairs,
                                          const MetricEvaluationSpec& spec);

}  // namespace cyxwiz
