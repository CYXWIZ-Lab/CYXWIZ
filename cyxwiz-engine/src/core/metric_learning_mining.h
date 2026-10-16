#pragma once

// Online mining for metric learning (TOFIX140 A5). With mining set to hard or
// semi-hard on a Pair / Triplet Dataset Builder, the batch goes through the
// encoder once as it is ([N, D] embeddings, class ids), and the loss picks its
// pairs or triplets from the [N, N] distance matrix of those embeddings. The
// picks are not differentiated (torch: chosen from detached distances).
// Everything runs on the ArrayFire device (TOFIX140 F7): the distances, the
// picks (row indices) and the gathers; picks never visit the host.

#include "metric_learning_batch.h"

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cyxwiz {

enum class MiningDistance {
    Euclidean,  // Triplet, Contrastive
    Cosine,     // Cosine Embedding: 1 - cos(a, b)
};

// Class ids of a batch's labels as Int64 [N]: Int32 / Int64 / Float32 whole
// numbers, shape [N] or [N, 1]. Throws for a wrong count or a fractional id.
Tensor BatchClassIds(const Tensor& labels, size_t rows);

// The same ids on the host, for the evaluation and export paths that collect
// a partition's rows there (a declared host boundary: checked on the device,
// downloaded once).
std::vector<int64_t> ReadBatchClassIds(const Tensor& labels, size_t rows);

// [N, N] distances between the rows of [N, D] embeddings (float64 where the
// device has it). Euclidean accumulates per feature (no Gram cancellation);
// cosine uses the Cosine Embedding loss's epsilon. The diagonal is 0.
Tensor MiningDistances(const Tensor& embeddings, MiningDistance kind);

// Picks as Int32 row indices, in anchor order; ties go to the lowest row.
//   Hard (batch-hard, Hermans et al. 2017): for every row with a positive and
//     a negative, its farthest same-class row and its closest other-class row.
//   SemiHard (FaceNet; the TF Addons triplet_semihard_loss rule): for every
//     same-class pair (anchor, positive), the closest negative farther from
//     the anchor than the positive; when there is none, the farthest negative.
struct MinedTriplets {
    Tensor anchors;
    Tensor positives;
    Tensor negatives;
    size_t count = 0;
};
MinedTriplets MineTriplets(const Tensor& class_ids, const Tensor& distances, MetricMining mining);

// Hard pairs: for every row, its farthest same-class row (similar = 1) and its
// closest other-class row (similar = 0), when it has them, in row order.
struct MinedPairs {
    Tensor firsts;
    Tensor seconds;
    Tensor similar;  // Float32 1 / 0
    size_t count = 0;
};
MinedPairs MinePairs(const Tensor& class_ids, const Tensor& distances);

// The rows of a [N, D] matrix at Int32 indices: [S, D].
Tensor GatherRows(const Tensor& matrix, const Tensor& rows);

// A [S, D] gradient summed back onto N rows by each row's source index.
Tensor ScatterAddRows(const Tensor& stacked, const Tensor& source_rows, size_t rows);

// distances[rows[i], columns[i]] for every i.
Tensor DistancesAt(const Tensor& distances, const Tensor& rows, const Tensor& columns);

}  // namespace cyxwiz
