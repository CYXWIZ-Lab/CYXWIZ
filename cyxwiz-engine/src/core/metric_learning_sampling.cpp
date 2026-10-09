#include "metric_learning_sampling.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace cyxwiz {

uint64_t MetricSamplingMix(uint64_t value) {
    uint64_t z = value + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

std::vector<std::array<int, 3>> SelectBatchTriplets(
    const std::vector<int64_t>& class_ids,
    uint64_t key,
    uint64_t batch_index) {
    std::map<int64_t, std::vector<int>> rows_of_class;
    for (size_t row = 0; row < class_ids.size(); ++row) {
        rows_of_class[class_ids[row]].push_back(static_cast<int>(row));
    }

    const uint64_t batch_key = MetricSamplingMix(key ^ MetricSamplingMix(batch_index));
    std::vector<std::array<int, 3>> triplets;
    triplets.reserve(class_ids.size());
    for (size_t row = 0; row < class_ids.size(); ++row) {
        const auto& same = rows_of_class[class_ids[row]];
        const size_t positives = same.size() - 1;
        const size_t negatives = class_ids.size() - same.size();
        if (positives == 0 || negatives == 0) continue;

        // The k-th row of the same class other than this one ...
        const uint64_t pick_positive = MetricSamplingMix(batch_key ^ MetricSamplingMix(2 * row)) % positives;
        int positive = -1;
        size_t seen = 0;
        for (int candidate : same) {
            if (candidate == static_cast<int>(row)) continue;
            if (seen++ == pick_positive) {
                positive = candidate;
                break;
            }
        }
        // ... and the k-th row of any other class, in row order.
        const uint64_t pick_negative = MetricSamplingMix(batch_key ^ MetricSamplingMix(2 * row + 1)) % negatives;
        int negative = -1;
        seen = 0;
        for (size_t candidate = 0; candidate < class_ids.size(); ++candidate) {
            if (class_ids[candidate] == class_ids[row]) continue;
            if (seen++ == pick_negative) {
                negative = static_cast<int>(candidate);
                break;
            }
        }
        triplets.push_back({static_cast<int>(row), positive, negative});
    }
    return triplets;
}

std::vector<int64_t> ReadBatchClassIds(const Tensor& labels, size_t rows) {
    if (labels.NumElements() != rows) {
        throw std::runtime_error(
            "Triplet Dataset Builder needs one class id per row; the labels have " +
            std::to_string(labels.NumElements()) + " values for " + std::to_string(rows) + " rows");
    }
    std::vector<int64_t> ids(rows);
    switch (labels.GetDataType()) {
        case DataType::Int32: {
            const int32_t* values = labels.ReadData<int32_t>();
            for (size_t i = 0; i < rows; ++i) ids[i] = values[i];
            break;
        }
        case DataType::Int64: {
            const int64_t* values = labels.ReadData<int64_t>();
            for (size_t i = 0; i < rows; ++i) ids[i] = values[i];
            break;
        }
        case DataType::Float32: {
            const float* values = labels.ReadData<float>();
            for (size_t i = 0; i < rows; ++i) {
                if (!std::isfinite(values[i]) || values[i] != std::round(values[i])) {
                    throw std::runtime_error(
                        "Triplet Dataset Builder needs whole-number class ids; a label is " +
                        std::to_string(values[i]));
                }
                ids[i] = static_cast<int64_t>(values[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("Triplet Dataset Builder needs Int32, Int64 or Float32 class ids");
    }
    return ids;
}

TripletBatchSampler::TripletBatchSampler(IBatcher& source, uint64_t seed)
    : source_(&source), seed_(seed), epoch_key_(seed) {}

bool TripletBatchSampler::SetEpochShuffleSeed(uint64_t seed) {
    epoch_key_ = seed;
    return source_->SetEpochShuffleSeed(seed);
}

void TripletBatchSampler::Reset() {
    source_->Reset();
    batch_index_ = 0;
    skipped_batches_ = 0;
}

Batch TripletBatchSampler::GetNextBatch() {
    while (true) {
        Batch batch = source_->GetNextBatch();
        if (!batch.IsValid()) return batch;
        if (batch.HasSparseFeatures()) {
            throw std::runtime_error("Triplet Dataset Builder needs dense rows, not sparse features");
        }

        const auto class_ids = ReadBatchClassIds(batch.labels, batch.size);
        const auto triplets = SelectBatchTriplets(class_ids, epoch_key_, batch_index_++);
        if (triplets.empty()) {
            ++skipped_batches_;
            continue;
        }

        const size_t count = triplets.size();
        std::vector<int> rows(3 * count);
        std::vector<int> anchors(count);
        for (size_t t = 0; t < count; ++t) {
            rows[t] = triplets[t][0];
            rows[count + t] = triplets[t][1];
            rows[2 * count + t] = triplets[t][2];
            anchors[t] = triplets[t][0];
        }

        Batch stacked;
        stacked.data = batch.data.IndexSelect(0, rows);
        stacked.labels = batch.labels.IndexSelect(0, anchors);
        stacked.size = count;
        stacked.inspection = std::move(batch.inspection);
        return stacked;
    }
}

}  // namespace cyxwiz
