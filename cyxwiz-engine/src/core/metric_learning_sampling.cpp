#include "metric_learning_sampling.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace cyxwiz {

namespace {

constexpr const char* kBuilderName = "The Pair / Triplet Dataset Builder";

// Rows of the batch grouped by class id, and the picks of one row.
class BatchClasses {
public:
    BatchClasses(const std::vector<int64_t>& class_ids, uint64_t key, uint64_t batch_index)
        : class_ids_(class_ids),
          batch_key_(MetricSamplingMix(key ^ MetricSamplingMix(batch_index))) {
        for (size_t row = 0; row < class_ids.size(); ++row) {
            rows_of_class_[class_ids[row]].push_back(static_cast<int>(row));
        }
    }

    uint64_t BatchKey() const { return batch_key_; }
    size_t Positives(size_t row) const { return rows_of_class_.at(class_ids_[row]).size() - 1; }
    size_t Negatives(size_t row) const {
        return class_ids_.size() - rows_of_class_.at(class_ids_[row]).size();
    }

    // The k-th row of the same class other than this one, in row order.
    int Positive(size_t row) const {
        const uint64_t pick = MetricSamplingMix(batch_key_ ^ MetricSamplingMix(2 * row)) % Positives(row);
        size_t seen = 0;
        for (int candidate : rows_of_class_.at(class_ids_[row])) {
            if (candidate == static_cast<int>(row)) continue;
            if (seen++ == pick) return candidate;
        }
        return -1;
    }

    // The k-th row of any other class, in row order.
    int Negative(size_t row) const {
        const uint64_t pick = MetricSamplingMix(batch_key_ ^ MetricSamplingMix(2 * row + 1)) % Negatives(row);
        size_t seen = 0;
        for (size_t candidate = 0; candidate < class_ids_.size(); ++candidate) {
            if (class_ids_[candidate] == class_ids_[row]) continue;
            if (seen++ == pick) return static_cast<int>(candidate);
        }
        return -1;
    }

private:
    const std::vector<int64_t>& class_ids_;
    uint64_t batch_key_;
    std::map<int64_t, std::vector<int>> rows_of_class_;
};

}  // namespace

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
    const BatchClasses classes(class_ids, key, batch_index);
    std::vector<std::array<int, 3>> triplets;
    triplets.reserve(class_ids.size());
    for (size_t row = 0; row < class_ids.size(); ++row) {
        if (classes.Positives(row) == 0 || classes.Negatives(row) == 0) continue;
        triplets.push_back({static_cast<int>(row), classes.Positive(row), classes.Negative(row)});
    }
    return triplets;
}

std::vector<std::array<int, 3>> SelectBatchPairs(
    const std::vector<int64_t>& class_ids,
    uint64_t key,
    uint64_t batch_index) {
    const BatchClasses classes(class_ids, key, batch_index);
    std::vector<std::array<int, 3>> pairs;
    pairs.reserve(class_ids.size());
    for (size_t row = 0; row < class_ids.size(); ++row) {
        const bool has_positive = classes.Positives(row) > 0;
        const bool has_negative = classes.Negatives(row) > 0;
        if (!has_positive && !has_negative) continue;
        const bool wants_similar = ((row + classes.BatchKey()) & 1u) == 0;
        const bool similar = has_positive && (wants_similar || !has_negative);
        pairs.push_back({static_cast<int>(row), similar ? classes.Positive(row) : classes.Negative(row),
                         similar ? 1 : 0});
    }
    return pairs;
}

std::vector<int64_t> ReadBatchClassIds(const Tensor& labels, size_t rows) {
    if (labels.NumElements() != rows) {
        throw std::runtime_error(
            std::string(kBuilderName) + " needs one class id per row; the labels have " +
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
                    throw std::runtime_error(std::string(kBuilderName) +
                                             " needs whole-number class ids; a label is " +
                                             std::to_string(values[i]));
                }
                ids[i] = static_cast<int64_t>(values[i]);
            }
            break;
        }
        default:
            throw std::runtime_error(std::string(kBuilderName) + " needs Int32, Int64 or Float32 class ids");
    }
    return ids;
}

MetricBatchSampler::MetricBatchSampler(IBatcher& source, MetricSampling sampling, uint64_t seed)
    : source_(&source), sampling_(sampling), seed_(seed), epoch_key_(seed) {
    if (sampling_ == MetricSampling::None) {
        throw std::invalid_argument("MetricBatchSampler needs pair or triplet sampling");
    }
}

bool MetricBatchSampler::SetEpochShuffleSeed(uint64_t seed) {
    epoch_key_ = seed;
    return source_->SetEpochShuffleSeed(seed);
}

void MetricBatchSampler::Reset() {
    source_->Reset();
    batch_index_ = 0;
    skipped_batches_ = 0;
}

Batch MetricBatchSampler::GetNextBatch() {
    while (true) {
        Batch batch = source_->GetNextBatch();
        if (!batch.IsValid()) return batch;
        if (batch.HasSparseFeatures()) {
            throw std::runtime_error(std::string(kBuilderName) + " needs dense rows, not sparse features");
        }

        const auto class_ids = ReadBatchClassIds(batch.labels, batch.size);
        const bool triplets = sampling_ == MetricSampling::Triplets;
        const auto picks = triplets ? SelectBatchTriplets(class_ids, epoch_key_, batch_index_)
                                    : SelectBatchPairs(class_ids, epoch_key_, batch_index_);
        ++batch_index_;
        if (picks.empty()) {
            ++skipped_batches_;
            continue;
        }

        const size_t count = picks.size();
        const size_t blocks = triplets ? 3 : 2;
        std::vector<int> rows(blocks * count);
        for (size_t i = 0; i < count; ++i) {
            for (size_t block = 0; block < blocks; ++block) rows[block * count + i] = picks[i][block];
        }

        Batch stacked;
        stacked.data = batch.data.IndexSelect(0, rows);
        if (triplets) {
            std::vector<int> anchors(count);
            for (size_t i = 0; i < count; ++i) anchors[i] = picks[i][0];
            stacked.labels = batch.labels.IndexSelect(0, anchors);
        } else {
            std::vector<float> similar(count);
            for (size_t i = 0; i < count; ++i) similar[i] = static_cast<float>(picks[i][2]);
            stacked.labels = Tensor({count}, similar.data(), DataType::Float32);
        }
        stacked.size = count;
        stacked.inspection = std::move(batch.inspection);
        return stacked;
    }
}

}  // namespace cyxwiz
