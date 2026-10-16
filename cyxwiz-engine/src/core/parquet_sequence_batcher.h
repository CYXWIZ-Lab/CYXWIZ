#pragma once

// Causal-LM token windows read a row group at a time (TOFIX154 step 6), for
// a training set kept on disk: memory holds one row group's windows, not the
// corpus. Each phase (Train / Validation / Test) is a list of units, a row
// group of a Parquet file or 128K rows of a table in memory. Train shuffles
// the unit order every epoch and the windows within a unit; the batches come
// from a SequenceBatcher over the loaded unit, so padding, targets and masks
// are the in-memory path's.

#include "parquet_backed_dataset.h"
#include "sequence_batcher.h"
#include "sequence_token_window_input.h"

#include <arrow/api.h>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

// One phase's windows: a Parquet file, or a table in memory.
struct SequenceWindowSource {
    static constexpr int64_t kTableUnitRows = 128 * 1024;
    std::shared_ptr<ParquetBackedDataset> file;
    std::shared_ptr<arrow::Table> table;

    bool Empty() const { return !file && !table; }
    std::shared_ptr<arrow::Schema> Schema() const { return file ? file->GetSchema() : table ? table->schema() : nullptr; }
    int Units() const {
        if (file) return file->GetNumRowGroups();
        return table ? static_cast<int>((table->num_rows() + kTableUnitRows - 1) / kTableUnitRows) : 0;
    }
    int64_t UnitRows(int unit) const {
        if (file) return file->GetRowGroupSize(unit);
        return std::min(kTableUnitRows, table->num_rows() - static_cast<int64_t>(unit) * kTableUnitRows);
    }
    std::shared_ptr<arrow::Table> ReadUnit(int unit) const {
        if (file) return file->ReadRowGroup(unit);
        return table->Slice(static_cast<int64_t>(unit) * kTableUnitRows, kTableUnitRows);
    }
};

class ParquetSequenceBatcher : public ISequenceBatcher {
public:
    // phases: Train, Validation, Test (Validation / Test may be empty).
    ParquetSequenceBatcher(std::array<SequenceWindowSource, 3> phases, std::string token_column, int context,
                           SequenceBatcherConfig config)
        : phases_(std::move(phases)), token_column_(std::move(token_column)), context_(context),
          config_(std::move(config)), rng_(config_.seed) {
        config_.batch_size = std::max<size_t>(1, config_.batch_size);
        Reset();
    }

    SequenceBatch GetNextSequenceBatch() override {
        while (!inner_ || inner_->IsEpochComplete()) {
            if (next_unit_ >= order_.size()) return {};
            Load(order_[next_unit_++]);
        }
        return inner_->GetNextSequenceBatch();
    }

    void Reset() override {
        const auto& source = Source();
        order_.resize(static_cast<size_t>(source.Units()));
        for (size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<int>(i);
        if (pending_seed_) {
            rng_.seed(static_cast<std::mt19937::result_type>(*pending_seed_));
            pending_seed_.reset();
        }
        if (phase_ == BatcherPhase::Train && config_.shuffle) std::shuffle(order_.begin(), order_.end(), rng_);
        next_unit_ = 0;
        inner_.reset();
    }

    bool SetEpochShuffleSeed(uint64_t seed) override {
        pending_seed_ = seed;
        return true;
    }

    bool IsEpochComplete() const override {
        return next_unit_ >= order_.size() && (!inner_ || inner_->IsEpochComplete());
    }

    size_t GetNumBatches() const override {
        const auto& source = Source();
        size_t batches = 0;
        for (int unit = 0; unit < source.Units(); ++unit) {
            const auto rows = static_cast<size_t>(std::max<int64_t>(0, source.UnitRows(unit)));
            batches += (rows + config_.batch_size - 1) / config_.batch_size;
        }
        return batches;
    }

    size_t GetNumSamples() const override {
        const auto& source = Source();
        size_t rows = 0;
        for (int unit = 0; unit < source.Units(); ++unit)
            rows += static_cast<size_t>(std::max<int64_t>(0, source.UnitRows(unit)));
        return rows;
    }

    void SetPhase(BatcherPhase phase) override { phase_ = phase; }

    bool HasPhase(BatcherPhase phase) const override {
        return phase == BatcherPhase::Train || !phases_[Index(phase)].Empty();
    }

private:
    static size_t Index(BatcherPhase phase) {
        return phase == BatcherPhase::Val ? 1 : phase == BatcherPhase::Test ? 2 : 0;
    }
    const SequenceWindowSource& Source() const { return phases_[Index(phase_)]; }

    void Load(int unit) {
        const auto table = Source().ReadUnit(unit);
        if (!table) throw std::runtime_error("Token windows: could not read row group " + std::to_string(unit));
        const int column = table->schema()->GetFieldIndex(token_column_);
        std::vector<SequenceSample> samples;
        Vocabulary vocabulary;
        int context = 0;
        std::string error;
        if (column < 0 || !ReadTokenWindowSamples(table, column, context_, samples, vocabulary, context, error))
            throw std::runtime_error(error.empty() ? "Token windows: no token column '" + token_column_ + "'" : error);
        SequenceBatcherConfig unit_config = config_;
        unit_config.train_indices.clear();
        unit_config.val_indices.clear();
        unit_config.test_indices.clear();
        unit_config.drop_last = false;
        unit_config.shuffle = phase_ == BatcherPhase::Train && config_.shuffle;
        unit_config.seed = static_cast<uint32_t>(rng_());
        inner_ = std::make_unique<SequenceBatcher>(std::move(samples), unit_config);
    }

    std::array<SequenceWindowSource, 3> phases_;
    std::string token_column_;
    int context_ = 0;
    SequenceBatcherConfig config_;
    std::mt19937 rng_;
    std::optional<uint64_t> pending_seed_;
    BatcherPhase phase_ = BatcherPhase::Train;
    std::vector<int> order_;
    size_t next_unit_ = 0;
    std::unique_ptr<SequenceBatcher> inner_;
};

}  // namespace cyxwiz
