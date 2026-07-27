#pragma once

#include "parquet_backed_dataset.h"
#include "dataset_batcher.h"  // for the shared Batch struct

#include <cyxwiz/tensor.h>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * ParquetArrowBatcher - Training batcher for disk-backed Parquet datasets.
 *
 * Mirrors the shape of ArrowDatasetBatcher (same Batch struct, same
 * lifecycle methods) but reads data lazily from a ParquetBackedDataset
 * one row group at a time instead of holding the whole table in RAM.
 *
 * Row group strategy:
 *   - At construction time, row groups are partitioned between train and
 *     val splits by index (first train_split fraction -> train,
 *     remainder -> val). This is the simplest correct split; fine-grained
 *     row-level splitting would need per-row group subsampling.
 *   - Shuffling happens at two levels: the *order* of row groups is
 *     permuted at epoch start (Reset()), and when a row group is loaded
 *     its rows are also shuffled into a local order. This keeps memory
 *     access sequential within a loaded group while still giving good
 *     statistical shuffling across an epoch.
 *   - Only one row group is resident at a time. Loading the next one
 *     discards the previous; the OS page cache handles eviction behind
 *     the scenes via the memory-mapped Parquet file.
 *
 * Output matches ArrowDatasetBatcher: float32 feature batches,
 * int-as-float labels (or one-hot if enabled). The per-cell type dispatch
 * for converting Parquet columns to float is currently duplicated from
 * ArrowDatasetBatcher; a shared helper is a follow-up.
 */
class ParquetArrowBatcher : public IBatcher {
public:
    /**
     * @param dataset         Parquet-backed dataset to iterate over.
     * @param label_column    Name of the label column in the dataset schema.
     * @param batch_size      Samples per batch.
     * @param shuffle         Shuffle row group order + intra-group rows each epoch.
     * @param train_split     Fraction (0..1) of row groups assigned to the train split.
     * @param is_training     True for train batcher, false for val batcher.
     *                        Determines which row group range this batcher iterates.
     */
    ParquetArrowBatcher(std::shared_ptr<ParquetBackedDataset> dataset,
                        const std::string& label_column,
                        size_t batch_size,
                        bool shuffle = true,
                        float train_split = 0.8f,
                        bool is_training = true,
                        const std::string& partition_column = "",
                        int partition_value = 0,
                        int num_workers = 0,
                        BatcherPhase split_phase = BatcherPhase::Train,
                        float val_split = 0.0f,
                        uint32_t seed = 42);

    // IBatcher interface
    Batch GetNextBatch() override;
    void Reset() override;
    bool IsEpochComplete() const override;
    size_t GetNumBatches() const override;
    size_t GetNumSamples() const override { return num_samples_; }

    // Preprocessing (IBatcher)
    void SetNormalization(float mean, float std_dev) override;
    void SetOneHotEncoding(size_t num_classes) override;
    void SetFlatten(bool flatten) override { flatten_ = flatten; }
    void SetScalarLabelMode(bool enable) override { scalar_label_mode_ = enable; }
    void SetRegressionMode(bool enable) { SetScalarLabelMode(enable); }

private:
    // Configuration
    std::shared_ptr<ParquetBackedDataset> dataset_;
    std::string label_column_;
    size_t batch_size_;
    bool shuffle_;
    bool is_training_;
    int num_workers_ = 0;
    float train_split_;
    float val_split_ = 0.0f;
    BatcherPhase split_phase_ = BatcherPhase::Train;
    std::string partition_column_;
    int partition_value_ = 0;

    // Column layout (populated by InitializeColumns at construction)
    std::vector<int> feature_cols_;   // column indices that carry feature data
    int label_col_idx_ = -1;          // column index of the label column, -1 = none
    size_t num_features_ = 0;

    // Row group assignment: which row groups this batcher is allowed to read
    std::vector<int> assigned_row_groups_;

    // Shuffled order of row groups for this epoch (same contents as
    // assigned_row_groups_, possibly permuted).
    std::vector<int> epoch_group_order_;

    // Current position within the epoch
    size_t current_group_position_ = 0;  // index into epoch_group_order_
    std::shared_ptr<arrow::Table> current_table_;
    std::vector<size_t> current_row_indices_;  // row order within the current table
    size_t current_row_cursor_ = 0;            // next row to emit (index into current_row_indices_)

    // Stats
    size_t num_samples_ = 0;    // total samples in this split
    size_t samples_served_ = 0; // samples served so far this epoch

    // Preprocessing state (same semantics as ArrowDatasetBatcher)
    bool normalize_ = false;
    float norm_mean_ = 0.0f;
    float norm_std_ = 1.0f;
    bool one_hot_ = false;
    size_t num_classes_ = 10;
    bool flatten_ = true;
    bool scalar_label_mode_ = false;

    std::mt19937 rng_;

    // Helpers
    void InitializeColumns();
    void AssignRowGroups();
    void ShuffleEpochGroupOrder();
    bool LoadNextRowGroup();  // returns false if no more groups
    size_t CountPartitionRowsInGroup(int row_group_idx) const;
};

} // namespace cyxwiz
