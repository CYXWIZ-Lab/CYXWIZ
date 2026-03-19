#pragma once

#include "dataset_types.h"
#include <vector>
#include <string>
#include <utility>

namespace cyxwiz {

/**
 * Abstract base class for all dataset implementations
 *
 * Provides common interface for:
 * - Data access (GetItem, GetBatch)
 * - Train/val/test splitting
 * - Streaming support
 * - Metadata query
 */
class Dataset {
public:
    virtual ~Dataset() = default;

    // Core interface - must be implemented by all datasets
    virtual size_t Size() const = 0;
    virtual std::pair<std::vector<float>, int> GetItem(size_t index) const = 0;
    virtual DatasetInfo GetInfo() const = 0;

    // Batch access - default implementation provided
    virtual std::pair<std::vector<std::vector<float>>, std::vector<int>>
        GetBatch(const std::vector<size_t>& indices) const;

    // Split management - default implementation provided
    virtual void SetSplit(const SplitConfig& config);
    virtual const std::vector<size_t>& GetTrainIndices() const { return train_indices_; }
    virtual const std::vector<size_t>& GetValIndices() const { return val_indices_; }
    virtual const std::vector<size_t>& GetTestIndices() const { return test_indices_; }
    virtual const std::vector<size_t>& GetSplitIndices(DatasetSplit split) const;

    // Streaming support - override if dataset supports streaming
    virtual bool IsStreaming() const { return false; }
    virtual bool HasNext() const { return false; }
    virtual std::pair<std::vector<float>, int> GetNext() { return {{}, -1}; }
    virtual void ResetStream() {}

    // Raw data access for preview - override if dataset provides this
    virtual std::vector<std::string> GetColumnNames() const { return {}; }
    virtual float GetFloatLabel(size_t index) const { (void)index; return 0.0f; }
    virtual bool HasFloatLabels() const { return false; }
    virtual int GetLabelColumnIndex() const { return -1; }
    virtual int GetOriginalColumnCount() const { return -1; }
    virtual const std::vector<std::string>& GetTextLines() const {
        static std::vector<std::string> empty;
        return empty;
    }
    virtual const void* GetRawJSON() const { return nullptr; }

protected:
    // Split indices - managed by base class
    std::vector<size_t> train_indices_;
    std::vector<size_t> val_indices_;
    std::vector<size_t> test_indices_;
    std::vector<size_t> all_indices_;
    SplitConfig split_config_;
};

} // namespace cyxwiz
