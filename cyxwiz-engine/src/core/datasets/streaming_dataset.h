#pragma once

#include "../dataset_base.h"
#include "../data_registry.h"
#include <vector>
#include <string>
#include <map>

namespace cyxwiz {

/**
 * Streaming dataset loader for large datasets
 *
 * Supports:
 * - CSV files (chunk-based line reading)
 * - Image folders (lazy image loading)
 * - Automatic type detection
 * - Chunk-based buffering with LRU eviction
 * - Forward streaming with GetNext()
 * - Random access with GetItem()
 *
 * Memory efficiency:
 * - Only keeps configurable number of chunks in memory
 * - Automatically evicts oldest chunks when buffer is full
 * - Estimates total size for progress tracking
 */
class StreamingDataset : public Dataset {
public:
    StreamingDataset(const std::string& path, const StreamingConfig& config);

    // Dataset interface
    size_t Size() const override;
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    DatasetInfo GetInfo() const override;

    // Streaming interface
    bool IsStreaming() const override;
    bool HasNext() const override;
    std::pair<std::vector<float>, int> GetNext() override;
    void ResetStream() override;

private:
    // Internal data structure
    struct DataChunk {
        std::vector<std::vector<float>> samples;
        std::vector<int> labels;
    };

    // Initialization
    void Initialize();
    void EstimateSize();

    // Chunk loading
    void LoadChunk(size_t chunk_idx) const;
    void LoadCSVChunk(DataChunk& chunk, size_t start_idx, size_t end_idx) const;
    void LoadImageChunk(DataChunk& chunk, size_t start_idx, size_t end_idx) const;
    void CollectImagePaths() const;

    // Utilities
    size_t GetBufferMemoryUsage() const;

    // Configuration
    std::string path_;
    StreamingConfig config_;
    DatasetType detected_type_ = DatasetType::None;
    mutable std::vector<size_t> shape_;
    mutable size_t num_classes_ = 0;
    size_t estimated_total_size_ = 0;
    mutable size_t current_position_ = 0;

    // Chunk buffer (LRU cache of chunks)
    mutable std::map<size_t, DataChunk> chunk_buffer_;
    mutable std::vector<size_t> chunk_access_order_;
    size_t max_chunks_in_buffer_ = 10;

    // Image paths for image folder streaming
    mutable std::vector<std::pair<std::string, int>> image_paths_;
};

} // namespace cyxwiz
