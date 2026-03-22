#pragma once

#include "../dataset_base.h"
#include "../data_registry.h"

#ifdef CYXWIZ_HAS_HDF5

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <set>
#include <unordered_map>
#include <list>

namespace cyxwiz {

/**
 * HDF5 dataset loader with lazy loading and prefetching
 *
 * Features:
 * - Auto schema detection (data/labels paths)
 * - Eager vs lazy loading modes
 * - Chunk-based caching with LRU eviction
 * - Multi-threaded prefetching for parallel I/O
 * - NCHW->NHWC transpose for PyTorch datasets
 * - Support for uint8, float, double, int32 data types
 * - Normalization (uint8 -> [0,1])
 *
 * Performance:
 * - Lazy loading for files > 10k samples (configurable)
 * - Cache up to 32 chunks (configurable)
 * - Prefetch 4 chunks ahead (configurable)
 * - 2 prefetch worker threads (configurable)
 */
class HDF5Dataset : public Dataset {
public:
    HDF5Dataset(const std::string& path, const HDF5Config& config = {});
    ~HDF5Dataset();

    // Dataset interface
    size_t Size() const override;
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    DatasetInfo GetInfo() const override;

    // HDF5-specific methods
    bool IsLazyLoading() const;
    size_t GetCacheHits() const;
    size_t GetCacheMisses() const;
    size_t GetCachedChunks() const;
    void ClearCache() const;

private:
    // Chunk data structure
    struct Chunk {
        std::vector<std::vector<float>> samples;  // samples in this chunk
        size_t start_index;                        // first sample index
        size_t count;                              // number of samples
    };

    // Initialization
    void Initialize();
    bool DetectSchema(HighFive::File& file);
    void LoadLabels(HighFive::File& file);
    void LoadAllData(HighFive::File& file);

    // Data reading helpers
    void ReadSampleInto(HighFive::DataSet& data_ds,
                        const std::vector<size_t>& offset,
                        const std::vector<size_t>& count,
                        std::vector<float>& out,
                        float scale) const;
    void ReadAndFlatten3D(HighFive::Selection& selection, std::vector<float>& out, float scale) const;
    void ReadAndFlatten4D(HighFive::Selection& selection, std::vector<float>& out, float scale) const;

    // Lazy loading and caching
    std::vector<float> GetSampleFromCache(size_t index) const;
    Chunk LoadChunk(size_t chunk_idx) const;

    // Prefetch methods
    void StartPrefetch() const;
    void StopPrefetch() const;
    void TriggerPrefetch(size_t current_index) const;
    void PrefetchWorker() const;

    // Utilities
    std::string GetShapeString() const;

    // Configuration
    std::string path_;
    HDF5Config config_;
    std::string data_path_;
    std::string label_path_;

    // Dataset metadata
    std::vector<size_t> dims_;
    std::vector<size_t> sample_shape_;
    size_t num_samples_ = 0;
    size_t sample_size_ = 0;
    int num_classes_ = 0;

    // Data type flags
    bool dtype_is_uint8_ = false;
    bool dtype_is_float_ = false;
    bool dtype_is_double_ = false;
    bool dtype_is_int32_ = false;

    // Loading mode
    bool use_lazy_loading_ = false;

    // NCHW transpose flag (for PyTorch-style datasets)
    bool is_nchw_ = false;

    // Eager loading storage (used when lazy_loading is disabled)
    std::vector<std::vector<float>> eager_samples_;

    // Labels (always loaded into memory)
    std::vector<int> labels_;

    // Lazy loading: chunk cache with LRU eviction
    mutable std::mutex cache_mutex_;
    mutable std::mutex file_mutex_;  // Serialize HDF5 file access (not thread-safe)
    mutable std::unordered_map<size_t, Chunk> chunk_cache_;
    mutable std::list<size_t> lru_order_;  // front = most recent, back = oldest
    mutable size_t cache_hits_ = 0;
    mutable size_t cache_misses_ = 0;

    // Prefetch thread pool for parallel I/O
    mutable std::vector<std::thread> prefetch_threads_;
    mutable std::queue<size_t> prefetch_queue_;
    mutable std::mutex prefetch_mutex_;
    mutable std::condition_variable prefetch_cv_;
    mutable std::atomic<bool> prefetch_stop_{false};
    mutable std::set<size_t> prefetch_in_progress_;  // Chunks currently being loaded
};

} // namespace cyxwiz

#endif // CYXWIZ_HAS_HDF5
