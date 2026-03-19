#include "hdf5_dataset.h"

#ifdef CYXWIZ_HAS_HDF5

#include <spdlog/spdlog.h>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace cyxwiz {

HDF5Dataset::HDF5Dataset(const std::string& path, const HDF5Config& config)
    : path_(path), config_(config) {
    Initialize();
}

HDF5Dataset::~HDF5Dataset() {
    // Stop prefetch threads
    StopPrefetch();
    // File is closed automatically when unique_ptr is destroyed
}

size_t HDF5Dataset::Size() const {
    return num_samples_;
}

std::pair<std::vector<float>, int> HDF5Dataset::GetItem(size_t index) const {
    if (index >= num_samples_) return {{}, -1};

    // Get label (labels are always in memory - they're small)
    int label = labels_.empty() ? 0 : labels_[index];

    // If not using lazy loading, return directly from memory
    if (!use_lazy_loading_) {
        return {eager_samples_[index], label};
    }

    // Trigger prefetch for upcoming chunks
    if (config_.prefetch_enabled && !prefetch_threads_.empty()) {
        TriggerPrefetch(index);
    }

    // Lazy loading path: get sample from chunk cache
    std::vector<float> sample = GetSampleFromCache(index);
    return {std::move(sample), label};
}

DatasetInfo HDF5Dataset::GetInfo() const {
    DatasetInfo info;
    info.name = fs::path(path_).stem().string();
    info.path = path_;
    info.type = DatasetType::HDF5;
    info.shape = sample_shape_;
    info.num_samples = num_samples_;
    info.num_classes = num_classes_;
    info.train_count = train_indices_.size();
    info.val_count = val_indices_.size();
    info.test_count = test_indices_.size();

    // Memory usage depends on loading mode
    if (use_lazy_loading_) {
        // Only cache is in memory
        std::lock_guard<std::mutex> lock(cache_mutex_);
        info.memory_usage = chunk_cache_.size() * config_.chunk_size * sample_size_ * sizeof(float);
        info.cache_usage = info.memory_usage;
        info.is_streaming = true;
    } else {
        info.memory_usage = num_samples_ * sample_size_ * sizeof(float);
        info.cache_usage = 0;
        info.is_streaming = false;
    }
    info.is_loaded = true;

    // Cache statistics
    info.cache_hits = cache_hits_;
    info.cache_misses = cache_misses_;

    return info;
}

bool HDF5Dataset::IsLazyLoading() const {
    return use_lazy_loading_;
}

size_t HDF5Dataset::GetCacheHits() const {
    return cache_hits_;
}

size_t HDF5Dataset::GetCacheMisses() const {
    return cache_misses_;
}

size_t HDF5Dataset::GetCachedChunks() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return chunk_cache_.size();
}

void HDF5Dataset::ClearCache() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    chunk_cache_.clear();
    lru_order_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
}

bool HDF5Dataset::DetectSchema(HighFive::File& file) {
    // Common data path names (priority order)
    const std::vector<std::string> data_names = {
        "images", "data", "features", "X", "x", "inputs",
        "x_train", "train/images", "train/data"
    };

    // Common label path names
    const std::vector<std::string> label_names = {
        "labels", "targets", "y", "Y", "outputs",
        "y_train", "train/labels", "train/targets"
    };

    // Try to find data dataset
    for (const auto& name : data_names) {
        if (file.exist(name)) {
            try {
                auto ds = file.getDataSet(name);
                auto dims = ds.getDimensions();
                if (dims.size() >= 1) {
                    data_path_ = name;
                    break;
                }
            } catch (...) {}
        }
    }

    // Try to find labels dataset
    for (const auto& name : label_names) {
        if (file.exist(name)) {
            try {
                auto ds = file.getDataSet(name);
                auto dims = ds.getDimensions();
                if (dims.size() == 1) {
                    label_path_ = name;
                    break;
                }
            } catch (...) {}
        }
    }

    return !data_path_.empty();
}

void HDF5Dataset::Initialize() {
    try {
        HighFive::File file(path_, HighFive::File::ReadOnly);

        // Use config paths or auto-detect
        if (config_.auto_detect || config_.data_path.empty()) {
            if (!DetectSchema(file)) {
                spdlog::error("HDF5: Could not auto-detect schema in {}", path_);
                return;
            }
        } else {
            data_path_ = config_.data_path;
            label_path_ = config_.label_path;
        }

        // Get data dataset info
        auto data_ds = file.getDataSet(data_path_);
        dims_ = data_ds.getDimensions();
        dtype_is_uint8_ = (data_ds.getDataType() == HighFive::AtomicType<uint8_t>());
        dtype_is_float_ = (data_ds.getDataType() == HighFive::AtomicType<float>());
        dtype_is_double_ = (data_ds.getDataType() == HighFive::AtomicType<double>());
        dtype_is_int32_ = (data_ds.getDataType() == HighFive::AtomicType<int32_t>());

        if (dims_.empty()) {
            spdlog::error("HDF5: Empty dataset at {}", data_path_);
            return;
        }

        num_samples_ = dims_[0];

        // Calculate sample shape and size
        sample_shape_.clear();
        sample_size_ = 1;
        for (size_t i = 1; i < dims_.size(); i++) {
            sample_shape_.push_back(dims_[i]);
            sample_size_ *= dims_[i];
        }

        // Handle 1D data (tabular)
        if (dims_.size() == 1) {
            sample_shape_ = {1};
            sample_size_ = 1;
        }

        // Detect NCHW format for 4D image data
        is_nchw_ = false;
        if (dims_.size() == 4) {
            // dims_ = [N, dim1, dim2, dim3]
            // NCHW: [N, C, H, W] where C is small (1, 3, 4) and H, W are larger
            // NHWC: [N, H, W, C] where C is small (1, 3, 4) and H, W are larger
            size_t dim1 = dims_[1];  // C in NCHW, H in NHWC
            size_t dim2 = dims_[2];  // H in NCHW, W in NHWC
            size_t dim3 = dims_[3];  // W in NCHW, C in NHWC

            bool dim1_is_channel = (dim1 == 1 || dim1 == 3 || dim1 == 4) && dim2 > 4 && dim3 > 4;
            bool dim3_is_channel = (dim3 == 1 || dim3 == 3 || dim3 == 4) && dim1 > 4 && dim2 > 4;

            if (config_.force_nchw) {
                // User explicitly said it's NCHW
                is_nchw_ = true;
                spdlog::info("HDF5: Forcing NCHW->NHWC transpose");
            } else if (config_.auto_detect_nchw && dim1_is_channel && !dim3_is_channel) {
                // Auto-detected NCHW format
                is_nchw_ = true;
                spdlog::info("HDF5: Auto-detected NCHW format [N,{},{},{}], will transpose to NHWC",
                             dim1, dim2, dim3);
            }

            // Adjust sample_shape_ if NCHW -> NHWC
            if (is_nchw_) {
                // Original: [C, H, W] -> Transposed: [H, W, C]
                sample_shape_ = {dim2, dim3, dim1};  // [H, W, C]
            }
        }

        // Load labels (always load into memory - they're small)
        LoadLabels(file);

        // Decide whether to use lazy loading
        use_lazy_loading_ = config_.lazy_loading &&
                            (num_samples_ > config_.lazy_threshold);

        if (use_lazy_loading_) {
            // Lazy loading: just store metadata, don't load data
            spdlog::info("HDF5 dataset initialized (LAZY): {} samples, shape [{}], {} classes, "
                         "chunk_size={}, cache_size={}",
                num_samples_, GetShapeString(), num_classes_,
                config_.chunk_size, config_.chunk_cache_size);

            // Start prefetch threads if enabled
            if (config_.prefetch_enabled && config_.prefetch_threads > 0) {
                StartPrefetch();
            }
        } else {
            // Eager loading: load all data into memory
            LoadAllData(file);
            spdlog::info("HDF5 dataset loaded (EAGER): {} samples, shape [{}], {} classes",
                num_samples_, GetShapeString(), num_classes_);
        }

        if (num_samples_ > 0) {
            SetSplit(SplitConfig{});
        }

    } catch (const std::exception& e) {
        spdlog::error("HDF5 initialization error: {}", e.what());
    }
}

void HDF5Dataset::LoadLabels(HighFive::File& file) {
    std::set<int> unique_labels;

    if (!label_path_.empty() && file.exist(label_path_)) {
        auto label_ds = file.getDataSet(label_path_);
        auto label_dtype = label_ds.getDataType();

        if (label_dtype == HighFive::AtomicType<int32_t>()) {
            label_ds.read(labels_);
        } else if (label_dtype == HighFive::AtomicType<int64_t>()) {
            std::vector<int64_t> labels64;
            label_ds.read(labels64);
            labels_.resize(labels64.size());
            for (size_t i = 0; i < labels64.size(); i++) {
                labels_[i] = static_cast<int>(labels64[i]);
            }
        } else if (label_dtype == HighFive::AtomicType<uint8_t>()) {
            std::vector<uint8_t> labels8;
            label_ds.read(labels8);
            labels_.resize(labels8.size());
            for (size_t i = 0; i < labels8.size(); i++) {
                labels_[i] = static_cast<int>(labels8[i]);
            }
        } else if (label_dtype == HighFive::AtomicType<float>()) {
            std::vector<float> labelsf;
            label_ds.read(labelsf);
            labels_.resize(labelsf.size());
            for (size_t i = 0; i < labelsf.size(); i++) {
                labels_[i] = static_cast<int>(labelsf[i]);
            }
        }

        for (int label : labels_) {
            unique_labels.insert(label);
        }
    }

    num_classes_ = unique_labels.empty() ? 0 : static_cast<int>(unique_labels.size());
}

void HDF5Dataset::LoadAllData(HighFive::File& file) {
    auto data_ds = file.getDataSet(data_path_);
    eager_samples_.resize(num_samples_);

    float scale = (config_.normalize && dtype_is_uint8_) ? 1.0f / 255.0f : 1.0f;

    // Build selection parameters
    std::vector<size_t> offset(dims_.size(), 0);
    std::vector<size_t> count(dims_.size(), 1);
    for (size_t d = 1; d < dims_.size(); d++) {
        count[d] = dims_[d];
    }

    for (size_t i = 0; i < num_samples_; i++) {
        offset[0] = i;
        eager_samples_[i].resize(sample_size_);
        ReadSampleInto(data_ds, offset, count, eager_samples_[i], scale);
    }
}

void HDF5Dataset::ReadSampleInto(HighFive::DataSet& data_ds,
                    const std::vector<size_t>& offset,
                    const std::vector<size_t>& count,
                    std::vector<float>& out,
                    float scale) const {
    auto selection = data_ds.select(offset, count);

    if (dims_.size() == 2) {
        // 2D: [N, features]
        if (dtype_is_float_) {
            selection.read(out);
        } else if (dtype_is_uint8_) {
            std::vector<uint8_t> raw(sample_size_);
            selection.read(raw);
            for (size_t j = 0; j < sample_size_; j++) {
                out[j] = static_cast<float>(raw[j]) * scale;
            }
        } else if (dtype_is_double_) {
            std::vector<double> raw(sample_size_);
            selection.read(raw);
            for (size_t j = 0; j < sample_size_; j++) {
                out[j] = static_cast<float>(raw[j]);
            }
        } else if (dtype_is_int32_) {
            std::vector<int32_t> raw(sample_size_);
            selection.read(raw);
            for (size_t j = 0; j < sample_size_; j++) {
                out[j] = static_cast<float>(raw[j]);
            }
        }
    } else if (dims_.size() == 3) {
        // 3D: [N, H, W]
        ReadAndFlatten3D(selection, out, scale);
    } else if (dims_.size() == 4) {
        // 4D: [N, H, W, C]
        ReadAndFlatten4D(selection, out, scale);
    }
}

void HDF5Dataset::ReadAndFlatten3D(HighFive::Selection& selection, std::vector<float>& out, float scale) const {
    size_t k = 0;
    if (dtype_is_uint8_) {
        std::vector<std::vector<std::vector<uint8_t>>> data_3d;
        selection.read(data_3d);
        for (const auto& plane : data_3d) {
            for (const auto& row : plane) {
                for (auto val : row) {
                    out[k++] = static_cast<float>(val) * scale;
                }
            }
        }
    } else if (dtype_is_float_) {
        std::vector<std::vector<std::vector<float>>> data_3d;
        selection.read(data_3d);
        for (const auto& plane : data_3d) {
            for (const auto& row : plane) {
                for (auto val : row) {
                    out[k++] = val;
                }
            }
        }
    } else if (dtype_is_double_) {
        std::vector<std::vector<std::vector<double>>> data_3d;
        selection.read(data_3d);
        for (const auto& plane : data_3d) {
            for (const auto& row : plane) {
                for (auto val : row) {
                    out[k++] = static_cast<float>(val);
                }
            }
        }
    } else if (dtype_is_int32_) {
        std::vector<std::vector<std::vector<int32_t>>> data_3d;
        selection.read(data_3d);
        for (const auto& plane : data_3d) {
            for (const auto& row : plane) {
                for (auto val : row) {
                    out[k++] = static_cast<float>(val);
                }
            }
        }
    }
}

void HDF5Dataset::ReadAndFlatten4D(HighFive::Selection& selection, std::vector<float>& out, float scale) const {
    size_t k = 0;

    // Helper lambda to flatten with optional NCHW->NHWC transpose
    auto flatten_4d = [&](auto& data_4d, float s) {
        for (const auto& vol : data_4d) {
            if (is_nchw_) {
                // NCHW -> NHWC transpose: vol is [C, H, W], output as [H, W, C]
                size_t C = vol.size();
                size_t H = C > 0 ? vol[0].size() : 0;
                size_t W = (C > 0 && H > 0) ? vol[0][0].size() : 0;
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        for (size_t c = 0; c < C; c++) {
                            out[k++] = static_cast<float>(vol[c][h][w]) * s;
                        }
                    }
                }
            } else {
                // NHWC: vol is [H, W, C], output as-is
                for (const auto& plane : vol) {
                    for (const auto& row : plane) {
                        for (auto val : row) {
                            out[k++] = static_cast<float>(val) * s;
                        }
                    }
                }
            }
        }
    };

    if (dtype_is_uint8_) {
        std::vector<std::vector<std::vector<std::vector<uint8_t>>>> data_4d;
        selection.read(data_4d);
        flatten_4d(data_4d, scale);
    } else if (dtype_is_float_) {
        std::vector<std::vector<std::vector<std::vector<float>>>> data_4d;
        selection.read(data_4d);
        flatten_4d(data_4d, 1.0f);
    } else if (dtype_is_double_) {
        std::vector<std::vector<std::vector<std::vector<double>>>> data_4d;
        selection.read(data_4d);
        flatten_4d(data_4d, 1.0f);
    } else if (dtype_is_int32_) {
        std::vector<std::vector<std::vector<std::vector<int32_t>>>> data_4d;
        selection.read(data_4d);
        flatten_4d(data_4d, 1.0f);
    }
}

std::vector<float> HDF5Dataset::GetSampleFromCache(size_t index) const {
    size_t chunk_idx = index / config_.chunk_size;
    size_t local_idx = index % config_.chunk_size;

    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Check if chunk is in cache
    auto it = chunk_cache_.find(chunk_idx);
    if (it != chunk_cache_.end()) {
        // Cache hit - update LRU order
        cache_hits_++;
        lru_order_.remove(chunk_idx);
        lru_order_.push_front(chunk_idx);
        return it->second.samples[local_idx];
    }

    // Cache miss - need to load chunk
    cache_misses_++;

    // Evict oldest chunks if cache is full
    while (chunk_cache_.size() >= config_.chunk_cache_size && !lru_order_.empty()) {
        size_t evict_idx = lru_order_.back();
        lru_order_.pop_back();
        chunk_cache_.erase(evict_idx);
    }

    // Load chunk from file
    Chunk chunk = LoadChunk(chunk_idx);
    chunk_cache_[chunk_idx] = std::move(chunk);
    lru_order_.push_front(chunk_idx);

    return chunk_cache_[chunk_idx].samples[local_idx];
}

HDF5Dataset::Chunk HDF5Dataset::LoadChunk(size_t chunk_idx) const {
    Chunk chunk;
    chunk.start_index = chunk_idx * config_.chunk_size;
    chunk.count = std::min(config_.chunk_size, num_samples_ - chunk.start_index);

    float scale = (config_.normalize && dtype_is_uint8_) ? 1.0f / 255.0f : 1.0f;

    try {
        // HDF5 is not thread-safe, so we need to serialize file access
        std::lock_guard<std::mutex> lock(file_mutex_);

        // Open file for reading
        HighFive::File file(path_, HighFive::File::ReadOnly);
        auto data_ds = file.getDataSet(data_path_);

        chunk.samples.resize(chunk.count);

        // Build selection parameters for the chunk
        std::vector<size_t> offset(dims_.size(), 0);
        std::vector<size_t> count(dims_.size(), 1);
        for (size_t d = 1; d < dims_.size(); d++) {
            count[d] = dims_[d];
        }

        for (size_t i = 0; i < chunk.count; i++) {
            offset[0] = chunk.start_index + i;
            chunk.samples[i].resize(sample_size_);
            ReadSampleInto(data_ds, offset, count, chunk.samples[i], scale);
        }

    } catch (const std::exception& e) {
        spdlog::error("HDF5: Failed to load chunk {}: {}", chunk_idx, e.what());
    }

    return chunk;
}

std::string HDF5Dataset::GetShapeString() const {
    std::string s;
    for (size_t i = 0; i < sample_shape_.size(); i++) {
        if (i > 0) s += ", ";
        s += std::to_string(sample_shape_[i]);
    }
    return s;
}

void HDF5Dataset::StartPrefetch() const {
    prefetch_stop_ = false;
    size_t num_threads = config_.prefetch_threads;
    spdlog::info("HDF5: Starting {} prefetch threads", num_threads);

    for (size_t i = 0; i < num_threads; i++) {
        prefetch_threads_.emplace_back([this]() { PrefetchWorker(); });
    }
}

void HDF5Dataset::StopPrefetch() const {
    if (prefetch_threads_.empty()) return;

    // Signal threads to stop
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        prefetch_stop_ = true;
    }
    prefetch_cv_.notify_all();

    // Wait for threads to finish
    for (auto& t : prefetch_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    prefetch_threads_.clear();
}

void HDF5Dataset::TriggerPrefetch(size_t current_index) const {
    size_t current_chunk = current_index / config_.chunk_size;
    size_t total_chunks = (num_samples_ + config_.chunk_size - 1) / config_.chunk_size;

    std::lock_guard<std::mutex> lock(prefetch_mutex_);

    // Queue chunks ahead of current position
    for (size_t i = 1; i <= config_.prefetch_ahead; i++) {
        size_t chunk_to_prefetch = current_chunk + i;
        if (chunk_to_prefetch >= total_chunks) break;

        // Skip if already cached or being loaded
        {
            std::lock_guard<std::mutex> cache_lock(cache_mutex_);
            if (chunk_cache_.find(chunk_to_prefetch) != chunk_cache_.end()) continue;
        }
        if (prefetch_in_progress_.find(chunk_to_prefetch) != prefetch_in_progress_.end()) continue;

        prefetch_queue_.push(chunk_to_prefetch);
        prefetch_in_progress_.insert(chunk_to_prefetch);
    }

    if (!prefetch_queue_.empty()) {
        prefetch_cv_.notify_one();
    }
}

void HDF5Dataset::PrefetchWorker() const {
    while (true) {
        size_t chunk_idx;

        // Wait for work
        {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);
            prefetch_cv_.wait(lock, [this]() {
                return prefetch_stop_ || !prefetch_queue_.empty();
            });

            if (prefetch_stop_ && prefetch_queue_.empty()) {
                return;
            }

            if (prefetch_queue_.empty()) continue;

            chunk_idx = prefetch_queue_.front();
            prefetch_queue_.pop();
        }

        // Load chunk (outside of lock)
        Chunk chunk = LoadChunk(chunk_idx);

        // Store in cache
        {
            std::lock_guard<std::mutex> cache_lock(cache_mutex_);

            // Evict if cache is full
            while (chunk_cache_.size() >= config_.chunk_cache_size && !lru_order_.empty()) {
                size_t evict_idx = lru_order_.back();
                lru_order_.pop_back();
                chunk_cache_.erase(evict_idx);
            }

            chunk_cache_[chunk_idx] = std::move(chunk);
            lru_order_.push_front(chunk_idx);
        }

        // Remove from in-progress set
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            prefetch_in_progress_.erase(chunk_idx);
        }
    }
}

} // namespace cyxwiz

#endif // CYXWIZ_HAS_HDF5
