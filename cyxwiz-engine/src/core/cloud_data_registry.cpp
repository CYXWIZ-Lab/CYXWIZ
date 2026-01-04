#include "cloud_data_registry.h"
#include <filesystem>

namespace cyxwiz {

static CloudDataRegistry* g_cloud_registry = nullptr;

CloudDataRegistry::CloudDataRegistry() {
    // Default cache directory
    std::string home;
#ifdef _WIN32
    const char* userprofile = std::getenv("USERPROFILE");
    if (userprofile) home = userprofile;
#else
    const char* home_env = std::getenv("HOME");
    if (home_env) home = home_env;
#endif

    if (!home.empty()) {
        cache_directory_ = (std::filesystem::path(home) / ".cyxwiz" / "cloud_cache").string();
    }
}

CloudDataRegistry::~CloudDataRegistry() {
    // Stop any active streaming
    for (auto& pair : cloud_datasets_) {
        if (pair.second.is_streaming && datastream_client_) {
            datastream_client_->StopStreaming();
        }
    }
}

CloudDataRegistry& CloudDataRegistry::Instance() {
    if (!g_cloud_registry) {
        g_cloud_registry = new CloudDataRegistry();
    }
    return *g_cloud_registry;
}

void CloudDataRegistry::SetDataStreamClient(network::DataStreamClient* client) {
    datastream_client_ = client;
}

DatasetHandle CloudDataRegistry::RegisterCloudDataset(const network::CloudDatasetInfo& cloud_dataset) {
    // Create a cloud dataset entry
    CloudDatasetEntry entry;
    entry.handle = DatasetHandle(next_cloud_handle_id_++);
    entry.info = cloud_dataset;
    entry.dataset_id = cloud_dataset.id;
    entry.is_streaming = false;

    cloud_datasets_[entry.handle.GetId()] = entry;

    return entry.handle;
}

bool CloudDataRegistry::GetCloudDatasetInfo(DatasetHandle handle, network::CloudDatasetInfo& out_info) const {
    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) {
        return false;
    }

    out_info = it->second.info;
    return true;
}

bool CloudDataRegistry::IsCloudDataset(DatasetHandle handle) const {
    return cloud_datasets_.find(handle.GetId()) != cloud_datasets_.end();
}

std::string CloudDataRegistry::GetCloudDatasetId(DatasetHandle handle) const {
    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) {
        return "";
    }
    return it->second.dataset_id;
}

bool CloudDataRegistry::StartCloudStreaming(DatasetHandle handle,
                                             int batch_size,
                                             network::BatchCallback on_batch,
                                             network::StreamErrorCallback on_error,
                                             network::StreamCompleteCallback on_complete,
                                             bool shuffle,
                                             int64_t seed) {
    if (!datastream_client_ || !datastream_client_->IsConnected()) {
        return false;
    }

    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) {
        return false;
    }

    // Check trust level
    if (it->second.info.trust_level > min_trust_level_) {
        if (on_error) {
            on_error("Dataset trust level below minimum requirement");
        }
        return false;
    }

    bool success = datastream_client_->StartStreaming(
        it->second.dataset_id,
        batch_size,
        on_batch,
        on_error,
        on_complete,
        0,  // start_index
        4,  // prefetch_batches
        min_trust_level_,
        shuffle,
        seed
    );

    if (success) {
        it->second.is_streaming = true;
    }

    return success;
}

void CloudDataRegistry::StopCloudStreaming(DatasetHandle handle) {
    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) return;

    if (it->second.is_streaming && datastream_client_) {
        datastream_client_->StopStreaming();
        it->second.is_streaming = false;
    }
}

bool CloudDataRegistry::IsStreaming(DatasetHandle handle) const {
    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) return false;
    return it->second.is_streaming;
}

float CloudDataRegistry::GetStreamingProgress(DatasetHandle handle) const {
    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) return 0.0f;

    if (!it->second.is_streaming || !datastream_client_) return 0.0f;

    return datastream_client_->GetStreamingProgress();
}

network::TrustLevel CloudDataRegistry::GetTrustLevel(DatasetHandle handle) const {
    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) {
        return network::TrustLevel::Untrusted;
    }
    return it->second.info.trust_level;
}

bool CloudDataRegistry::VerifyCloudDataset(DatasetHandle handle,
                                            network::DatasetVerificationResult& out_result) {
    if (!datastream_client_ || !datastream_client_->IsConnected()) {
        return false;
    }

    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) {
        return false;
    }

    bool success = datastream_client_->VerifyDataset(it->second.dataset_id, out_result);

    if (success) {
        // Update trust level based on verification
        it->second.info.trust_level = out_result.computed_trust_level;
    }

    return success;
}

bool CloudDataRegistry::ListPublicDatasets(std::vector<network::PublicDatasetInfo>& out_datasets,
                                            const std::string& filter) {
    if (!datastream_client_ || !datastream_client_->IsConnected()) {
        return false;
    }

    return datastream_client_->ListPublicDatasets(out_datasets, filter);
}

DatasetHandle CloudDataRegistry::UsePublicDataset(const network::PublicDatasetInfo& public_dataset) {
    // If already cached in CyxCloud, use the cached version
    if (public_dataset.cached && !public_dataset.cached_dataset_id.empty()) {
        network::CloudDatasetInfo info;
        info.id = public_dataset.cached_dataset_id;
        info.name = public_dataset.name;
        info.trust_level = network::TrustLevel::Verified;
        return RegisterCloudDataset(info);
    }

    // Otherwise, we'd need to trigger a download
    // For now, return invalid handle
    return DatasetHandle();
}

bool CloudDataRegistry::CacheCloudDataset(DatasetHandle handle, const std::string& cache_path) {
    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) {
        return false;
    }

    // Create cache directory if needed
    std::filesystem::create_directories(cache_path);

    // TODO: Download all files from the dataset and cache locally
    // This would require streaming all data and writing to disk

    it->second.cache_path = cache_path;
    return true;
}

bool CloudDataRegistry::IsCachedLocally(DatasetHandle handle) const {
    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) {
        return false;
    }

    if (it->second.cache_path.empty()) {
        return false;
    }

    return std::filesystem::exists(it->second.cache_path);
}

void CloudDataRegistry::ClearCache(DatasetHandle handle) {
    auto it = cloud_datasets_.find(handle.GetId());
    if (it == cloud_datasets_.end()) return;

    if (!it->second.cache_path.empty() && std::filesystem::exists(it->second.cache_path)) {
        std::filesystem::remove_all(it->second.cache_path);
    }

    it->second.cache_path.clear();
}

// ============================================================================
// UnifiedDatasetHandle
// ============================================================================

UnifiedDatasetHandle::UnifiedDatasetHandle(DatasetHandle local_handle)
    : is_valid_(local_handle.IsValid())
    , is_local_(true)
    , local_handle_(local_handle) {
    if (is_valid_) {
        const auto& info = DataRegistry::Instance().GetDatasetInfo(local_handle);
        name_ = info.name;
    }
}

UnifiedDatasetHandle::UnifiedDatasetHandle(const network::CloudDatasetInfo& cloud_info)
    : is_valid_(true)
    , is_local_(false)
    , cloud_dataset_id_(cloud_info.id)
    , name_(cloud_info.name)
    , cloud_info_(cloud_info) {

    // Also register with CloudDataRegistry
    local_handle_ = CloudDataRegistry::Instance().RegisterCloudDataset(cloud_info);
}

DataLoaderConfig UnifiedDatasetHandle::GetLoaderConfig(int batch_size, bool shuffle) const {
    DataLoaderConfig config;
    config.batch_size = batch_size;
    config.shuffle = shuffle;

    if (is_local_) {
        // Standard local config
    } else {
        // Cloud config - batch size handled by streaming
        // The actual data will come from the streaming callback
    }

    return config;
}

} // namespace cyxwiz
