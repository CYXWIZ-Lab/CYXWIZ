#include "plugin_data_loader_registry.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <algorithm>
#include <unordered_set>

namespace cyxwiz::plugin {

PluginDataLoaderRegistry& PluginDataLoaderRegistry::Instance() {
    static PluginDataLoaderRegistry instance;
    return instance;
}

void PluginDataLoaderRegistry::Register(const std::string& plugin_id, IDataProvider* provider) {
    if (!provider) return;

    auto loaders = provider->GetLoaders();
    std::lock_guard lock(mutex_);

    for (auto& info : loaders) {
        if (loaders_.count(info.loader_id)) {
            spdlog::warn("PluginDataLoaderRegistry: Duplicate loader {}, skipping", info.loader_id);
            continue;
        }

        // Build extension map
        for (const auto& ext : info.extensions) {
            std::string lower_ext = ext;
            std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);
            ext_map_[lower_ext].push_back(info.loader_id);
        }

        spdlog::info("PluginDataLoaderRegistry: Registered loader '{}' ({})",
                     info.format_name, info.loader_id);
        loaders_[info.loader_id] = LoaderEntry{plugin_id, std::move(info), provider};
    }
}

void PluginDataLoaderRegistry::RemoveByPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);

    // Collect loader_ids to remove
    std::vector<std::string> to_remove;
    for (const auto& [id, entry] : loaders_) {
        if (entry.plugin_id == plugin_id)
            to_remove.push_back(id);
    }

    // Remove from extension map
    std::unordered_set<std::string> to_remove_set(to_remove.begin(), to_remove.end());
    for (auto& [ext, ids] : ext_map_) {
        ids.erase(std::remove_if(ids.begin(), ids.end(), [&](const std::string& id) {
            return to_remove_set.count(id) > 0;
        }), ids.end());
    }

    // Remove empty extension entries
    for (auto it = ext_map_.begin(); it != ext_map_.end();) {
        if (it->second.empty()) it = ext_map_.erase(it);
        else ++it;
    }

    // Remove loaders
    for (const auto& id : to_remove)
        loaders_.erase(id);
}

std::vector<PluginDataLoaderInfo> PluginDataLoaderRegistry::GetAllLoaders() const {
    std::lock_guard lock(mutex_);
    std::vector<PluginDataLoaderInfo> result;
    result.reserve(loaders_.size());
    for (const auto& [_, entry] : loaders_)
        result.push_back(entry.info);
    return result;
}

bool PluginDataLoaderRegistry::HasLoaderForExtension(const std::string& extension) const {
    std::string lower = extension;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    std::lock_guard lock(mutex_);
    return ext_map_.count(lower) > 0;
}

std::shared_ptr<PluginDataset> PluginDataLoaderRegistry::TryLoadDataset(
    const std::string& file_path,
    const std::string& dataset_name
) const {
    // Get extension
    std::string ext = std::filesystem::path(file_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Snapshot providers under lock, then try outside lock (I/O can be slow)
    std::vector<std::pair<std::string, IDataProvider*>> providers_to_try;
    {
        std::lock_guard lock(mutex_);
        auto ext_it = ext_map_.find(ext);
        if (ext_it == ext_map_.end()) return nullptr;

        for (const auto& loader_id : ext_it->second) {
            auto it = loaders_.find(loader_id);
            if (it != loaders_.end() && it->second.provider)
                providers_to_try.emplace_back(loader_id, it->second.provider);
        }
    }

    for (const auto& [loader_id, provider] : providers_to_try) {
        try {
            if (provider->CanLoad(file_path)) {
                auto result = provider->LoadDataset(file_path, dataset_name);
                if (result) return result;
            }
        } catch (const std::exception& e) {
            spdlog::error("PluginDataLoaderRegistry: Loader {} failed for {}: {}",
                         loader_id, file_path, e.what());
        }
    }
    return nullptr;
}

size_t PluginDataLoaderRegistry::GetCount() const {
    std::lock_guard lock(mutex_);
    return loaders_.size();
}

} // namespace cyxwiz::plugin
