#pragma once

#include "../interfaces/i_data_provider.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <memory>

namespace cyxwiz::plugin {

class PluginDataLoaderRegistry {
public:
    static PluginDataLoaderRegistry& Instance();

    PluginDataLoaderRegistry(const PluginDataLoaderRegistry&) = delete;
    PluginDataLoaderRegistry& operator=(const PluginDataLoaderRegistry&) = delete;

    void Register(const std::string& plugin_id, IDataProvider* provider);
    void RemoveByPlugin(const std::string& plugin_id);

    // Query
    std::vector<PluginDataLoaderInfo> GetAllLoaders() const;
    bool HasLoaderForExtension(const std::string& extension) const;

    // Try all registered loaders for a file path
    std::shared_ptr<PluginDataset> TryLoadDataset(
        const std::string& file_path,
        const std::string& dataset_name
    ) const;

    size_t GetCount() const;

private:
    PluginDataLoaderRegistry() = default;

    struct LoaderEntry {
        std::string plugin_id;
        PluginDataLoaderInfo info;
        IDataProvider* provider = nullptr;   // non-owning
    };

    mutable std::mutex mutex_;
    std::map<std::string, LoaderEntry> loaders_;             // loader_id -> entry
    std::map<std::string, std::vector<std::string>> ext_map_;// ".ext" -> loader_ids
};

} // namespace cyxwiz::plugin
