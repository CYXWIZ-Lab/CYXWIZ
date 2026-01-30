#pragma once

#include <string>
#include <vector>
#include <memory>

namespace cyxwiz::plugin {

struct PluginDataLoaderInfo {
    std::string loader_id;              // Unique ID
    std::string format_name;            // Display name (e.g. "Parquet Dataset")
    std::vector<std::string> extensions;// e.g. {".parquet", ".pq"}
    std::string description;
};

// Opaque dataset handle returned by plugin loaders.
// Phase 3 will bridge this to the engine's Dataset class.
struct PluginDataset {
    std::string name;
    size_t num_samples = 0;
    std::vector<std::string> column_names;
    // Plugin-owned opaque pointer; plugin is responsible for lifetime.
    void* opaque = nullptr;
};

class IDataProvider {
public:
    virtual ~IDataProvider() = default;

    virtual std::vector<PluginDataLoaderInfo> GetLoaders() = 0;

    virtual bool CanLoad(const std::string& file_path) = 0;

    // Load dataset from file. Returns nullptr on failure.
    virtual std::shared_ptr<PluginDataset> LoadDataset(
        const std::string& file_path,
        const std::string& dataset_name
    ) = 0;
};

} // namespace cyxwiz::plugin
