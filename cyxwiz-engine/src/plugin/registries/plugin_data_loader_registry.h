#pragma once

#include "../interfaces/i_data_provider.h"
#include "plugin_use_counter.h"

#include <arrow/record_batch.h>
#include <arrow/result.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

// The dataset formats plugins add (IDataProvider, TOFIX181). Data Input lists
// them and reads their files through Open, as Arrow record batches.
class PluginDataLoaderRegistry {
public:
    static PluginDataLoaderRegistry& Instance();

    PluginDataLoaderRegistry(const PluginDataLoaderRegistry&) = delete;
    PluginDataLoaderRegistry& operator=(const PluginDataLoaderRegistry&) = delete;

    void Register(const std::string& plugin_id, IDataProvider* provider);
    // Waits until no stream of the plugin is being read any more.
    void RemoveByPlugin(const std::string& plugin_id);

    struct Loader {
        std::string plugin_id;
        PluginDataLoaderInfo info;  // extensions lower case, without the dot
    };
    std::vector<Loader> Loaders() const;  // ordered by format name
    bool Find(const std::string& loader_id, Loader& out) const;

    // The loader for a file: its extension matches and the loader's CanLoad
    // accepts it. Empty when none does.
    std::string LoaderForFile(const std::string& file_path) const;

    // Opens the file with the loader. The reader streams the plugin's
    // batches; the plugin stays loaded until the reader is destroyed.
    arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> Open(
        const std::string& loader_id, const std::string& file_path) const;

    // Changes whenever a loader is added or removed.
    uint64_t Generation() const { return generation_.load(); }

private:
    PluginDataLoaderRegistry() = default;

    struct Entry {
        Loader loader;
        IDataProvider* provider = nullptr;  // non-owning
    };

    mutable std::mutex mutex_;
    std::map<std::string, Entry> loaders_;  // loader_id -> entry
    std::atomic<uint64_t> generation_{0};
    mutable PluginUseCounter uses_;
};

} // namespace cyxwiz::plugin
