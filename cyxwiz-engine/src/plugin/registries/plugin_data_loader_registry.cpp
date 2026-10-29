#include "plugin_data_loader_registry.h"

#include <arrow/c/bridge.h>
#include <arrow/device.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <filesystem>

namespace cyxwiz::plugin {

namespace {

std::string NormalizeExtension(std::string extension) {
    if (!extension.empty() && extension.front() == '.') extension.erase(0, 1);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension;
}

// The plugin's reader, holding the plugin's use until it is destroyed: the
// stream's callbacks live in the plugin's library. Each batch is copied into
// the Engine's memory and the plugin's released at once, so data read from
// a plugin outlives the plugin (unloading it frees its library).
class PluginStreamReader final : public arrow::RecordBatchReader {
public:
    PluginStreamReader(std::shared_ptr<arrow::RecordBatchReader> inner, std::shared_ptr<void> use)
        : inner_(std::move(inner)), use_(std::move(use)) {}
    ~PluginStreamReader() override { inner_.reset(); }

    std::shared_ptr<arrow::Schema> schema() const override { return inner_->schema(); }
    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
        std::shared_ptr<arrow::RecordBatch> plugin_batch;
        ARROW_RETURN_NOT_OK(inner_->ReadNext(&plugin_batch));
        if (!plugin_batch) {
            batch->reset();
            return arrow::Status::OK();
        }
        ARROW_ASSIGN_OR_RAISE(*batch, plugin_batch->CopyTo(arrow::default_cpu_memory_manager()));
        return arrow::Status::OK();
    }
    arrow::Status Close() override { return inner_->Close(); }

private:
    std::shared_ptr<arrow::RecordBatchReader> inner_;
    std::shared_ptr<void> use_;  // released after inner_ (declared after it)
};

}  // namespace

PluginDataLoaderRegistry& PluginDataLoaderRegistry::Instance() {
    static PluginDataLoaderRegistry instance;
    return instance;
}

void PluginDataLoaderRegistry::Register(const std::string& plugin_id, IDataProvider* provider) {
    if (!provider) return;

    auto loaders = provider->GetLoaders();
    std::lock_guard lock(mutex_);
    for (auto& info : loaders) {
        if (info.loader_id.empty() || info.format_name.empty() || info.extensions.empty()) {
            spdlog::warn("PluginDataLoaderRegistry: {} declares a loader without an id, a name or "
                         "extensions; skipped", plugin_id);
            continue;
        }
        if (const auto it = loaders_.find(info.loader_id); it != loaders_.end()) {
            if (it->second.loader.plugin_id != plugin_id || it->second.provider != provider)
                spdlog::warn("PluginDataLoaderRegistry: loader {} of {} is already registered by {}; skipped",
                             info.loader_id, plugin_id, it->second.loader.plugin_id);
            continue;
        }
        for (auto& extension : info.extensions) extension = NormalizeExtension(extension);
        spdlog::info("PluginDataLoaderRegistry: registered '{}' ({}) from {}",
                     info.format_name, info.loader_id, plugin_id);
        // The key first: the right side (which moves info) is evaluated
        // before the subscript (C++17), which would read an empty id.
        const std::string loader_id = info.loader_id;
        loaders_[loader_id] = Entry{Loader{plugin_id, std::move(info)}, provider};
        ++generation_;
    }
}

void PluginDataLoaderRegistry::RemoveByPlugin(const std::string& plugin_id) {
    {
        std::lock_guard lock(mutex_);
        for (auto it = loaders_.begin(); it != loaders_.end();) {
            if (it->second.loader.plugin_id == plugin_id) {
                it = loaders_.erase(it);
                ++generation_;
            } else {
                ++it;
            }
        }
    }
    if (uses_.InUse(plugin_id))
        spdlog::info("PluginDataLoaderRegistry: waiting for a dataset of {} to finish reading", plugin_id);
    uses_.WaitUntilUnused(plugin_id);
}

std::vector<PluginDataLoaderRegistry::Loader> PluginDataLoaderRegistry::Loaders() const {
    std::vector<Loader> result;
    {
        std::lock_guard lock(mutex_);
        result.reserve(loaders_.size());
        for (const auto& [_, entry] : loaders_) result.push_back(entry.loader);
    }
    std::sort(result.begin(), result.end(), [](const Loader& a, const Loader& b) {
        return a.info.format_name < b.info.format_name;
    });
    return result;
}

bool PluginDataLoaderRegistry::Find(const std::string& loader_id, Loader& out) const {
    std::lock_guard lock(mutex_);
    const auto it = loaders_.find(loader_id);
    if (it == loaders_.end()) return false;
    out = it->second.loader;
    return true;
}

std::string PluginDataLoaderRegistry::LoaderForFile(const std::string& file_path) const {
    const std::string extension = NormalizeExtension(std::filesystem::path(file_path).extension().string());
    if (extension.empty()) return {};

    struct Candidate {
        std::string loader_id;
        IDataProvider* provider;
        std::shared_ptr<void> use;
    };
    std::vector<Candidate> candidates;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [id, entry] : loaders_) {
            const auto& extensions = entry.loader.info.extensions;
            if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end())
                candidates.push_back({id, entry.provider, uses_.Acquire(entry.loader.plugin_id)});
        }
    }
    // CanLoad may read the file: outside the lock.
    for (const auto& candidate : candidates) {
        try {
            if (candidate.provider->CanLoad(candidate.loader_id, file_path)) return candidate.loader_id;
        } catch (const std::exception& e) {
            spdlog::warn("PluginDataLoaderRegistry: {} CanLoad({}) threw: {}",
                         candidate.loader_id, file_path, e.what());
        }
    }
    return {};
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> PluginDataLoaderRegistry::Open(
    const std::string& loader_id, const std::string& file_path) const {
    IDataProvider* provider = nullptr;
    std::string format_name;
    std::shared_ptr<void> use;
    {
        std::lock_guard lock(mutex_);
        const auto it = loaders_.find(loader_id);
        if (it == loaders_.end())
            return arrow::Status::Invalid("The format '", loader_id, "' is not installed (its plugin is not loaded)");
        provider = it->second.provider;
        format_name = it->second.loader.info.format_name;
        use = uses_.Acquire(it->second.loader.plugin_id);
    }

    ArrowArrayStream stream{};
    std::string error;
    bool opened = false;
    try {
        opened = provider->OpenDataset(loader_id, file_path, &stream, error);
    } catch (const std::exception& e) {
        error = std::string("the plugin threw: ") + e.what();
    }
    if (!opened || !stream.release) {
        if (stream.release) stream.release(&stream);
        if (error.empty()) error = opened ? "the plugin returned an empty stream" : "the plugin gave no reason";
        return arrow::Status::IOError(format_name, " could not open ", file_path, ": ", error);
    }
    auto reader = arrow::ImportRecordBatchReader(&stream);
    if (!reader.ok())
        return arrow::Status::IOError(format_name, " gave data the Engine cannot read: ",
                                      reader.status().message());
    return std::shared_ptr<arrow::RecordBatchReader>(
        std::make_shared<PluginStreamReader>(std::move(reader).ValueUnsafe(), std::move(use)));
}

}  // namespace cyxwiz::plugin
