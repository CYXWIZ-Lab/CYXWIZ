#include "plugin_data_loader_registry.h"

#include "../security/safe_execute.h"

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
// a plugin outlives the plugin (unloading it frees its library). Every call
// into the plugin (get_next, the batch's and the stream's release) runs in
// the crash guard: a reader that crashes fails the read, not the Engine.
class PluginStreamReader final : public arrow::RecordBatchReader {
public:
    PluginStreamReader(std::shared_ptr<arrow::RecordBatchReader> inner, std::shared_ptr<void> use,
                       std::string plugin_id)
        : inner_(std::move(inner)), use_(std::move(use)), plugin_id_(std::move(plugin_id)) {}
    ~PluginStreamReader() override {
        security::SafeExecute(plugin_id_, "release", [&] { inner_.reset(); });
    }

    std::shared_ptr<arrow::Schema> schema() const override { return inner_->schema(); }
    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
        batch->reset();
        if (crashed_) return arrow::Status::IOError(crash_message_);
        arrow::Status status;
        std::shared_ptr<arrow::RecordBatch> copy;
        const auto run = security::SafeExecute(plugin_id_, "get_next", [&] {
            std::shared_ptr<arrow::RecordBatch> plugin_batch;
            status = inner_->ReadNext(&plugin_batch);
            if (!status.ok() || !plugin_batch) return;
            auto copied = plugin_batch->CopyTo(arrow::default_cpu_memory_manager());
            plugin_batch.reset();  // the plugin's release, inside the guard
            if (copied.ok()) copy = std::move(copied).ValueUnsafe();
            else status = copied.status();
        });
        if (!run.success) {
            crashed_ = true;
            crash_message_ = run.error_message + "; the Engine kept running";
            return arrow::Status::IOError(crash_message_);
        }
        ARROW_RETURN_NOT_OK(status);
        *batch = std::move(copy);
        return arrow::Status::OK();
    }
    arrow::Status Close() override {
        arrow::Status status;
        const auto run = security::SafeExecute(plugin_id_, "Close", [&] { status = inner_->Close(); });
        if (!run.success) crashed_ = true;
        return run.success ? status : arrow::Status::IOError(run.error_message);
    }

private:
    std::shared_ptr<arrow::RecordBatchReader> inner_;
    std::shared_ptr<void> use_;  // released after inner_ (declared after it)
    std::string plugin_id_;
    bool crashed_ = false;
    std::string crash_message_;
};

}  // namespace

PluginDataLoaderRegistry& PluginDataLoaderRegistry::Instance() {
    static PluginDataLoaderRegistry instance;
    return instance;
}

void PluginDataLoaderRegistry::Register(const std::string& plugin_id, const std::string& plugin_version,
                                        IDataProvider* provider) {
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
        loaders_[loader_id] = Entry{Loader{plugin_id, plugin_version, std::move(info)}, provider};
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
        std::string plugin_id;
        IDataProvider* provider;
        std::shared_ptr<void> use;
    };
    std::vector<Candidate> candidates;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [id, entry] : loaders_) {
            const auto& extensions = entry.loader.info.extensions;
            if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end())
                candidates.push_back({id, entry.loader.plugin_id, entry.provider, uses_.Acquire(entry.loader.plugin_id)});
        }
    }
    // CanLoad may read the file: outside the lock, in the crash guard (a
    // loader that throws or crashes does not take the file).
    for (const auto& candidate : candidates) {
        bool accepts = false;
        const auto run = security::SafeExecute(candidate.plugin_id, "CanLoad", [&] {
            accepts = candidate.provider->CanLoad(candidate.loader_id, file_path);
        });
        if (run.success && accepts) return candidate.loader_id;
        if (!run.success)
            spdlog::warn("PluginDataLoaderRegistry: {} CanLoad({}): {}", candidate.loader_id, file_path,
                         run.error_message);
    }
    return {};
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> PluginDataLoaderRegistry::Open(
    const std::string& loader_id, const std::string& file_path) const {
    IDataProvider* provider = nullptr;
    std::string format_name;
    std::string plugin_id;
    std::shared_ptr<void> use;
    {
        std::lock_guard lock(mutex_);
        const auto it = loaders_.find(loader_id);
        if (it == loaders_.end())
            return arrow::Status::Invalid("The format '", loader_id, "' is not installed (its plugin is not loaded)");
        provider = it->second.provider;
        format_name = it->second.loader.info.format_name;
        plugin_id = it->second.loader.plugin_id;
        use = uses_.Acquire(it->second.loader.plugin_id);
    }

    // OpenDataset and the stream's get_schema (read by the import) run in the
    // crash guard; after a crash the stream is not released (its state is
    // unknown) and the Engine keeps running.
    ArrowArrayStream stream{};
    std::string error;
    bool opened = false;
    bool imported = false;
    arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> reader = arrow::Status::UnknownError("not opened");
    const auto run = security::SafeExecute(plugin_id, "OpenDataset", [&] {
        try {
            opened = provider->OpenDataset(loader_id, file_path, &stream, error);
        } catch (const std::exception& e) {
            error = std::string("the plugin threw: ") + e.what();
        }
        if (!opened || !stream.release) {
            if (stream.release) stream.release(&stream);
            return;
        }
        imported = true;
        reader = arrow::ImportRecordBatchReader(&stream);
    });
    if (!run.success)
        return arrow::Status::IOError(format_name, " could not open ", file_path, ": ", run.error_message,
                                      "; the Engine kept running");
    if (!imported) {
        if (error.empty()) error = opened ? "the plugin returned an empty stream" : "the plugin gave no reason";
        return arrow::Status::IOError(format_name, " could not open ", file_path, ": ", error);
    }
    if (!reader.ok())
        return arrow::Status::IOError(format_name, " gave data the Engine cannot read: ",
                                      reader.status().message());
    return std::shared_ptr<arrow::RecordBatchReader>(
        std::make_shared<PluginStreamReader>(std::move(reader).ValueUnsafe(), std::move(use), plugin_id));
}

}  // namespace cyxwiz::plugin
