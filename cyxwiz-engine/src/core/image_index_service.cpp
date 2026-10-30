#include "image_index_service.h"

#include "arrow_dataset.h"
#include "async_task_manager.h"
#include "image_index.h"
#include "prepared_image_set.h"

#include <arrow/api.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>

namespace cyxwiz {

namespace {

struct Build {
    uint64_t generation = 0;
    std::shared_ptr<std::atomic<bool>> cancel;
    ImageIndexStatus status;
};

std::mutex g_mutex;
std::map<std::string, Build> g_builds;
uint64_t g_generation = 0;

struct Quality {
    std::shared_ptr<const ImageQualityAnalysis> analysis;
    ImageQualityVerdict verdict;
};
std::map<std::string, Quality> g_quality;  // by dataset

// The index with the dataset's measured quality, when there is one.
std::shared_ptr<arrow::Table> WithQuality(const std::string& dataset, std::shared_ptr<arrow::Table> table) {
    Quality quality;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_quality.find(dataset);
        if (it == g_quality.end()) return table;
        quality = it->second;
    }
    return AddImageQualityColumns(table, *quality.analysis, quality.verdict);
}

// Updates the status when this build is still the current one.
template <typename F>
void Update(const std::string& dataset, uint64_t generation, F&& f) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_builds.find(dataset);
    if (it != g_builds.end() && it->second.generation == generation) f(it->second.status);
}

bool Current(const std::string& dataset, uint64_t generation) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_builds.find(dataset);
    return it != g_builds.end() && it->second.generation == generation;
}

}  // namespace

void BuildImageIndexInBackground(const std::string& dataset, const DataRegistry::ImageDatasetEntry& entry,
                                 const std::vector<std::string>& label_names) {
    auto cancel = std::make_shared<std::atomic<bool>>(false);
    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto& build = g_builds[dataset];
        if (build.cancel) build.cancel->store(true);
        generation = ++g_generation;
        build.generation = generation;
        build.cancel = cancel;
        build.status = {ImageIndexStatus::State::Building, 0, entry.num_images, {}};
    }
    AsyncTaskManager::Instance().RunAsync(
        "Indexing images " + dataset,
        [dataset, entry, label_names, cancel, generation](LambdaTask& task) {
            const auto started = std::chrono::steady_clock::now();
            const ImageSetListing listing = ListImageSet(entry);
            auto table = BuildImageIndex(
                listing, label_names,
                [&](size_t done, size_t total) {
                    if (task.IsCancelRequested()) cancel->store(true);
                    Update(dataset, generation, [&](ImageIndexStatus& s) {
                        s.done = done;
                        s.total = total;
                    });
                    if (total > 0) task.ReportProgress(static_cast<float>(done) / total, std::to_string(done) + " of " + std::to_string(total));
                },
                cancel.get());
            if (!table) {
                Update(dataset, generation, [](ImageIndexStatus& s) { s.state = ImageIndexStatus::State::None; });
                return;  // cancelled: a newer build or an unload took over
            }
            // Register only while this build is current and the images are still loaded.
            if (!Current(dataset, generation) || !DataRegistry::Instance().GetImageDatasetEntry(dataset)) return;
            table = WithQuality(dataset, table);
            DataRegistry::Instance().RegisterArrowTable(table, DataRegistry::ImageIndexName(dataset));
            Update(dataset, generation, [](ImageIndexStatus& s) { s.state = ImageIndexStatus::State::Ready; });
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            spdlog::info("Image index '{}': {} images in {:.1f} s", DataRegistry::ImageIndexName(dataset),
                         table->num_rows(), seconds);
        },
        nullptr,
        [dataset, generation](bool success, const std::string& error) {
            if (success) return;
            spdlog::error("Image index for '{}' failed: {}", dataset, error);
            Update(dataset, generation, [&](ImageIndexStatus& s) {
                s.state = ImageIndexStatus::State::Failed;
                s.error = error;
            });
        });
}

void AttachImageQuality(const std::string& dataset, std::shared_ptr<const ImageQualityAnalysis> analysis,
                        const ImageQualityVerdict& verdict) {
    if (!analysis) return;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_quality[dataset] = {std::move(analysis), verdict};
    }
    auto& registry = DataRegistry::Instance();
    const std::string name = DataRegistry::ImageIndexName(dataset);
    const auto index = registry.GetArrowDataset(name);
    if (!index) return;  // the build in progress adds them
    registry.RegisterArrowTable(WithQuality(dataset, index->GetArrowTable()), name);
}

bool RenameImageClasses(const std::string& dataset, const std::vector<std::string>& display_names) {
    auto& registry = DataRegistry::Instance();
    if (!registry.SetImageClassDisplayNames(dataset, display_names)) return false;
    const DataRegistry::ImageDatasetEntry* entry = registry.GetImageDatasetEntry(dataset);
    if (!entry) return false;
    BuildImageIndexInBackground(dataset, *entry, display_names);
    return true;
}

ImageIndexStatus GetImageIndexStatus(const std::string& dataset) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_builds.find(dataset);
    return it == g_builds.end() ? ImageIndexStatus{} : it->second.status;
}

}  // namespace cyxwiz
