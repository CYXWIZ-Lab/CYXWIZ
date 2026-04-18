#include "image_loader.h"

#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"

#include <spdlog/spdlog.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace cyxwiz::loaders {

// ImageLayout values must match DataInputDialog::ImageLayout int
// casts (0 = ClassSubdirs, 1 = FlatWithCSV). Captured as local
// constants so the dispatch inside the worker reads clearly without
// pulling the dialog's private enum across module boundaries.
namespace {
constexpr int kLayoutClassSubdirs = 0;
constexpr int kLayoutFlatWithCSV  = 1;
}  // namespace

bool ImageLoader::ValidateApplyContext(const ApplyContext& ctx,
                                       std::string& err) const {
    if (ctx.source_path.empty()) {
        err = "Image load needs a folder path";
        return false;
    }
    if (ctx.dataset_name.empty()) {
        err = "Dataset name is empty";
        return false;
    }
    // FlatWithCSV requires a labels CSV — cheap UI-thread check so the
    // user sees the error immediately, not after a worker round-trip.
    if (ctx.image_layout == kLayoutFlatWithCSV && ctx.image_labels_csv.empty()) {
        err = "Flat folder layout requires a Labels CSV - "
              "pick one or switch to 'Class subdirectories'";
        return false;
    }
    return true;
}

uint64_t ImageLoader::LaunchAsyncLoad(const ApplyContext& ctx,
                                      std::shared_ptr<AsyncLoadState> state) {
    if (!state) return 0;

    // Cross-Apply cleanup: if the user re-Applied with a different
    // folder, drop the prior image entry so the registry doesn't leak.
    auto& registry = cyxwiz::DataRegistry::Instance();
    if (!ctx.previous_dataset_name.empty() &&
        ctx.previous_dataset_name != ctx.dataset_name) {
        registry.UnregisterImageDataset(ctx.previous_dataset_name);
    }

    // Snapshot everything by value for the worker.
    const std::string folder = ctx.source_path;
    const std::string csv    = ctx.image_labels_csv;
    const std::string name   = ctx.dataset_name;
    const int layout         = ctx.image_layout;
    const int w              = ctx.image_width  > 0 ? ctx.image_width  : 224;
    const int h              = ctx.image_height > 0 ? ctx.image_height : 224;
    const int c              = ctx.image_channels > 0 ? ctx.image_channels : 3;

    state->dataset_name = name;
    state->source_path  = folder;

    auto& mgr = cyxwiz::AsyncTaskManager::Instance();
    return mgr.RunAsync(
        "Loading images " + name,
        [folder, csv, name, layout, w, h, c, state]
        (cyxwiz::LambdaTask& task) {
            try {
                task.ReportProgress(0.1f, "Scanning folder");

                auto& reg = cyxwiz::DataRegistry::Instance();
                // Clear any stale Arrow / Parquet entry under the
                // same name so IsArrowDataset doesn't mis-route
                // training dispatch if the user toggled categories.
                reg.UnregisterTabularDataset(name);

                cyxwiz::DatasetHandle handle;
                if (layout == kLayoutFlatWithCSV) {
                    handle = reg.LoadImageCSV(folder, csv, name);
                } else {
                    handle = reg.LoadImageFolder(folder, name);
                }

                if (!handle.IsValid()) {
                    state->success = false;
                    state->message = "Failed to load image folder - "
                        "expected either class subdirectories "
                        "(cats/, dogs/, ...) OR a flat folder with "
                        "a Labels CSV selected above";
                    state->done.store(true);
                    return;
                }

                task.ReportProgress(0.9f, "Registering dataset");

                auto info = handle.GetInfo();

                cyxwiz::DataRegistry::ImageDatasetEntry img_entry;
                img_entry.folder_path = folder;
                img_entry.labels_csv  = csv;
                img_entry.layout      = layout;
                img_entry.num_images  = info.num_samples;
                img_entry.num_classes = info.num_classes;
                img_entry.class_names = info.class_names;
                // LoadImageFolder / LoadImageCSV may uniquify the name
                // if another handle is live; use info.name as the
                // canonical registration key.
                reg.RegisterImageDataset(info.name, img_entry);

                size_t per_image = static_cast<size_t>(w) *
                                   static_cast<size_t>(h) *
                                   static_cast<size_t>(c) * sizeof(float);

                state->success      = true;
                state->backend      = 3;
                state->dataset_name = info.name;  // may have been uniquified
                state->rows         = static_cast<int64_t>(info.num_samples);
                state->cols         = 1;
                state->bytes        = info.num_samples * per_image;
                state->num_classes  = info.num_classes;
                state->message      = "Loaded " +
                    std::to_string(info.num_samples) + " images (" +
                    std::to_string(info.num_classes) + " classes) from " +
                    fs::path(folder).filename().string();
            } catch (const std::exception& e) {
                state->success = false;
                state->message = std::string("Error loading images: ") + e.what();
                spdlog::error("ImageLoader async: {}", state->message);
            }
            // Publish barrier — must be set LAST.
            state->done.store(true);
        });
}

bool ImageLoader::IsRegistered(const std::string& name) const {
    return cyxwiz::DataRegistry::Instance().IsImageDataset(name);
}

void ImageLoader::Unregister(const std::string& name) {
    cyxwiz::DataRegistry::Instance().UnregisterImageDataset(name);
}

}  // namespace cyxwiz::loaders
