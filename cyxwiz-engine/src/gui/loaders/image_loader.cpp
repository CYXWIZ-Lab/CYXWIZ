#include "image_loader.h"

#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/graph_compiler.h"  // PreprocessingDomain
#include "../../core/training_manager.h"
#include "../node_editor.h"  // gui::MLNode for RestoreFromRegistry

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

    // Cross-Apply cleanup: drop the prior image registration owned by
    // this node before probing again. Re-applying the same folder on
    // the same node keeps the same base dataset_name ("cat_dog"), so
    // only clearing when the name changes leaves the old datasets_
    // entry alive and forces DataRegistry::GenerateUniqueName() to
    // drift to "cat_dog_1". That in turn desynchronizes the node's
    // persisted dataset_name from the registry if the dialog closes
    // before the async completion callback updates the node params.
    auto& registry = cyxwiz::DataRegistry::Instance();
    if (!ctx.previous_dataset_name.empty()) {
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
                const std::string canonical_name = handle.GetName();

                cyxwiz::DataRegistry::ImageDatasetEntry img_entry;
                img_entry.folder_path = folder;
                img_entry.labels_csv  = csv;
                img_entry.layout      = layout;
                img_entry.num_images  = info.num_samples;
                img_entry.num_classes = info.num_classes;
                img_entry.class_names = info.class_names;
                // LoadImageFolder / LoadImageCSV may uniquify the
                // registry key. Use the DatasetHandle name here, not
                // DatasetInfo::name: for ImageCSVDataset the info name
                // is a display label like "cat_dog (CSV)", which is not
                // the actual key stored in DataRegistry::datasets_.
                reg.RegisterImageDataset(canonical_name, img_entry);

                size_t per_image = static_cast<size_t>(w) *
                                   static_cast<size_t>(h) *
                                   static_cast<size_t>(c) * sizeof(float);

                state->success      = true;
                state->backend      = 3;
                state->dataset_name = canonical_name;  // may have been uniquified
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

bool ImageLoader::RestoreFromRegistry(const std::string& name,
                                      const gui::MLNode& node,
                                      RestoreState& out) const {
    auto* entry = cyxwiz::DataRegistry::Instance().GetImageDatasetEntry(name);
    if (!entry) return false;

    // Read the persisted target_width / target_height / rgb params off
    // the node so the restored Memory tab shows the same byte estimate
    // the user saw at Apply. Falls back to the 224×224×3 default for
    // nodes that predate those params.
    int w = 224, h = 224, c = 3;
    auto tw_it  = node.parameters.find("target_width");
    auto th_it  = node.parameters.find("target_height");
    auto rgb_it = node.parameters.find("rgb");
    if (tw_it != node.parameters.end()) {
        try { int v = std::stoi(tw_it->second); if (v > 0) w = v; } catch (...) {}
    }
    if (th_it != node.parameters.end()) {
        try { int v = std::stoi(th_it->second); if (v > 0) h = v; } catch (...) {}
    }
    if (rgb_it != node.parameters.end()) {
        c = (rgb_it->second == "false" || rgb_it->second == "0") ? 1 : 3;
    }
    const size_t per_image = static_cast<size_t>(w) *
                             static_cast<size_t>(h) *
                             static_cast<size_t>(c) * sizeof(float);

    out.found   = true;
    out.rows    = static_cast<int64_t>(entry->num_images);
    out.cols    = 1;
    out.bytes   = entry->num_images * per_image;
    out.backend = 3;
    out.memory_is_estimate = true;
    out.status_message = "Loaded " + name + " (" +
        std::to_string(entry->num_images) + " images, " +
        std::to_string(entry->num_classes) + " classes)";
    return true;
}

CompletedLoadDescription ImageLoader::DescribeCompletedLoad(
    const AsyncLoadState& state) const {
    CompletedLoadDescription d;
    d.memory_is_estimate = true;
    d.node_description_suffix =
        std::to_string(state.rows) + " images, " +
        std::to_string(state.num_classes) + " classes";

    fs::path p(state.source_path);
    d.default_status_message = std::string("Loaded images from ") +
        p.filename().string();
    return d;
}

cyxwiz::PreprocessingDomain ImageLoader::Domain(
    const std::string& /*file_category*/) const {
    return cyxwiz::PreprocessingDomain::Image;
}

bool ImageLoader::LaunchTraining(
    cyxwiz::TrainingConfiguration config,
    const std::string& dataset_name,
    const std::string& /*label_column*/,
    int epochs,
    int batch_size,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback) {
    auto* entry = cyxwiz::DataRegistry::Instance().GetImageDatasetEntry(dataset_name);
    if (!entry) {
        spdlog::error("ImageLoader: image dataset '{}' is registered but entry "
                      "could not be retrieved", dataset_name);
        return false;
    }
    spdlog::info("ImageLoader: Starting image training: dataset={}, epochs={}, "
                 "batch_size={}, {} images, {} classes",
                 dataset_name, epochs, batch_size,
                 entry->num_images, entry->num_classes);
    return cyxwiz::TrainingManager::Instance().StartTrainingImage(
        std::move(config), *entry, epochs, batch_size, plot_panel,
        std::move(node_editor_callback));
}

std::vector<ParamSchema> ImageLoader::NodeParams() const {
    return {
        {"image_layout",   "0",    "0=ClassSubdirs, 1=FlatWithCSV"},
        {"labels_csv",     "",     "Labels CSV path (FlatWithCSV mode)"},
        {"target_width",   "224",  "Resize target width"},
        {"target_height",  "224",  "Resize target height"},
        {"normalize",      "true", "Normalize to [0,1]"},
        {"rgb",            "true", "RGB (true) vs grayscale (false)"},
    };
}

SyntheticBatch ImageLoader::MakeSynthetic(
    const cyxwiz::TrainingConfiguration& /*config*/, uint32_t /*seed*/) const {
    return SyntheticBatch{};
}

}  // namespace cyxwiz::loaders
