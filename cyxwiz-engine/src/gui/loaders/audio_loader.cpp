#include "audio_loader.h"

#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/formats/audio_dataset.h"
#include "../../core/training_manager.h"

#include <spdlog/spdlog.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace cyxwiz::loaders {

bool AudioLoader::ValidateApplyContext(const ApplyContext& ctx,
                                       std::string& err) const {
    if (ctx.source_path.empty()) {
        err = "Audio load needs a folder path";
        return false;
    }
    if (ctx.dataset_name.empty()) {
        err = "Dataset name is empty";
        return false;
    }
    // Flat layout requires a labels CSV — cheap UI-thread precheck.
    if (!ctx.audio_labeled_subdirs && ctx.audio_labels_csv.empty()) {
        err = "Flat folder layout requires a Labels CSV - "
              "pick one or switch to 'Class subdirectories'";
        return false;
    }
    return true;
}

uint64_t AudioLoader::LaunchAsyncLoad(const ApplyContext& ctx,
                                      std::shared_ptr<AsyncLoadState> state) {
    if (!state) return 0;

    // Cross-Apply cleanup for a changed folder.
    auto& registry = cyxwiz::DataRegistry::Instance();
    if (!ctx.previous_dataset_name.empty() &&
        ctx.previous_dataset_name != ctx.dataset_name) {
        registry.UnregisterAudioDataset(ctx.previous_dataset_name);
    }

    // Build the registry entry skeleton on the UI thread; the worker
    // populates runtime fields (num_samples / num_classes / feature
    // shape) from the AudioDataset probe. Captured by value — the
    // dialog can close mid-load without racing.
    cyxwiz::DataRegistry::AudioDatasetEntry entry;
    entry.folder_path     = ctx.source_path;
    entry.labeled_subdirs = ctx.audio_labeled_subdirs;
    if (!ctx.audio_labeled_subdirs) {
        entry.csv_path     = ctx.audio_labels_csv;
        entry.filename_col = ctx.audio_filename_col;
        entry.label_col    = ctx.audio_label_col;
    }
    entry.feature_type = ctx.audio_feature_type;
    entry.target_sr    = ctx.audio_target_sr;
    entry.max_duration = ctx.audio_max_duration > 0.0f ? ctx.audio_max_duration : 5.0f;
    entry.n_fft        = ctx.audio_n_fft;
    entry.hop_length   = ctx.audio_hop_length;
    entry.n_mels       = ctx.audio_n_mels;
    entry.n_mfcc       = ctx.audio_n_mfcc;

    const std::string folder = ctx.source_path;
    const std::string name   = ctx.dataset_name;

    state->dataset_name = name;
    state->source_path  = folder;

    auto& mgr = cyxwiz::AsyncTaskManager::Instance();
    return mgr.RunAsync(
        "Loading audio " + name,
        [folder, name, entry, state]
        (cyxwiz::LambdaTask& task) {
            try {
                task.ReportProgress(0.1f, "Scanning folder");

                auto& reg = cyxwiz::DataRegistry::Instance();
                // Clear stale cross-category entries under the same
                // name so IsArrowDataset / IsAudioDataset don't mis-
                // route training dispatch.
                reg.UnregisterTabularDataset(name);
                reg.UnregisterAudioDataset(name);

                cyxwiz::AudioDatasetConfig probe_cfg;
                probe_cfg.feature_type    = cyxwiz::AudioDatasetConfig::FeatureType::MelSpectrogram;
                probe_cfg.target_sr       = entry.target_sr;
                probe_cfg.max_duration    = entry.max_duration;
                probe_cfg.labeled_subdirs = entry.labeled_subdirs;
                probe_cfg.n_fft           = entry.n_fft;
                probe_cfg.hop_length      = entry.hop_length;
                probe_cfg.n_mels          = entry.n_mels;
                probe_cfg.csv_path        = entry.csv_path;
                probe_cfg.filename_col    = entry.filename_col;
                probe_cfg.label_col       = entry.label_col;

                cyxwiz::AudioDataset probe(folder, probe_cfg);
                auto info = probe.GetInfo();

                cyxwiz::DataRegistry::AudioDatasetEntry final_entry = entry;
                final_entry.num_samples = info.num_samples;
                final_entry.num_classes = info.num_classes;
                final_entry.class_names = probe.GetClassNames();
                if (info.shape.size() >= 2) {
                    final_entry.feature_rows = static_cast<int>(info.shape[0]);
                    final_entry.feature_cols = static_cast<int>(info.shape[1]);
                }

                task.ReportProgress(0.9f, "Registering dataset");
                reg.RegisterAudioDataset(name, final_entry);

                // Estimate bytes from the probed feature shape; fall
                // back to n_mels * 313 (the default MelSpectrogram
                // frame count at target_sr=16kHz, duration=5s) if the
                // probe didn't report a 2D shape.
                size_t per_sample = (final_entry.feature_rows > 0 && final_entry.feature_cols > 0)
                    ? static_cast<size_t>(final_entry.feature_rows) *
                      static_cast<size_t>(final_entry.feature_cols) * sizeof(float)
                    : static_cast<size_t>(final_entry.n_mels) * 313 * sizeof(float);

                state->success      = true;
                state->backend      = 4;
                state->rows         = static_cast<int64_t>(info.num_samples);
                state->cols         = 1;
                state->bytes        = info.num_samples * per_sample;
                state->num_classes  = info.num_classes;
                state->message      = "Loaded " +
                    std::to_string(info.num_samples) + " audio files (" +
                    std::to_string(info.num_classes) + " classes) from " +
                    fs::path(folder).filename().string();
            } catch (const std::exception& e) {
                state->success = false;
                state->message = std::string("Error loading audio: ") + e.what();
                spdlog::error("AudioLoader async: {}", state->message);
            }
            // Publish barrier — must be set LAST.
            state->done.store(true);
        });
}

bool AudioLoader::IsRegistered(const std::string& name) const {
    return cyxwiz::DataRegistry::Instance().IsAudioDataset(name);
}

void AudioLoader::Unregister(const std::string& name) {
    cyxwiz::DataRegistry::Instance().UnregisterAudioDataset(name);
}

bool AudioLoader::RestoreFromRegistry(const std::string& name,
                                      const gui::MLNode& /*node*/,
                                      RestoreState& out) const {
    auto* entry = cyxwiz::DataRegistry::Instance().GetAudioDatasetEntry(name);
    if (!entry) return false;

    // Prefer the persisted feature shape (probed at Apply time) over
    // re-probing on every reopen. Fall back to n_mels × 313 (the
    // default MelSpectrogram frame count at 16kHz, 5s duration) when
    // the shape wasn't stashed.
    const size_t per_sample = (entry->feature_rows > 0 && entry->feature_cols > 0)
        ? static_cast<size_t>(entry->feature_rows) *
          static_cast<size_t>(entry->feature_cols) * sizeof(float)
        : static_cast<size_t>(entry->n_mels) * 313 * sizeof(float);

    out.found   = true;
    out.rows    = static_cast<int64_t>(entry->num_samples);
    out.cols    = 1;
    out.bytes   = entry->num_samples * per_sample;
    out.backend = 4;
    out.memory_is_estimate = true;
    out.status_message = "Loaded " + name + " (" +
        std::to_string(entry->num_samples) + " audio files, " +
        std::to_string(entry->num_classes) + " classes)";
    return true;
}

CompletedLoadDescription AudioLoader::DescribeCompletedLoad(
    const AsyncLoadState& state) const {
    CompletedLoadDescription d;
    d.memory_is_estimate = true;
    d.node_description_suffix =
        std::to_string(state.rows) + " audio files, " +
        std::to_string(state.num_classes) + " classes";

    fs::path p(state.source_path);
    d.default_status_message = std::string("Loaded audio from ") +
        p.filename().string();
    return d;
}

bool AudioLoader::LaunchTraining(
    cyxwiz::TrainingConfiguration config,
    const std::string& dataset_name,
    const std::string& /*label_column*/,
    int epochs,
    int batch_size,
    cyxwiz::TrainingPlotPanel* plot_panel,
    std::function<void(bool)> node_editor_callback) {
    auto* entry = cyxwiz::DataRegistry::Instance().GetAudioDatasetEntry(dataset_name);
    if (!entry) {
        spdlog::error("AudioLoader: audio dataset '{}' is registered but entry "
                      "could not be retrieved", dataset_name);
        return false;
    }
    spdlog::info("AudioLoader: Starting audio training: dataset={}, epochs={}, "
                 "batch_size={}, {} samples, {} classes, feature_type={}",
                 dataset_name, epochs, batch_size,
                 entry->num_samples, entry->num_classes, entry->feature_type);
    return cyxwiz::TrainingManager::Instance().StartTrainingAudio(
        std::move(config), *entry, epochs, batch_size, plot_panel,
        std::move(node_editor_callback));
}

}  // namespace cyxwiz::loaders
