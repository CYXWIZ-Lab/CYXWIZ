#include <stdexcept>
#include "audio_loader.h"

#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_audit.h"
#include "../../core/formats/audio_dataset.h"
#include "../../core/graph_compiler.h"  // PreprocessingDomain
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

    // A previous dataset under another name stays until this load succeeds:
    // PollAsyncLoadResult drops it then, and keeps it when this one fails.

    // Build the registry entry skeleton on the UI thread; the worker
    // fills num_samples / num_classes / class_names from a folder scan.
    // Captured by value — the dialog can close mid-load without racing.
    cyxwiz::DataRegistry::AudioDatasetEntry entry;
    entry.folder_path     = ctx.source_path;
    entry.labeled_subdirs = ctx.audio_labeled_subdirs;
    if (!ctx.audio_labeled_subdirs) {
        entry.csv_path     = ctx.audio_labels_csv;
        entry.filename_col = ctx.audio_filename_col;
        entry.label_col    = ctx.audio_label_col;
    }
    entry.target_sr    = ctx.audio_target_sr;
    entry.max_duration = ctx.audio_max_duration > 0.0f ? ctx.audio_max_duration : 5.0f;

    const std::string folder = ctx.source_path;
    const std::string name   = ctx.dataset_name;

    state->dataset_name = name;
    state->previous_dataset_name = ctx.previous_dataset_name;
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

                cyxwiz::AudioDatasetConfig scan_cfg;
                scan_cfg.probe_feature_shape = false;
                scan_cfg.target_sr           = entry.target_sr;
                scan_cfg.max_duration        = entry.max_duration;
                scan_cfg.labeled_subdirs     = entry.labeled_subdirs;
                scan_cfg.csv_path            = entry.csv_path;
                scan_cfg.filename_col        = entry.filename_col;
                scan_cfg.label_col           = entry.label_col;

                cyxwiz::AudioDataset scan(folder, scan_cfg);
                auto info = scan.GetInfo();

                cyxwiz::DataRegistry::AudioDatasetEntry final_entry = entry;
                final_entry.num_samples = info.num_samples;
                final_entry.num_classes = info.num_classes;
                final_entry.class_names = scan.GetClassNames();

                task.ReportProgress(0.9f, "Registering dataset");
                reg.RegisterAudioDataset(name, final_entry);

                state->success      = true;
                state->backend      = 4;
                state->rows         = static_cast<int64_t>(info.num_samples);
                state->cols         = 1;
                state->bytes        = 0;  // decoded per batch by the feature node
                state->num_classes  = info.num_classes;
                state->message      = "Loaded " +
                    std::to_string(info.num_samples) + " audio files (" +
                    std::to_string(info.num_classes) + " classes) from " +
                    fs::path(folder).filename().string();
                auto audit = cyxwiz::DatasetAudit::AuditAudio(name, final_entry);
                state->audit_errors = audit.ErrorCount();
                state->audit_warnings = audit.WarningCount();
                state->audit_message = cyxwiz::FormatAuditSummary(audit);
                state->audit_issue_lines = cyxwiz::FormatAuditIssueLines(audit);
            } catch (const std::exception& e) {
                state->success = false;
                state->message = std::string("Error loading audio: ") + e.what();
                spdlog::error("AudioLoader async: {}", state->message);
            }
            // Publish barrier — must be set LAST.
            state->done.store(true);
            // The dialog reads the published state; the task itself must also
            // end Failed (not Completed) so the Tasks panel tells the truth.
            if (!state->success && !task.ShouldStop()) {
                throw std::runtime_error(state->message.empty() ? std::string("Load failed") : state->message);
            }
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

    out.found   = true;
    out.rows    = static_cast<int64_t>(entry->num_samples);
    out.classes = static_cast<int64_t>(entry->num_classes);
    out.cols    = 1;
    out.bytes   = 0;  // decoded per batch by the feature node
    out.backend = 4;
    out.memory_is_estimate = false;
    out.status_message = "Loaded " + name + " (" +
        std::to_string(entry->num_samples) + " audio files, " +
        std::to_string(entry->num_classes) + " classes)";
    return true;
}

CompletedLoadDescription AudioLoader::DescribeCompletedLoad(
    const AsyncLoadState& state) const {
    CompletedLoadDescription d;
    d.memory_is_estimate = false;
    d.node_description_suffix =
        std::to_string(state.rows) + " audio files, " +
        std::to_string(state.num_classes) + " classes";

    fs::path p(state.source_path);
    d.default_status_message = std::string("Loaded audio from ") +
        p.filename().string();
    return d;
}

cyxwiz::PreprocessingDomain AudioLoader::Domain(
    const std::string& /*file_category*/) const {
    return cyxwiz::PreprocessingDomain::Audio;
}

bool AudioLoader::LaunchTraining(
    cyxwiz::TrainingConfiguration config,
    const std::string& dataset_name,
    const std::string& /*label_column*/,
    int epochs,
    int batch_size,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback) {
    auto* entry = cyxwiz::DataRegistry::Instance().GetAudioDatasetEntry(dataset_name);
    if (!entry) {
        spdlog::error("AudioLoader: audio dataset '{}' is registered but entry "
                      "could not be retrieved", dataset_name);
        return false;
    }
    spdlog::info("AudioLoader: Starting audio training: dataset={}, epochs={}, "
                 "batch_size={}, {} samples, {} classes, feature_type={}",
                 dataset_name, epochs, batch_size, entry->num_samples, entry->num_classes,
                 static_cast<int>(config.audio_preprocessing.feature_type));
    return cyxwiz::TrainingManager::Instance().StartTrainingAudio(
        std::move(config), *entry, epochs, batch_size, plot_panel,
        std::move(node_editor_callback));
}

std::vector<ParamSchema> AudioLoader::NodeParams() const {
    return {
        {"audio_layout",        "0",     "0=ClassSubdirs, 1=FlatWithCSV"},
        {"audio_labels_csv",    "",      "Labels CSV path (FlatWithCSV)"},
        {"audio_filename_col",  "",      "Filename column (auto-detect if empty)"},
        {"audio_label_col",     "",      "Label column (auto-detect if empty)"},
        {"sample_rate",         "16000", "Target sample rate (Hz)"},
        {"mono",                "true",  "Mix to mono"},
    };
}

SyntheticBatch AudioLoader::MakeSynthetic(
    const cyxwiz::TrainingConfiguration& config, uint32_t seed) const {
    return MakeSyntheticForDomain(
        config, cyxwiz::PreprocessingDomain::Audio, seed, "Audio");
}

}  // namespace cyxwiz::loaders
