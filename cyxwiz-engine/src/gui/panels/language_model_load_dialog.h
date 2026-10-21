#pragma once

#include "../../core/language_model_loader.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz {

// "Load a model" for the LM Generation panel (TOFIX159, design approved
// 2026-10-18): this session's model, a training checkpoint with the graph it
// was trained from, or a .cyxmodel package. Loads in the background with
// Cancel; the session's active model is not replaced.
class LanguageModelLoadDialog {
public:
    // `source` preselects a source (the empty state's buttons).
    void Open(LanguageModelLoadRequest::Source source = LanguageModelLoadRequest::Source::Checkpoint);
    // Draws the dialog while it is open. Returns the loaded model once, on
    // the frame its load finishes (the dialog then closes).
    std::shared_ptr<LoadedLanguageModel> Render();

private:
    struct LoadState {
        std::atomic<bool> done{false};
        bool ok = false;
        std::mutex mutex;
        std::string step;  // Guarded by mutex.
        int index = 0;
        int count = 0;
        std::string error;
        std::shared_ptr<LoadedLanguageModel> result;
        uint64_t task_id = 0;
    };

    void RenderSources();
    void RenderSession();
    void RenderCheckpoint();
    void RenderPackage();
    void RenderProgress();
    void RefreshGraphs();
    void RefreshRuns();
    void Start();
    std::string Blocker() const;

    bool open_request_ = false;
    LanguageModelLoadRequest::Source source_ = LanguageModelLoadRequest::Source::Checkpoint;
    std::vector<std::filesystem::path> graphs_;  // The project's .cyxgraph files.
    int graph_index_ = -1;
    std::filesystem::path runs_folder_;
    bool runs_folder_chosen_ = false;
    std::vector<CheckpointRunRow> runs_;
    int run_index_ = 0;
    std::string package_path_;
    std::shared_ptr<LoadState> load_;
    std::string error_;
    std::string device_label_;  // The selected device, read when the dialog opens.
};

}  // namespace cyxwiz
