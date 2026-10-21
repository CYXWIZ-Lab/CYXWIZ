#pragma once

// Data Loader > Generation (TOFIX159 A3, design approved 2026-10-18): what the
// model is asked during training, how it is decoded and when. Generic for any
// language model: continuation for base models, a chat template for
// instruction-tuned ones.

#include "../core/training_generation_preview_settings.h"
#include "generation_preview_view.h"

#include <cyxwiz/tokenizer.h>

#include <atomic>
#include <filesystem>
#include <map>
#include <memory>
#include <string>

namespace gui {

// What the tab knows about the training data (the Data Input upstream of the
// Data Loader, when it is loaded).
struct GenerationTabDataset {
    bool found = false;
    std::string name;
    std::string supervision;  // cyxwiz.token_windows.supervision ("" or "response_only")
    int context = 0;
    std::shared_ptr<cyxwiz::Tokenizer> tokenizer;  // For token counts; null when not loaded.
};

class NodeEditor;
struct MLNode;

class DataLoaderGenerationTab {
public:
    // Reads the generation_preview_* parameters. A Data Loader saved with the
    // pre-TOFIX159 keys starts from the defaults and says so.
    void Load(const std::map<std::string, std::string>& params);
    // Writes the parameters and removes the pre-TOFIX159 keys.
    void Store(std::map<std::string, std::string>& params) const;
    // Returns true when a setting changed. `editor` and `loader` let Try now
    // build the graph's model; either may be null (no Try now then).
    bool Render(const GenerationTabDataset& dataset, NodeEditor* editor, const MLNode* loader);
    // "" when the settings are valid (or off).
    std::string Problem() const;

private:
    bool RenderPrompts();
    bool RenderPromptEditor();
    void RenderModelSees(const GenerationTabDataset& dataset);
    bool RenderTemplate();
    bool RenderStops();
    bool RenderDecoding();
    bool RenderSchedule();
    // Try now (TOFIX159): the current settings on a checkpoint, without training.
    void RenderTryNow(const GenerationTabDataset& dataset, NodeEditor* editor, const MLNode* loader);
    void StartTry(NodeEditor* editor, const MLNode* loader, const std::filesystem::path& checkpoint);
    struct TryState {
        std::atomic<bool> done{false};
        bool ok = false;
        std::string error;
        std::filesystem::path folder;  // Where the report went.
        uint64_t task_id = 0;
    };
    std::shared_ptr<TryState> try_;
    std::filesystem::path try_checkpoint_;  // Chosen with Choose...; empty = the newest.
    cyxwiz::GenerationPreviewView try_view_;
    std::string try_note_;
    bool try_failed_ = false;

    cyxwiz::TrainingGenerationPreviewSettings settings_;
    int selected_ = 0;
    std::string new_stop_;
    std::string old_format_note_;
};

// Finds the Data Input upstream of `loader_node_id` and reads its token-window
// metadata from the loaded dataset.
class NodeEditor;
GenerationTabDataset ResolveGenerationTabDataset(NodeEditor* editor, int loader_node_id);

}  // namespace gui
