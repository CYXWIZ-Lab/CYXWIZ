#pragma once

// Data Loader > Generation (TOFIX159 A3, design approved 2026-10-18): what the
// model is asked during training, how it is decoded and when. Generic for any
// language model: continuation for base models, a chat template for
// instruction-tuned ones.

#include "../core/training_generation_preview_settings.h"

#include <cyxwiz/tokenizer.h>

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

class DataLoaderGenerationTab {
public:
    // Reads the generation_preview_* parameters. A Data Loader saved with the
    // pre-TOFIX159 keys starts from the defaults and says so.
    void Load(const std::map<std::string, std::string>& params);
    // Writes the parameters and removes the pre-TOFIX159 keys.
    void Store(std::map<std::string, std::string>& params) const;
    // Returns true when a setting changed.
    bool Render(const GenerationTabDataset& dataset);
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
