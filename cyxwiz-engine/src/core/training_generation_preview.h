#pragma once
#include "training_generation_preview_settings.h"
#include <cyxwiz/tokenizer.h>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {
class SequentialModel;

// Where in the run a generation pass happens.
struct GenerationPreviewPoint {
    std::string kind = "epoch";  // baseline | epoch | step | try
    int epoch = 0;               // Completed epochs.
    int total_epochs = 0;
    long long step = 0;          // Completed optimizer steps.
    std::optional<double> validation_loss;
    std::string weights = "current";  // The weights actually used: current | best (a checkpoint's name for try)
};

// Bound once before the first update. Owns small host-side prompt/tokenizer state.
class TrainingGenerationPreview {
public:
    explicit TrainingGenerationPreview(const TrainingGenerationPreviewSettings& settings,
                                       size_t model_vocabulary);
    // Generates for every prompt, writes the report under `output_directory`
    // and logs one readable block (Info) plus each sample's JSON (Debug).
    // Returns false when cancelled.
    bool Run(SequentialModel& model, const GenerationPreviewPoint& point, const std::string& run_id,
             const std::filesystem::path& output_directory, const std::function<bool()>& should_stop);
    const TrainingGenerationPreviewSettings& Settings() const { return settings_; }

private:
    TrainingGenerationPreviewSettings settings_;
    Tokenizer tokenizer_;
    std::vector<std::string> rendered_;
    std::vector<std::vector<int64_t>> prompt_ids_;
    std::vector<std::string> stop_texts_;
    std::string vocabulary_hash_;
    std::string settings_hash_;
};
} // namespace cyxwiz
