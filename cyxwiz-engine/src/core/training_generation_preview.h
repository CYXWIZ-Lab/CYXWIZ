#pragma once
#include "training_generation_preview_settings.h"
#include <cyxwiz/tokenizer.h>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {
class SequentialModel;
// Bound once before the first update. Owns small host-side prompt/tokenizer state.
class TrainingGenerationPreview {
public:
    explicit TrainingGenerationPreview(const TrainingGenerationPreviewSettings& settings,
                                       size_t model_vocabulary);
    bool Run(SequentialModel& model, int epoch, const std::string& run_id,
             const std::string& output_directory, const std::function<bool()>& should_stop);
private:
    TrainingGenerationPreviewSettings settings_;
    Tokenizer tokenizer_;
    std::vector<std::string> prompts_;
    std::vector<std::vector<int64_t>> prompt_ids_;
    std::string vocabulary_hash_;
    std::string prompt_hash_;
};
} // namespace cyxwiz
