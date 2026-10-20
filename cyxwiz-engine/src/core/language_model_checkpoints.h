#pragma once

#include "training_generation_preview_settings.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

// Checkpoint and graph helpers of the LM Generation panel (TOFIX159). Pure:
// files and JSON only.

// "8,192".
std::string LanguageModelThousands(size_t value);
// "2026-10-14T13:26:55Z" -> "2026-10-14 13:26".
std::string CheckpointSavedLabel(const std::string& timestamp);

// The training runs of a checkpoint folder, newest first (the Load dialog).
struct CheckpointRunRow {
    std::string run;      // train-1791984271862
    std::string weights;  // best | latest | the folder's name
    int epoch = 0;
    std::optional<double> validation_loss;
    std::string saved;    // "2026-10-14 13:26"
    std::filesystem::path folder;  // The checkpoint itself (…/best).
};
std::vector<CheckpointRunRow> ListCheckpointRuns(const std::filesystem::path& folder);

// The folder a graph's runs are saved in: its Data Loader's checkpoint_dir
// (relative to the project), else the project's checkpoints folder.
std::filesystem::path GraphCheckpointFolder(const std::string& graph_path, const std::string& project_root,
                                            const std::string& project_checkpoints);

// "Epoch 4 · validation loss 3.42 · context 768 · vocabulary 8,192 (Byte BPE)".
std::string LanguageModelSummary(int epoch, std::optional<double> validation_loss, int context, size_t vocabulary,
                                 const std::string& tokenizer_kind);

// Writes the generation settings into the graph's Data Loader (the other
// generation_preview_* keys, such as on/off and prompts, are kept).
bool SaveGenerationSettingsToGraph(const std::string& graph_path, int loader_node_id,
                                   const TrainingGenerationPreviewSettings& settings, std::string& error);

}  // namespace cyxwiz
