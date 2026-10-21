#pragma once

#include "language_model_chat.h"
#include "language_model_checkpoints.h"

#include <cyxwiz/device.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

// One loader for a trained language model used outside training (TOFIX159,
// LM Generation panel): the model this session holds, a training checkpoint
// with the graph it was trained from, or a .cyxmodel package. Works for any
// causal language model: a layer stack or a graph with branches.

struct LanguageModelFact {
    std::string key;
    std::string value;
};

struct LoadedLanguageModel {
    std::shared_ptr<IExecutableModel> model;
    std::shared_ptr<GenerationTokenizer> tokenizer;
    // The graph Data Loader's generation settings (context set), else defaults.
    TrainingGenerationPreviewSettings settings;
    bool settings_from_graph = false;
    std::string graph_path;  // The .cyxgraph the settings can be saved to ("" when none).
    int loader_node_id = -1;  // Its Data Loader.

    std::string source;    // "Training checkpoint", "This session's model", "Model package"
    std::string name;      // The graph or package name.
    std::string subtitle;  // "checkpoint · best of train-1791984271862"
    std::string summary;   // "Epoch 4 · validation loss 3.42 · context 768 · vocabulary 8,192 (Byte BPE)"
    std::optional<ProcessDeviceSelection> device;  // Where the weights live.
    std::string device_label;                      // "CUDA · NVIDIA GeForce GTX 1050 Ti"
    std::string location;                          // Checkpoint folder or package file.
    std::filesystem::path chats_folder;            // Conversation reports ("" = not saved).
    LanguageModelReportSource report;
    std::vector<LanguageModelFact> details;
    std::vector<std::string> checks;    // What was verified while loading, in words.
    std::vector<std::string> warnings;  // What loaded with a caveat (settings not used, ...).
};

struct LanguageModelLoadRequest {
    enum class Source { Session, Checkpoint, Package };
    Source source = Source::Checkpoint;
    std::string graph_path;       // Checkpoint: the .cyxgraph it was trained from.
    std::string checkpoint_path;  // A run folder (its best/) or a checkpoint folder.
    std::string package_path;     // Package: the .cyxmodel.
    std::string project_root;     // Relative Data Input files resolve here.
};

// Loads on the calling thread (a task worker) on the selected device.
// `progress` names each step (index from 1 of count). Throws
// std::runtime_error with a reason for the user.
LoadedLanguageModel LoadLanguageModel(
    const LanguageModelLoadRequest& request, const std::function<bool()>& cancel,
    const std::function<void(const std::string& step, int index, int count)>& progress);

// Makes the model's device current on this thread; "" or the reason it failed.
std::string ActivateLanguageModelDevice(const LoadedLanguageModel& model);

}  // namespace cyxwiz
