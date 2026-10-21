#include "language_model_loader.h"

#include "executable_model.h"
#include "graph_document.h"
#include "graph_training_job.h"
#include "model_importer.h"
#include "parquet_backed_dataset.h"
#include "training_manager.h"
#include "formats/cyxmodel_format.h"
#include "../inference/language_model_inference_contract.h"
#include "../inference/text_inference_input.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>

namespace cyxwiz {
namespace {

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

std::string Param(const gui::MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}


std::string Fixed(double value, int decimals) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    return buffer;
}

std::string Percent(double fraction) { return Fixed(fraction * 100.0, 1) + "%"; }

const char* DeviceTypeLabel(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::OPENCL: return "OpenCL";
        default: return "Device";
    }
}


// Activates the process device on this worker (task threads start on
// ArrayFire's default backend) and records it as the model's.
void UseSelectedDevice(LoadedLanguageModel& loaded) {
    if (const auto selected = Device::GetProcessDevice()) {
        const auto activation = Device(selected->type, selected->device_id).ActivateExact(false);
        if (!activation.success)
            throw std::runtime_error("Cannot use the selected device: " + activation.message +
                                     ". Choose an available device in Preferences > Devices.");
        loaded.device = *selected;
    }
    if (const auto* current = Device::GetCurrentDevice()) {
        if (!loaded.device) loaded.device = ProcessDeviceSelection{current->GetType(), current->GetDeviceId()};
        const auto info = current->GetInfo();
        loaded.device_label = std::string(DeviceTypeLabel(info.type)) + (info.name.empty() ? "" : " \xC2\xB7 " + info.name);
    }
}

void RequireNotCancelled(const std::function<bool()>& cancel) {
    if (cancel && cancel()) throw std::runtime_error("Loading was cancelled");
}

// One forward over a single token: the model must give next-token scores
// for every token of its tokenizer.
void CheckNextTokenScores(LoadedLanguageModel& loaded) {
    const size_t vocabulary = loaded.tokenizer->VocabularySize();
    std::vector<int64_t> token = {std::min<int64_t>(1, static_cast<int64_t>(vocabulary) - 1)};
    const Tensor logits = loaded.model->Forward(Tensor({1, 1}, token.data(), DataType::Int64));
    const auto output = ValidateLanguageModelRuntimeOutput(logits, 1, vocabulary);
    if (!output.compatible)
        throw std::runtime_error("This model does not give next-token scores for its tokenizer: " + output.error);
    loaded.checks.push_back("Gives next-token scores for " + LanguageModelThousands(vocabulary) + " tokens");
}

void AddDetail(LoadedLanguageModel& loaded, const std::string& key, const std::string& value) {
    if (!value.empty()) loaded.details.push_back({key, value});
}

// Settings, tokenizer and facts from a training configuration (a checkpoint's
// prepared graph, or the session's evaluation configuration).
void FromTrainingConfiguration(LoadedLanguageModel& loaded, const TrainingConfiguration& config) {
    const auto& sequence = config.sequence_batch;
    if (!sequence.enabled || !sequence.create_causal_lm_targets)
        throw std::runtime_error("This is not a language model: its graph does not train next-token prediction");
    if (sequence.tokenizer_config_json.empty() || sequence.tokenizer_vocabulary_artifact.empty())
        throw std::runtime_error("The model's tokenizer is not recorded (its training data has no tokenizer metadata)");
    std::string error;
    loaded.tokenizer = GenerationTokenizer::Load(sequence.tokenizer_config_json, sequence.tokenizer_vocabulary_artifact,
                                                 "", error);
    if (!loaded.tokenizer) throw std::runtime_error("The model's tokenizer cannot be used: " + error);
    loaded.settings = config.generation_preview;
    loaded.settings_from_graph = true;
    loaded.settings.context = static_cast<int>(sequence.max_sequence_length);
}

void Describe(LoadedLanguageModel& loaded, int epoch, std::optional<double> validation_loss) {
    loaded.summary = LanguageModelSummary(epoch, validation_loss, loaded.settings.context,
                                          loaded.tokenizer->VocabularySize(), loaded.tokenizer->Kind());
    if (loaded.settings_from_graph) {
        loaded.summary += loaded.settings.mode == GenerationPreviewMode::Chat
                              ? std::string(" \xC2\xB7 chat, ") + GenerationPreviewTemplateLabel(loaded.settings.chat_template)
                              : std::string(" \xC2\xB7 continuation");
    }
    loaded.report.context = loaded.settings.context;
    loaded.report.tokenizer_sha256 = loaded.tokenizer->Sha256();
    AddDetail(loaded, "Source", loaded.source);
    AddDetail(loaded, "Context", loaded.settings.context > 0 ? std::to_string(loaded.settings.context) + " tokens" : "");
    AddDetail(loaded, "Output", "scores [1, seq, " + LanguageModelThousands(loaded.tokenizer->VocabularySize()) + "]");
    AddDetail(loaded, "Tokenizer", loaded.tokenizer->Kind() + " \xC2\xB7 " + LanguageModelThousands(loaded.tokenizer->VocabularySize()));
    const auto& sha = loaded.tokenizer->Sha256();
    AddDetail(loaded, "Tokenizer hash", sha.size() > 14 ? sha.substr(0, 8) + "..." + sha.substr(sha.size() - 6) : sha);
    AddDetail(loaded, "Device", loaded.device_label);
    AddDetail(loaded, "Model", loaded.model->AsSequentialModel() ? "layer stack" : "graph with branches");
}

LoadedLanguageModel LoadFromCheckpoint(const LanguageModelLoadRequest& request, const std::function<bool()>& cancel,
                                       const std::function<void(const std::string&, int, int)>& progress) {
    namespace fs = std::filesystem;
    LoadedLanguageModel loaded;
    loaded.source = "Training checkpoint";
    if (request.graph_path.empty()) throw std::runtime_error("Choose the graph the checkpoint was trained from");
    if (request.checkpoint_path.empty()) throw std::runtime_error("Choose a checkpoint");
    if (progress) progress("Reading the graph", 1, 3);
    GraphTrainingJobRequest job;
    job.graph_json = ReadFile(request.graph_path);
    GraphDocument graph;
    std::string error;
    if (!ParseGraphDocument(job.graph_json, graph, error)) throw std::runtime_error("The graph cannot be read: " + error);
    std::map<std::string, std::string> loader_parameters;
    for (const auto& node : graph.nodes) {
        if (node.type == gui::NodeType::DataLoader && loaded.loader_node_id < 0) {
            loaded.loader_node_id = node.id;
            loader_parameters = node.parameters;
        }
        if (node.type != gui::NodeType::DataInput) continue;
        // Relative Data Input files are relative to the project.
        const fs::path file = Param(node, "file_path");
        if (!file.empty() && file.is_relative() && !request.project_root.empty())
            job.dataset_files[Param(node, "dataset_name")] = (fs::path(request.project_root) / file).string();
    }
    // Building the model does not use the graph's generation previews: they
    // are compiled out here and read on their own below, so settings an
    // older Engine saved do not stop the model from loading.
    {
        auto document = nlohmann::json::parse(job.graph_json);
        for (auto& node : document["nodes"]) {
            if (!node.contains("parameters") || !node["parameters"].is_object()) continue;
            auto& parameters = node["parameters"];
            for (auto it = parameters.begin(); it != parameters.end();)
                it = it.key().rfind("generation_preview_", 0) == 0 ? parameters.erase(it) : std::next(it);
        }
        job.graph_json = document.dump();
    }
    std::string settings_problem;
    TrainingGenerationPreviewSettings graph_settings;
    try {
        graph_settings = ParseGenerationSettings(loader_parameters);
    } catch (const std::exception& exception) {
        settings_problem = exception.what();
    }
    RequireNotCancelled(cancel);

    if (progress) progress("Loading the graph's training data", 2, 3);
    PreparedGraphJob prepared;
    if (const auto failure = PrepareGraphJob(job, prepared); !failure.error.empty())
        throw std::runtime_error("The graph cannot be prepared: " + failure.error);
    const auto dataset = prepared.datasets.count(prepared.config.dataset_name)
                             ? prepared.datasets.at(prepared.config.dataset_name)
                             : nullptr;
    if (!dataset && prepared.disk_datasets.count(prepared.config.dataset_name))
        throw std::runtime_error("The training data '" + prepared.config.dataset_name +
                                 "' is too large to load for its vocabulary; raise the data budget in Preferences");
    RequireNotCancelled(cancel);

    if (progress) progress("Building the model and loading the weights", 3, 3);
    UseSelectedDevice(loaded);
    auto built = BuildModelFromCheckpoint(prepared.config, dataset, request.checkpoint_path, cancel);
    if (!built.model) throw std::runtime_error(built.error_message.empty() ? "The checkpoint cannot be loaded"
                                                                           : built.error_message);
    loaded.model = std::shared_ptr<IExecutableModel>(std::move(built.model));
    FromTrainingConfiguration(loaded, built.prepared);
    const int context = loaded.settings.context;
    loaded.settings = graph_settings;
    loaded.settings.context = context;
    loaded.settings_from_graph = settings_problem.empty();
    CheckNextTokenScores(loaded);

    const fs::path resolved = built.resolved_checkpoint_path;
    const auto& meta = built.metadata;
    loaded.graph_path = request.graph_path;
    loaded.name = fs::path(request.graph_path).stem().string();
    loaded.location = resolved.string();
    loaded.chats_folder = resolved.parent_path() / "generation_chats";
    const std::string run = resolved.parent_path().filename().string();
    loaded.subtitle = "checkpoint \xC2\xB7 " + resolved.filename().string() + " of " + run;
    loaded.report.run_id = run;
    loaded.report.epoch = meta.epoch;
    loaded.report.step = meta.global_step;
    loaded.report.weights = resolved.filename().string();
    std::optional<double> validation;
    if (meta.val_loss > 0.0f) validation = meta.val_loss;
    loaded.report.validation_loss = validation;
    if (built.tokenizer_from_checkpoint) {
        loaded.checks.push_back("Tokenizer stored in the checkpoint");
    } else if (!meta.tokenizer_sha256.empty()) {
        if (meta.tokenizer_sha256 != loaded.tokenizer->Sha256())
            throw std::runtime_error("The training data's tokenizer is not the one this checkpoint was trained with");
        loaded.checks.push_back("Tokenizer rebuilt from the training data matches the one the checkpoint was trained with");
    }
    if (!settings_problem.empty())
        loaded.warnings.push_back("The graph's generation settings are not used (" + settings_problem +
                                  "); the defaults are");
    else
        loaded.checks.push_back(loaded.settings.mode == GenerationPreviewMode::Chat
                                    ? std::string("Generation settings from the graph's Data Loader: chat, ") +
                                          GenerationPreviewTemplateLabel(loaded.settings.chat_template)
                                    : "Generation settings from the graph's Data Loader: continuation");
    Describe(loaded, meta.epoch, validation);
    AddDetail(loaded, "Weights", resolved.filename().string() + " \xC2\xB7 epoch " + std::to_string(meta.epoch) +
                                     " \xC2\xB7 step " + std::to_string(meta.global_step));
    AddDetail(loaded, "Graph", fs::path(request.graph_path).filename().string());
    AddDetail(loaded, "Tokenizer from", built.tokenizer_from_checkpoint
                                            ? "the checkpoint"
                                            : "the training data (this checkpoint stores none)");
    AddDetail(loaded, "Saved", CheckpointSavedLabel(meta.timestamp));
    if (validation)
        AddDetail(loaded, "Validation", "loss " + Fixed(*validation, 2) + " \xC2\xB7 accuracy " + Percent(meta.val_accuracy));
    if (meta.train_loss > 0.0f)
        AddDetail(loaded, "Training", "loss " + Fixed(meta.train_loss, 2) + " \xC2\xB7 accuracy " + Percent(meta.train_accuracy));
    if (!meta.initialized_from.empty()) {
        const fs::path from = meta.initialized_from;
        AddDetail(loaded, "Started from", from.parent_path().parent_path().filename().string() + " \xC2\xB7 " +
                                              from.filename().string());
    }
    return loaded;
}

LoadedLanguageModel LoadFromSession() {
    LoadedLanguageModel loaded;
    loaded.source = "This session's model";
    auto& training = TrainingManager::Instance();
    if (training.IsTrainingActive()) throw std::runtime_error("Training is running; use the model when it ends");
    loaded.model = training.GetActiveModel();
    const auto info = training.GetActiveModelInfo();
    if (!loaded.model || !info.evaluation_config)
        throw std::runtime_error("This session has no trained or loaded model");
    loaded.device = info.model_device;
    if (loaded.device) {
        const auto activation = Device(loaded.device->type, loaded.device->device_id).ActivateExact(false);
        if (!activation.success) throw std::runtime_error("Cannot use the model's device: " + activation.message);
    }
    if (const auto* current = Device::GetCurrentDevice()) {
        const auto device_info = current->GetInfo();
        loaded.device_label = std::string(DeviceTypeLabel(device_info.type)) +
                              (device_info.name.empty() ? "" : " \xC2\xB7 " + device_info.name);
    }
    FromTrainingConfiguration(loaded, *info.evaluation_config);
    CheckNextTokenScores(loaded);
    const auto& meta = info.checkpoint_metadata;
    std::optional<double> validation;
    if (meta.val_loss > 0.0f) validation = meta.val_loss;
    if (info.origin == ActiveModelOrigin::LoadedCheckpoint && !info.checkpoint_path.empty()) {
        const std::filesystem::path resolved = info.checkpoint_path;
        loaded.location = resolved.string();
        loaded.chats_folder = resolved.parent_path() / "generation_chats";
        loaded.subtitle = "loaded checkpoint \xC2\xB7 " + resolved.filename().string() + " of " +
                          resolved.parent_path().filename().string();
        loaded.report.run_id = resolved.parent_path().filename().string();
        loaded.report.weights = resolved.filename().string();
    } else {
        loaded.subtitle = "trained in this session";
        loaded.report.run_id = "session";
        loaded.report.weights = "session";
    }
    loaded.name = info.effective_dataset_name.empty() ? "Session model" : info.effective_dataset_name;
    loaded.report.epoch = meta.epoch;
    loaded.report.step = meta.global_step;
    loaded.report.validation_loss = validation;
    loaded.checks.push_back("Generation settings from the trained graph's Data Loader");
    Describe(loaded, meta.epoch, validation);
    AddDetail(loaded, "Weights", loaded.subtitle);
    return loaded;
}

LoadedLanguageModel LoadFromPackage(const LanguageModelLoadRequest& request, const std::function<bool()>& cancel) {
    namespace fs = std::filesystem;
    LoadedLanguageModel loaded;
    loaded.source = "Model package";
    if (request.package_path.empty()) throw std::runtime_error("Choose a .cyxmodel package");
    UseSelectedDevice(loaded);
    ModelImporter importer;
    ImportOptions options;
    std::shared_ptr<IExecutableModel> model;
    const auto imported = importer.ImportExecutableModel(request.package_path, model, options);
    if (!imported.success || !model)
        throw std::runtime_error("The package cannot be imported: " +
                                 (imported.error_message.empty() ? importer.GetLastError() : imported.error_message));
    RequireNotCancelled(cancel);
    formats::CyxModelFormat format;
    std::string config_json, vocabulary, model_data;
    if (!format.ExtractTextTokenizerAssets(request.package_path, config_json, vocabulary, model_data))
        throw std::runtime_error("The package carries no tokenizer (tokenizer/config.json and vocab.txt)");
    TextTokenizerPackage contract_tokenizer;
    std::string error;
    if (!LoadTextTokenizerPackage(config_json, vocabulary, model_data, contract_tokenizer, error))
        throw std::runtime_error("The package's tokenizer cannot be used: " + error);
    const auto contract = ValidateLanguageModelPackageContract(importer.ProbeFile(request.package_path),
                                                                &contract_tokenizer, request.package_path);
    if (!contract.compatible) throw std::runtime_error("The package is not a language model: " + contract.error);
    loaded.tokenizer = GenerationTokenizer::Load(config_json, vocabulary, model_data, error);
    if (!loaded.tokenizer) throw std::runtime_error("The package's tokenizer cannot be used: " + error);
    loaded.model = std::move(model);
    loaded.settings.context = static_cast<int>(contract.max_sequence_length);
    CheckNextTokenScores(loaded);
    loaded.checks.push_back("Carries its tokenizer (" + loaded.tokenizer->Kind() + ")");
    const fs::path path = request.package_path;
    loaded.name = path.stem().string();
    loaded.location = path.string();
    loaded.subtitle = "package \xC2\xB7 " + path.filename().string();
    loaded.report.run_id = loaded.name;
    loaded.report.weights = "package";
    Describe(loaded, 0, std::nullopt);
    AddDetail(loaded, "Family", contract.model_family);
    return loaded;
}

}  // namespace

LoadedLanguageModel LoadLanguageModel(const LanguageModelLoadRequest& request, const std::function<bool()>& cancel,
                                      const std::function<void(const std::string&, int, int)>& progress) {
    LoadedLanguageModel loaded;
    switch (request.source) {
        case LanguageModelLoadRequest::Source::Checkpoint: loaded = LoadFromCheckpoint(request, cancel, progress); break;
        case LanguageModelLoadRequest::Source::Session: loaded = LoadFromSession(); break;
        case LanguageModelLoadRequest::Source::Package: loaded = LoadFromPackage(request, cancel); break;
    }
    loaded.settings.enabled = true;
    spdlog::info("LM Generation: loaded {} ({}) on {}", loaded.name, loaded.subtitle, loaded.device_label);
    return loaded;
}

std::string ActivateLanguageModelDevice(const LoadedLanguageModel& model) {
    if (!model.device) return {};
    const auto activation = Device(model.device->type, model.device->device_id).ActivateExact(false);
    return activation.success ? std::string() : "Cannot use the model's device: " + activation.message;
}

}  // namespace cyxwiz
