#include "training_resume_checkpoint.h"

#include "checkpoint_payload_io.h"
#include "executable_model.h"
#include "sha256_digest.h"

#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/optimizer.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace cyxwiz {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr const char* kModelPath = "model/parameters.bin";
constexpr const char* kOptimizerPath = "optimizer/state.bin";
constexpr const char* kRuntimePath = "runtime.json";
constexpr const char* kGraphPath = "graph.cyxgraph";
constexpr const char* kDatasetsPath = "datasets.json";

std::string UtcNow() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

// A text payload written in place (the directory is new and the manifest is
// published last), described with its size and SHA-256.
bool WriteTextPayload(const fs::path& root, const std::string& relative, CheckpointPayloadKind kind,
                      const std::string& text, CheckpointPayloadDescriptor& descriptor, std::string& error) {
    const fs::path path = root / relative;
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out || !(out << text)) {
            error = "cannot write " + path.string();
            return false;
        }
    }
    descriptor.kind = kind;
    descriptor.relative_path = relative;
    descriptor.byte_size = static_cast<std::uint64_t>(fs::file_size(path, ec));
    descriptor.required = true;
    return Sha256File(path, descriptor.sha256, error);
}

bool ReadTextPayload(const fs::path& root, const CheckpointPayloadDescriptor& descriptor, std::string& text,
                     std::string& error) {
    if (!VerifyCheckpointPayloadFile(root, descriptor, error)) return false;
    std::ifstream in(root / descriptor.relative_path, std::ios::binary);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    text = buffer.str();
    return true;
}

json SchedulerStateToJson(const SchedulerState& s) {
    return {{"schema_version", s.schema_version}, {"scheduler_type", s.scheduler_type},
            {"base_learning_rate", s.base_learning_rate}, {"current_learning_rate", s.current_learning_rate},
            {"last_step", s.last_step}, {"hyperparameters", s.hyperparameters},
            {"string_hyperparameters", s.string_hyperparameters}, {"values", s.values}};
}

SchedulerState SchedulerStateFromJson(const json& j) {
    SchedulerState s;
    s.schema_version = j.value("schema_version", 1);
    s.scheduler_type = j.value("scheduler_type", std::string{});
    s.base_learning_rate = j.value("base_learning_rate", 0.0);
    s.current_learning_rate = j.value("current_learning_rate", 0.0);
    s.last_step = j.value("last_step", 0);
    s.hyperparameters = j.value("hyperparameters", std::map<std::string, double>{});
    s.string_hyperparameters = j.value("string_hyperparameters", std::map<std::string, std::string>{});
    s.values = j.value("values", std::map<std::string, double>{});
    return s;
}

json RuntimeToJson(const TrainingResumeRuntime& r) {
    json j{{"completed_epoch", r.completed_epoch},
           {"total_epochs", r.total_epochs},
           {"optimizer_step_count", r.optimizer_step_count},
           {"scheduler_step_count", r.scheduler_step_count},
           {"best_val_loss_set", r.best_val_loss_set},
           {"best_val_loss", r.best_val_loss_set ? r.best_val_loss : 0.0f},
           {"epochs_without_improvement", r.epochs_without_improvement},
           {"model_seed", r.model_seed},
           {"dataloader_seed", r.dataloader_seed},
           {"data_order_exact", r.data_order_exact},
           {"loss_history", r.loss_history},
           {"accuracy_history", r.accuracy_history},
           {"mae_history", r.mae_history},
           {"rmse_history", r.rmse_history},
           {"val_loss_history", r.val_loss_history},
           {"val_accuracy_history", r.val_accuracy_history},
           {"val_mae_history", r.val_mae_history},
           {"val_rmse_history", r.val_rmse_history},
           {"learning_rate_history", r.learning_rate_history}};
    if (r.in_epoch) {
        const auto& e = *r.in_epoch;
        j["in_epoch"] = {{"epoch", e.epoch},
                         {"next_batch", e.next_batch},
                         {"epoch_loss", e.epoch_loss},
                         {"loss_weight_sum", e.loss_weight_sum},
                         {"sample_count", e.sample_count},
                         {"correct_tokens", e.metrics.correct_tokens},
                         {"total_tokens", e.metrics.total_tokens},
                         {"token_accuracy", e.metrics.token_accuracy},
                         {"predicted_entities", e.metrics.predicted_entities},
                         {"gold_entities", e.metrics.gold_entities},
                         {"matched_entities", e.metrics.matched_entities},
                         {"entity_precision", e.metrics.entity_precision},
                         {"entity_recall", e.metrics.entity_recall},
                         {"entity_f1", e.metrics.entity_f1}};
    }
    if (r.scheduler) {
        j["scheduler"] = {{"state", SchedulerStateToJson(r.scheduler->scheduler_state)},
                          {"completed_epochs", r.scheduler->completed_epochs},
                          {"completed_optimizer_steps", r.scheduler->completed_optimizer_steps}};
    }
    return j;
}

TrainingResumeRuntime RuntimeFromJson(const json& j) {
    TrainingResumeRuntime r;
    r.completed_epoch = j.value("completed_epoch", 0);
    r.total_epochs = j.value("total_epochs", 0);
    r.optimizer_step_count = j.value("optimizer_step_count", 0);
    r.scheduler_step_count = j.value("scheduler_step_count", 0);
    r.best_val_loss_set = j.value("best_val_loss_set", false);
    r.best_val_loss = r.best_val_loss_set ? j.value("best_val_loss", 0.0f) : std::numeric_limits<float>::infinity();
    r.epochs_without_improvement = j.value("epochs_without_improvement", 0);
    r.model_seed = j.value("model_seed", std::int64_t{-1});
    r.dataloader_seed = j.value("dataloader_seed", std::int64_t{-1});
    r.data_order_exact = j.value("data_order_exact", false);
    const auto floats = [&j](const char* key) { return j.value(key, std::vector<float>{}); };
    r.loss_history = floats("loss_history");
    r.accuracy_history = floats("accuracy_history");
    r.mae_history = floats("mae_history");
    r.rmse_history = floats("rmse_history");
    r.val_loss_history = floats("val_loss_history");
    r.val_accuracy_history = floats("val_accuracy_history");
    r.val_mae_history = floats("val_mae_history");
    r.val_rmse_history = floats("val_rmse_history");
    r.learning_rate_history = j.value("learning_rate_history", std::vector<double>{});
    if (j.contains("in_epoch") && j["in_epoch"].is_object()) {
        const auto& e = j["in_epoch"];
        TrainingEpochProgress p;
        p.epoch = e.value("epoch", 0);
        p.next_batch = e.value("next_batch", 0);
        p.epoch_loss = e.value("epoch_loss", 0.0f);
        p.loss_weight_sum = e.value("loss_weight_sum", 0.0f);
        p.sample_count = e.value("sample_count", std::uint64_t{0});
        p.metrics.correct_tokens = e.value("correct_tokens", size_t{0});
        p.metrics.total_tokens = e.value("total_tokens", size_t{0});
        p.metrics.token_accuracy = e.value("token_accuracy", 0.0);
        p.metrics.predicted_entities = e.value("predicted_entities", size_t{0});
        p.metrics.gold_entities = e.value("gold_entities", size_t{0});
        p.metrics.matched_entities = e.value("matched_entities", size_t{0});
        p.metrics.entity_precision = e.value("entity_precision", 0.0);
        p.metrics.entity_recall = e.value("entity_recall", 0.0);
        p.metrics.entity_f1 = e.value("entity_f1", 0.0);
        r.in_epoch = p;
    }
    if (j.contains("scheduler") && j["scheduler"].is_object()) {
        TrainingSchedulerResumeState s;
        s.scheduler_state = SchedulerStateFromJson(j["scheduler"].value("state", json::object()));
        s.completed_epochs = j["scheduler"].value("completed_epochs", 0);
        s.completed_optimizer_steps = j["scheduler"].value("completed_optimizer_steps", 0);
        r.scheduler = s;
    }
    return r;
}

const CheckpointPayloadDescriptor* FindPayload(const CheckpointManifestV2& manifest, CheckpointPayloadKind kind) {
    for (const auto& payload : manifest.payloads) {
        if (payload.kind == kind) return &payload;
    }
    return nullptr;
}

}  // namespace

std::uint64_t TrainingEpochSeed(std::uint64_t run_seed, int epoch) {
    std::uint64_t z = run_seed + 0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(epoch + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

bool SaveTrainingResumeCheckpoint(const fs::path& directory, const TrainingResumeIdentity& identity,
                                  IExecutableModel& model, const Optimizer& optimizer,
                                  const TrainingResumeRuntime& runtime, std::string& error) {
    std::error_code ec;
    if (fs::exists(directory, ec)) {
        error = "resume checkpoint already exists: " + directory.string();
        return false;
    }
    fs::create_directories(directory, ec);
    if (ec) {
        error = "cannot create " + directory.string() + ": " + ec.message();
        return false;
    }

    CheckpointManifestV2 manifest;
    manifest.checkpoint_id = identity.run_id + "-" + directory.filename().string();
    manifest.run_id = identity.run_id;
    manifest.created_at = UtcNow();
    manifest.engine_version = GetVersionString();
    manifest.backend_version = GetVersionString();
    manifest.graph_fingerprint = identity.graph_fingerprint.empty() ? "unrecorded" : identity.graph_fingerprint;
    manifest.dataset_fingerprint = identity.dataset_fingerprint.empty() ? "unrecorded" : identity.dataset_fingerprint;
    manifest.partition_fingerprint =
        identity.partition_fingerprint.empty() ? "unrecorded" : identity.partition_fingerprint;
    manifest.model_type = identity.model_type;
    manifest.optimizer_type = identity.optimizer_type.empty() ? "unrecorded" : identity.optimizer_type;
    manifest.scheduler_type = runtime.scheduler ? runtime.scheduler->scheduler_state.scheduler_type : std::string{};
    manifest.loss_type = identity.loss_type.empty() ? "unrecorded" : identity.loss_type;
    manifest.precision = "float32";
    manifest.completed_epoch = runtime.completed_epoch;
    manifest.next_batch = runtime.in_epoch ? runtime.in_epoch->next_batch : 0;
    manifest.optimizer_step = runtime.optimizer_step_count;
    manifest.rng_state_present = runtime.model_seed >= 0;       // reseeded per epoch
    manifest.sampler_state_present = runtime.data_order_exact;  // per-epoch shuffle seed
    manifest.early_stopping_enabled = true;
    manifest.early_stopping_state_present = true;               // in runtime.json

    CheckpointPayloadDescriptor payload;
    if (!SaveModelPayloadV2(directory, kModelPath, model, payload, error)) return false;
    manifest.payloads.push_back(payload);
    if (!SaveOptimizerPayloadV2(directory, kOptimizerPath, optimizer, payload, error)) return false;
    manifest.payloads.push_back(payload);
    if (!WriteTextPayload(directory, kRuntimePath, CheckpointPayloadKind::RuntimeState,
                          RuntimeToJson(runtime).dump(2), payload, error)) {
        return false;
    }
    manifest.payloads.push_back(payload);
    if (!identity.graph_json.empty()) {
        if (!WriteTextPayload(directory, kGraphPath, CheckpointPayloadKind::GraphSnapshot, identity.graph_json,
                              payload, error)) {
            return false;
        }
        manifest.payloads.push_back(payload);
    }
    if (!identity.dataset_manifest_json.empty()) {
        if (!WriteTextPayload(directory, kDatasetsPath, CheckpointPayloadKind::DatasetManifest,
                              identity.dataset_manifest_json, payload, error)) {
            return false;
        }
        manifest.payloads.push_back(payload);
    }
    return SaveCheckpointManifestV2Atomic(directory, manifest, error);
}

bool LoadTrainingResumeCheckpoint(const fs::path& directory, IExecutableModel& model, Optimizer& optimizer,
                                  TrainingResumeRuntime& runtime, TrainingResumeIdentity& identity,
                                  std::string& error) {
    const auto manifest = LoadCheckpointManifestV2(directory, error);
    if (!manifest) return false;
    for (const auto& descriptor : manifest->payloads) {
        if (!VerifyCheckpointPayloadFile(directory, descriptor, error)) return false;
    }
    const auto* model_payload = FindPayload(*manifest, CheckpointPayloadKind::ModelParameters);
    const auto* optimizer_payload = FindPayload(*manifest, CheckpointPayloadKind::OptimizerState);
    const auto* runtime_payload = FindPayload(*manifest, CheckpointPayloadKind::RuntimeState);
    if (!model_payload || !optimizer_payload || !runtime_payload) {
        error = "resume checkpoint is missing its model, optimizer or runtime payload";
        return false;
    }
    std::string text;
    if (!ReadTextPayload(directory, *runtime_payload, text, error)) return false;
    const json runtime_json = json::parse(text, nullptr, false);
    if (runtime_json.is_discarded()) {
        error = "resume checkpoint runtime state is not valid JSON";
        return false;
    }
    if (!LoadModelPayloadV2(directory, *model_payload, model, error)) return false;
    if (!LoadOptimizerPayloadV2(directory, *optimizer_payload, optimizer, error)) return false;
    runtime = RuntimeFromJson(runtime_json);

    identity = {};
    identity.run_id = manifest->run_id;
    identity.graph_fingerprint = manifest->graph_fingerprint;
    identity.dataset_fingerprint = manifest->dataset_fingerprint;
    identity.partition_fingerprint = manifest->partition_fingerprint;
    identity.model_type = manifest->model_type;
    identity.optimizer_type = manifest->optimizer_type;
    identity.loss_type = manifest->loss_type;
    identity.scheduler_type = manifest->scheduler_type;
    if (const auto* graph = FindPayload(*manifest, CheckpointPayloadKind::GraphSnapshot)) {
        if (!ReadTextPayload(directory, *graph, identity.graph_json, error)) return false;
    }
    if (const auto* datasets = FindPayload(*manifest, CheckpointPayloadKind::DatasetManifest)) {
        if (!ReadTextPayload(directory, *datasets, identity.dataset_manifest_json, error)) return false;
    }
    return true;
}

std::string TrainingResumeCheckpointName(int epoch, int next_batch) {
    char name[40];
    std::snprintf(name, sizeof(name), "resume-%04d-%07d", epoch, next_batch);
    return name;
}

std::uint64_t TrainingStepSeed(std::uint64_t run_seed, int epoch, int batch) {
    return TrainingEpochSeed(TrainingEpochSeed(run_seed, epoch), batch);
}

std::optional<fs::path> FindLatestTrainingResumeCheckpoint(const fs::path& root) {
    std::error_code ec;
    std::optional<fs::path> latest;
    for (const auto& entry : fs::directory_iterator(root, ec)) {
        const std::string name = entry.path().filename().string();
        if (!entry.is_directory() || name.rfind("resume-", 0) != 0) continue;
        if (!fs::exists(entry.path() / "manifest.json", ec)) continue;  // incomplete
        if (!latest || name > latest->filename().string()) latest = entry.path();
    }
    return latest;
}

void PruneTrainingResumeCheckpoints(const fs::path& root, int keep) {
    std::error_code ec;
    std::vector<fs::path> checkpoints;
    for (const auto& entry : fs::directory_iterator(root, ec)) {
        if (entry.is_directory() && entry.path().filename().string().rfind("resume-", 0) == 0 &&
            fs::exists(entry.path() / "manifest.json", ec)) {
            checkpoints.push_back(entry.path());
        }
    }
    std::sort(checkpoints.begin(), checkpoints.end());
    for (size_t i = 0; i + static_cast<size_t>(std::max(0, keep)) < checkpoints.size(); ++i) {
        fs::remove_all(checkpoints[i], ec);
    }
}

}  // namespace cyxwiz
