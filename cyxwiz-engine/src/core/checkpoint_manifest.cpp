#include "checkpoint_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <set>
#include <stdexcept>

namespace cyxwiz {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr const char* kCheckpointFormat = "cyxwiz_checkpoint";

bool IsSha256(const std::string& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char ch) {
               return std::isxdigit(ch) != 0;
           });
}

json ToJson(const CheckpointManifestV2& manifest) {
    json payloads = json::array();
    for (const auto& payload : manifest.payloads) {
        payloads.push_back({
            {"kind", ToString(payload.kind)},
            {"relative_path", payload.relative_path},
            {"byte_size", payload.byte_size},
            {"sha256", payload.sha256},
            {"required", payload.required},
        });
    }

    return {
        {"format", kCheckpointFormat},
        {"schema_version", manifest.schema_version},
        {"checkpoint_id", manifest.checkpoint_id},
        {"run_id", manifest.run_id},
        {"parent_checkpoint_id", manifest.parent_checkpoint_id},
        {"created_at", manifest.created_at},
        {"engine_version", manifest.engine_version},
        {"backend_version", manifest.backend_version},
        {"graph_fingerprint", manifest.graph_fingerprint},
        {"dataset_fingerprint", manifest.dataset_fingerprint},
        {"partition_fingerprint", manifest.partition_fingerprint},
        {"model_type", manifest.model_type},
        {"optimizer_type", manifest.optimizer_type},
        {"scheduler_type", manifest.scheduler_type},
        {"loss_type", manifest.loss_type},
        {"precision", manifest.precision},
        {"cursor", {
            {"completed_epoch", manifest.completed_epoch},
            {"next_batch", manifest.next_batch},
            {"optimizer_step", manifest.optimizer_step},
            {"accumulation_step", manifest.accumulation_step},
        }},
        {"runtime_state", {
            {"rng_present", manifest.rng_state_present},
            {"sampler_present", manifest.sampler_state_present},
            {"early_stopping_enabled", manifest.early_stopping_enabled},
            {"early_stopping_present", manifest.early_stopping_state_present},
        }},
        {"payloads", std::move(payloads)},
    };
}

CheckpointManifestV2 FromJson(const json& value) {
    if (!value.is_object() ||
        value.value("format", std::string{}) != kCheckpointFormat) {
        throw std::runtime_error("manifest format must be 'cyxwiz_checkpoint'");
    }

    CheckpointManifestV2 manifest;
    manifest.schema_version = value.at("schema_version").get<int>();
    manifest.checkpoint_id = value.at("checkpoint_id").get<std::string>();
    manifest.run_id = value.at("run_id").get<std::string>();
    manifest.parent_checkpoint_id =
        value.value("parent_checkpoint_id", std::string{});
    manifest.created_at = value.at("created_at").get<std::string>();
    manifest.engine_version = value.at("engine_version").get<std::string>();
    manifest.backend_version = value.at("backend_version").get<std::string>();
    manifest.graph_fingerprint =
        value.at("graph_fingerprint").get<std::string>();
    manifest.dataset_fingerprint =
        value.at("dataset_fingerprint").get<std::string>();
    manifest.partition_fingerprint =
        value.at("partition_fingerprint").get<std::string>();
    manifest.model_type = value.at("model_type").get<std::string>();
    manifest.optimizer_type = value.at("optimizer_type").get<std::string>();
    manifest.scheduler_type = value.value("scheduler_type", std::string{});
    manifest.loss_type = value.at("loss_type").get<std::string>();
    manifest.precision = value.at("precision").get<std::string>();

    const auto& cursor = value.at("cursor");
    manifest.completed_epoch = cursor.at("completed_epoch").get<int>();
    manifest.next_batch = cursor.at("next_batch").get<int>();
    manifest.optimizer_step = cursor.at("optimizer_step").get<int>();
    manifest.accumulation_step = cursor.at("accumulation_step").get<int>();

    const auto& runtime = value.at("runtime_state");
    manifest.rng_state_present = runtime.at("rng_present").get<bool>();
    manifest.sampler_state_present = runtime.at("sampler_present").get<bool>();
    manifest.early_stopping_enabled =
        runtime.at("early_stopping_enabled").get<bool>();
    manifest.early_stopping_state_present =
        runtime.at("early_stopping_present").get<bool>();

    for (const auto& payload_value : value.at("payloads")) {
        const auto kind = CheckpointPayloadKindFromString(
            payload_value.at("kind").get<std::string>());
        if (!kind) throw std::runtime_error("manifest contains unknown payload kind");
        CheckpointPayloadDescriptor payload;
        payload.kind = *kind;
        payload.relative_path =
            payload_value.at("relative_path").get<std::string>();
        payload.byte_size = payload_value.at("byte_size").get<std::uint64_t>();
        payload.sha256 = payload_value.at("sha256").get<std::string>();
        payload.required = payload_value.value("required", true);
        manifest.payloads.push_back(std::move(payload));
    }
    return manifest;
}

} // namespace

std::string ToString(CheckpointPayloadKind kind) {
    switch (kind) {
        case CheckpointPayloadKind::ModelParameters: return "model_parameters";
        case CheckpointPayloadKind::OptimizerState: return "optimizer_state";
        case CheckpointPayloadKind::SchedulerState: return "scheduler_state";
        case CheckpointPayloadKind::RuntimeState: return "runtime_state";
        case CheckpointPayloadKind::GraphSnapshot: return "graph_snapshot";
        case CheckpointPayloadKind::DatasetManifest: return "dataset_manifest";
    }
    return "unknown";
}

std::optional<CheckpointPayloadKind> CheckpointPayloadKindFromString(
    const std::string& value)
{
    if (value == "model_parameters") return CheckpointPayloadKind::ModelParameters;
    if (value == "optimizer_state") return CheckpointPayloadKind::OptimizerState;
    if (value == "scheduler_state") return CheckpointPayloadKind::SchedulerState;
    if (value == "runtime_state") return CheckpointPayloadKind::RuntimeState;
    if (value == "graph_snapshot") return CheckpointPayloadKind::GraphSnapshot;
    if (value == "dataset_manifest") return CheckpointPayloadKind::DatasetManifest;
    return std::nullopt;
}

bool IsSafeCheckpointPayloadPath(const std::string& value) {
    if (value.empty()) return false;
    const fs::path path(value);
    if (path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        return false;
    }
    for (const auto& part : path) {
        if (part == ".." || part == ".") return false;
    }
    return true;
}

CheckpointManifestValidation ValidateCheckpointManifestV2(
    const CheckpointManifestV2& manifest)
{
    CheckpointManifestValidation result;
    if (manifest.schema_version != 2) {
        result.errors.push_back("checkpoint manifest schema_version must be 2");
    }

    const std::pair<const char*, const std::string*> required_strings[] = {
        {"checkpoint_id", &manifest.checkpoint_id},
        {"run_id", &manifest.run_id},
        {"created_at", &manifest.created_at},
        {"engine_version", &manifest.engine_version},
        {"backend_version", &manifest.backend_version},
        {"graph_fingerprint", &manifest.graph_fingerprint},
        {"dataset_fingerprint", &manifest.dataset_fingerprint},
        {"partition_fingerprint", &manifest.partition_fingerprint},
        {"model_type", &manifest.model_type},
        {"optimizer_type", &manifest.optimizer_type},
        {"loss_type", &manifest.loss_type},
        {"precision", &manifest.precision},
    };
    for (const auto& [name, value] : required_strings) {
        if (value->empty()) result.errors.push_back(std::string(name) + " is required");
    }
    if (manifest.completed_epoch < 0 || manifest.next_batch < 0 ||
        manifest.optimizer_step < 0 || manifest.accumulation_step < 0) {
        result.errors.push_back("checkpoint training cursor values cannot be negative");
    }

    std::set<CheckpointPayloadKind> required_kinds;
    std::set<std::string> paths;
    for (const auto& payload : manifest.payloads) {
        if (payload.required) required_kinds.insert(payload.kind);
        if (!IsSafeCheckpointPayloadPath(payload.relative_path)) {
            result.errors.push_back("payload path must be a safe relative path: " +
                                    payload.relative_path);
        } else if (!paths.insert(payload.relative_path).second) {
            result.errors.push_back("payload path is duplicated: " +
                                    payload.relative_path);
        }
        if (payload.byte_size == 0) {
            result.errors.push_back("payload byte_size must be non-zero: " +
                                    payload.relative_path);
        }
        if (!IsSha256(payload.sha256)) {
            result.errors.push_back("payload sha256 must contain 64 hex digits: " +
                                    payload.relative_path);
        }
    }

    const std::pair<CheckpointPayloadKind, const char*> exact_required[] = {
        {CheckpointPayloadKind::ModelParameters, "model parameters payload"},
        {CheckpointPayloadKind::OptimizerState, "optimizer state payload"},
        {CheckpointPayloadKind::RuntimeState, "runtime state payload"},
        {CheckpointPayloadKind::GraphSnapshot, "graph snapshot payload"},
        {CheckpointPayloadKind::DatasetManifest, "dataset manifest payload"},
    };
    for (const auto& [kind, label] : exact_required) {
        if (required_kinds.count(kind) == 0) {
            result.missing_exact_resume_state.push_back(label);
        }
    }
    if (!manifest.scheduler_type.empty() &&
        required_kinds.count(CheckpointPayloadKind::SchedulerState) == 0) {
        result.missing_exact_resume_state.push_back("scheduler state payload");
    }
    if (!manifest.rng_state_present) {
        result.missing_exact_resume_state.push_back("RNG state");
    }
    if (!manifest.sampler_state_present) {
        result.missing_exact_resume_state.push_back("sampler state");
    }
    if (manifest.early_stopping_enabled &&
        !manifest.early_stopping_state_present) {
        result.missing_exact_resume_state.push_back("early-stopping state");
    }

    result.valid = result.errors.empty();
    result.declares_exact_resume_state =
        result.valid && result.missing_exact_resume_state.empty();
    return result;
}

bool SaveCheckpointManifestV2Atomic(
    const fs::path& checkpoint_directory,
    const CheckpointManifestV2& manifest,
    std::string& error)
{
    error.clear();
    const auto validation = ValidateCheckpointManifestV2(manifest);
    if (!validation.valid) {
        error = validation.errors.front();
        return false;
    }

    std::error_code ec;
    fs::create_directories(checkpoint_directory, ec);
    if (ec) {
        error = "could not create checkpoint directory: " + ec.message();
        return false;
    }

    const fs::path manifest_path = checkpoint_directory / "manifest.json";
    if (fs::exists(manifest_path)) {
        error = "checkpoint v2 manifest already exists and is immutable: " +
                manifest_path.string();
        return false;
    }

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path temporary_path = manifest_path;
    temporary_path += ".tmp." + std::to_string(nonce);
    try {
        std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
            error = "could not open temporary checkpoint manifest";
            return false;
        }
        output << ToJson(manifest).dump(2);
        output.flush();
        if (!output.good()) {
            error = "could not write temporary checkpoint manifest";
            output.close();
            fs::remove(temporary_path, ec);
            return false;
        }
        output.close();
        fs::rename(temporary_path, manifest_path, ec);
        if (ec) {
            error = "could not publish checkpoint manifest atomically: " +
                    ec.message();
            fs::remove(temporary_path, ec);
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        error = std::string("checkpoint manifest save failed: ") + exception.what();
        fs::remove(temporary_path, ec);
        return false;
    }
}

std::optional<CheckpointManifestV2> LoadCheckpointManifestV2(
    const fs::path& checkpoint_directory,
    std::string& error)
{
    error.clear();
    const fs::path manifest_path = checkpoint_directory / "manifest.json";
    try {
        std::ifstream input(manifest_path, std::ios::binary);
        if (!input.is_open()) {
            error = "checkpoint v2 manifest is unreadable: " +
                    manifest_path.string();
            return std::nullopt;
        }
        json value;
        input >> value;
        auto manifest = FromJson(value);
        const auto validation = ValidateCheckpointManifestV2(manifest);
        if (!validation.valid) {
            error = validation.errors.front();
            return std::nullopt;
        }
        return manifest;
    } catch (const std::exception& exception) {
        error = std::string("checkpoint v2 manifest is invalid: ") +
                exception.what();
        return std::nullopt;
    }
}

} // namespace cyxwiz
