#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

enum class CheckpointPayloadKind {
    ModelParameters,
    OptimizerState,
    SchedulerState,
    RuntimeState,
    GraphSnapshot,
    DatasetManifest,
};

struct CheckpointPayloadDescriptor {
    CheckpointPayloadKind kind = CheckpointPayloadKind::ModelParameters;
    std::string relative_path;
    std::uint64_t byte_size = 0;
    std::string sha256;
    bool required = true;
};

/**
 * Checkpoint v2 manifest. Payload files are written before this manifest is
 * published; manifest publication is the checkpoint commit point.
 */
struct CheckpointManifestV2 {
    int schema_version = 2;
    std::string checkpoint_id;
    std::string run_id;
    std::string parent_checkpoint_id;
    std::string created_at;
    std::string engine_version;
    std::string backend_version;

    std::string graph_fingerprint;
    std::string dataset_fingerprint;
    std::string partition_fingerprint;
    std::string model_type;
    std::string optimizer_type;
    std::string scheduler_type;
    std::string loss_type;
    std::string precision;

    int completed_epoch = 0;
    int next_batch = 0;
    int optimizer_step = 0;
    int accumulation_step = 0;

    bool rng_state_present = false;
    bool sampler_state_present = false;
    bool early_stopping_enabled = false;
    bool early_stopping_state_present = false;

    std::vector<CheckpointPayloadDescriptor> payloads;
};

struct CheckpointManifestValidation {
    bool valid = false;
    // This validates the declared inventory only. Payload bytes and hashes
    // must still be verified before exact resume is enabled.
    bool declares_exact_resume_state = false;
    std::vector<std::string> errors;
    std::vector<std::string> missing_exact_resume_state;
};

std::string ToString(CheckpointPayloadKind kind);
std::optional<CheckpointPayloadKind> CheckpointPayloadKindFromString(
    const std::string& value);

bool IsSafeCheckpointPayloadPath(const std::string& value);

CheckpointManifestValidation ValidateCheckpointManifestV2(
    const CheckpointManifestV2& manifest);

bool SaveCheckpointManifestV2Atomic(
    const std::filesystem::path& checkpoint_directory,
    const CheckpointManifestV2& manifest,
    std::string& error);

std::optional<CheckpointManifestV2> LoadCheckpointManifestV2(
    const std::filesystem::path& checkpoint_directory,
    std::string& error);

} // namespace cyxwiz
