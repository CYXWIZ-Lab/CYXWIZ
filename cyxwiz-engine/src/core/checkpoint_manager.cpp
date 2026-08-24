#include "checkpoint_manager.h"
#include "algorithms/arrayfire_backend_utils.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <limits>

namespace cyxwiz {

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr const char* kCheckpointFormatV1 = "1.0";

bool IsSupportedCheckpointDataType(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:
        case DataType::Float64:
        case DataType::Int32:
        case DataType::Int64:
        case DataType::UInt8:
            return true;
    }
    return false;
}

CheckpointInspection ClassifyCheckpointVersion(const std::string& version) {
    CheckpointInspection inspection;
    inspection.format_version = version;

    if (version == kCheckpointFormatV1) {
        inspection.valid = true;
        inspection.can_load_for_testing = true;
        inspection.can_warm_start = true;
        inspection.can_exact_resume = false;
        inspection.exact_resume_reason =
            "Checkpoint format 1.0 stores model parameters and a learning "
            "rate, but not optimizer tensors, scheduler state, RNG/sampler "
            "state, or the exact training cursor.";
        return inspection;
    }

    inspection.error = "Unsupported checkpoint format version '" + version +
                       "'. This build supports format 1.0 for loading model "
                       "weights only; exact resume requires checkpoint v2.";
    inspection.exact_resume_reason = inspection.error;
    return inspection;
}

} // namespace

// ============================================================================
// CheckpointManager Implementation
// ============================================================================

CheckpointManager::CheckpointManager(const std::string& checkpoint_dir)
    : checkpoint_dir_(checkpoint_dir)
{
    // Create checkpoint directory if it doesn't exist
    if (!fs::exists(checkpoint_dir_)) {
        fs::create_directories(checkpoint_dir_);
        spdlog::info("CheckpointManager: Created checkpoint directory: {}", checkpoint_dir);
    }
}

std::string CheckpointManager::SaveCheckpoint(
    const SequentialModel& model,
    const Optimizer* optimizer,
    const TrainingMetrics& metrics,
    const std::string& checkpoint_name)
{
    // Generate checkpoint name if not provided
    std::string name = checkpoint_name.empty() ? GenerateCheckpointName(metrics) : checkpoint_name;
    fs::path checkpoint_path = GetCheckpointPath(name);

    spdlog::info("CheckpointManager: Saving checkpoint to {}", checkpoint_path.string());

    // Create checkpoint directory
    if (!fs::create_directories(checkpoint_path)) {
        if (!fs::exists(checkpoint_path)) {
            spdlog::error("CheckpointManager: Failed to create directory {}", checkpoint_path.string());
            return "";
        }
    }

    // Prepare metadata
    CheckpointMetadata metadata;
    metadata.epoch = metrics.current_epoch;
    metadata.global_step = metrics.current_batch;
    metadata.train_loss = metrics.train_loss;
    metadata.train_accuracy = metrics.train_accuracy;
    metadata.train_mae = metrics.train_mae;
    metadata.train_rmse = metrics.train_rmse;
    metadata.val_loss = metrics.val_loss;
    metadata.val_accuracy = metrics.val_accuracy;
    metadata.val_mae = metrics.val_mae;
    metadata.val_rmse = metrics.val_rmse;
    metadata.loss_history = metrics.loss_history;
    metadata.accuracy_history = metrics.accuracy_history;
    metadata.mae_history = metrics.mae_history;
    metadata.rmse_history = metrics.rmse_history;
    metadata.val_loss_history = metrics.val_loss_history;
    metadata.val_accuracy_history = metrics.val_accuracy_history;
    metadata.val_mae_history = metrics.val_mae_history;
    metadata.val_rmse_history = metrics.val_rmse_history;

    // Add timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    metadata.timestamp = ss.str();

    // Add optimizer info if available
    if (optimizer) {
        metadata.learning_rate = static_cast<float>(optimizer->GetLearningRate());
        // Note: OptimizerType to string mapping would need to be added
        metadata.optimizer_type = "Unknown";
    }

    // Save metadata
    if (!SaveMetadata(checkpoint_path, metadata)) {
        spdlog::error("CheckpointManager: Failed to save metadata");
        return "";
    }

    // Save model parameters
    if (!SaveModelParameters(checkpoint_path, model)) {
        spdlog::error("CheckpointManager: Failed to save model parameters");
        return "";
    }

    spdlog::info("CheckpointManager: Checkpoint saved successfully to {}", checkpoint_path.string());

    // Clean up old checkpoints
    CleanupOldCheckpoints();

    return checkpoint_path.string();
}

std::optional<CheckpointMetadata> CheckpointManager::LoadCheckpoint(
    SequentialModel& model,
    Optimizer* optimizer,
    const std::string& checkpoint_name)
{
    last_error_.clear();

    // Find checkpoint path
    fs::path checkpoint_path;
    if (checkpoint_name.empty()) {
        std::string latest = GetLatestCheckpoint();
        if (latest.empty()) {
            last_error_ = "No checkpoints were found in '" +
                          checkpoint_dir_.string() + "'.";
            spdlog::warn("CheckpointManager: No checkpoints found");
            return std::nullopt;
        }
        checkpoint_path = latest;
    } else {
        checkpoint_path = GetCheckpointPath(checkpoint_name);
    }

    if (!fs::exists(checkpoint_path)) {
        last_error_ = "Checkpoint directory was not found: " +
                      checkpoint_path.string();
        spdlog::error("CheckpointManager: Checkpoint not found: {}", checkpoint_path.string());
        return std::nullopt;
    }

    spdlog::info("CheckpointManager: Loading checkpoint from {}", checkpoint_path.string());

    // Load metadata
    auto metadata = LoadMetadata(checkpoint_path);
    if (!metadata) {
        if (last_error_.empty()) {
            last_error_ = "Checkpoint metadata is missing or invalid at '" +
                          checkpoint_path.string() + "'.";
        }
        spdlog::error("CheckpointManager: Failed to load metadata");
        return std::nullopt;
    }

    const auto inspection = ClassifyCheckpointVersion(metadata->version);
    if (!inspection.can_load_for_testing) {
        last_error_ = inspection.error;
        spdlog::error("CheckpointManager: {}", last_error_);
        return std::nullopt;
    }

    // Load model parameters
    if (!LoadModelParameters(checkpoint_path, model)) {
        if (last_error_.empty()) {
            last_error_ = "Checkpoint model parameters are invalid at '" +
                          checkpoint_path.string() + "'.";
        }
        spdlog::error("CheckpointManager: Failed to load model parameters");
        return std::nullopt;
    }

    // Restore optimizer learning rate if available
    if (optimizer && metadata->learning_rate > 0) {
        optimizer->SetLearningRate(metadata->learning_rate);
    }

    spdlog::info("CheckpointManager: Checkpoint loaded successfully from epoch {}",
                 metadata->epoch);

    return metadata;
}

CheckpointInspection CheckpointManager::InspectCheckpoint(
    const std::string& checkpoint_name)
{
    last_error_.clear();

    fs::path checkpoint_path;
    if (checkpoint_name.empty()) {
        const std::string latest = GetLatestCheckpoint();
        if (latest.empty()) {
            CheckpointInspection inspection;
            inspection.error = "No checkpoints were found in '" +
                               checkpoint_dir_.string() + "'.";
            last_error_ = inspection.error;
            return inspection;
        }
        checkpoint_path = latest;
    } else {
        checkpoint_path = GetCheckpointPath(checkpoint_name);
    }

    if (!fs::exists(checkpoint_path)) {
        CheckpointInspection inspection;
        inspection.error = "Checkpoint directory was not found: " +
                           checkpoint_path.string();
        last_error_ = inspection.error;
        return inspection;
    }

    const auto metadata = LoadMetadata(checkpoint_path);
    if (!metadata) {
        CheckpointInspection inspection;
        inspection.error = last_error_.empty()
            ? "Checkpoint metadata is missing or invalid at '" +
                  checkpoint_path.string() + "'."
            : last_error_;
        return inspection;
    }

    auto inspection = ClassifyCheckpointVersion(metadata->version);
    if (!inspection.valid) {
        last_error_ = inspection.error;
    }
    return inspection;
}

bool CheckpointManager::SaveBestModel(
    const SequentialModel& model,
    const Optimizer* optimizer,
    const TrainingMetrics& metrics,
    float val_loss)
{
    if (val_loss < best_val_loss_) {
        best_val_loss_ = val_loss;
        std::string path = SaveCheckpoint(model, optimizer, metrics, "best");
        if (!path.empty()) {
            spdlog::info("CheckpointManager: New best model saved (val_loss={:.4f})", val_loss);
            return true;
        }
    }
    return false;
}

std::vector<std::string> CheckpointManager::ListCheckpoints() const {
    std::vector<std::pair<std::string, fs::file_time_type>> checkpoints;

    if (!fs::exists(checkpoint_dir_)) {
        return {};
    }

    for (const auto& entry : fs::directory_iterator(checkpoint_dir_)) {
        if (entry.is_directory()) {
            fs::path metadata_path = entry.path() / "metadata.json";
            if (fs::exists(metadata_path)) {
                checkpoints.push_back({
                    entry.path().filename().string(),
                    entry.last_write_time()
                });
            }
        }
    }

    // Sort by time (newest first)
    std::sort(checkpoints.begin(), checkpoints.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<std::string> names;
    names.reserve(checkpoints.size());
    for (const auto& [name, time] : checkpoints) {
        names.push_back(name);
    }

    return names;
}

bool CheckpointManager::DeleteCheckpoint(const std::string& checkpoint_name) {
    fs::path checkpoint_path = GetCheckpointPath(checkpoint_name);

    if (!fs::exists(checkpoint_path)) {
        spdlog::warn("CheckpointManager: Checkpoint not found: {}", checkpoint_name);
        return false;
    }

    std::error_code ec;
    fs::remove_all(checkpoint_path, ec);
    if (ec) {
        spdlog::error("CheckpointManager: Failed to delete checkpoint: {}", ec.message());
        return false;
    }

    spdlog::info("CheckpointManager: Deleted checkpoint: {}", checkpoint_name);
    return true;
}

std::string CheckpointManager::GetLatestCheckpoint() const {
    auto checkpoints = ListCheckpoints();

    // Filter out "best" checkpoint
    for (const auto& name : checkpoints) {
        if (name != "best") {
            return GetCheckpointPath(name).string();
        }
    }

    return "";
}

std::string CheckpointManager::GetBestCheckpoint() const {
    fs::path best_path = GetCheckpointPath("best");
    return fs::exists(best_path) ? best_path.string() : "";
}

// ============================================================================
// Private Helpers
// ============================================================================

std::string CheckpointManager::GenerateCheckpointName(const TrainingMetrics& metrics) const {
    std::stringstream ss;
    ss << "epoch_" << std::setfill('0') << std::setw(4) << metrics.current_epoch;
    return ss.str();
}

bool CheckpointManager::SaveMetadata(const fs::path& dir, const CheckpointMetadata& metadata) {
    try {
        json j;
        j["version"] = metadata.version;
        j["epoch"] = metadata.epoch;
        j["global_step"] = metadata.global_step;
        j["train_loss"] = metadata.train_loss;
        j["train_accuracy"] = metadata.train_accuracy;
        j["train_mae"] = metadata.train_mae;
        j["train_rmse"] = metadata.train_rmse;
        j["val_loss"] = metadata.val_loss;
        j["val_accuracy"] = metadata.val_accuracy;
        j["val_mae"] = metadata.val_mae;
        j["val_rmse"] = metadata.val_rmse;
        j["model_name"] = metadata.model_name;
        j["optimizer_type"] = metadata.optimizer_type;
        j["learning_rate"] = metadata.learning_rate;
        j["timestamp"] = metadata.timestamp;

        // Save history arrays
        j["loss_history"] = metadata.loss_history;
        j["accuracy_history"] = metadata.accuracy_history;
        j["mae_history"] = metadata.mae_history;
        j["rmse_history"] = metadata.rmse_history;
        j["val_loss_history"] = metadata.val_loss_history;
        j["val_accuracy_history"] = metadata.val_accuracy_history;
        j["val_mae_history"] = metadata.val_mae_history;
        j["val_rmse_history"] = metadata.val_rmse_history;

        fs::path metadata_path = dir / "metadata.json";
        std::ofstream file(metadata_path);
        if (!file.is_open()) {
            spdlog::error("CheckpointManager: Failed to open metadata file for writing: {}", metadata_path.string());
            return false;
        }
        file << j.dump(2);  // Pretty print with 2-space indent
        if (!file.good()) {
            spdlog::error("CheckpointManager: Failed to write metadata to file: {}", metadata_path.string());
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        spdlog::error("CheckpointManager: Error saving metadata: {}", e.what());
        return false;
    }
}

std::optional<CheckpointMetadata> CheckpointManager::LoadMetadata(const fs::path& dir) {
    try {
        fs::path metadata_path = dir / "metadata.json";
        std::ifstream file(metadata_path);
        if (!file.is_open()) {
            last_error_ = "Checkpoint metadata file is unreadable: " +
                          metadata_path.string();
            return std::nullopt;
        }

        json j;
        file >> j;

        CheckpointMetadata metadata;
        metadata.version = j.value("version", "1.0");
        metadata.epoch = j.value("epoch", 0);
        metadata.global_step = j.value("global_step", 0);
        metadata.train_loss = j.value("train_loss", 0.0f);
        metadata.train_accuracy = j.value("train_accuracy", 0.0f);
        metadata.train_mae = j.value("train_mae", 0.0f);
        metadata.train_rmse = j.value("train_rmse", 0.0f);
        metadata.val_loss = j.value("val_loss", 0.0f);
        metadata.val_accuracy = j.value("val_accuracy", 0.0f);
        metadata.val_mae = j.value("val_mae", 0.0f);
        metadata.val_rmse = j.value("val_rmse", 0.0f);
        metadata.model_name = j.value("model_name", "");
        metadata.optimizer_type = j.value("optimizer_type", "");
        metadata.learning_rate = j.value("learning_rate", 0.0f);
        metadata.timestamp = j.value("timestamp", "");

        // Load history arrays
        if (j.contains("loss_history")) {
            metadata.loss_history = j["loss_history"].get<std::vector<float>>();
        }
        if (j.contains("accuracy_history")) {
            metadata.accuracy_history = j["accuracy_history"].get<std::vector<float>>();
        }
        if (j.contains("mae_history")) {
            metadata.mae_history = j["mae_history"].get<std::vector<float>>();
        }
        if (j.contains("rmse_history")) {
            metadata.rmse_history = j["rmse_history"].get<std::vector<float>>();
        }
        if (j.contains("val_loss_history")) {
            metadata.val_loss_history = j["val_loss_history"].get<std::vector<float>>();
        }
        if (j.contains("val_accuracy_history")) {
            metadata.val_accuracy_history = j["val_accuracy_history"].get<std::vector<float>>();
        }
        if (j.contains("val_mae_history")) {
            metadata.val_mae_history = j["val_mae_history"].get<std::vector<float>>();
        }
        if (j.contains("val_rmse_history")) {
            metadata.val_rmse_history = j["val_rmse_history"].get<std::vector<float>>();
        }

        return metadata;
    } catch (const std::exception& e) {
        last_error_ = "Checkpoint metadata is invalid: " +
                      std::string(e.what());
        spdlog::error("CheckpointManager: Error loading metadata: {}", e.what());
        return std::nullopt;
    }
}

bool CheckpointManager::SaveModelParameters(const fs::path& dir, const SequentialModel& model) {
    try {
        fs::path model_dir = dir / "model";
        fs::create_directories(model_dir);

        // Get all parameters from the model
        // Note: We need const_cast here because GetParameters isn't const
        auto& mutable_model = const_cast<SequentialModel&>(model);
        auto params = mutable_model.GetParameters();

        spdlog::debug("CheckpointManager: Saving {} parameters", params.size());

        for (const auto& [name, tensor] : params) {
            // Convert param name to filename (replace dots with underscores)
            std::string filename = name;
            std::replace(filename.begin(), filename.end(), '.', '_');
            filename += ".bin";

            fs::path tensor_path = model_dir / filename;
            if (!SaveTensor(tensor_path, tensor)) {
                spdlog::error("CheckpointManager: Failed to save parameter: {}", name);
                return false;
            }
        }

        // Save parameter manifest (maps filenames back to parameter names)
        json manifest;
        for (const auto& [name, tensor] : params) {
            std::string filename = name;
            std::replace(filename.begin(), filename.end(), '.', '_');
            filename += ".bin";
            manifest[filename] = name;
        }

        fs::path manifest_path = model_dir / "manifest.json";
        std::ofstream manifest_file(manifest_path);
        if (!manifest_file.is_open()) {
            spdlog::error("CheckpointManager: Failed to open manifest file for writing: {}", manifest_path.string());
            return false;
        }
        manifest_file << manifest.dump(2);
        if (!manifest_file.good()) {
            spdlog::error("CheckpointManager: Failed to write manifest to file: {}", manifest_path.string());
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::error("CheckpointManager: Error saving model parameters: {}", e.what());
        return false;
    }
}

bool CheckpointManager::LoadModelParameters(const fs::path& dir, SequentialModel& model) {
    try {
        fs::path model_dir = dir / "model";
        if (!fs::exists(model_dir)) {
            last_error_ = "Checkpoint model directory was not found: " +
                          model_dir.string();
            spdlog::error("CheckpointManager: Model directory not found: {}", model_dir.string());
            return false;
        }

        // Load manifest
        fs::path manifest_path = model_dir / "manifest.json";
        if (!fs::exists(manifest_path)) {
            last_error_ = "Checkpoint parameter manifest was not found: " +
                          manifest_path.string();
            spdlog::error("CheckpointManager: Manifest not found");
            return false;
        }

        std::ifstream manifest_file(manifest_path);
        json manifest;
        manifest_file >> manifest;
        if (!manifest.is_object()) {
            last_error_ = "Checkpoint parameter manifest must be a JSON object: " +
                          manifest_path.string();
            return false;
        }

        // Load parameters
        std::map<std::string, Tensor> params;
        for (auto& [filename, param_name] : manifest.items()) {
            fs::path tensor_path = model_dir / filename;
            auto tensor = LoadTensor(tensor_path);
            if (!tensor) {
                if (last_error_.empty()) {
                    last_error_ = "Could not load checkpoint parameter '" +
                                  param_name.get<std::string>() + "'.";
                }
                spdlog::error("CheckpointManager: Failed to load parameter: {}", param_name.get<std::string>());
                return false;
            }
            params[param_name.get<std::string>()] = std::move(*tensor);
        }

        // Validate the complete checkpoint before mutating the destination
        // model. A failed load must leave the previously active model intact.
        const auto expected = model.GetParameters();
        if (expected.size() != params.size()) {
            last_error_ = "Checkpoint parameter count (" +
                          std::to_string(params.size()) +
                          ") does not match the active graph model (" +
                          std::to_string(expected.size()) + ").";
            return false;
        }
        for (const auto& [name, expected_tensor] : expected) {
            const auto found = params.find(name);
            if (found == params.end()) {
                last_error_ = "Checkpoint is missing model parameter '" +
                              name + "'.";
                return false;
            }
            if (found->second.Shape() != expected_tensor.Shape()) {
                last_error_ = "Checkpoint shape mismatch for parameter '" +
                              name + "'.";
                return false;
            }
            if (found->second.GetDataType() != expected_tensor.GetDataType()) {
                last_error_ = "Checkpoint data type mismatch for parameter '" +
                              name + "'.";
                return false;
            }
        }

        // Set parameters only after compatibility has passed for every tensor.
        model.SetParameters(params);

        spdlog::debug("CheckpointManager: Loaded {} parameters", params.size());
        return true;
    } catch (const std::exception& e) {
        last_error_ = "Checkpoint parameter loading failed: " +
                      std::string(e.what());
        spdlog::error("CheckpointManager: Error loading model parameters: {}", e.what());
        return false;
    }
}

bool CheckpointManager::SaveTensor(const fs::path& path, const Tensor& tensor) {
    try {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            spdlog::error("CheckpointManager: Failed to open tensor file for writing: {}", path.string());
            return false;
        }

        // Write header: ndims, shape, dtype
        const auto& shape = tensor.Shape();
        uint32_t ndims = static_cast<uint32_t>(shape.size());
        uint32_t dtype = static_cast<uint32_t>(tensor.GetDataType());

        file.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
        if (!file.good()) {
            spdlog::error("CheckpointManager: Failed to write tensor ndims to file: {}", path.string());
            return false;
        }

        for (size_t dim : shape) {
            uint64_t d = static_cast<uint64_t>(dim);
            file.write(reinterpret_cast<const char*>(&d), sizeof(d));
            if (!file.good()) {
                spdlog::error("CheckpointManager: Failed to write tensor shape to file: {}", path.string());
                return false;
            }
        }

        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
        if (!file.good()) {
            spdlog::error("CheckpointManager: Failed to write tensor dtype to file: {}", path.string());
            return false;
        }

        // Write data
        const ScopedArrayFireHostSyncAttribution checkpoint_attribution(
            ArrayFireHostSyncCategory::CheckpointOutput,
            "CheckpointManager::SaveTensor");
        const size_t num_bytes = tensor.NumBytes();
        if (num_bytes > static_cast<size_t>(
                (std::numeric_limits<std::streamsize>::max)())) {
            spdlog::error(
                "CheckpointManager: Tensor payload is too large to serialize: {}",
                path.string());
            return false;
        }
        const void* data = tensor.ReadData();
        file.write(reinterpret_cast<const char*>(data),
                   static_cast<std::streamsize>(num_bytes));
        if (!file.good()) {
            spdlog::error("CheckpointManager: Failed to write tensor data to file: {}", path.string());
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::error("CheckpointManager: Error saving tensor: {}", e.what());
        return false;
    }
}

std::optional<Tensor> CheckpointManager::LoadTensor(const fs::path& path) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            last_error_ = "Checkpoint tensor is unreadable: " + path.string();
            return std::nullopt;
        }

        // Read header
        uint32_t ndims;
        file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
        if (!file.good() || ndims > 16) {
            last_error_ = "Checkpoint tensor has an invalid rank header: " +
                          path.string();
            return std::nullopt;
        }

        std::vector<size_t> shape(ndims);
        for (uint32_t i = 0; i < ndims; ++i) {
            uint64_t d;
            file.read(reinterpret_cast<char*>(&d), sizeof(d));
            if (!file.good() || d == 0) {
                last_error_ = "Checkpoint tensor has an invalid shape header: " +
                              path.string();
                return std::nullopt;
            }
            shape[i] = static_cast<size_t>(d);
        }

        uint32_t dtype;
        file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
        if (!file.good()) {
            last_error_ = "Checkpoint tensor has an invalid data type header: " +
                          path.string();
            return std::nullopt;
        }
        const auto data_type = static_cast<DataType>(dtype);
        if (!IsSupportedCheckpointDataType(data_type)) {
            last_error_ = "Checkpoint tensor has an unsupported data type header: " +
                          path.string();
            return std::nullopt;
        }

        // Create tensor with shape
        Tensor tensor(shape, data_type);

        // Read data
        const size_t num_bytes = tensor.NumBytes();
        if (num_bytes > static_cast<size_t>(
                (std::numeric_limits<std::streamsize>::max)())) {
            last_error_ = "Checkpoint tensor payload is too large to load: " +
                          path.string();
            return std::nullopt;
        }
        void* data = tensor.MutableData();
        file.read(reinterpret_cast<char*>(data),
                  static_cast<std::streamsize>(num_bytes));
        if (!file.good()) {
            last_error_ = "Checkpoint tensor data is truncated: " + path.string();
            return std::nullopt;
        }

        return tensor;
    } catch (const std::exception& e) {
        last_error_ = "Checkpoint tensor loading failed for '" + path.string() +
                      "': " + e.what();
        spdlog::error("CheckpointManager: Error loading tensor: {}", e.what());
        return std::nullopt;
    }
}

void CheckpointManager::CleanupOldCheckpoints() {
    if (max_to_keep_ == 0) {
        return;  // Unlimited checkpoints
    }

    auto checkpoints = ListCheckpoints();

    // Filter out "best" checkpoint (should not be deleted)
    std::vector<std::string> deletable;
    for (const auto& name : checkpoints) {
        if (name != "best") {
            deletable.push_back(name);
        }
    }

    // Delete oldest checkpoints beyond limit
    while (deletable.size() > max_to_keep_) {
        const std::string& oldest = deletable.back();
        spdlog::debug("CheckpointManager: Deleting old checkpoint: {}", oldest);
        DeleteCheckpoint(oldest);
        deletable.pop_back();
    }
}

fs::path CheckpointManager::GetCheckpointPath(const std::string& name) const {
    return checkpoint_dir_ / name;
}

} // namespace cyxwiz
