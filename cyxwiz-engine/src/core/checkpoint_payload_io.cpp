#include "checkpoint_payload_io.h"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>

namespace cyxwiz {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr std::array<char, 8> kArchiveMagic = {
    'C', 'Y', 'X', 'V', '2', 'A', 'R', '1'};
constexpr std::uint32_t kArchiveSchemaVersion = 1;
constexpr std::uint64_t kMaxArchiveHeaderBytes = 16ULL * 1024ULL * 1024ULL;
constexpr std::size_t kHashBlockBytes = 64 * 1024;

using TensorMap = std::map<std::string, Tensor>;

struct LoadedArchive {
    std::string payload_type;
    json metadata;
    TensorMap tensors;
};

std::size_t DataTypeBytes(DataType type) {
    switch (type) {
        case DataType::Float32: return 4;
        case DataType::Float64: return 8;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        case DataType::UInt8: return 1;
    }
    return 0;
}

bool ComputeFileSha256(
    const fs::path& path,
    std::string& digest,
    std::string& error)
{
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        error = "checkpoint payload is unreadable: " + path.string();
        return false;
    }

    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
        EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        error = "could not initialize SHA-256 for checkpoint payload";
        return false;
    }

    std::array<char, kHashBlockBytes> buffer{};
    while (input.good()) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0 &&
            EVP_DigestUpdate(context.get(), buffer.data(),
                             static_cast<std::size_t>(count)) != 1) {
            error = "could not update checkpoint payload SHA-256";
            return false;
        }
    }
    if (!input.eof()) {
        error = "could not read checkpoint payload for SHA-256";
        return false;
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(context.get(), bytes.data(), &length) != 1 ||
        length != 32) {
        error = "could not finalize checkpoint payload SHA-256";
        return false;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < length; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(bytes[index]);
    }
    digest = output.str();
    return true;
}

bool ResolvePayloadPath(
    const fs::path& directory,
    const std::string& relative_path,
    fs::path& resolved,
    std::string& error)
{
    if (!IsSafeCheckpointPayloadPath(relative_path)) {
        error = "checkpoint payload path must be safe and relative: " +
                relative_path;
        return false;
    }
    resolved = directory / fs::path(relative_path);
    return true;
}

bool WriteArchiveAtomic(
    const fs::path& checkpoint_directory,
    const std::string& relative_path,
    const std::string& payload_type,
    const json& metadata,
    const TensorMap& tensors,
    CheckpointPayloadKind kind,
    CheckpointPayloadDescriptor& descriptor,
    std::string& error)
{
    error.clear();
    fs::path final_path;
    if (!ResolvePayloadPath(checkpoint_directory, relative_path,
                            final_path, error)) {
        return false;
    }
    if (payload_type == "model_parameters" && tensors.empty()) {
        error = "checkpoint tensor payload cannot be empty";
        return false;
    }

    json tensor_inventory = json::array();
    for (const auto& [name, tensor] : tensors) {
        if (name.empty() || tensor.NumBytes() == 0) {
            error = "checkpoint tensor payload contains an invalid tensor";
            return false;
        }
        tensor_inventory.push_back({
            {"name", name},
            {"shape", tensor.Shape()},
            {"data_type", static_cast<int>(tensor.GetDataType())},
            {"byte_size", tensor.NumBytes()},
        });
    }
    const json header = {
        {"archive_schema_version", kArchiveSchemaVersion},
        {"payload_type", payload_type},
        {"metadata", metadata},
        {"tensors", std::move(tensor_inventory)},
    };
    const std::string header_bytes = header.dump();
    if (header_bytes.size() > kMaxArchiveHeaderBytes) {
        error = "checkpoint payload header is too large";
        return false;
    }

    std::error_code ec;
    fs::create_directories(final_path.parent_path(), ec);
    if (ec) {
        error = "could not create checkpoint payload directory: " + ec.message();
        return false;
    }
    if (fs::exists(final_path)) {
        error = "checkpoint payload already exists and is immutable: " +
                final_path.string();
        return false;
    }
    fs::path temporary_path = final_path;
    temporary_path += ".tmp." + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());

    try {
        std::ofstream output(temporary_path,
                             std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
            error = "could not open temporary checkpoint payload";
            return false;
        }
        const std::uint64_t header_size = header_bytes.size();
        output.write(kArchiveMagic.data(), kArchiveMagic.size());
        output.write(reinterpret_cast<const char*>(&header_size),
                     sizeof(header_size));
        output.write(header_bytes.data(),
                     static_cast<std::streamsize>(header_bytes.size()));
        for (const auto& [name, tensor] : tensors) {
            (void)name;
            output.write(static_cast<const char*>(tensor.ReadData()),
                         static_cast<std::streamsize>(tensor.NumBytes()));
        }
        output.flush();
        if (!output.good()) {
            error = "could not write checkpoint payload";
            output.close();
            fs::remove(temporary_path, ec);
            return false;
        }
        output.close();

        std::string sha256;
        if (!ComputeFileSha256(temporary_path, sha256, error)) {
            fs::remove(temporary_path, ec);
            return false;
        }
        const std::uint64_t byte_size = fs::file_size(temporary_path, ec);
        if (ec || byte_size == 0) {
            error = "could not measure checkpoint payload";
            fs::remove(temporary_path, ec);
            return false;
        }
        fs::rename(temporary_path, final_path, ec);
        if (ec) {
            error = "could not publish checkpoint payload atomically: " +
                    ec.message();
            fs::remove(temporary_path, ec);
            return false;
        }
        descriptor = {kind, relative_path, byte_size, sha256, true};
        return true;
    } catch (const std::exception& exception) {
        error = std::string("checkpoint payload save failed: ") +
                exception.what();
        fs::remove(temporary_path, ec);
        return false;
    }
}

bool ReadArchive(
    const fs::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    const std::string& expected_payload_type,
    LoadedArchive& archive,
    std::string& error)
{
    if (!VerifyCheckpointPayloadFile(checkpoint_directory, descriptor, error)) {
        return false;
    }
    fs::path path;
    if (!ResolvePayloadPath(checkpoint_directory, descriptor.relative_path,
                            path, error)) {
        return false;
    }

    try {
        std::ifstream input(path, std::ios::binary);
        std::array<char, 8> magic{};
        std::uint64_t header_size = 0;
        input.read(magic.data(), magic.size());
        input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        if (!input.good() || magic != kArchiveMagic || header_size == 0 ||
            header_size > kMaxArchiveHeaderBytes ||
            header_size + magic.size() + sizeof(header_size) >
                descriptor.byte_size) {
            error = "checkpoint payload archive header is invalid";
            return false;
        }
        std::string header_bytes(static_cast<std::size_t>(header_size), '\0');
        input.read(header_bytes.data(),
                   static_cast<std::streamsize>(header_bytes.size()));
        if (!input.good()) {
            error = "checkpoint payload archive header is truncated";
            return false;
        }
        const json header = json::parse(header_bytes);
        if (header.at("archive_schema_version").get<std::uint32_t>() !=
            kArchiveSchemaVersion) {
            error = "checkpoint payload archive schema is unsupported";
            return false;
        }
        archive.payload_type = header.at("payload_type").get<std::string>();
        if (archive.payload_type != expected_payload_type) {
            error = "checkpoint payload type '" + archive.payload_type +
                    "' does not match expected type '" + expected_payload_type + "'";
            return false;
        }
        archive.metadata = header.at("metadata");
        const auto& inventory = header.at("tensors");
        if (!inventory.is_array() ||
            (expected_payload_type == "model_parameters" && inventory.empty()) ||
            inventory.size() > 100000) {
            error = "checkpoint payload tensor inventory is invalid";
            return false;
        }

        std::uint64_t expected_data_bytes = 0;
        for (const auto& item : inventory) {
            const std::string name = item.at("name").get<std::string>();
            const int type_value = item.at("data_type").get<int>();
            const auto shape_values =
                item.at("shape").get<std::vector<std::uint64_t>>();
            const std::uint64_t declared_bytes =
                item.at("byte_size").get<std::uint64_t>();
            if (name.empty() || type_value < static_cast<int>(DataType::Float32) ||
                type_value > static_cast<int>(DataType::UInt8) ||
                shape_values.empty() || shape_values.size() > 16) {
                error = "checkpoint payload tensor metadata is invalid";
                return false;
            }
            const auto type = static_cast<DataType>(type_value);
            std::size_t elements = 1;
            std::vector<std::size_t> shape;
            shape.reserve(shape_values.size());
            for (const auto dimension : shape_values) {
                if (dimension == 0 ||
                    dimension > std::numeric_limits<std::size_t>::max() ||
                    elements > std::numeric_limits<std::size_t>::max() /
                                   static_cast<std::size_t>(dimension)) {
                    error = "checkpoint payload tensor shape is invalid";
                    return false;
                }
                shape.push_back(static_cast<std::size_t>(dimension));
                elements *= static_cast<std::size_t>(dimension);
            }
            const std::size_t element_bytes = DataTypeBytes(type);
            if (element_bytes == 0 ||
                elements > std::numeric_limits<std::size_t>::max() /
                               element_bytes ||
                declared_bytes != elements * element_bytes ||
                declared_bytes > descriptor.byte_size - expected_data_bytes) {
                error = "checkpoint payload tensor byte size is invalid";
                return false;
            }
            Tensor tensor(shape, type);
            input.read(static_cast<char*>(tensor.MutableData()),
                       static_cast<std::streamsize>(declared_bytes));
            if (!input.good() || !archive.tensors.emplace(name, std::move(tensor)).second) {
                error = "checkpoint payload tensor data is truncated or duplicated";
                return false;
            }
            expected_data_bytes += declared_bytes;
        }
        const std::uint64_t archive_overhead =
            magic.size() + sizeof(header_size) + header_size;
        if (archive_overhead + expected_data_bytes != descriptor.byte_size ||
            input.peek() != std::ifstream::traits_type::eof()) {
            error = "checkpoint payload archive size does not match its inventory";
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        error = std::string("checkpoint payload archive is invalid: ") +
                exception.what();
        return false;
    }
}

} // namespace

bool VerifyCheckpointPayloadFile(
    const fs::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    std::string& error)
{
    error.clear();
    fs::path path;
    if (!ResolvePayloadPath(checkpoint_directory, descriptor.relative_path,
                            path, error)) {
        return false;
    }
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) {
        error = "checkpoint payload file was not found: " + path.string();
        return false;
    }
    const auto size = fs::file_size(path, ec);
    if (ec || size != descriptor.byte_size) {
        error = "checkpoint payload size mismatch: " + descriptor.relative_path;
        return false;
    }
    std::string actual_hash;
    if (!ComputeFileSha256(path, actual_hash, error)) return false;
    std::string expected_hash = descriptor.sha256;
    std::transform(expected_hash.begin(), expected_hash.end(),
                   expected_hash.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    if (actual_hash != expected_hash) {
        error = "checkpoint payload SHA-256 mismatch: " +
                descriptor.relative_path;
        return false;
    }
    return true;
}

bool SaveModelPayloadV2(
    const fs::path& checkpoint_directory,
    const std::string& relative_path,
    const SequentialModel& model,
    CheckpointPayloadDescriptor& descriptor,
    std::string& error)
{
    auto& mutable_model = const_cast<SequentialModel&>(model);
    const auto parameters = mutable_model.GetParameters();
    return WriteArchiveAtomic(
        checkpoint_directory, relative_path, "model_parameters", json::object(),
        parameters, CheckpointPayloadKind::ModelParameters, descriptor, error);
}

bool LoadModelPayloadV2(
    const fs::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    SequentialModel& model,
    std::string& error)
{
    if (descriptor.kind != CheckpointPayloadKind::ModelParameters) {
        error = "checkpoint descriptor is not a model-parameters payload";
        return false;
    }
    LoadedArchive archive;
    if (!ReadArchive(checkpoint_directory, descriptor, "model_parameters",
                     archive, error)) {
        return false;
    }
    const auto expected = model.GetParameters();
    if (expected.size() != archive.tensors.size()) {
        error = "checkpoint model parameter count mismatch";
        return false;
    }
    for (const auto& [name, expected_tensor] : expected) {
        const auto found = archive.tensors.find(name);
        if (found == archive.tensors.end() ||
            found->second.Shape() != expected_tensor.Shape() ||
            found->second.GetDataType() != expected_tensor.GetDataType()) {
            error = "checkpoint model parameter is incompatible: " + name;
            return false;
        }
    }
    model.SetParameters(archive.tensors);
    return true;
}

bool SaveOptimizerPayloadV2(
    const fs::path& checkpoint_directory,
    const std::string& relative_path,
    const Optimizer& optimizer,
    CheckpointPayloadDescriptor& descriptor,
    std::string& error)
{
    OptimizerState state;
    if (!optimizer.ExportState(state, error)) return false;
    const json metadata = {
        {"optimizer_state_schema_version", state.schema_version},
        {"optimizer_type", state.optimizer_type},
        {"learning_rate", state.learning_rate},
        {"step_count", state.step_count},
        {"hyperparameters", state.hyperparameters},
    };
    return WriteArchiveAtomic(
        checkpoint_directory, relative_path, "optimizer_state", metadata,
        state.tensors, CheckpointPayloadKind::OptimizerState, descriptor, error);
}

bool LoadOptimizerPayloadV2(
    const fs::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    Optimizer& optimizer,
    std::string& error)
{
    if (descriptor.kind != CheckpointPayloadKind::OptimizerState) {
        error = "checkpoint descriptor is not an optimizer-state payload";
        return false;
    }
    LoadedArchive archive;
    if (!ReadArchive(checkpoint_directory, descriptor, "optimizer_state",
                     archive, error)) {
        return false;
    }
    try {
        OptimizerState state;
        state.schema_version =
            archive.metadata.at("optimizer_state_schema_version").get<int>();
        state.optimizer_type =
            archive.metadata.at("optimizer_type").get<std::string>();
        state.learning_rate =
            archive.metadata.at("learning_rate").get<double>();
        state.step_count = archive.metadata.at("step_count").get<int>();
        state.hyperparameters =
            archive.metadata.at("hyperparameters")
                .get<std::map<std::string, double>>();
        state.tensors = std::move(archive.tensors);
        return optimizer.ImportState(state, error);
    } catch (const std::exception& exception) {
        error = std::string("checkpoint optimizer metadata is invalid: ") +
                exception.what();
        return false;
    }
}

bool SaveSchedulerPayloadV2(
    const fs::path& checkpoint_directory,
    const std::string& relative_path,
    const LRScheduler& scheduler,
    CheckpointPayloadDescriptor& descriptor,
    std::string& error)
{
    SchedulerState state;
    if (!scheduler.ExportState(state, error)) return false;
    const json metadata = {
        {"scheduler_state_schema_version", state.schema_version},
        {"scheduler_type", state.scheduler_type},
        {"base_learning_rate", state.base_learning_rate},
        {"current_learning_rate", state.current_learning_rate},
        {"last_step", state.last_step},
        {"hyperparameters", state.hyperparameters},
        {"string_hyperparameters", state.string_hyperparameters},
        {"values", state.values},
    };
    return WriteArchiveAtomic(
        checkpoint_directory, relative_path, "scheduler_state", metadata, {},
        CheckpointPayloadKind::SchedulerState, descriptor, error);
}

bool LoadSchedulerPayloadV2(
    const fs::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    LRScheduler& scheduler,
    std::string& error)
{
    if (descriptor.kind != CheckpointPayloadKind::SchedulerState) {
        error = "checkpoint descriptor is not a scheduler-state payload";
        return false;
    }
    LoadedArchive archive;
    if (!ReadArchive(checkpoint_directory, descriptor, "scheduler_state",
                     archive, error)) {
        return false;
    }
    if (!archive.tensors.empty()) {
        error = "checkpoint scheduler state must not contain tensors";
        return false;
    }
    try {
        SchedulerState state;
        state.schema_version =
            archive.metadata.at("scheduler_state_schema_version").get<int>();
        state.scheduler_type =
            archive.metadata.at("scheduler_type").get<std::string>();
        state.base_learning_rate =
            archive.metadata.at("base_learning_rate").get<double>();
        state.current_learning_rate =
            archive.metadata.at("current_learning_rate").get<double>();
        state.last_step = archive.metadata.at("last_step").get<int>();
        state.hyperparameters =
            archive.metadata.at("hyperparameters")
                .get<std::map<std::string, double>>();
        state.string_hyperparameters =
            archive.metadata.at("string_hyperparameters")
                .get<std::map<std::string, std::string>>();
        state.values = archive.metadata.at("values")
                           .get<std::map<std::string, double>>();
        return scheduler.ImportState(state, error);
    } catch (const std::exception& exception) {
        error = std::string("checkpoint scheduler metadata is invalid: ") +
                exception.what();
        return false;
    }
}

bool SaveLRWarmupPayloadV2(
    const fs::path& checkpoint_directory,
    const std::string& relative_path,
    const LRWarmup& warmup,
    CheckpointPayloadDescriptor& descriptor,
    std::string& error)
{
    LRWarmupState state;
    if (!warmup.ExportState(state, error)) return false;

    std::string warmup_type;
    switch (state.warmup_type) {
        case WarmupType::None: warmup_type = "none"; break;
        case WarmupType::Linear: warmup_type = "linear"; break;
        case WarmupType::Cosine: warmup_type = "cosine"; break;
        default:
            error = "LRWarmup state contains an invalid warmup type";
            return false;
    }

    const json metadata = {
        {"lr_warmup_state_schema_version", state.schema_version},
        {"warmup_steps", state.warmup_steps},
        {"warmup_type", warmup_type},
        {"base_learning_rate", state.base_learning_rate},
        {"current_step", state.current_step},
        {"optimizer_state_schema_version",
         state.optimizer_state.schema_version},
        {"optimizer_type", state.optimizer_state.optimizer_type},
        {"optimizer_learning_rate", state.optimizer_state.learning_rate},
        {"optimizer_step_count", state.optimizer_state.step_count},
        {"optimizer_hyperparameters", state.optimizer_state.hyperparameters},
    };
    return WriteArchiveAtomic(
        checkpoint_directory, relative_path, "lr_warmup_state", metadata,
        state.optimizer_state.tensors, CheckpointPayloadKind::SchedulerState,
        descriptor, error);
}

bool LoadLRWarmupPayloadV2(
    const fs::path& checkpoint_directory,
    const CheckpointPayloadDescriptor& descriptor,
    LRWarmup& warmup,
    std::string& error)
{
    if (descriptor.kind != CheckpointPayloadKind::SchedulerState) {
        error = "checkpoint descriptor is not a scheduler-state payload";
        return false;
    }
    LoadedArchive archive;
    if (!ReadArchive(checkpoint_directory, descriptor, "lr_warmup_state",
                     archive, error)) {
        return false;
    }
    try {
        LRWarmupState state;
        state.schema_version =
            archive.metadata.at("lr_warmup_state_schema_version").get<int>();
        state.warmup_steps = archive.metadata.at("warmup_steps").get<int>();
        const auto warmup_type =
            archive.metadata.at("warmup_type").get<std::string>();
        if (warmup_type == "none") {
            state.warmup_type = WarmupType::None;
        } else if (warmup_type == "linear") {
            state.warmup_type = WarmupType::Linear;
        } else if (warmup_type == "cosine") {
            state.warmup_type = WarmupType::Cosine;
        } else {
            error = "checkpoint LRWarmup metadata has an invalid warmup type";
            return false;
        }
        state.base_learning_rate =
            archive.metadata.at("base_learning_rate").get<double>();
        state.current_step = archive.metadata.at("current_step").get<int>();
        state.optimizer_state.schema_version =
            archive.metadata.at("optimizer_state_schema_version").get<int>();
        state.optimizer_state.optimizer_type =
            archive.metadata.at("optimizer_type").get<std::string>();
        state.optimizer_state.learning_rate =
            archive.metadata.at("optimizer_learning_rate").get<double>();
        state.optimizer_state.step_count =
            archive.metadata.at("optimizer_step_count").get<int>();
        state.optimizer_state.hyperparameters =
            archive.metadata.at("optimizer_hyperparameters")
                .get<std::map<std::string, double>>();
        state.optimizer_state.tensors = std::move(archive.tensors);
        return warmup.ImportState(state, error);
    } catch (const std::exception& exception) {
        error = std::string("checkpoint LRWarmup metadata is invalid: ") +
                exception.what();
        return false;
    }
}

} // namespace cyxwiz
