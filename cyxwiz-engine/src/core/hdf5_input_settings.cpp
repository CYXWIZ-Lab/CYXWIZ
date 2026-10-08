#include "hdf5_input_settings.h"
#include "hdf5_object_path.h"

#include <array>
#include <charconv>

namespace cyxwiz {
namespace {

constexpr uint64_t kDefaultMaxMaterializedBytes = 256ULL * 1024 * 1024;

constexpr const char* kSelectionVersion = "hdf5_selection_version";
constexpr const char* kDataPath = "hdf5_data_path";
constexpr const char* kLabelPath = "hdf5_label_path";
constexpr const char* kImportMode = "hdf5_import_mode";
constexpr const char* kNumericPolicy = "hdf5_numeric_policy";
constexpr const char* kMaxMaterializedBytes = "hdf5_max_materialized_bytes";
constexpr const char* kLegacyDataset = "hdf5_dataset";

constexpr std::array<const char*, 6> kCanonicalKeys = {
    kSelectionVersion,
    kDataPath,
    kLabelPath,
    kImportMode,
    kNumericPolicy,
    kMaxMaterializedBytes,
};

bool HasKey(const std::map<std::string, std::string>& parameters,
            const char* key) {
    return parameters.find(key) != parameters.end();
}

std::string Value(const std::map<std::string, std::string>& parameters,
                  const char* key) {
    const auto it = parameters.find(key);
    return it == parameters.end() ? std::string{} : it->second;
}

bool HasAnyCanonical(const std::map<std::string, std::string>& parameters) {
    for (const char* key : kCanonicalKeys) {
        if (HasKey(parameters, key)) return true;
    }
    return false;
}

bool MissingCanonical(const std::map<std::string, std::string>& parameters,
                      std::string& missing) {
    for (const char* key : kCanonicalKeys) {
        if (!HasKey(parameters, key)) {
            missing = key;
            return true;
        }
    }
    return false;
}

bool ParseBudget(const std::string& text, uint64_t& value,
                 std::string& error) {
    if (text.empty()) {
        error = "hdf5_max_materialized_bytes must not be empty";
        return false;
    }
    uint64_t parsed = 0;
    const auto* first = text.data();
    const auto* last = text.data() + text.size();
    const auto result = std::from_chars(first, last, parsed);
    if (result.ec != std::errc{} || result.ptr != last) {
        error = "hdf5_max_materialized_bytes must be an unsigned integer";
        return false;
    }
    if (parsed == 0 || parsed > kDefaultMaxMaterializedBytes) {
        error = "hdf5_max_materialized_bytes must be > 0 and <= 268435456";
        return false;
    }
    value = parsed;
    return true;
}

std::string PolicyText(Hdf5NumericPolicy policy) {
    return policy == Hdf5NumericPolicy::Float64 ? "float64" : "preserve";
}

bool ParsePolicy(const std::string& text, Hdf5NumericPolicy& policy,
                 std::string& error) {
    if (text == "preserve") {
        policy = Hdf5NumericPolicy::Preserve;
        return true;
    }
    if (text == "float64") {
        policy = Hdf5NumericPolicy::Float64;
        return true;
    }
    error = "hdf5_numeric_policy must be 'preserve' or 'float64'";
    return false;
}

std::string NormalizeLegacyDatasetPath(const std::string& path) {
    if (!path.empty() && path.front() == '/') return path;
    return "/" + path;
}

void WriteCanonical(const Hdf5InputSettings& settings,
                    std::map<std::string, std::string>& parameters) {
    parameters[kSelectionVersion] = "1";
    parameters[kDataPath] = settings.selection.data_path;
    parameters[kLabelPath] = settings.selection.label_path;
    parameters[kImportMode] = "numeric_table";
    parameters[kNumericPolicy] = PolicyText(settings.numeric_policy);
    parameters[kMaxMaterializedBytes] =
        std::to_string(settings.max_materialized_bytes);
    parameters.erase(kLegacyDataset);
}

} // namespace

bool ValidateHdf5InputSettings(const Hdf5InputSettings& settings,
                               std::string& error) {
    error.clear();
    if (!ValidateHdf5ObjectPath(settings.selection.data_path, error)) {
        return false;
    }
    if (!settings.selection.label_path.empty() &&
        !ValidateHdf5ObjectPath(settings.selection.label_path, error)) {
        return false;
    }
    if (settings.numeric_policy != Hdf5NumericPolicy::Preserve &&
        settings.numeric_policy != Hdf5NumericPolicy::Float64) {
        error = "hdf5_numeric_policy is invalid";
        return false;
    }
    if (settings.max_materialized_bytes == 0 ||
        settings.max_materialized_bytes > kDefaultMaxMaterializedBytes) {
        error = "hdf5_max_materialized_bytes must be > 0 and <= 268435456";
        return false;
    }
    return true;
}

Hdf5InputSettingsResult ReadHdf5InputSettings(
    const std::map<std::string, std::string>& parameters) {
    Hdf5InputSettingsResult result;
    result.settings.max_materialized_bytes = kDefaultMaxMaterializedBytes;

    if (HasAnyCanonical(parameters)) {
        std::string missing;
        if (MissingCanonical(parameters, missing)) {
            result.error = "Missing canonical HDF5 setting: " + missing;
            return result;
        }
        if (Value(parameters, kSelectionVersion) != "1") {
            result.error = "hdf5_selection_version must be '1'";
            return result;
        }
        if (Value(parameters, kImportMode) != "numeric_table") {
            result.error = "hdf5_import_mode must be 'numeric_table'";
            return result;
        }
        result.settings.selection.data_path = Value(parameters, kDataPath);
        result.settings.selection.label_path = Value(parameters, kLabelPath);
        if (!ParsePolicy(Value(parameters, kNumericPolicy),
                         result.settings.numeric_policy,
                         result.error)) {
            return result;
        }
        if (!ParseBudget(Value(parameters, kMaxMaterializedBytes),
                         result.settings.max_materialized_bytes,
                         result.error)) {
            return result;
        }
        if (!ValidateHdf5InputSettings(result.settings, result.error)) {
            return result;
        }
        result.ok = true;
        return result;
    }

    if (HasKey(parameters, kLegacyDataset)) {
        const auto legacy = Value(parameters, kLegacyDataset);
        if (legacy.empty()) {
            result.error = "hdf5_dataset legacy path must not be empty";
            return result;
        }
        result.settings.selection.data_path = NormalizeLegacyDatasetPath(legacy);
        result.settings.selection.label_path.clear();
        result.settings.numeric_policy = Hdf5NumericPolicy::Preserve;
        result.settings.max_materialized_bytes = kDefaultMaxMaterializedBytes;
        if (!ValidateHdf5InputSettings(result.settings, result.error)) {
            return result;
        }
        result.ok = true;
        result.migrated_legacy = true;
        return result;
    }

    result.settings.selection.data_path = "/data";
    result.settings.selection.label_path.clear();
    result.settings.numeric_policy = Hdf5NumericPolicy::Preserve;
    result.settings.max_materialized_bytes = kDefaultMaxMaterializedBytes;
    result.ok = true;
    return result;
}

bool WriteHdf5InputSettings(const Hdf5InputSettings& settings,
                            std::map<std::string, std::string>& parameters,
                            std::string& error) {
    error.clear();
    if (!ValidateHdf5InputSettings(settings, error)) {
        return false;
    }
    auto updated = parameters;
    WriteCanonical(settings, updated);
    parameters = std::move(updated);
    return true;
}

} // namespace cyxwiz
