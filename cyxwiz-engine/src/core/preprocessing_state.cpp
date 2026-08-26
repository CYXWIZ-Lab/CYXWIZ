#include "preprocessing_state.h"

#include <arrow/type.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <unordered_set>

namespace cyxwiz {
namespace {

using json = nlohmann::json;

constexpr const char* kArtifactType = "cyxwiz.preprocessing_state";
constexpr int kArtifactVersion = 1;

std::string TrimAndLower(std::string value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }).base();
    value = first < last ? std::string(first, last) : std::string{};
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

bool ReadBooleanOption(
    const std::map<std::string, std::string>& parameters,
    const char* name,
    bool default_value,
    const std::string& operator_name,
    bool& result,
    std::string& error) {
    const auto it = parameters.find(name);
    if (it == parameters.end() || it->second.empty()) {
        result = default_value;
        return true;
    }
    const std::string value = TrimAndLower(it->second);
    if (value == "true") {
        result = true;
        return true;
    }
    if (value == "false") {
        result = false;
        return true;
    }
    error = operator_name + ": '" + name +
            "' must be 'true' or 'false' (got '" + it->second + "')";
    return false;
}

bool IsNumericArrowType(const std::shared_ptr<arrow::DataType>& type) {
    if (!type) {
        return false;
    }
    switch (type->id()) {
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
    case arrow::Type::HALF_FLOAT:
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
        return true;
    default:
        return false;
    }
}

bool IsNumericArrowTypeName(const std::string& name) {
    return name == "int8" || name == "int16" || name == "int32" ||
           name == "int64" || name == "uint8" || name == "uint16" ||
           name == "uint32" || name == "uint64" ||
           name == "halffloat" || name == "float" || name == "double";
}

bool IsStringArrowType(const std::shared_ptr<arrow::DataType>& type) {
    return type && (type->id() == arrow::Type::STRING ||
                    type->id() == arrow::Type::LARGE_STRING);
}

bool IsStringArrowTypeName(const std::string& name) {
    return name == "string" || name == "large_string";
}

std::string AbsolutePathForMessage(const std::filesystem::path& path) {
    std::error_code ec;
    const auto absolute = std::filesystem::absolute(path, ec);
    return ec ? path.string() : absolute.string();
}

json ToJson(const FittedPreprocessingState& state) {
    json features = json::array();
    for (const auto& feature : state.features) {
        features.push_back({
            {"name", feature.name},
            {"data_type", feature.data_type},
            {"numeric_values", feature.numeric_values},
            {"string_values", feature.string_values},
        });
    }
    return {
        {"artifact_type", state.artifact_type},
        {"artifact_version", state.artifact_version},
        {"operator_name", state.operator_name},
        {"operator_version", state.operator_version},
        {"fit_rows", state.fit_rows},
        {"input_schema_fingerprint", state.input_schema_fingerprint},
        {"configuration", state.configuration},
        {"features", std::move(features)},
    };
}

bool FromJson(const json& value,
              FittedPreprocessingState& state,
              std::string& error) {
    try {
        state.artifact_type = value.at("artifact_type").get<std::string>();
        state.artifact_version = value.at("artifact_version").get<int>();
        state.operator_name = value.at("operator_name").get<std::string>();
        state.operator_version = value.at("operator_version").get<int>();
        state.fit_rows = value.at("fit_rows").get<int64_t>();
        state.input_schema_fingerprint =
            value.at("input_schema_fingerprint").get<std::string>();
        state.configuration =
            value.at("configuration").get<std::map<std::string, std::string>>();
        state.features.clear();
        for (const auto& feature_json : value.at("features")) {
            PreprocessingFeatureState feature;
            feature.name = feature_json.at("name").get<std::string>();
            feature.data_type =
                feature_json.at("data_type").get<std::string>();
            feature.numeric_values =
                feature_json.value("numeric_values",
                                   std::map<std::string, double>{});
            feature.string_values =
                feature_json.value("string_values",
                                   std::map<std::string, std::string>{});
            state.features.push_back(std::move(feature));
        }
    } catch (const std::exception& exception) {
        error = std::string("invalid preprocessing state structure: ") +
                exception.what();
        return false;
    }
    return true;
}

}  // namespace

bool ParseFittedPreprocessingOptions(
    const std::map<std::string, std::string>& parameters,
    const std::string& operator_name,
    FittedPreprocessingOptions& options,
    std::string& error) {
    options = {};

    const auto mode = parameters.find("operation_mode");
    if (mode != parameters.end() && !mode->second.empty()) {
        options.operation_mode = TrimAndLower(mode->second);
    }
    if (options.operation_mode != "fit_transform" &&
        options.operation_mode != "transform_only") {
        error = operator_name +
                ": 'operation_mode' must be 'fit_transform' or "
                "'transform_only' (got '" + options.operation_mode + "')";
        return false;
    }

    const auto path = parameters.find("state_path");
    if (path != parameters.end()) {
        options.state_path = path->second;
        const auto first = std::find_if_not(
            options.state_path.begin(), options.state_path.end(),
            [](unsigned char ch) { return std::isspace(ch) != 0; });
        const auto last = std::find_if_not(
            options.state_path.rbegin(), options.state_path.rend(),
            [](unsigned char ch) { return std::isspace(ch) != 0; }).base();
        options.state_path = first < last
            ? std::string(first, last)
            : std::string{};
    }

    if (!ReadBooleanOption(parameters, "save_state", false, operator_name,
                           options.save_state, error) ||
        !ReadBooleanOption(parameters, "state_overwrite", false,
                           operator_name, options.state_overwrite, error)) {
        return false;
    }
    if (options.IsTransformOnly() && options.state_path.empty()) {
        error = operator_name +
                ": Transform Only requires 'state_path'. Fit this node on "
                "training data with 'Save fitted state' enabled, then select "
                "that artifact.";
        return false;
    }
    if (!options.IsTransformOnly() && options.save_state &&
        options.state_path.empty()) {
        error = operator_name +
                ": 'Save fitted state' is enabled but 'state_path' is empty. "
                "Choose a .cyxstate.json path or disable saving.";
        return false;
    }
    return true;
}

bool ValidateFittedPreprocessingConfiguration(
    const FittedPreprocessingState& state,
    const std::map<std::string, std::string>& expected,
    std::string& error) {
    for (const auto& [name, expected_value] : expected) {
        const auto actual = state.configuration.find(name);
        if (actual == state.configuration.end()) {
            error = state.operator_name + ": fitted state is missing '" +
                    name + "'. Refit the artifact with this CyxWiz version.";
            return false;
        }
        if (actual->second != expected_value) {
            error = state.operator_name + ": node setting '" + name +
                    "' is '" + expected_value +
                    "', but the fitted artifact requires '" +
                    actual->second +
                    "'. Match the training setting or choose the correct "
                    "artifact.";
            return false;
        }
    }
    return true;
}

bool SaveFittedTextVectorizerState(
    const std::string& path,
    const std::string& operator_name,
    int64_t fit_rows,
    const std::string& input_schema_fingerprint,
    const std::map<std::string, std::string>& configuration,
    const std::vector<FittedTextVectorizerFeature>& features,
    bool overwrite,
    std::string& error) {
    FittedPreprocessingState persisted;
    persisted.operator_name = operator_name;
    persisted.operator_version = 1;
    persisted.fit_rows = fit_rows;
    persisted.input_schema_fingerprint = input_schema_fingerprint;
    persisted.configuration = configuration;
    persisted.features.reserve(features.size());
    for (const auto& feature : features) {
        PreprocessingFeatureState persisted_feature;
        persisted_feature.name = feature.term;
        persisted_feature.data_type = "text_token";
        persisted_feature.numeric_values["weight"] = feature.weight;
        persisted.features.push_back(std::move(persisted_feature));
    }
    return SaveFittedPreprocessingState(
        path, persisted, overwrite, error);
}

bool LoadFittedTextVectorizerState(
    const std::string& path,
    const std::string& expected_operator,
    const std::map<std::string, std::string>& expected_configuration,
    size_t max_features,
    FittedTextVectorizerState& state,
    std::string& error) {
    FittedPreprocessingState persisted;
    if (!LoadFittedPreprocessingState(
            path, expected_operator, persisted, error)) {
        return false;
    }
    if (persisted.operator_version != 1) {
        error = expected_operator +
                ": fitted state uses unsupported operator version " +
                std::to_string(persisted.operator_version) +
                ". Refit it with this CyxWiz version.";
        return false;
    }
    if (!ValidateFittedPreprocessingConfiguration(
            persisted, expected_configuration, error)) {
        return false;
    }
    if (persisted.features.size() > max_features) {
        error = expected_operator +
                ": fitted vocabulary exceeds max_features. Select the "
                "matching artifact or refit it.";
        return false;
    }

    state = {};
    state.fit_rows = persisted.fit_rows;
    state.input_schema_fingerprint = persisted.input_schema_fingerprint;
    state.features.reserve(persisted.features.size());
    std::unordered_set<std::string> unique_terms;
    for (const auto& feature : persisted.features) {
        const auto weight = feature.numeric_values.find("weight");
        if (feature.name.empty() || feature.data_type != "text_token" ||
            weight == feature.numeric_values.end() ||
            !std::isfinite(weight->second) || weight->second <= 0.0 ||
            !unique_terms.insert(feature.name).second) {
            error = expected_operator +
                    ": fitted state contains an invalid or duplicate "
                    "vocabulary term. Refit the artifact.";
            state = {};
            return false;
        }
        state.features.push_back({feature.name, weight->second});
    }
    return true;
}

std::string FingerprintPreprocessingSchema(
    const std::shared_ptr<arrow::Schema>& schema) {
    // Stable FNV-1a over ordered field names and Arrow data types.
    uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&hash](const std::string& text) {
        for (const unsigned char byte : text) {
            hash ^= static_cast<uint64_t>(byte);
            hash *= 1099511628211ULL;
        }
    };
    if (schema) {
        for (const auto& field : schema->fields()) {
            mix(field->name());
            mix(":");
            mix(field->type()->ToString());
            mix(";");
        }
    }
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

bool SaveFittedPreprocessingState(
    const std::string& path_text,
    const FittedPreprocessingState& state,
    bool overwrite,
    std::string& error) {
    if (path_text.empty()) {
        error = "state artifact path is empty";
        return false;
    }

    const std::filesystem::path path(path_text);
    std::error_code ec;
    if (std::filesystem::exists(path, ec) && !overwrite) {
        error = "state artifact already exists at '" +
                AbsolutePathForMessage(path) +
                "'. Enable 'Allow state overwrite' or choose a new path.";
        return false;
    }
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            error = "could not create state artifact directory '" +
                    AbsolutePathForMessage(path.parent_path()) + "': " +
                    ec.message();
            return false;
        }
    }

    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = "could not open temporary state artifact '" +
                    AbsolutePathForMessage(temporary) + "' for writing";
            return false;
        }
        output << ToJson(state).dump(2) << '\n';
        output.flush();
        if (!output) {
            error = "failed while writing temporary state artifact '" +
                    AbsolutePathForMessage(temporary) + "'";
            return false;
        }
    }

    if (overwrite && std::filesystem::exists(path, ec)) {
        std::filesystem::remove(path, ec);
        if (ec) {
            std::filesystem::remove(temporary);
            error = "could not replace state artifact '" +
                    AbsolutePathForMessage(path) + "': " + ec.message();
            return false;
        }
    }
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        std::filesystem::remove(temporary);
        error = "could not finalize state artifact '" +
                AbsolutePathForMessage(path) + "': " + ec.message();
        return false;
    }
    return true;
}

bool LoadFittedPreprocessingState(
    const std::string& path_text,
    const std::string& expected_operator,
    FittedPreprocessingState& state,
    std::string& error) {
    if (path_text.empty()) {
        error = "state artifact path is empty";
        return false;
    }
    const std::filesystem::path path(path_text);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "state artifact was not found or is unreadable at '" +
                AbsolutePathForMessage(path) +
                "'. Fit the preprocessing node on training data first, then "
                "select the saved artifact.";
        return false;
    }

    json value;
    try {
        input >> value;
    } catch (const std::exception& exception) {
        error = "state artifact '" + AbsolutePathForMessage(path) +
                "' is not valid JSON: " + exception.what();
        return false;
    }
    if (!FromJson(value, state, error)) {
        error = "state artifact '" + AbsolutePathForMessage(path) +
                "' is invalid: " + error;
        return false;
    }
    if (state.artifact_type != kArtifactType ||
        state.artifact_version != kArtifactVersion) {
        error = "state artifact '" + AbsolutePathForMessage(path) +
                "' has unsupported type/version ('" + state.artifact_type +
                "', version " + std::to_string(state.artifact_version) +
                "). Refit it with this CyxWiz version.";
        return false;
    }
    if (state.operator_name != expected_operator) {
        error = "state artifact '" + AbsolutePathForMessage(path) +
                "' belongs to " + state.operator_name + ", not " +
                expected_operator + ". Select the artifact produced by the " +
                expected_operator + " node.";
        return false;
    }
    if (state.features.empty()) {
        error = "state artifact '" + AbsolutePathForMessage(path) +
                "' contains no fitted features. Refit it on non-empty "
                "training data.";
        return false;
    }
    return true;
}

bool ValidateFittedPreprocessingStateSchema(
    const FittedPreprocessingState& state,
    const std::shared_ptr<arrow::Schema>& schema,
    std::string& error) {
    if (!schema) {
        error = state.operator_name + ": input schema is unavailable";
        return false;
    }
    for (const auto& feature : state.features) {
        const int index = schema->GetFieldIndex(feature.name);
        if (index < 0) {
            error = state.operator_name + ": state expects column '" +
                    feature.name +
                    "', but it is missing from this dataset. Use a dataset "
                    "with the training schema or fit a new state.";
            return false;
        }
        const std::string actual =
            schema->field(index)->type()->ToString();
        const auto& actual_type = schema->field(index)->type();
        const bool compatible_numeric =
            IsNumericArrowTypeName(feature.data_type) &&
            IsNumericArrowType(actual_type);
        const bool compatible_string =
            IsStringArrowTypeName(feature.data_type) &&
            IsStringArrowType(actual_type);
        if (actual != feature.data_type && !compatible_numeric &&
            !compatible_string) {
            error = state.operator_name + ": state expects column '" +
                    feature.name + "' to have type " + feature.data_type +
                    ", but this dataset has " + actual +
                    ". Align the test schema with training or fit a new state.";
            return false;
        }
    }
    return true;
}

}  // namespace cyxwiz
