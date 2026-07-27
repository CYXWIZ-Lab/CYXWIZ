#include "preprocessing_state.h"

#include <arrow/type.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

namespace cyxwiz {
namespace {

using json = nlohmann::json;

constexpr const char* kArtifactType = "cyxwiz.preprocessing_state";
constexpr int kArtifactVersion = 1;

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
