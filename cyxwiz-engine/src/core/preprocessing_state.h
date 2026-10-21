#pragma once

#include <arrow/type_fwd.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

struct PreprocessingFeatureState {
    std::string name;
    std::string data_type;
    std::map<std::string, double> numeric_values;
    std::map<std::string, std::string> string_values;
};

struct FittedPreprocessingState {
    std::string artifact_type = "cyxwiz.preprocessing_state";
    int artifact_version = 1;
    std::string operator_name;
    int operator_version = 1;
    int64_t fit_rows = 0;
    std::string input_schema_fingerprint;
    std::map<std::string, std::string> configuration;
    std::vector<PreprocessingFeatureState> features;
};

struct FittedPreprocessingOptions {
    std::string operation_mode = "fit_transform";
    std::string state_path;
    bool save_state = false;
    bool state_overwrite = false;

    bool IsTransformOnly() const {
        return operation_mode == "transform_only";
    }
};

struct FittedTextVectorizerFeature {
    std::string term;
    double weight = 1.0;
};

struct FittedTextVectorizerState {
    int64_t fit_rows = 0;
    std::string input_schema_fingerprint;
    std::vector<FittedTextVectorizerFeature> features;
};

bool ParseFittedPreprocessingOptions(
    const std::map<std::string, std::string>& parameters,
    const std::string& operator_name,
    FittedPreprocessingOptions& options,
    std::string& error);

bool ValidateFittedPreprocessingConfiguration(
    const FittedPreprocessingState& state,
    const std::map<std::string, std::string>& expected,
    std::string& error);

bool SaveFittedTextVectorizerState(
    const std::string& path,
    const std::string& operator_name,
    int64_t fit_rows,
    const std::string& input_schema_fingerprint,
    const std::map<std::string, std::string>& configuration,
    const std::vector<FittedTextVectorizerFeature>& features,
    bool overwrite,
    std::string& error);

bool LoadFittedTextVectorizerState(
    const std::string& path,
    const std::string& expected_operator,
    const std::map<std::string, std::string>& expected_configuration,
    size_t max_features,
    FittedTextVectorizerState& state,
    std::string& error);

std::string FingerprintPreprocessingSchema(
    const std::shared_ptr<arrow::Schema>& schema);

bool SaveFittedPreprocessingState(
    const std::string& path,
    const FittedPreprocessingState& state,
    bool overwrite,
    std::string& error);

bool LoadFittedPreprocessingState(
    const std::string& path,
    const std::string& expected_operator,
    FittedPreprocessingState& state,
    std::string& error);

bool ValidateFittedPreprocessingStateSchema(
    const FittedPreprocessingState& state,
    const std::shared_ptr<arrow::Schema>& schema,
    std::string& error);

}  // namespace cyxwiz
