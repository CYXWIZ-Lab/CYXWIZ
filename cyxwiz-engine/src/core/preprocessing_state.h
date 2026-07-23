#pragma once

#include <arrow/type_fwd.h>

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
