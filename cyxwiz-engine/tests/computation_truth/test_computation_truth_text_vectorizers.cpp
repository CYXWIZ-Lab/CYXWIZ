#include "core/node_executors/count_vectorizer_operator.h"
#include "core/node_executors/tfidf_vectorizer_operator.h"
#include "core/preprocessing_state.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

fs::path ResolveFixture(const fs::path& executable,
                        const char* override_path) {
    if (override_path != nullptr && *override_path != '\0') {
        return fs::path(override_path);
    }
    return fs::absolute(executable).parent_path() /
           "computation_truth_fixtures" /
           "text_vectorizers_sklearn.json";
}

json LoadFixture(const fs::path& path) {
    std::ifstream input(path);
    Check(input.is_open(), "could not open fixture: " + path.string());
    json fixture;
    input >> fixture;
    Check(fixture.at("schema_version").get<int>() == 1,
          "unsupported text vectorizer fixture schema");
    Check(fixture.at("oracle").at("name").get<std::string>() ==
              "scikit-learn",
          "text vectorizer fixture must identify scikit-learn as its oracle");
    return fixture;
}

std::shared_ptr<arrow::Table> MakeTextTable(
    const std::vector<std::string>& documents) {
    arrow::StringBuilder builder;
    for (const auto& document : documents) {
        Check(builder.Append(document).ok(), "fixture text append failed");
    }
    std::shared_ptr<arrow::Array> values;
    Check(builder.Finish(&values).ok(), "fixture text finish failed");
    return arrow::Table::Make(
        arrow::schema({arrow::field("text", arrow::utf8())}),
        {values}, static_cast<int64_t>(documents.size()));
}

float ReadFloat(const std::shared_ptr<arrow::Table>& table,
                const std::string& column,
                int64_t row) {
    const auto chunked = table->GetColumnByName(column);
    Check(chunked != nullptr, "missing output column: " + column);
    const auto values =
        std::static_pointer_cast<arrow::FloatArray>(chunked->chunk(0));
    return values->Value(row);
}

std::map<std::string, std::string> BuildParameters(
    const json& test_case,
    const fs::path& state_path) {
    std::map<std::string, std::string> parameters;
    for (const auto& item : test_case.at("parameters").items()) {
        parameters[item.key()] = item.value().get<std::string>();
    }
    parameters["text_col"] = "text";
    parameters["label_col"] = "";
    if (parameters.find("max_features") == parameters.end()) {
        parameters["max_features"] = "128";
    }
    parameters["min_df"] = "1";
    parameters["stop_words"] = "none";
    parameters["output_format"] = "dense";
    parameters["operation_mode"] = "fit_transform";
    parameters["save_state"] = "true";
    parameters["state_path"] = state_path.string();
    parameters["state_overwrite"] = "true";
    return parameters;
}

std::shared_ptr<arrow::Table> RunOperator(
    const std::string& operator_name,
    const std::map<std::string, std::string>& parameters,
    const std::shared_ptr<arrow::Table>& input) {
    std::string error;
    if (operator_name == "CountVectorizer") {
        cyxwiz::CountVectorizerOperator operation;
        Check(operation.Configure(parameters, error), error);
        auto result = operation.Apply(input);
        Check(result.ok(), result.status().ToString());
        return result.ValueOrDie();
    }
    Check(operator_name == "TFIDFVectorizer",
          "unsupported fixture operator: " + operator_name);
    cyxwiz::TFIDFVectorizerOperator operation;
    Check(operation.Configure(parameters, error), error);
    auto result = operation.Apply(input);
    Check(result.ok(), result.status().ToString());
    return result.ValueOrDie();
}

void CheckCase(const json& test_case,
               double tolerance,
               const fs::path& state_root) {
    const std::string name = test_case.at("name").get<std::string>();
    const std::string operator_name =
        test_case.at("operator").get<std::string>();
    const auto documents =
        test_case.at("documents").get<std::vector<std::string>>();
    const auto feature_names =
        test_case.at("feature_names").get<std::vector<std::string>>();
    const auto expected =
        test_case.at("expected").get<std::vector<std::vector<double>>>();
    const fs::path state_path = state_root / (name + ".cyxstate.json");

    const auto parameters = BuildParameters(test_case, state_path);
    const auto output = RunOperator(
        operator_name, parameters, MakeTextTable(documents));
    Check(output->num_columns() == static_cast<int>(feature_names.size()),
          name + ": output feature width differs from sklearn");
    Check(output->num_rows() == static_cast<int64_t>(documents.size()),
          name + ": output row count differs from sklearn");

    cyxwiz::FittedPreprocessingState state;
    std::string error;
    Check(cyxwiz::LoadFittedPreprocessingState(
              state_path.string(), operator_name, state, error),
          name + ": " + error);
    Check(state.features.size() == feature_names.size(),
          name + ": fitted vocabulary width differs from sklearn");

    const std::string prefix =
        operator_name == "TFIDFVectorizer" ? "tfidf_" : "count_";
    for (size_t feature = 0; feature < feature_names.size(); ++feature) {
        Check(state.features[feature].name == feature_names[feature],
              name + ": feature order mismatch at index " +
                  std::to_string(feature) + " (CyxWiz='" +
                  state.features[feature].name + "', sklearn='" +
                  feature_names[feature] + "')");
        if (test_case.contains("idf")) {
            const double actual_idf =
                state.features[feature].numeric_values.at("weight");
            const double expected_idf =
                test_case.at("idf").at(feature).get<double>();
            Check(std::abs(actual_idf - expected_idf) <= tolerance,
                  name + ": IDF mismatch for '" + feature_names[feature] +
                      "'");
        }
        for (size_t row = 0; row < documents.size(); ++row) {
            const double actual = ReadFloat(
                output, prefix + std::to_string(feature),
                static_cast<int64_t>(row));
            const double reference = expected[row][feature];
            Check(std::abs(actual - reference) <= tolerance,
                  name + ": value mismatch at row " + std::to_string(row) +
                      ", feature '" + feature_names[feature] +
                      "' (CyxWiz=" + std::to_string(actual) +
                      ", sklearn=" + std::to_string(reference) + ")");
        }
    }
}

void CheckOperatorRangeValidation() {
    const std::vector<std::string> invalid_ranges = {
        "0,1", "1,0", "2,1", "1,4", "1,2junk", "1,2,3"
    };
    for (const auto& range : invalid_ranges) {
        std::string count_error;
        cyxwiz::CountVectorizerOperator count;
        Check(!count.Configure(
                  {{"text_col", "text"}, {"ngram_range", range}},
                  count_error),
              "CountVectorizer should reject ngram_range='" + range + "'");
        Check(!count_error.empty(),
              "CountVectorizer invalid range should explain the error");

        std::string tfidf_error;
        cyxwiz::TFIDFVectorizerOperator tfidf;
        Check(!tfidf.Configure(
                  {{"text_col", "text"}, {"ngram_range", range}},
                  tfidf_error),
              "TFIDFVectorizer should reject ngram_range='" + range + "'");
        Check(!tfidf_error.empty(),
              "TFIDFVectorizer invalid range should explain the error");
    }
}

}  // namespace

int main(int argc, char** argv) {
    CheckOperatorRangeValidation();
    const fs::path fixture_path = ResolveFixture(
        argc > 0 ? fs::path(argv[0]) : fs::path{},
        argc > 1 ? argv[1] : nullptr);
    const json fixture = LoadFixture(fixture_path);
    const double tolerance = fixture.at("tolerance").get<double>();
    const fs::path state_root =
        fs::temp_directory_path() / "cyxwiz_text_vectorizer_parity";
    std::error_code remove_error;
    fs::remove_all(state_root, remove_error);
    Check(fs::create_directories(state_root),
          "could not create text vectorizer parity state directory");

    for (const auto& test_case : fixture.at("cases")) {
        CheckCase(test_case, tolerance, state_root);
    }

    fs::remove_all(state_root, remove_error);
    std::cout << "Text vectorizer sklearn parity checks passed\n";
    return 0;
}
