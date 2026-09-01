#include "core/node_executors/count_vectorizer_operator.h"
#include "core/node_executors/tfidf_vectorizer_operator.h"
#include "core/preprocessing_state.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<arrow::Table> MakeTextTable(
    const std::vector<std::string>& texts) {
    arrow::StringBuilder builder;
    for (const auto& text : texts) {
        Check(builder.Append(text).ok(), "text fixture should append");
    }
    std::shared_ptr<arrow::Array> values;
    Check(builder.Finish(&values).ok(), "text fixture should finish");
    return arrow::Table::Make(
        arrow::schema({arrow::field("text", arrow::utf8())}),
        {values}, static_cast<int64_t>(texts.size()));
}

double ReadFloat(const std::shared_ptr<arrow::Table>& table,
                 const std::string& column,
                 int64_t row) {
    const auto values = std::static_pointer_cast<arrow::FloatArray>(
        table->GetColumnByName(column)->chunk(0));
    return static_cast<double>(values->Value(row));
}

std::map<std::string, std::string> TFIDFParameters(
    const std::filesystem::path& state_path) {
    return {
        {"text_col", "text"},
        {"label_col", ""},
        {"max_features", "3"},
        {"min_df", "1"},
        {"use_idf", "true"},
        {"smooth_idf", "true"},
        {"norm", "none"},
        {"ngram_range", "1,1"},
        {"stop_words", "none"},
        {"output_format", "dense"},
        {"operation_mode", "fit_transform"},
        {"save_state", "true"},
        {"state_path", state_path.string()},
        {"state_overwrite", "false"},
    };
}

std::map<std::string, std::string> CountParameters(
    const std::filesystem::path& state_path) {
    return {
        {"text_col", "text"},
        {"label_col", ""},
        {"max_features", "3"},
        {"norm", "none"},
        {"ngram_range", "1,1"},
        {"stop_words", "none"},
        {"binary", "false"},
        {"output_format", "dense"},
        {"operation_mode", "fit_transform"},
        {"save_state", "true"},
        {"state_path", state_path.string()},
        {"state_overwrite", "false"},
    };
}

size_t FindFeature(const cyxwiz::FittedPreprocessingState& state,
                   const std::string& term) {
    for (size_t index = 0; index < state.features.size(); ++index) {
        if (state.features[index].name == term) {
            return index;
        }
    }
    Check(false, "fitted state should contain term '" + term + "'");
    return 0;
}

void CheckTFIDFState(const std::shared_ptr<arrow::Table>& training,
                     const std::shared_ptr<arrow::Table>& inference,
                     const std::filesystem::path& state_path) {
    auto fit_params = TFIDFParameters(state_path);
    cyxwiz::TFIDFVectorizerOperator fit;
    std::string error;
    Check(fit.Configure(fit_params, error), error);
    Check(!fit.IsCacheable(),
          "TF-IDF state writes should bypass materialization cache");
    auto fitted_output = fit.Apply(training);
    Check(fitted_output.ok(), fitted_output.status().ToString());
    Check(std::filesystem::exists(state_path),
          "TF-IDF fit should save its state artifact");

    cyxwiz::FittedPreprocessingState state;
    Check(cyxwiz::LoadFittedPreprocessingState(
              state_path.string(), "TFIDFVectorizer", state, error),
          error);
    Check(state.features.size() == 3,
          "TF-IDF state should preserve its bounded vocabulary");
    Check(state.configuration.at("value_semantics") ==
              "sklearn_raw_count_v1",
          "TF-IDF state should pin its numerical value semantics");
    const size_t carrot = FindFeature(state, "carrot");
    const double carrot_idf =
        state.features[carrot].numeric_values.at("weight");

    auto transform_params = fit_params;
    transform_params["operation_mode"] = "transform_only";
    transform_params["save_state"] = "false";
    cyxwiz::TFIDFVectorizerOperator transform;
    error.clear();
    Check(transform.Configure(transform_params, error), error);
    Check(transform.IsCacheable(),
          "TF-IDF Transform Only should be cacheable with state identity");
    std::vector<cyxwiz::PipelineOperatorCacheDependency> dependencies;
    Check(transform.CollectCacheDependencies(dependencies, error), error);
    Check(dependencies.size() == 1 &&
              dependencies[0].role == "fitted_state" &&
              dependencies[0].path == state_path.string(),
          "TF-IDF Transform Only should declare its fitted-state dependency");
    auto transformed = transform.Apply(inference);
    Check(transformed.ok(), transformed.status().ToString());
    Check((*transformed)->num_columns() == 3,
          "TF-IDF Transform Only should retain training feature width");
    Check(std::abs(ReadFloat(*transformed,
                             "tfidf_" + std::to_string(carrot), 0) -
                   carrot_idf) < 1e-5,
          "TF-IDF Transform Only should reuse the fitted term/IDF mapping");
    for (int column = 0; column < (*transformed)->num_columns(); ++column) {
        Check(std::abs(ReadFloat(
                  *transformed, "tfidf_" + std::to_string(column), 1)) <
                  1e-7,
              "unknown TF-IDF terms should produce a fixed-width zero row");
    }

    auto mismatched_params = transform_params;
    mismatched_params["norm"] = "l2";
    cyxwiz::TFIDFVectorizerOperator mismatched;
    error.clear();
    Check(mismatched.Configure(mismatched_params, error), error);
    dependencies.clear();
    error.clear();
    Check(!mismatched.CollectCacheDependencies(dependencies, error) &&
              error.find("node setting 'norm'") != std::string::npos,
          "TF-IDF cache lookup should reject mismatched fitted settings");
    auto mismatch = mismatched.Apply(inference);
    Check(!mismatch.ok() &&
              mismatch.status().message().find("node setting 'norm'") !=
                  std::string::npos,
          "TF-IDF Transform Only should reject mismatched fitted settings");

    cyxwiz::TFIDFVectorizerOperator no_overwrite;
    error.clear();
    Check(no_overwrite.Configure(fit_params, error), error);
    auto overwrite_result = no_overwrite.Apply(training);
    Check(!overwrite_result.ok() && overwrite_result.status().IsIOError(),
          "TF-IDF fit should protect an existing artifact by default");
}

void CheckCountState(const std::shared_ptr<arrow::Table>& training,
                     const std::shared_ptr<arrow::Table>& inference,
                     const std::filesystem::path& state_path,
                     const std::filesystem::path& tfidf_state_path) {
    auto fit_params = CountParameters(state_path);
    cyxwiz::CountVectorizerOperator fit;
    std::string error;
    Check(fit.Configure(fit_params, error), error);
    Check(!fit.IsCacheable(),
          "Count state writes should bypass materialization cache");
    auto fitted_output = fit.Apply(training);
    Check(fitted_output.ok(), fitted_output.status().ToString());
    Check(std::filesystem::exists(state_path),
          "Count fit should save its state artifact");

    cyxwiz::FittedPreprocessingState state;
    Check(cyxwiz::LoadFittedPreprocessingState(
              state_path.string(), "CountVectorizer", state, error),
          error);
    Check(state.configuration.at("value_semantics") ==
              "sklearn_raw_count_v1",
          "Count state should pin its numerical value semantics");
    const size_t carrot = FindFeature(state, "carrot");

    auto transform_params = fit_params;
    transform_params["operation_mode"] = "transform_only";
    transform_params["save_state"] = "false";
    cyxwiz::CountVectorizerOperator transform;
    error.clear();
    Check(transform.Configure(transform_params, error), error);
    Check(transform.IsCacheable(),
          "Count Transform Only should be cacheable with state identity");
    std::vector<cyxwiz::PipelineOperatorCacheDependency> dependencies;
    Check(transform.CollectCacheDependencies(dependencies, error), error);
    Check(dependencies.size() == 1 &&
              dependencies[0].path == state_path.string(),
          "Count Transform Only should declare its fitted-state dependency");
    auto transformed = transform.Apply(inference);
    Check(transformed.ok(), transformed.status().ToString());
    Check((*transformed)->num_columns() == 3,
          "Count Transform Only should retain training feature width");
    Check(std::abs(ReadFloat(*transformed,
                             "count_" + std::to_string(carrot), 0) -
                   1.0) < 1e-7,
          "Count Transform Only should reuse the fitted term mapping");

    auto wrong_artifact_params = transform_params;
    wrong_artifact_params["state_path"] = tfidf_state_path.string();
    cyxwiz::CountVectorizerOperator wrong_artifact;
    error.clear();
    Check(wrong_artifact.Configure(wrong_artifact_params, error), error);
    dependencies.clear();
    error.clear();
    Check(!wrong_artifact.CollectCacheDependencies(dependencies, error) &&
              error.find("belongs to TFIDFVectorizer") != std::string::npos,
          "Count cache lookup should reject a wrong-operator state artifact");
    auto wrong = wrong_artifact.Apply(inference);
    Check(!wrong.ok() &&
              wrong.status().message().find("belongs to TFIDFVectorizer") !=
                  std::string::npos,
          "Count Transform Only should reject a TF-IDF artifact");

    transform_params["state_path"] = "";
    cyxwiz::CountVectorizerOperator missing_path;
    error.clear();
    Check(!missing_path.Configure(transform_params, error) &&
              error.find("Transform Only requires 'state_path'") !=
                  std::string::npos,
          "Transform Only should fail configuration without an artifact");
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const auto test_root =
        fs::temp_directory_path() / "cyxwiz_text_vectorizer_fitted_state";
    std::error_code remove_error;
    fs::remove_all(test_root, remove_error);
    fs::create_directories(test_root);

    const auto training = MakeTextTable({
        "apple apple banana",
        "banana carrot",
    });
    const auto inference = MakeTextTable({"carrot", "unknown"});
    const auto tfidf_state = test_root / "tfidf.cyxstate.json";
    const auto count_state = test_root / "count.cyxstate.json";

    CheckTFIDFState(training, inference, tfidf_state);
    CheckCountState(training, inference, count_state, tfidf_state);

    fs::remove_all(test_root, remove_error);
    std::cout << "Text vectorizer fitted-state tests passed\n";
    return 0;
}
