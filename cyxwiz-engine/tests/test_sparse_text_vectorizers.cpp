#include "core/node_executors/count_vectorizer_operator.h"
#include "core/node_executors/tfidf_vectorizer_operator.h"
#include "core/sparse_feature_dataset.h"

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
    const std::vector<std::string>& texts,
    const std::vector<int32_t>& labels) {
    Check(texts.size() == labels.size(), "fixture shape mismatch");
    arrow::StringBuilder text_builder;
    arrow::Int32Builder label_builder;
    Check(text_builder.AppendValues(texts).ok(), "text fixture append failed");
    Check(label_builder.AppendValues(labels).ok(),
          "label fixture append failed");
    auto texts_result = text_builder.Finish();
    auto labels_result = label_builder.Finish();
    Check(texts_result.ok(), texts_result.status().ToString());
    Check(labels_result.ok(), labels_result.status().ToString());
    return arrow::Table::Make(
        arrow::schema({
            arrow::field("text", arrow::utf8()),
            arrow::field("label", arrow::int32()),
        }),
        {texts_result.ValueOrDie(), labels_result.ValueOrDie()},
        static_cast<int64_t>(texts.size()));
}

void CheckDenseSparseParity(
    const std::shared_ptr<arrow::Table>& dense,
    const std::shared_ptr<cyxwiz::SparseFeatureDataset>& sparse,
    const std::string& feature_prefix) {
    Check(dense != nullptr && sparse != nullptr, "outputs must be present");
    Check(dense->num_rows() == sparse->GetNumRows(), "row count mismatch");
    Check(dense->num_columns() == sparse->GetNumFeatures() + 1,
          "feature width mismatch");
    Check(sparse->GetFeatureNames().size() ==
              static_cast<size_t>(sparse->GetNumFeatures()),
          "sparse vocabulary width mismatch");

    const auto& offsets = sparse->GetRowOffsets();
    const auto& columns = sparse->GetColumnIndices();
    const auto& values = sparse->GetValues();
    for (int64_t row = 0; row < sparse->GetNumRows(); ++row) {
        size_t cursor = static_cast<size_t>(offsets[static_cast<size_t>(row)]);
        const size_t end =
            static_cast<size_t>(offsets[static_cast<size_t>(row + 1)]);
        for (int64_t feature = 0; feature < sparse->GetNumFeatures(); ++feature) {
            float sparse_value = 0.0f;
            if (cursor < end && columns[cursor] == feature) {
                sparse_value = values[cursor++];
            }
            const auto dense_values =
                std::static_pointer_cast<arrow::FloatArray>(
                    dense->GetColumnByName(
                        feature_prefix + std::to_string(feature))->chunk(0));
            Check(std::fabs(dense_values->Value(row) - sparse_value) < 1e-6f,
                  "dense/CSR numerical mismatch at row " +
                      std::to_string(row) + ", feature " +
                      std::to_string(feature));
        }
        Check(cursor == end, "CSR row contains an unmatched feature");
    }

    Check(sparse->GetLabelName() == "y" && sparse->GetLabels() != nullptr,
          "sparse labels must preserve the training label role");
    const auto dense_labels = std::static_pointer_cast<arrow::Int32Array>(
        dense->GetColumnByName("y")->chunk(0));
    const auto sparse_labels = std::static_pointer_cast<arrow::Int32Array>(
        sparse->GetLabels()->chunk(0));
    for (int64_t row = 0; row < dense->num_rows(); ++row) {
        Check(dense_labels->Value(row) == sparse_labels->Value(row),
              "dense/CSR label mismatch");
    }
}

std::map<std::string, std::string> CountParameters() {
    return {
        {"text_col", "text"},
        {"label_col", "label"},
        {"max_features", "8"},
        {"norm", "l2"},
        {"ngram_range", "1,2"},
        {"stop_words", "none"},
        {"binary", "false"},
    };
}

std::map<std::string, std::string> TFIDFParameters() {
    return {
        {"text_col", "text"},
        {"label_col", "label"},
        {"max_features", "8"},
        {"min_df", "1"},
        {"use_idf", "true"},
        {"smooth_idf", "true"},
        {"norm", "l2"},
        {"ngram_range", "1,2"},
        {"stop_words", "none"},
    };
}

template <typename Operator>
void CheckDirectRepresentationParity(
    const std::shared_ptr<arrow::Table>& input,
    const std::map<std::string, std::string>& base_parameters,
    const std::string& dataset_name,
    const std::string& feature_prefix) {
    std::string error;
    auto dense_parameters = base_parameters;
    dense_parameters["output_format"] = "dense";
    Operator dense_operator;
    Check(dense_operator.Configure(dense_parameters, error), error);
    auto dense = dense_operator.Apply(input);
    Check(dense.ok(), dense.status().ToString());
    Check(!dense_operator.ApplySparse(input, dataset_name).ok(),
          "dense configuration must reject typed sparse publication");

    auto sparse_parameters = base_parameters;
    sparse_parameters["output_format"] = "sparse";
    Operator sparse_operator;
    error.clear();
    Check(sparse_operator.Configure(sparse_parameters, error), error);
    Check(!sparse_operator.Apply(input).ok(),
          "sparse configuration must reject the Arrow table route");
    Check(!sparse_operator.ApplySparse(input, {}).ok(),
          "sparse publication must require an artifact name");
    auto sparse = sparse_operator.ApplySparse(input, dataset_name);
    Check(sparse.ok(), sparse.status().ToString());
    Check((*sparse)->GetName() == dataset_name,
          "sparse artifact identity mismatch");
    CheckDenseSparseParity(*dense, *sparse, feature_prefix);
}

template <typename Operator>
void CheckFittedStateCrossesRepresentationBoundary(
    const std::shared_ptr<arrow::Table>& training,
    const std::shared_ptr<arrow::Table>& inference,
    std::map<std::string, std::string> parameters,
    const std::filesystem::path& state_path,
    const std::string& dataset_name,
    const std::string& feature_prefix) {
    parameters["operation_mode"] = "fit_transform";
    parameters["output_format"] = "dense";
    parameters["save_state"] = "true";
    parameters["state_path"] = state_path.string();
    parameters["state_overwrite"] = "true";

    std::string error;
    Operator fit;
    Check(fit.Configure(parameters, error), error);
    auto fitted = fit.Apply(training);
    Check(fitted.ok(), fitted.status().ToString());

    parameters["operation_mode"] = "transform_only";
    parameters["save_state"] = "false";
    parameters["output_format"] = "dense";
    Operator dense_transform;
    error.clear();
    Check(dense_transform.Configure(parameters, error), error);
    auto dense = dense_transform.Apply(inference);
    Check(dense.ok(), dense.status().ToString());

    parameters["output_format"] = "sparse";
    Operator sparse_transform;
    error.clear();
    Check(sparse_transform.Configure(parameters, error), error);
    auto sparse = sparse_transform.ApplySparse(inference, dataset_name);
    Check(sparse.ok(), sparse.status().ToString());
    CheckDenseSparseParity(*dense, *sparse, feature_prefix);
    Check((*sparse)->GetRowOffsets()[1] == (*sparse)->GetRowOffsets()[2],
          "out-of-vocabulary inference row must remain an empty CSR row");
}

template <typename Operator>
void CheckInvalidOutputFormat(
    std::map<std::string, std::string> parameters) {
    parameters["output_format"] = "compressed-ish";
    Operator op;
    std::string error;
    Check(!op.Configure(parameters, error) &&
              error.find("dense") != std::string::npos &&
              error.find("sparse") != std::string::npos,
          "unknown output format must fail configuration");
}

template <typename Operator>
void CheckSparseCancellation(
    const std::shared_ptr<arrow::Table>& input,
    std::map<std::string, std::string> parameters,
    const std::string& dataset_name) {
    parameters["output_format"] = "sparse";
    Operator op;
    std::string error;
    Check(op.Configure(parameters, error), error);
    cyxwiz::PipelineOperatorExecutionContext context;
    context.cancellation_requested = []() { return true; };
    op.SetExecutionContext(std::move(context));
    const auto result = op.ApplySparse(input, dataset_name);
    Check(!result.ok() && result.status().IsCancelled(),
          "sparse vectorizer must propagate cancellation");
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    const auto training = MakeTextTable(
        {"apple apple banana", "banana carrot", "delta"},
        {1, 0, 1});
    const auto inference = MakeTextTable(
        {"carrot apple", "unknown"},
        {0, 1});

    CheckDirectRepresentationParity<cyxwiz::CountVectorizerOperator>(
        training, CountParameters(), "count_sparse", "count_");
    CheckDirectRepresentationParity<cyxwiz::TFIDFVectorizerOperator>(
        training, TFIDFParameters(), "tfidf_sparse", "tfidf_");
    CheckInvalidOutputFormat<cyxwiz::CountVectorizerOperator>(
        CountParameters());
    CheckInvalidOutputFormat<cyxwiz::TFIDFVectorizerOperator>(
        TFIDFParameters());
    CheckSparseCancellation<cyxwiz::CountVectorizerOperator>(
        training, CountParameters(), "count_cancelled_sparse");
    CheckSparseCancellation<cyxwiz::TFIDFVectorizerOperator>(
        training, TFIDFParameters(), "tfidf_cancelled_sparse");

    const auto test_root =
        fs::temp_directory_path() / "cyxwiz_sparse_text_vectorizers";
    std::error_code filesystem_error;
    fs::remove_all(test_root, filesystem_error);
    Check(fs::create_directories(test_root, filesystem_error) &&
              !filesystem_error,
          "could not create fitted-state test directory");
    CheckFittedStateCrossesRepresentationBoundary<
        cyxwiz::CountVectorizerOperator>(
            training, inference, CountParameters(),
            test_root / "count.cyxstate.json",
            "count_transform_sparse", "count_");
    CheckFittedStateCrossesRepresentationBoundary<
        cyxwiz::TFIDFVectorizerOperator>(
            training, inference, TFIDFParameters(),
            test_root / "tfidf.cyxstate.json",
            "tfidf_transform_sparse", "tfidf_");
    fs::remove_all(test_root, filesystem_error);

    std::cout << "Sparse text vectorizer parity tests passed\n";
    return 0;
}
