#include "../src/core/node_executors/decision_tree_trainer.h"
#include "../src/core/node_executors/gradient_boosting_trainer.h"
#include "../src/core/node_executors/random_forest_trainer.h"
#include "../src/core/node_executors/tree_model_artifact.h"
#include "../src/core/node_executors/tree_model_predictor_operator.h"

#include <arrow/api.h>

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
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishDoubleArray(
    const std::vector<double>& values) {
    arrow::DoubleBuilder builder;
    for (double value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeInferenceTable() {
    auto schema = arrow::schema({
        arrow::field("x", arrow::float64()),
        arrow::field("z", arrow::float64()),
    });
    return arrow::Table::Make(schema, {
        FinishDoubleArray({0.0, 0.1, 0.9, 1.0}),
        FinishDoubleArray({4.0, 3.0, 2.0, 1.0}),
    });
}

std::vector<std::vector<double>> Features() {
    return {
        {0.0, 4.0},
        {0.1, 3.0},
        {0.9, 2.0},
        {1.0, 1.0},
    };
}

std::vector<int> Labels() {
    return {0, 0, 1, 1};
}

std::vector<std::string> FeatureNames() {
    return {"x", "z"};
}

double ReadDoubleValue(const std::shared_ptr<arrow::Table>& table,
                       const std::string& column_name,
                       int64_t row) {
    auto column = table->GetColumnByName(column_name);
    Check(column != nullptr, "numeric prediction column exists");
    auto scalar_result = column->GetScalar(row);
    Check(scalar_result.ok(), scalar_result.status().ToString());
    auto scalar = *scalar_result;
    Check(scalar && scalar->is_valid, "numeric prediction is valid");
    Check(scalar->type->id() == arrow::Type::DOUBLE,
          "numeric prediction column is double");
    return std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
}

std::string ReadStringValue(const std::shared_ptr<arrow::Table>& table,
                            const std::string& column_name,
                            int64_t row) {
    auto column = table->GetColumnByName(column_name);
    Check(column != nullptr, "string prediction column exists");
    auto chunk = column->chunk(0);
    Check(chunk->type_id() == arrow::Type::STRING,
          "string prediction column is utf8");
    return std::static_pointer_cast<arrow::StringArray>(chunk)->GetString(row);
}

void TestDecisionTreeArtifactInference(const std::filesystem::path& root) {
    cyxwiz::DecisionTreeTrainingOptions options;
    options.max_depth = 2;
    cyxwiz::DecisionTreeTrainer trainer(options);
    auto model = trainer.Fit(
        Features(), Labels(), 2, FeatureNames(), {"0", "1"}, true);

    const auto path = root / "decision_tree.cyx-tree.json";
    std::string error;
    Check(cyxwiz::SaveDecisionTreeModelArtifact(model, path.string(), &error),
          error);

    cyxwiz::TreeModelPredictorOperator op;
    Check(op.Configure({
        {"model_path", path.string()},
        {"prediction_col", "tree_pred"},
    }, error), error);

    auto result = op.Apply(MakeInferenceTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_columns() == 3, "predictor appends one column");
    Check(ReadDoubleValue(output, "tree_pred", 0) == 0.0,
          "DecisionTree artifact predicts first row");
    Check(ReadDoubleValue(output, "tree_pred", 3) == 1.0,
          "DecisionTree artifact predicts last row");
}

void TestRandomForestStringLabelInference(const std::filesystem::path& root) {
    cyxwiz::RandomForestTrainingOptions options;
    options.n_estimators = 7;
    options.max_depth = 2;
    options.max_features = "all";
    options.seed = 11;
    cyxwiz::RandomForestTrainer trainer(options);
    auto model = trainer.Fit(
        Features(), Labels(), 2, FeatureNames(), {"low", "high"}, false);

    const auto path = root / "random_forest.cyx-tree.json";
    std::string error;
    Check(cyxwiz::SaveRandomForestModelArtifact(model, path.string(), &error),
          error);

    cyxwiz::TreeModelPredictorOperator op;
    Check(op.Configure({
        {"model_path", path.string()},
        {"prediction_col", "rf_pred"},
    }, error), error);

    auto result = op.Apply(MakeInferenceTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(ReadStringValue(output, "rf_pred", 0) == "low",
          "RandomForest artifact preserves string labels");
    Check(ReadStringValue(output, "rf_pred", 3) == "high",
          "RandomForest artifact predicts high class");
}

void TestGradientBoostingExplicitFeatureInference(
    const std::filesystem::path& root) {
    cyxwiz::GradientBoostingTrainingOptions options;
    options.n_estimators = 30;
    options.learning_rate = 0.4;
    options.max_depth = 2;
    cyxwiz::GradientBoostingTrainer trainer(options);
    auto model = trainer.Fit(
        Features(), Labels(), 2, FeatureNames(), {"0", "1"}, true);

    const auto path = root / "gradient_boosting.cyx-tree.json";
    std::string error;
    Check(cyxwiz::SaveGradientBoostingModelArtifact(
              model, path.string(), &error),
          error);

    cyxwiz::TreeModelPredictorOperator op;
    Check(op.Configure({
        {"model_path", path.string()},
        {"feature_cols", "x,z"},
        {"prediction_col", "gb_pred"},
    }, error), error);

    auto result = op.Apply(MakeInferenceTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(ReadDoubleValue(output, "gb_pred", 0) == 0.0,
          "GradientBoosting artifact predicts first row");
    Check(ReadDoubleValue(output, "gb_pred", 3) == 1.0,
          "GradientBoosting artifact predicts last row");
}

void TestValidation(const std::filesystem::path& root) {
    cyxwiz::TreeModelPredictorOperator op;
    std::string error;
    Check(!op.Configure({{"prediction_col", "pred"}}, error),
          "missing model_path should fail configure");
    Check(error.find("model_path") != std::string::npos,
          "missing model_path error names parameter");

    cyxwiz::DecisionTreeTrainingOptions options;
    options.max_depth = 2;
    cyxwiz::DecisionTreeTrainer trainer(options);
    auto model = trainer.Fit(
        Features(), Labels(), 2, FeatureNames(), {"0", "1"}, true);

    const auto path = root / "validation_tree.cyx-tree.json";
    Check(cyxwiz::SaveDecisionTreeModelArtifact(model, path.string(), &error),
          error);
    Check(op.Configure({
        {"model_path", path.string()},
        {"feature_cols", "missing"},
    }, error), error);
    auto result = op.Apply(MakeInferenceTable());
    Check(!result.ok(), "missing configured feature should fail apply");
    Check(result.status().ToString().find("missing") != std::string::npos,
          "missing feature error should name missing column");
}

} // namespace

int main() {
    const auto root =
        std::filesystem::temp_directory_path() / "cyxwiz_tree_model_predictor";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    TestDecisionTreeArtifactInference(root);
    TestRandomForestStringLabelInference(root);
    TestGradientBoostingExplicitFeatureInference(root);
    TestValidation(root);

    std::filesystem::remove_all(root);
    std::cout << "Tree model predictor operator passed\n";
    return 0;
}
