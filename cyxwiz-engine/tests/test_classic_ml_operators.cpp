// Classic ML node operators (TOFIX145 Group C): text and numeric labels in
// scikit-learn's sorted class order, the prediction column, and the saved
// cyxwiz_classic_model artifact.

#include "../src/core/node_executors/knn_operator.h"
#include "../src/core/node_executors/logistic_regression_operator.h"
#include "../src/core/node_executors/naive_bayes_operator.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> Doubles(const std::vector<double>& values) {
    arrow::DoubleBuilder builder;
    Check(builder.AppendValues(values).ok(), "append doubles");
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "finish doubles");
    return array;
}

std::shared_ptr<arrow::Array> Strings(const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    Check(builder.AppendValues(values).ok(), "append strings");
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "finish strings");
    return array;
}

// "zebra" comes first in the rows; sorted order makes "apple" class 0.
std::shared_ptr<arrow::Table> TextLabelTable() {
    auto schema = arrow::schema({
        arrow::field("x", arrow::float64()),
        arrow::field("z", arrow::float64()),
        arrow::field("animal", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {
        Doubles({0.9, 1.0, 0.8, 0.0, 0.1, 0.2}),
        Doubles({2.0, 1.0, 1.5, 8.0, 7.0, 7.5}),
        Strings({"zebra", "zebra", "zebra", "apple", "apple", "apple"}),
    });
}

// Numeric labels 10 and 2: numbers sort by value, so 2 is class 0.
std::shared_ptr<arrow::Table> NumericLabelTable() {
    auto schema = arrow::schema({
        arrow::field("x", arrow::float64()),
        arrow::field("label", arrow::float64()),
    });
    return arrow::Table::Make(schema, {
        Doubles({5.0, 5.2, 4.9, 0.0, 0.1, -0.1}),
        Doubles({10.0, 10.0, 10.0, 2.0, 2.0, 2.0}),
    });
}

nlohmann::json ReadJson(const fs::path& path) {
    std::ifstream in(path);
    Check(in.good(), "artifact readable: " + path.string());
    return nlohmann::json::parse(in);
}

void CheckTextLabels(const fs::path& model_path) {
    cyxwiz::NaiveBayesClassifierOperator op;
    std::string error;
    Check(op.Configure({{"target_col", "animal"}, {"prediction_col", "guess"},
                        {"model_path", model_path.string()}},
                       error),
          "NB configures: " + error);
    auto result = op.Apply(TextLabelTable());
    Check(result.ok(), "NB applies on text labels: " + result.status().ToString());
    const auto table = *result;
    Check(table->num_columns() == 4 && table->num_rows() == 6, "NB appends one column");
    const auto guess = table->GetColumnByName("guess");
    Check(guess != nullptr && guess->type()->id() == arrow::Type::STRING,
          "text labels predict text");
    const auto chunk = std::static_pointer_cast<arrow::StringArray>(guess->chunk(0));
    Check(chunk->GetString(0) == "zebra" && chunk->GetString(3) == "apple",
          "NB predicts the separable rows");

    const auto doc = ReadJson(model_path);
    Check(doc["format"] == "cyxwiz_classic_model" &&
              doc["model_type"] == "NaiveBayesClassifier",
          "artifact envelope names the classic model");
    const auto& model = doc["model"];
    Check(model["class_labels"] == nlohmann::json::array({"apple", "zebra"}),
          "class labels in sorted order: " + model["class_labels"].dump());
    Check(model["feature_names"] == nlohmann::json::array({"x", "z"}) &&
              model["numeric_labels"] == false,
          "feature names auto-detected without the target");
    const auto& params = model["parameters"];
    Check(params["class_prior"].size() == 2 && params["theta"].size() == 2 &&
              params["var"].size() == 2,
          "fitted parameters per class");
    // apple (class 0) has small x.
    Check(params["theta"][0][0].get<double>() < 0.5 &&
              params["theta"][1][0].get<double>() > 0.5,
          "class 0 is apple after sorting");
}

void CheckNumericLabels(const fs::path& model_path) {
    cyxwiz::NaiveBayesClassifierOperator op;
    std::string error;
    Check(op.Configure({{"target_col", "label"}, {"model_path", model_path.string()}}, error),
          "NB configures: " + error);
    auto result = op.Apply(NumericLabelTable());
    Check(result.ok(), "NB applies on numeric labels: " + result.status().ToString());
    const auto prediction = (*result)->GetColumnByName("prediction");
    Check(prediction != nullptr && prediction->type()->id() == arrow::Type::DOUBLE,
          "numeric labels predict numbers");
    const auto chunk = std::static_pointer_cast<arrow::DoubleArray>(prediction->chunk(0));
    Check(chunk->Value(0) == 10.0 && chunk->Value(3) == 2.0, "NB predicts numeric labels");
    const auto doc = ReadJson(model_path);
    Check(std::stod(doc["model"]["class_labels"][0].get<std::string>()) == 2.0 &&
              doc["model"]["numeric_labels"] == true,
          "numeric classes sort by value: " + doc["model"]["class_labels"].dump());
}

void CheckKNN(const fs::path& model_path) {
    cyxwiz::KNNClassifierOperator op;
    std::string error;
    Check(op.Configure({{"target_col", "animal"}, {"n_neighbors", "3"}, {"weights", "distance"},
                        {"metric", "manhattan"}, {"model_path", model_path.string()}},
                       error),
          "KNN configures: " + error);
    auto result = op.Apply(TextLabelTable());
    Check(result.ok(), "KNN applies on text labels: " + result.status().ToString());
    const auto chunk =
        std::static_pointer_cast<arrow::StringArray>((*result)->GetColumnByName("prediction")->chunk(0));
    Check(chunk->GetString(0) == "zebra" && chunk->GetString(5) == "apple", "KNN predicts the separable rows");
    const auto doc = ReadJson(model_path);
    const auto& model = doc["model"];
    Check(doc["model_type"] == "KNNClassifier" &&
              model["class_labels"] == nlohmann::json::array({"apple", "zebra"}),
          "KNN artifact in sorted class order");
    const auto& params = model["parameters"];
    Check(params["n_neighbors"] == 3 && params["weights"] == "distance" && params["metric"] == "manhattan" &&
              params["x"].size() == 6 && params["labels"][0] == 1,
          "KNN artifact keeps its training rows and settings: " + params.dump());

    cyxwiz::KNNClassifierOperator too_many;
    Check(too_many.Configure({{"target_col", "animal"}, {"n_neighbors", "7"}}, error), "KNN configures k=7");
    Check(!too_many.Apply(TextLabelTable()).ok(), "KNN refuses more neighbours than rows");
}

void CheckLogistic(const fs::path& model_path) {
    cyxwiz::LogisticRegressionOperator op;
    std::string error;
    Check(op.Configure({{"target_col", "animal"}, {"C", "10"}, {"model_path", model_path.string()}}, error),
          "Logistic configures: " + error);
    auto result = op.Apply(TextLabelTable());
    Check(result.ok(), "Logistic applies on text labels: " + result.status().ToString());
    const auto chunk =
        std::static_pointer_cast<arrow::StringArray>((*result)->GetColumnByName("prediction")->chunk(0));
    Check(chunk->GetString(0) == "zebra" && chunk->GetString(5) == "apple", "Logistic predicts the separable rows");
    const auto doc = ReadJson(model_path);
    const auto& params = doc["model"]["parameters"];
    Check(doc["model_type"] == "LogisticRegressionNode" && params["coef"].size() == 1 &&
              params["coef"][0].size() == 2 && params["intercept"].size() == 1 && params["converged"] == true,
          "two classes save one coefficient row: " + params.dump());
}

void CheckRefusals() {
    cyxwiz::NaiveBayesClassifierOperator op;
    std::string error;
    Check(!op.Configure({}, error) && error.find("target_col") != std::string::npos,
          "NB requires target_col");
    Check(op.Configure({{"target_col", "missing"}}, error), "NB configures");
    auto result = op.Apply(NumericLabelTable());
    Check(!result.ok(), "NB refuses a missing target column");
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_classic_ml_operators";
    fs::create_directories(dir);
    CheckTextLabels(dir / "nb_text.cyxclassic.json");
    CheckNumericLabels(dir / "nb_numeric.cyxclassic.json");
    CheckKNN(dir / "knn_text.cyxclassic.json");
    CheckLogistic(dir / "logistic_text.cyxclassic.json");
    CheckRefusals();
    fs::remove_all(dir);
    std::cout << "classic ML operator tests passed\n";
    return 0;
}
