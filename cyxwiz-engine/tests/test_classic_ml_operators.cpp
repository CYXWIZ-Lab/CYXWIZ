// Classic ML node operators (TOFIX145 Group C): text and numeric labels in
// scikit-learn's sorted class order, the prediction column, the saved
// cyxwiz_classic_model artifact, and the Classic Model Predictor reading it back.

#include "../src/core/node_executors/classic_model_predictor_operator.h"
#include "../src/core/node_executors/knn_operator.h"
#include "../src/core/node_executors/logistic_regression_operator.h"
#include "../src/core/node_executors/naive_bayes_operator.h"
#include "../src/core/node_executors/svm_operators.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cmath>
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

std::shared_ptr<arrow::Table> CheckTextLabels(const fs::path& model_path) {
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
    return table;
}

std::shared_ptr<arrow::Table> CheckNumericLabels(const fs::path& model_path) {
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
    return *result;
}

std::shared_ptr<arrow::Table> CheckKNN(const fs::path& model_path) {
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
    return *result;
}

std::shared_ptr<arrow::Table> CheckLogistic(const fs::path& model_path) {
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
    return *result;
}

std::shared_ptr<arrow::Table> CheckSVC(const fs::path& model_path) {
    cyxwiz::SVMClassifierOperator op;
    std::string error;
    Check(op.Configure({{"target_col", "animal"}, {"kernel", "rbf"}, {"gamma", "auto"},
                        {"model_path", model_path.string()}},
                       error),
          "SVC configures: " + error);
    auto result = op.Apply(TextLabelTable());
    Check(result.ok(), "SVC applies on text labels: " + result.status().ToString());
    const auto chunk =
        std::static_pointer_cast<arrow::StringArray>((*result)->GetColumnByName("prediction")->chunk(0));
    Check(chunk->GetString(0) == "zebra" && chunk->GetString(5) == "apple", "SVC predicts the separable rows");
    const auto doc = ReadJson(model_path);
    const auto& params = doc["model"]["parameters"];
    Check(doc["model_type"] == "SVMClassifier" && params["gamma"] == 0.5 && params["gamma_setting"] == "auto" &&
              params["pair_first"].size() == 1 && params["rho"].size() == 1 && params["converged"] == true &&
              !params["support_vectors"].empty(),
          "SVC artifact keeps the support vectors and resolved gamma: " + params.dump());
    return *result;
}

std::shared_ptr<arrow::Table> CheckSVR(const fs::path& model_path) {
    // y = 2 x + 1 on six rows; a linear SVR with a tight tube recovers it.
    auto schema = arrow::schema({arrow::field("x", arrow::float64()), arrow::field("y", arrow::float64())});
    const auto table = arrow::Table::Make(schema, {Doubles({0.0, 1.0, 2.0, 3.0, 4.0, 5.0}),
                                                   Doubles({1.0, 3.0, 5.0, 7.0, 9.0, 11.0})});
    cyxwiz::SVMRegressorOperator op;
    std::string error;
    Check(op.Configure({{"target_col", "y"}, {"kernel", "linear"}, {"C", "100"}, {"epsilon", "0.01"},
                        {"model_path", model_path.string()}},
                       error),
          "SVR configures: " + error);
    auto result = op.Apply(table);
    Check(result.ok(), "SVR applies: " + result.status().ToString());
    const auto chunk = std::static_pointer_cast<arrow::DoubleArray>((*result)->GetColumnByName("prediction")->chunk(0));
    for (int64_t r = 0; r < 6; ++r)
        Check(std::abs(chunk->Value(r) - (2.0 * static_cast<double>(r) + 1.0)) <= 0.011,
              "SVR follows y = 2x + 1 within epsilon at row " + std::to_string(r));
    const auto doc = ReadJson(model_path);
    Check(doc["model_type"] == "SVMRegressor" && doc["model"]["parameters"]["epsilon"] == 0.01 &&
              doc["model"]["target"] == "y",
          "SVR artifact names its target and epsilon");

    cyxwiz::SVMRegressorOperator text_target;
    Check(text_target.Configure({{"target_col", "animal"}}, error), "SVR configures on text");
    Check(!text_target.Apply(TextLabelTable()).ok(), "SVR refuses a text target");
    return *result;
}

// The trainer's input without its target and with the columns reversed: the
// predictor finds features by name.
std::shared_ptr<arrow::Table> ScoringTable(const std::shared_ptr<arrow::Table>& trained_on, const std::string& target) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    for (int c = trained_on->num_columns() - 1; c >= 0; --c) {
        if (trained_on->schema()->field(c)->name() == target) continue;
        fields.push_back(trained_on->schema()->field(c));
        columns.push_back(trained_on->column(c));
    }
    return arrow::Table::Make(arrow::schema(fields), columns);
}

// The saved model, read back by the Classic Model Predictor, predicts what the
// trainer predicted.
void CheckPredictor(const fs::path& model_path, const std::shared_ptr<arrow::Table>& trained,
                    const std::string& target, const std::string& trainer_prediction_col) {
    cyxwiz::ClassicModelPredictorOperator op;
    std::string error;
    Check(op.Configure({{"model_path", model_path.string()}, {"prediction_col", "scored"}}, error),
          "predictor configures: " + error);
    std::vector<cyxwiz::PipelineOperatorCacheDependency> dependencies;
    Check(op.CollectCacheDependencies(dependencies, error) && dependencies.size() == 1 &&
              dependencies[0].path == model_path.string(),
          "predictor depends on the model file: " + error);
    auto result = op.Apply(ScoringTable(trained, target));
    Check(result.ok(), "predictor applies " + model_path.filename().string() + ": " + result.status().ToString());
    const auto scored = (*result)->GetColumnByName("scored");
    const auto expected = trained->GetColumnByName(trainer_prediction_col);
    Check(scored != nullptr && expected != nullptr && scored->Equals(*expected),
          "predictor reproduces the trainer's predictions for " + model_path.filename().string());
}

// Probability columns: <prediction>_p_<class> in class order, rows summing to 1;
// the predictor writes the same ones from the saved model.
void CheckProbabilityColumns(const fs::path& dir) {
    std::string error;
    const fs::path nb_path = dir / "nb_proba.cyxclassic.json";
    cyxwiz::NaiveBayesClassifierOperator nb;
    Check(nb.Configure({{"target_col", "animal"}, {"probability_columns", "true"}, {"model_path", nb_path.string()}},
                       error),
          "NB configures with probability columns: " + error);
    auto nb_result = nb.Apply(TextLabelTable());
    Check(nb_result.ok(), "NB applies with probability columns: " + nb_result.status().ToString());
    const auto apple = (*nb_result)->GetColumnByName("prediction_p_apple");
    const auto zebra = (*nb_result)->GetColumnByName("prediction_p_zebra");
    Check(apple && zebra && apple->type()->id() == arrow::Type::DOUBLE, "NB writes one probability column per class");
    const auto a = std::static_pointer_cast<arrow::DoubleArray>(apple->chunk(0));
    const auto z = std::static_pointer_cast<arrow::DoubleArray>(zebra->chunk(0));
    for (int64_t r = 0; r < a->length(); ++r)
        Check(std::abs(a->Value(r) + z->Value(r) - 1.0) < 1e-9, "probabilities sum to 1 in row " + std::to_string(r));
    Check(z->Value(0) > 0.5 && a->Value(3) > 0.5, "the probabilities favour the predicted class");

    cyxwiz::NaiveBayesClassifierOperator bad;
    Check(!bad.Configure({{"target_col", "animal"}, {"probability_columns", "maybe"}}, error) &&
              error.find("probability_columns") != std::string::npos,
          "probability_columns must be true or false");

    // SVC: Platt calibration on 5 folds, saved with the model.
    const fs::path svc_path = dir / "svc_proba.cyxclassic.json";
    cyxwiz::SVMClassifierOperator svc;
    Check(svc.Configure({{"target_col", "animal"}, {"kernel", "linear"}, {"probability_columns", "true"},
                         {"random_state", "3"}, {"model_path", svc_path.string()}},
                        error),
          "SVC configures with probability columns: " + error);
    auto svc_result = svc.Apply(TextLabelTable());
    Check(svc_result.ok(), "SVC applies with probability columns: " + svc_result.status().ToString());
    Check((*svc_result)->GetColumnByName("prediction_p_apple") && (*svc_result)->GetColumnByName("prediction_p_zebra"),
          "SVC writes probability columns");
    const auto doc = ReadJson(svc_path);
    const auto& params = doc["model"]["parameters"];
    Check(params["prob_a"].size() == 1 && params["prob_b"].size() == 1 && params["random_state"] == 3,
          "SVC saves its Platt sigmoid and random_state: " + params.dump());

    cyxwiz::SVMClassifierOperator negative_seed;
    Check(!negative_seed.Configure({{"target_col", "animal"}, {"random_state", "-1"}}, error),
          "SVC refuses a negative random_state");

    // The predictor writes the trainer's probability columns from the saved models.
    for (const auto& [path, trained] : {std::pair{nb_path, *nb_result}, std::pair{svc_path, *svc_result}}) {
        cyxwiz::ClassicModelPredictorOperator predictor;
        Check(predictor.Configure({{"model_path", path.string()}, {"probability_columns", "true"}}, error),
              "predictor configures with probability columns: " + error);
        auto scored = predictor.Apply(ScoringTable(TextLabelTable(), "animal"));
        Check(scored.ok(), "predictor writes probability columns: " + scored.status().ToString());
        for (const char* column : {"prediction_p_apple", "prediction_p_zebra"}) {
            const auto got = (*scored)->GetColumnByName(column);
            const auto want = trained->GetColumnByName(column);
            Check(got && want && got->ApproxEquals(*want), std::string("predictor reproduces ") + column + " from " +
                                                               path.filename().string());
        }
    }

    // An SVC saved without probabilities: the predictor says how to get them.
    cyxwiz::ClassicModelPredictorOperator no_probabilities;
    Check(no_probabilities.Configure({{"model_path", (dir / "svc_text.cyxclassic.json").string()},
                                      {"probability_columns", "true"}},
                                     error),
          "predictor configures");
    auto refused = no_probabilities.Apply(ScoringTable(TextLabelTable(), "animal"));
    Check(!refused.ok() && refused.status().ToString().find("without probabilities") != std::string::npos,
          "predictor refuses SVC probabilities the model lacks: " + refused.status().ToString());
}

void CheckPredictorRefusals(const fs::path& dir) {
    std::string error;
    cyxwiz::ClassicModelPredictorOperator no_model;
    Check(!no_model.Configure({}, error) && error.find("Model output") != std::string::npos,
          "predictor asks for a model: " + error);

    const fs::path other = dir / "other_format.json";
    std::ofstream(other) << R"({"format":"cyxwiz_tree_model","version":1,"model_type":"DecisionTreeClassifier","model":{}})";
    cyxwiz::ClassicModelPredictorOperator wrong_format;
    Check(wrong_format.Configure({{"model_path", other.string()}}, error), "predictor configures on another format");
    auto result = wrong_format.Apply(TextLabelTable());
    Check(!result.ok() && result.status().ToString().find("cyxwiz_classic_model") != std::string::npos,
          "predictor refuses another artifact format: " + result.status().ToString());

    const fs::path unknown = dir / "unknown_type.json";
    std::ofstream(unknown) << R"({"format":"cyxwiz_classic_model","version":1,"model_type":"Perceptron","model":)"
                              R"({"feature_names":["x"],"class_labels":["a","b"],"parameters":{}}})";
    cyxwiz::ClassicModelPredictorOperator unknown_type;
    Check(unknown_type.Configure({{"model_path", unknown.string()}}, error), "predictor configures on unknown type");
    result = unknown_type.Apply(TextLabelTable());
    Check(!result.ok() && result.status().ToString().find("Perceptron") != std::string::npos,
          "predictor names an unknown model type: " + result.status().ToString());

    cyxwiz::ClassicModelPredictorOperator missing_feature;
    Check(missing_feature.Configure({{"model_path", (dir / "nb_text.cyxclassic.json").string()}}, error),
          "predictor configures");
    result = missing_feature.Apply(NumericLabelTable());
    Check(!result.ok() && result.status().ToString().find("no column z") != std::string::npos,
          "predictor names the missing feature column: " + result.status().ToString());
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
    CheckPredictor(dir / "nb_text.cyxclassic.json", CheckTextLabels(dir / "nb_text.cyxclassic.json"), "animal", "guess");
    CheckPredictor(dir / "nb_numeric.cyxclassic.json", CheckNumericLabels(dir / "nb_numeric.cyxclassic.json"), "label",
                   "prediction");
    CheckPredictor(dir / "knn_text.cyxclassic.json", CheckKNN(dir / "knn_text.cyxclassic.json"), "animal", "prediction");
    CheckPredictor(dir / "logistic_text.cyxclassic.json", CheckLogistic(dir / "logistic_text.cyxclassic.json"), "animal",
                   "prediction");
    CheckPredictor(dir / "svc_text.cyxclassic.json", CheckSVC(dir / "svc_text.cyxclassic.json"), "animal", "prediction");
    const fs::path svr_path = dir / "run_folder_not_made_yet" / "svr_line.cyxclassic.json";  // the writer makes the folder
    CheckPredictor(svr_path, CheckSVR(svr_path), "y", "prediction");
    CheckPredictorRefusals(dir);
    CheckProbabilityColumns(dir);
    CheckRefusals();
    fs::remove_all(dir);
    std::cout << "classic ML operator tests passed\n";
    return 0;
}
