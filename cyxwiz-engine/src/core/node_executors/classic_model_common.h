#pragma once

// Shared pieces of the classic ML nodes (TOFIX145 Group C): Naive Bayes, KNN,
// Logistic Regression, SVM. Data in -> the input plus a prediction column, and
// the fitted model saved as a cyxwiz_classic_model artifact (the Model pin).
// Classes are numbered in scikit-learn's order (sorted classes_), so ties and
// probability columns agree with it.

#include "pipeline_operator.h"

#include <cyxwiz/classic_ml.h>

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

// Columns every classic model node takes.
struct ClassicColumns {
    std::vector<std::string> feature_cols;  // empty: every numeric column but the target
    std::string target_col;
    std::string prediction_col = "prediction";
    std::string model_path;
};

bool ParseClassicColumns(const std::map<std::string, std::string>& params, const std::string& op_name,
                         ClassicColumns& out, std::string& error);

// The training rows of a classic model node.
struct ClassicTrainingData {
    std::vector<std::string> features;   // resolved feature columns, in order
    ClassicMatrix x;                     // [n, d]
    std::vector<int> labels;             // 0..k-1 in sorted class order
    std::vector<std::string> classes;    // class label text, sorted
    bool numeric_labels = false;
};

// Features, labels and the sorted class order; numeric labels sort by value.
arrow::Status ReadClassicTrainingData(const std::shared_ptr<arrow::Table>& input, const ClassicColumns& columns,
                                      const std::string& op_name, const PipelineOperatorCancellationQuery& cancelled,
                                      ClassicTrainingData& out);

// Renumbers labels so classes are sorted (numbers by value, text by bytes).
void SortClassOrder(std::vector<int>& labels, std::vector<std::string>& classes, bool numeric_labels);

// {"format":"cyxwiz_classic_model","version":1,"model_type":...,"model":
// {feature_names, class_labels, numeric_labels, parameters}} written to `path`.
bool SaveClassicModelArtifact(const std::string& path, const std::string& model_type,
                              const ClassicTrainingData& data, const nlohmann::json& parameters, std::string* error);

// The input plus a float64 prediction column (regressors); refuses a column
// name already in the input.
arrow::Result<std::shared_ptr<arrow::Table>> AppendNumericPredictions(const std::shared_ptr<arrow::Table>& input,
                                                                     const std::string& prediction_col,
                                                                     const std::vector<double>& values,
                                                                     const std::string& op_name);

}  // namespace cyxwiz
