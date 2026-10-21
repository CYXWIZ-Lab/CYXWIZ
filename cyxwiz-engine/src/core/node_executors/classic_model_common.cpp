#include "classic_model_common.h"

#include "feature_matrix_utils.h"
#include "model_artifact_json_io.h"
#include "tree_classification_utils.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>

namespace cyxwiz {

bool ParseClassicColumns(const std::map<std::string, std::string>& params, const std::string& op_name,
                         ClassicColumns& out, std::string& error) {
    out = ClassicColumns{};
    if (const auto it = params.find("feature_cols"); it != params.end()) ParseCommaList(it->second, out.feature_cols);
    const auto target = params.find("target_col");
    if (target == params.end() || TrimAscii(target->second).empty()) {
        error = op_name + ": 'target_col' parameter is required";
        return false;
    }
    out.target_col = TrimAscii(target->second);
    if (const auto it = params.find("prediction_col"); it != params.end() && !TrimAscii(it->second).empty())
        out.prediction_col = TrimAscii(it->second);
    if (const auto it = params.find("model_path"); it != params.end() && !TrimAscii(it->second).empty())
        out.model_path = TrimAscii(it->second);
    return ParseProbabilityColumns(params, op_name, out.probability_columns, error);
}

bool ParseProbabilityColumns(const std::map<std::string, std::string>& params, const std::string& op_name, bool& out,
                             std::string& error) {
    out = false;
    const auto it = params.find("probability_columns");
    if (it == params.end()) return true;
    const std::string value = TrimAscii(it->second);
    if (value.empty() || value == "false" || value == "0") return true;
    if (value == "true" || value == "1") {
        out = true;
        return true;
    }
    error = op_name + ": 'probability_columns' must be true or false";
    return false;
}

void SortClassOrder(std::vector<int>& labels, std::vector<std::string>& classes, bool numeric_labels) {
    std::vector<int> order(classes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (numeric_labels) return std::strtod(classes[a].c_str(), nullptr) < std::strtod(classes[b].c_str(), nullptr);
        return classes[a] < classes[b];
    });
    std::vector<int> new_id(classes.size());
    std::vector<std::string> sorted(classes.size());
    for (size_t i = 0; i < order.size(); ++i) {
        new_id[static_cast<size_t>(order[i])] = static_cast<int>(i);
        sorted[i] = classes[static_cast<size_t>(order[i])];
    }
    for (int& label : labels) label = new_id[static_cast<size_t>(label)];
    classes = std::move(sorted);
}

arrow::Status ReadClassicTrainingData(const std::shared_ptr<arrow::Table>& input, const ClassicColumns& columns,
                                      const std::string& op_name, const PipelineOperatorCancellationQuery& cancelled,
                                      ClassicTrainingData& out) {
    if (!input) return arrow::Status::Invalid(op_name + ": input table is null");
    out = ClassicTrainingData{};
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(input, columns.feature_cols, columns.target_col, op_name, out.features));
    int64_t n = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(input, out.features, op_name, out.x, n, cancelled));
    if (n <= 0) return arrow::Status::Invalid(op_name + ": input table has no rows");
    ARROW_RETURN_NOT_OK(
        ReadClassificationLabels(input, columns.target_col, op_name, out.labels, out.classes, out.numeric_labels));
    if (out.labels.size() != static_cast<size_t>(n)) return arrow::Status::Invalid(op_name + ": feature/label row mismatch");
    SortClassOrder(out.labels, out.classes, out.numeric_labels);
    return arrow::Status::OK();
}

bool SaveClassicModelArtifact(const std::string& path, const std::string& model_type, const ClassicTrainingData& data,
                              const nlohmann::json& parameters, std::string* error) {
    const nlohmann::json model = {
        {"feature_names", data.features},
        {"class_labels", data.classes},
        {"numeric_labels", data.numeric_labels},
        {"parameters", parameters},
    };
    return artifact_json::WriteJsonFile(path, artifact_json::MakeEnvelope("cyxwiz_classic_model", model_type, model),
                                        error);
}

arrow::Result<std::shared_ptr<arrow::Table>> AppendClassicPredictions(const std::shared_ptr<arrow::Table>& input,
                                                                     const std::string& prediction_col,
                                                                     bool probability_columns,
                                                                     const std::vector<std::string>& classes,
                                                                     bool numeric_labels,
                                                                     const ClassicPrediction& prediction,
                                                                     const std::string& op_name) {
    ARROW_ASSIGN_OR_RAISE(auto table, AppendClassificationPredictions(input, prediction_col, classes, numeric_labels,
                                                                      prediction.labels, op_name));
    if (!probability_columns) return table;
    if (prediction.probabilities.size() != prediction.labels.size())
        return arrow::Status::Invalid(op_name + ": this model gives no class probabilities");
    for (size_t k = 0; k < classes.size(); ++k) {
        const std::string name = prediction_col + "_p_" + classes[k];
        if (table->schema()->GetFieldIndex(name) >= 0)
            return arrow::Status::Invalid(op_name + ": probability column '" + name + "' already exists");
        arrow::DoubleBuilder builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(prediction.probabilities.size())));
        for (const auto& row : prediction.probabilities) {
            if (row.size() != classes.size())
                return arrow::Status::Invalid(op_name + ": probabilities and classes differ in number");
            builder.UnsafeAppend(row[k]);
        }
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        ARROW_ASSIGN_OR_RAISE(table, table->AddColumn(table->num_columns(), arrow::field(name, arrow::float64()),
                                                      std::make_shared<arrow::ChunkedArray>(array)));
    }
    return table;
}

arrow::Result<std::shared_ptr<arrow::Table>> AppendNumericPredictions(const std::shared_ptr<arrow::Table>& input,
                                                                     const std::string& prediction_col,
                                                                     const std::vector<double>& values,
                                                                     const std::string& op_name) {
    if (input->schema()->GetFieldIndex(prediction_col) >= 0)
        return arrow::Status::Invalid(op_name + ": prediction column '" + prediction_col + "' already exists");
    arrow::DoubleBuilder builder;
    ARROW_RETURN_NOT_OK(builder.AppendValues(values));
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return input->AddColumn(input->num_columns(), arrow::field(prediction_col, arrow::float64()),
                            std::make_shared<arrow::ChunkedArray>(array));
}

}  // namespace cyxwiz
