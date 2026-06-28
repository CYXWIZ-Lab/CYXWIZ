#include "decision_tree_operator.h"
#include "feature_matrix_utils.h"

#include <arrow/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace cyxwiz {

namespace {

std::string TrimString(const std::string& value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(),
        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(),
        [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

bool ParseIntParam(const std::map<std::string, std::string>& params,
                   const std::string& key,
                   int& out,
                   const std::string& op_name,
                   std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        return true;
    }
    try {
        size_t parsed = 0;
        const int value = std::stoi(it->second, &parsed);
        if (parsed != it->second.size()) {
            throw std::runtime_error("trailing characters");
        }
        out = value;
        return true;
    } catch (...) {
        error = op_name + ": '" + key + "' is not a valid integer: " +
                it->second;
        return false;
    }
}

bool IsNumericType(const std::shared_ptr<arrow::DataType>& type) {
    if (!type) return false;
    switch (type->id()) {
        case arrow::Type::DOUBLE:
        case arrow::Type::FLOAT:
        case arrow::Type::INT64:
        case arrow::Type::INT32:
        case arrow::Type::INT16:
        case arrow::Type::INT8:
        case arrow::Type::UINT64:
        case arrow::Type::UINT32:
        case arrow::Type::UINT16:
        case arrow::Type::UINT8:
            return true;
        default:
            return false;
    }
}

bool IsStringType(const std::shared_ptr<arrow::DataType>& type) {
    return type && (type->id() == arrow::Type::STRING ||
                    type->id() == arrow::Type::LARGE_STRING);
}

arrow::Status ReadLabels(const std::shared_ptr<arrow::Table>& table,
                         const std::string& target_col,
                         std::vector<int>& labels,
                         std::vector<std::string>& class_labels,
                         bool& numeric_labels) {
    auto column = table->GetColumnByName(target_col);
    if (!column) {
        return arrow::Status::KeyError(
            "DecisionTreeClassifier: target column '" + target_col +
            "' not found");
    }
    if (column->num_chunks() == 0) {
        return arrow::Status::Invalid(
            "DecisionTreeClassifier: target column has no chunks");
    }
    const auto type = column->type();
    numeric_labels = IsNumericType(type);
    if (!numeric_labels && !IsStringType(type)) {
        return arrow::Status::TypeError(
            "DecisionTreeClassifier: target column must be numeric or string");
    }

    std::map<std::string, int> label_to_id;
    labels.clear();
    class_labels.clear();
    labels.reserve(static_cast<size_t>(table->num_rows()));

    const auto record_label = [&](const std::string& key) {
        auto it = label_to_id.find(key);
        if (it == label_to_id.end()) {
            const int id = static_cast<int>(class_labels.size());
            it = label_to_id.emplace(key, id).first;
            class_labels.push_back(key);
        }
        labels.push_back(it->second);
    };

    for (int chunk_index = 0; chunk_index < column->num_chunks(); ++chunk_index) {
        const auto chunk = column->chunk(chunk_index);
        if (chunk->type_id() == arrow::Type::STRING) {
            const auto strings = std::static_pointer_cast<arrow::StringArray>(chunk);
            for (int64_t row = 0; row < strings->length(); ++row) {
                if (strings->IsNull(row)) {
                    return arrow::Status::Invalid(
                        "DecisionTreeClassifier: target column contains null labels");
                }
                record_label(strings->GetString(row));
            }
            continue;
        }
        if (chunk->type_id() == arrow::Type::LARGE_STRING) {
            const auto strings =
                std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            for (int64_t row = 0; row < strings->length(); ++row) {
                if (strings->IsNull(row)) {
                    return arrow::Status::Invalid(
                        "DecisionTreeClassifier: target column contains null labels");
                }
                record_label(strings->GetString(row));
            }
            continue;
        }
        for (int64_t row = 0; row < chunk->length(); ++row) {
            if (chunk->IsNull(row)) {
                return arrow::Status::Invalid(
                    "DecisionTreeClassifier: target column contains null labels");
            }
            ARROW_ASSIGN_OR_RAISE(auto scalar, chunk->GetScalar(row));
            record_label(scalar->ToString());
        }
    }

    if (class_labels.size() < 2) {
        return arrow::Status::Invalid(
            "DecisionTreeClassifier: target column must contain at least two classes");
    }
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> AppendPredictions(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& prediction_col,
    const DecisionTreeModel& model,
    const std::vector<int>& predicted_classes) {
    if (input->num_rows() != static_cast<int64_t>(predicted_classes.size())) {
        return arrow::Status::Invalid(
            "DecisionTreeClassifier: prediction count does not match row count");
    }

    if (model.HasNumericLabels()) {
        arrow::DoubleBuilder builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(input->num_rows()));
        for (int cls : predicted_classes) {
            const auto& raw = model.ClassLabels()[static_cast<size_t>(cls)];
            try {
                ARROW_RETURN_NOT_OK(builder.Append(std::stod(raw)));
            } catch (...) {
                return arrow::Status::Invalid(
                    "DecisionTreeClassifier: numeric label could not be restored: " +
                    raw);
            }
        }
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        return input->AddColumn(
            input->num_columns(),
            arrow::field(prediction_col, arrow::float64()),
            std::make_shared<arrow::ChunkedArray>(array));
    }

    arrow::StringBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(input->num_rows()));
    for (int cls : predicted_classes) {
        ARROW_RETURN_NOT_OK(
            builder.Append(model.ClassLabels()[static_cast<size_t>(cls)]));
    }
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return input->AddColumn(
        input->num_columns(),
        arrow::field(prediction_col, arrow::utf8()),
        std::make_shared<arrow::ChunkedArray>(array));
}

} // namespace

bool DecisionTreeClassifierOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    feature_cols_.clear();
    target_col_.clear();
    prediction_col_ = "prediction";
    options_ = DecisionTreeTrainingOptions{};

    auto fc = params.find("feature_cols");
    if (fc != params.end()) {
        ParseCommaList(fc->second, feature_cols_);
    }

    auto target = params.find("target_col");
    if (target == params.end() || TrimString(target->second).empty()) {
        error = "DecisionTreeClassifier: 'target_col' parameter is required";
        return false;
    }
    target_col_ = TrimString(target->second);

    auto prediction = params.find("prediction_col");
    if (prediction != params.end() && !TrimString(prediction->second).empty()) {
        prediction_col_ = TrimString(prediction->second);
    }

    if (!ParseIntParam(params, "max_depth", options_.max_depth, GetName(), error) ||
        !ParseIntParam(params, "min_samples_split",
                       options_.min_samples_split, GetName(), error) ||
        !ParseIntParam(params, "min_samples_leaf",
                       options_.min_samples_leaf, GetName(), error)) {
        return false;
    }

    auto criterion = params.find("criterion");
    if (criterion != params.end() && !criterion->second.empty()) {
        options_.criterion = ToLowerAscii(TrimString(criterion->second));
    }
    if (options_.criterion != "gini" && options_.criterion != "entropy") {
        error = "DecisionTreeClassifier: 'criterion' must be 'gini' or "
                "'entropy' (got '" + options_.criterion + "')";
        return false;
    }
    if (options_.max_depth < 1) {
        error = "DecisionTreeClassifier: max_depth must be >= 1";
        return false;
    }
    if (options_.min_samples_split < 2) {
        error = "DecisionTreeClassifier: min_samples_split must be >= 2";
        return false;
    }
    if (options_.min_samples_leaf < 1) {
        error = "DecisionTreeClassifier: min_samples_leaf must be >= 1";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
DecisionTreeClassifierOperator::Apply(
    const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid(
            "DecisionTreeClassifier: input table is null");
    }

    std::vector<std::string> resolved_features;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        input, feature_cols_, target_col_, GetName(), resolved_features));

    std::vector<std::vector<double>> features;
    int64_t n_samples = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(
        input, resolved_features, GetName(), features, n_samples));
    if (n_samples <= 0) {
        return arrow::Status::Invalid(
            "DecisionTreeClassifier: input table has no rows");
    }

    std::vector<int> labels;
    std::vector<std::string> class_labels;
    bool numeric_labels = false;
    ARROW_RETURN_NOT_OK(ReadLabels(
        input, target_col_, labels, class_labels, numeric_labels));
    if (labels.size() != static_cast<size_t>(n_samples)) {
        return arrow::Status::Invalid(
            "DecisionTreeClassifier: feature/label row mismatch");
    }

    DecisionTreeTrainer trainer(options_);
    DecisionTreeModel model;
    try {
        model = trainer.Fit(features, labels, class_labels.size(),
                            resolved_features, class_labels, numeric_labels);
    } catch (const std::exception& ex) {
        return arrow::Status::Invalid(ex.what());
    }

    const std::vector<int> predictions = model.PredictClasses(features);
    spdlog::info(
        "DecisionTreeClassifier: fit {} rows x {} features, classes={}, "
        "nodes={}, depth={}, criterion={}",
        n_samples, resolved_features.size(), class_labels.size(),
        model.Nodes().size(), model.MaxDepth(), options_.criterion);

    return AppendPredictions(input, prediction_col_, model, predictions);
}

} // namespace cyxwiz
