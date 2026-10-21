#include "../src/core/node_executors/gradient_boosting_operator.h"

#include <arrow/api.h>

#include <cstdlib>
#include <iostream>
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

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeNumericLabelTable() {
    auto schema = arrow::schema({
        arrow::field("x", arrow::float64()),
        arrow::field("z", arrow::float64()),
        arrow::field("label", arrow::float64()),
    });
    return arrow::Table::Make(schema, {
        FinishDoubleArray({0.00, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 1.00}),
        FinishDoubleArray({8.00, 7.00, 6.00, 5.00, 4.00, 3.00, 2.00, 1.00}),
        FinishDoubleArray({0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 1.00}),
    });
}

std::shared_ptr<arrow::Table> MakeStringLabelTable() {
    auto schema = arrow::schema({
        arrow::field("x", arrow::float64()),
        arrow::field("z", arrow::float64()),
        arrow::field("label", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {
        FinishDoubleArray({0.00, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 1.00}),
        FinishDoubleArray({8.00, 7.00, 6.00, 5.00, 4.00, 3.00, 2.00, 1.00}),
        FinishStringArray({"low", "low", "low", "low",
                           "high", "high", "high", "high"}),
    });
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

void TestNumericLabels() {
    cyxwiz::GradientBoostingClassifierOperator op;
    std::string error;
    Check(op.Configure({
        {"target_col", "label"},
        {"feature_cols", "x,z"},
        {"prediction_col", "gb_pred"},
        {"n_estimators", "30"},
        {"learning_rate", "0.4"},
        {"max_depth", "2"},
    }, error), error);

    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });

    auto result = op.Apply(MakeNumericLabelTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_columns() == 4, "prediction column should be appended");
    Check(ReadDoubleValue(output, "gb_pred", 0) == 0.0,
          "first numeric prediction should match class 0");
    Check(ReadDoubleValue(output, "gb_pred", 7) == 1.0,
          "last numeric prediction should match class 1");
    Check(progress_events.size() > 1,
          "GradientBoosting should emit a memory preflight event");
    Check(progress_events[1].stage ==
              "GradientBoostingClassifier memory preflight",
          "GradientBoosting should preflight after resolving features");
    Check(progress_events[1].memory_risk_level == "safe",
          "small GradientBoosting fixture should report safe memory risk");
    Check(progress_events[1].estimated_memory_bytes >
              8ULL * 35ULL * static_cast<uint64_t>(sizeof(double)),
          "GradientBoosting preflight should include estimator workspace");
}

void TestStringLabels() {
    cyxwiz::GradientBoostingClassifierOperator op;
    std::string error;
    Check(op.Configure({
        {"target_col", "label"},
        {"feature_cols", "x,z"},
        {"prediction_col", "gb_class"},
        {"n_estimators", "30"},
        {"learning_rate", "0.4"},
        {"max_depth", "2"},
    }, error), error);

    auto result = op.Apply(MakeStringLabelTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(ReadStringValue(output, "gb_class", 0) == "low",
          "first string prediction should match low class");
    Check(ReadStringValue(output, "gb_class", 7) == "high",
          "last string prediction should match high class");
}

void TestValidation() {
    cyxwiz::GradientBoostingClassifierOperator op;
    std::string error;
    Check(!op.Configure({{"feature_cols", "x"}}, error),
          "missing target_col should fail configure");
    Check(error.find("target_col") != std::string::npos,
          "missing target_col error should name target_col");

    Check(!op.Configure({
        {"target_col", "label"},
        {"n_estimators", "0"},
    }, error), "zero n_estimators should fail configure");
    Check(error.find("n_estimators") != std::string::npos,
          "bad n_estimators error should name n_estimators");

    Check(!op.Configure({
        {"target_col", "label"},
        {"learning_rate", "0"},
    }, error), "zero learning_rate should fail configure");
    Check(error.find("learning_rate") != std::string::npos,
          "bad learning_rate error should name learning_rate");

    Check(!op.Configure({
        {"target_col", "label"},
        {"learning_rate", "fast"},
    }, error), "non-numeric learning_rate should fail configure");
    Check(error.find("learning_rate") != std::string::npos,
          "invalid learning_rate error should name learning_rate");
}

} // namespace

int main() {
    TestNumericLabels();
    TestStringLabels();
    TestValidation();
    std::cout << "Gradient boosting operator passed\n";
    return 0;
}
