#include "../src/core/node_executors/decision_tree_operator.h"

#include <arrow/api.h>

#include <cstdlib>
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
        FinishDoubleArray({0.0, 0.1, 0.9, 1.0}),
        FinishDoubleArray({4.0, 3.0, 2.0, 1.0}),
        FinishDoubleArray({0.0, 0.0, 1.0, 1.0}),
    });
}

std::shared_ptr<arrow::Table> MakeStringLabelTable() {
    auto schema = arrow::schema({
        arrow::field("x", arrow::float64()),
        arrow::field("z", arrow::float64()),
        arrow::field("label", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {
        FinishDoubleArray({0.0, 0.1, 0.9, 1.0}),
        FinishDoubleArray({4.0, 3.0, 2.0, 1.0}),
        FinishStringArray({"low", "low", "high", "high"}),
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
    cyxwiz::DecisionTreeClassifierOperator op;
    std::string error;
    Check(op.Configure({
        {"target_col", "label"},
        {"feature_cols", "x,z"},
        {"prediction_col", "pred"},
        {"max_depth", "2"},
        {"criterion", "gini"},
    }, error), error);

    auto result = op.Apply(MakeNumericLabelTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_columns() == 4, "prediction column should be appended");
    Check(ReadDoubleValue(output, "pred", 0) == 0.0,
          "first numeric prediction should match class 0");
    Check(ReadDoubleValue(output, "pred", 3) == 1.0,
          "last numeric prediction should match class 1");
}

void TestStringLabels() {
    cyxwiz::DecisionTreeClassifierOperator op;
    std::string error;
    Check(op.Configure({
        {"target_col", "label"},
        {"feature_cols", "x,z"},
        {"prediction_col", "class_pred"},
        {"max_depth", "2"},
        {"criterion", "entropy"},
    }, error), error);

    auto result = op.Apply(MakeStringLabelTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(ReadStringValue(output, "class_pred", 0) == "low",
          "first string prediction should match low class");
    Check(ReadStringValue(output, "class_pred", 3) == "high",
          "last string prediction should match high class");
}

void TestValidation() {
    cyxwiz::DecisionTreeClassifierOperator op;
    std::string error;
    Check(!op.Configure({{"feature_cols", "x"}}, error),
          "missing target_col should fail configure");
    Check(error.find("target_col") != std::string::npos,
          "missing target_col error should name target_col");

    Check(!op.Configure({
        {"target_col", "label"},
        {"criterion", "gain_ratio"},
    }, error), "bad criterion should fail configure");
    Check(error.find("criterion") != std::string::npos,
          "bad criterion error should name criterion");
}

} // namespace

int main() {
    TestNumericLabels();
    TestStringLabels();
    TestValidation();
    std::cout << "Decision tree operator passed\n";
    return 0;
}
