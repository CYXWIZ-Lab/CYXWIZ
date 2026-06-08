#include "../src/core/node_executors/count_vectorizer_operator.h"
#include "../src/core/node_executors/time_series_features_operator.h"
#include "../src/core/node_executors/time_series_window_operator.h"

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

std::shared_ptr<arrow::Array> FinishFloatArray(
    const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeTextTable() {
    auto schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
        arrow::field("label", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {
        FinishStringArray({
            "alpha beta",
            "alpha gamma",
            "delta epsilon",
        }),
        FinishStringArray({"yes", "yes", "no"}),
    }, 3);
}

std::shared_ptr<arrow::Table> MakeTimeSeriesTable() {
    auto schema = arrow::schema({
        arrow::field("value", arrow::float32()),
        arrow::field("extra", arrow::float32()),
        arrow::field("time", arrow::float32()),
    });
    return arrow::Table::Make(schema, {
        FinishFloatArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
        FinishFloatArray({10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f}),
        FinishFloatArray({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}),
    }, 6);
}

void TestCountVectorizerResetsOptionalLabelAndMaxFeatures() {
    cyxwiz::CountVectorizerOperator op;
    const auto input = MakeTextTable();
    std::string error;

    Check(op.Configure({
        {"text_col", "text"},
        {"label_col", "label"},
        {"max_features", "1"},
    }, error), error);
    auto first = op.Apply(input);
    Check(first.ok(), first.status().ToString());
    Check(first.ValueOrDie()->GetColumnByName("y") != nullptr,
          "first count-vectorizer configure should emit label column");
    Check(first.ValueOrDie()->num_columns() == 2,
          "first count-vectorizer configure should honor max_features=1 plus y");

    Check(op.Configure({{"text_col", "text"}}, error), error);
    auto second = op.Apply(input);
    Check(second.ok(), second.status().ToString());
    auto second_table = second.ValueOrDie();
    Check(second_table->GetColumnByName("y") == nullptr,
          "second count-vectorizer configure should clear stale label_col");
    Check(second_table->num_columns() > 1,
          "second count-vectorizer configure should restore default max_features");
}

void TestTimeSeriesFeaturesClearsStaleFeatureLists() {
    cyxwiz::TimeSeriesFeaturesOperator op;
    const auto input = MakeTimeSeriesTable();
    std::string error;

    Check(op.Configure({
        {"value_col", "value"},
        {"lag_values", "2"},
    }, error), error);
    auto first = op.Apply(input);
    Check(first.ok(), first.status().ToString());
    Check(first.ValueOrDie()->GetColumnByName("value_lag_2") != nullptr,
          "first time-series features configure should emit lag column");

    Check(op.Configure({
        {"value_col", "value"},
        {"rolling_windows", "2"},
    }, error), error);
    auto second = op.Apply(input);
    Check(second.ok(), second.status().ToString());
    auto second_table = second.ValueOrDie();
    Check(second_table->GetColumnByName("value_lag_2") == nullptr,
          "second time-series features configure should clear stale lags");
    Check(second_table->GetColumnByName("value_roll_2_mean") != nullptr,
          "second time-series features configure should emit requested rolling mean");
}

void TestTimeSeriesWindowClearsOptionalFeatureAndTimeColumns() {
    cyxwiz::TimeSeriesWindowOperator op;
    const auto input = MakeTimeSeriesTable();
    std::string error;

    Check(op.Configure({
        {"value_col", "value"},
        {"feature_cols", "extra"},
        {"time_col", "time"},
        {"input_width", "2"},
    }, error), error);
    auto first = op.Apply(input);
    Check(first.ok(), first.status().ToString());
    Check(first.ValueOrDie()->GetColumnByName("extra_x_0") != nullptr,
          "first time-series window configure should emit extra feature block");
    Check(first.ValueOrDie()->GetColumnByName("__window_start_time") != nullptr,
          "first time-series window configure should emit time metadata");

    Check(op.Configure({
        {"value_col", "value"},
        {"input_width", "2"},
    }, error), error);
    auto second = op.Apply(input);
    Check(second.ok(), second.status().ToString());
    auto second_table = second.ValueOrDie();
    Check(second_table->GetColumnByName("extra_x_0") == nullptr,
          "second time-series window configure should clear stale feature_cols");
    Check(second_table->GetColumnByName("__window_start_time") == nullptr,
          "second time-series window configure should clear stale time_col");
    Check(second_table->num_columns() == 3,
          "second time-series window configure should emit x_0, x_1, y only");
}

} // namespace

int main() {
    TestCountVectorizerResetsOptionalLabelAndMaxFeatures();
    TestTimeSeriesFeaturesClearsStaleFeatureLists();
    TestTimeSeriesWindowClearsOptionalFeatureAndTimeColumns();
    std::cout << "Operator Configure reset regressions passed\n";
    return 0;
}
