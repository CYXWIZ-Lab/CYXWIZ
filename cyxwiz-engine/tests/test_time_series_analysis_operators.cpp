#include "../src/core/node_executors/time_series_analysis_operators.h"

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

std::shared_ptr<arrow::Array> FinishDoubleArray(const std::vector<double>& values) {
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

std::shared_ptr<arrow::Table> MakeTimeSeriesTable() {
    std::vector<double> signal;
    signal.reserve(32);
    for (int i = 0; i < 32; ++i) {
        const double seasonal = static_cast<double>(i % 4);
        signal.push_back(10.0 + 0.25 * static_cast<double>(i) + seasonal);
    }

    std::vector<double> index(signal.size());
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = static_cast<double>(i);
    }

    auto schema = arrow::schema({
        arrow::field("t", arrow::float64()),
        arrow::field("signal", arrow::float64()),
    });
    return arrow::Table::Make(schema, {
        FinishDoubleArray(index),
        FinishDoubleArray(signal),
    });
}

void CheckHasColumn(const std::shared_ptr<arrow::Table>& table,
                    const std::string& column_name) {
    Check(table != nullptr, "table should not be null");
    Check(table->GetColumnByName(column_name) != nullptr,
          "missing expected column: " + column_name);
}

void TestDecomposition() {
    cyxwiz::TimeSeriesDecompositionOperator op;
    std::string error;
    Check(op.Configure({
        {"signal_col", "signal"},
        {"period", "4"},
        {"method", "additive"},
        {"algorithm", "classical"},
    }, error), error);

    auto input = MakeTimeSeriesTable();
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == input->num_rows(),
          "decomposition should preserve row count");
    CheckHasColumn(output, "trend");
    CheckHasColumn(output, "seasonal");
    CheckHasColumn(output, "residual");
}

void TestArima() {
    cyxwiz::ARIMAOperator op;
    std::string error;
    Check(op.Configure({
        {"signal_col", "signal"},
        {"p", "1"},
        {"d", "0"},
        {"q", "0"},
    }, error), error);

    auto input = MakeTimeSeriesTable();
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == input->num_rows(),
          "ARIMA in-sample fit should preserve row count");
    CheckHasColumn(output, "fitted");
    CheckHasColumn(output, "residual");
}

void TestExponentialSmoothing() {
    cyxwiz::ExponentialSmoothingOperator op;
    std::string error;
    Check(op.Configure({
        {"signal_col", "signal"},
        {"method", "simple"},
        {"alpha", "0.3"},
    }, error), error);

    auto input = MakeTimeSeriesTable();
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == input->num_rows(),
          "exponential smoothing in-sample fit should preserve row count");
    CheckHasColumn(output, "fitted");
    CheckHasColumn(output, "residual");
}

} // namespace

int main() {
    TestDecomposition();
    TestArima();
    TestExponentialSmoothing();
    std::cout << "Time-series analysis operators passed\n";
    return 0;
}
