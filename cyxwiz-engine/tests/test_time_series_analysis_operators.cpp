#include "../src/core/node_executors/time_series_analysis_operators.h"

#include <arrow/api.h>

#include <cstdlib>
#include <cmath>
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

std::shared_ptr<arrow::Array> FinishFloatArray(const std::vector<float>& values) {
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

std::shared_ptr<arrow::Array> FinishInt64Array(const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    for (int64_t value : values) {
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

    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });

    auto input = MakeTimeSeriesTable();
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    Check(!progress_events.empty(),
          "TimeSeriesDecomposition should emit materialization progress events");
    Check(progress_events.front().stage ==
              "TimeSeriesDecomposition memory preflight",
          "TimeSeriesDecomposition first progress event should be memory preflight");
    Check(progress_events.front().status == "running",
          "safe TimeSeriesDecomposition preflight should stay in running status");
    Check(progress_events.front().memory_risk_level == "safe",
          "safe TimeSeriesDecomposition preflight should report safe risk");
    Check(progress_events.front().estimated_memory_bytes >
              32ULL * 4ULL * static_cast<uint64_t>(sizeof(double)),
          "TimeSeriesDecomposition preflight should include peak allocation overhead");
    Check(progress_events.front().total_items == 32ULL * 4ULL,
          "TimeSeriesDecomposition preflight should report planned materialized cells");
    Check(progress_events.front().message.find("Suggestion:") !=
              std::string::npos,
          "TimeSeriesDecomposition preflight should include mitigation guidance");
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

void TestACF() {
    cyxwiz::ACFOperator op;
    std::string error;
    Check(op.Configure({
        {"signal_col", "signal"},
        {"max_lag", "6"},
    }, error), error);

    auto result = op.Apply(MakeTimeSeriesTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == 7, "ACF should emit max_lag + 1 rows");
    CheckHasColumn(output, "lag");
    CheckHasColumn(output, "acf");
    CheckHasColumn(output, "confidence_lower");
    CheckHasColumn(output, "confidence_upper");
    CheckHasColumn(output, "significant");
}

void TestPACF() {
    cyxwiz::PACFOperator op;
    std::string error;
    Check(op.Configure({
        {"signal_col", "signal"},
        {"max_lag", "6"},
    }, error), error);

    auto result = op.Apply(MakeTimeSeriesTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == 7, "PACF should emit max_lag + 1 rows");
    CheckHasColumn(output, "lag");
    CheckHasColumn(output, "pacf");
    CheckHasColumn(output, "confidence_lower");
    CheckHasColumn(output, "confidence_upper");
    CheckHasColumn(output, "significant");
}

void TestStationarity() {
    cyxwiz::StationarityTestOperator op;
    std::string error;
    Check(op.Configure({
        {"signal_col", "signal"},
        {"max_lags", "-1"},
    }, error), error);

    auto result = op.Apply(MakeTimeSeriesTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == 1, "stationarity should emit one summary row");
    CheckHasColumn(output, "adf_statistic");
    CheckHasColumn(output, "adf_pvalue");
    CheckHasColumn(output, "adf_stationary");
    CheckHasColumn(output, "kpss_statistic");
    CheckHasColumn(output, "kpss_pvalue");
    CheckHasColumn(output, "kpss_stationary");
    CheckHasColumn(output, "is_stationary");
    CheckHasColumn(output, "suggested_differencing");
    CheckHasColumn(output, "rolling_window");
    CheckHasColumn(output, "analysis");
}

void TestSeasonality() {
    cyxwiz::SeasonalityDetectorOperator op;
    std::string error;
    Check(op.Configure({
        {"signal_col", "signal"},
        {"min_period", "2"},
        {"max_period", "-1"},
    }, error), error);

    auto result = op.Apply(MakeTimeSeriesTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() >= 1, "seasonality should emit at least one row");
    CheckHasColumn(output, "period");
    CheckHasColumn(output, "strength");
    CheckHasColumn(output, "is_primary");
    CheckHasColumn(output, "has_seasonality");
    CheckHasColumn(output, "analysis");
}

void TestSeasonalNaiveLongFormForecast() {
    auto input = arrow::Table::Make(
        arrow::schema({
            arrow::field("x_0", arrow::float32()),
            arrow::field("x_1", arrow::float32()),
            arrow::field("x_2", arrow::float32()),
            arrow::field("x_3", arrow::float32()),
            arrow::field("y", arrow::float32()),
            arrow::field("y_1", arrow::float32()),
            arrow::field("__target_start_index", arrow::int64()),
            // FilterRows/DuckDB may widen the compact split metadata. The
            // baseline operator must accept integral physical widths.
            arrow::field("__partition__", arrow::int64()),
        }),
        {
            FinishFloatArray({10.0f, 11.0f}),
            FinishFloatArray({11.0f, 20.0f}),
            FinishFloatArray({20.0f, 21.0f}),
            FinishFloatArray({21.0f, 30.0f}),
            FinishFloatArray({30.0f, 31.0f}),
            FinishFloatArray({31.0f, 40.0f}),
            FinishInt64Array({100, 101}),
            FinishInt64Array({1, 2}),
        });

    cyxwiz::SeasonalNaiveOperator op;
    std::string error;
    Check(op.Configure({{"seasonal_period", "2"}}, error), error);
    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });

    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == 4,
          "seasonal naive should emit one row per window and horizon");
    CheckHasColumn(output, "actual");
    CheckHasColumn(output, "prediction");
    CheckHasColumn(output, "error");
    CheckHasColumn(output, "__target_index");
    CheckHasColumn(output, "__partition__");

    auto actual = std::static_pointer_cast<arrow::DoubleArray>(
        output->GetColumnByName("actual")->chunk(0));
    auto prediction = std::static_pointer_cast<arrow::DoubleArray>(
        output->GetColumnByName("prediction")->chunk(0));
    auto target_index = std::static_pointer_cast<arrow::Int64Array>(
        output->GetColumnByName("__target_index")->chunk(0));
    auto partition = std::static_pointer_cast<arrow::Int8Array>(
        output->GetColumnByName("__partition__")->chunk(0));
    Check(std::fabs(actual->Value(0) - 30.0) < 1e-9 &&
              std::fabs(actual->Value(3) - 40.0) < 1e-9,
          "seasonal naive should preserve ordered actual targets");
    Check(std::fabs(prediction->Value(0) - 20.0) < 1e-9 &&
              std::fabs(prediction->Value(1) - 21.0) < 1e-9 &&
              std::fabs(prediction->Value(2) - 21.0) < 1e-9 &&
              std::fabs(prediction->Value(3) - 30.0) < 1e-9,
          "seasonal naive should repeat the latest complete seasonal cycle");
    Check(target_index->Value(0) == 100 && target_index->Value(3) == 102,
          "seasonal naive should preserve target timeline indices");
    Check(partition->Value(0) == 1 && partition->Value(3) == 2,
          "seasonal naive should expand partition metadata with each horizon");
    Check(!progress_events.empty() &&
              progress_events.front().stage ==
                  "SeasonalNaive memory preflight",
          "seasonal naive should emit a memory preflight before expansion");

    cyxwiz::SeasonalNaiveOperator invalid;
    Check(invalid.Configure({{"seasonal_period", "5"}}, error), error);
    auto invalid_result = invalid.Apply(input);
    Check(!invalid_result.ok() &&
              invalid_result.status().ToString().find(
                  "exceeds available input width") != std::string::npos,
          "seasonal period larger than lookback should fail clearly");
}

} // namespace

int main() {
    TestDecomposition();
    TestArima();
    TestExponentialSmoothing();
    TestACF();
    TestPACF();
    TestStationarity();
    TestSeasonality();
    TestSeasonalNaiveLongFormForecast();
    std::cout << "Time-series analysis operators passed\n";
    return 0;
}
