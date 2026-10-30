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

// A date32 column "day" (daily from 2024-01-01) next to the signal.
std::shared_ptr<arrow::Table> MakeDailyTable() {
    auto base = MakeTimeSeriesTable();
    arrow::Date32Builder days;
    const int32_t start = 19723;  // 2024-01-01 in days since 1970-01-01
    for (int64_t i = 0; i < base->num_rows(); ++i) (void)days.Append(start + static_cast<int32_t>(i));
    std::shared_ptr<arrow::Array> day;
    (void)days.Finish(&day);
    return base->AddColumn(0, arrow::field("day", arrow::date32()), std::make_shared<arrow::ChunkedArray>(day)).ValueOrDie();
}

double DoubleAt(const std::shared_ptr<arrow::Table>& table, const std::string& column, int64_t row) {
    auto col = table->GetColumnByName(column);
    Check(col != nullptr, "missing column " + column);
    int64_t offset = row;
    for (const auto& chunk : col->chunks()) {
        if (offset < chunk->length()) {
            Check(chunk->type_id() == arrow::Type::DOUBLE, column + " should be float64");
            const auto& a = static_cast<const arrow::DoubleArray&>(*chunk);
            return a.IsNull(offset) ? std::nan("") : a.Value(offset);
        }
        offset -= chunk->length();
    }
    Check(false, "row out of range in " + column);
    return 0.0;
}

bool IsNullAt(const std::shared_ptr<arrow::Table>& table, const std::string& column, int64_t row) {
    auto col = table->GetColumnByName(column);
    Check(col != nullptr, "missing column " + column);
    int64_t offset = row;
    for (const auto& chunk : col->chunks()) {
        if (offset < chunk->length()) return chunk->IsNull(offset);
        offset -= chunk->length();
    }
    return true;
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
    // Classical: half a period empty at each end, the middle filled.
    Check(IsNullAt(output, "trend", 0) && IsNullAt(output, "trend", 31) && !IsNullAt(output, "trend", 2),
          "classical decomposition leaves its edges empty");
    auto schema = op.InferOutputSchema(input->schema()).ValueOrDie();
    Check(schema->num_fields() == input->num_columns() + 3 && schema->GetFieldIndex("residual") >= 0,
          "decomposition schema adds the three parts");

    // STL with the period found: the saw of period 4 stands out.
    cyxwiz::TimeSeriesDecompositionOperator stl;
    Check(stl.Configure({{"signal_col", "signal"}, {"period", "-1"}, {"algorithm", "stl"}, {"robust", "true"}}, error), error);
    auto stl_result = stl.Apply(input);
    Check(stl_result.ok(), stl_result.status().ToString());
    Check(!IsNullAt(stl_result.ValueOrDie(), "trend", 0), "STL fills the edges");
    Check(!stl.Configure({{"signal_col", "signal"}, {"robust", "maybe"}}, error) &&
              error.find("'robust' must be 'true' or 'false'") != std::string::npos,
          "a bad robust flag is named: " + error);
    Check(!stl.Configure({{"signal_col", "signal"}, {"period", "1"}}, error), "period 1 is refused");
}

void TestArima() {
    cyxwiz::ARIMAOperator op;
    std::string error;
    Check(op.Configure({
        {"signal_col", "signal"},
        {"p", "1"},
        {"d", "0"},
        {"q", "0"},
        {"horizon", "0"},
    }, error), error);

    auto input = MakeTimeSeriesTable();
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == input->num_rows(),
          "ARIMA with horizon 0 keeps the row count");
    CheckHasColumn(output, "fitted");
    CheckHasColumn(output, "residual");
    Check(output->GetColumnByName("forecast") == nullptr, "no forecast columns without a horizon");
    Check(!op.Configure({{"signal_col", "signal"}, {"p", "7"}}, error) && error.find("'p' must be between 0 and 5") != std::string::npos, "p 7 refused: " + error);
    Check(!op.Configure({{"signal_col", "signal"}, {"level", "1.5"}}, error) && error.find("'level' must be between") != std::string::npos, "level 1.5 refused: " + error);
}

void TestExponentialSmoothing() {
    cyxwiz::ExponentialSmoothingOperator op;
    std::string error;
    Check(op.Configure({
        {"signal_col", "signal"},
        {"method", "simple"},
        {"alpha", "0.3"},
        {"horizon", "0"},
    }, error), error);

    auto input = MakeTimeSeriesTable();
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == input->num_rows(),
          "exponential smoothing with horizon 0 keeps the row count");
    CheckHasColumn(output, "fitted");
    CheckHasColumn(output, "residual");
    Check(!op.Configure({{"signal_col", "signal"}, {"damped", "yes"}}, error) &&
              error.find("'damped' must be 'true' or 'false'") != std::string::npos,
          "a bad damped flag is named: " + error);
    Check(!op.Configure({{"signal_col", "signal"}, {"method", "ets"}}, error) && error.find("'method' must be") != std::string::npos, "method ets refused");
    Check(!op.Configure({{"signal_col", "signal"}, {"alpha", "1.5"}}, error) && error.find("'alpha' must be between 0 and 1") != std::string::npos, "alpha 1.5 refused");
}

// A horizon adds future rows: the input columns empty there (the time column
// continued), fitted / residual in-sample, forecast / low / high ahead.
void TestForecastRows() {
    cyxwiz::ExponentialSmoothingOperator op;
    std::string error;
    Check(op.Configure({{"signal_col", "signal"}, {"method", "holt_winters"}, {"period", "4"}, {"horizon", "3"}, {"level", "0.9"}, {"time_col", "t"}}, error), error);
    auto input = MakeTimeSeriesTable();
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == 35, "32 rows + 3 future rows: " + std::to_string(output->num_rows()));
    for (const char* c : {"fitted", "residual", "forecast", "low", "high"}) CheckHasColumn(output, c);
    Check(IsNullAt(output, "signal", 32) && IsNullAt(output, "signal", 34), "the signal is empty on the future rows");
    Check(!IsNullAt(output, "fitted", 31) && IsNullAt(output, "fitted", 32), "fitted stops with the data");
    Check(IsNullAt(output, "forecast", 31) && !IsNullAt(output, "forecast", 32), "the forecast starts after the data");
    const double f0 = DoubleAt(output, "forecast", 32), lo = DoubleAt(output, "low", 32), hi = DoubleAt(output, "high", 32);
    Check(lo <= f0 && f0 <= hi, "low <= forecast <= high (the saw fits exactly, so the range may be a point)");
    // The saw continues: 10 + 0.25 i + (i % 4) at i = 32 is 18.
    Check(std::fabs(f0 - 18.0) < 1.0, "Holt-Winters continues the saw: " + std::to_string(f0));
    Check(std::fabs(DoubleAt(output, "t", 32) - 32.0) < 1e-9 && std::fabs(DoubleAt(output, "t", 34) - 34.0) < 1e-9,
          "a numeric time column continues its step: " + std::to_string(DoubleAt(output, "t", 32)));
    auto schema = op.InferOutputSchema(input->schema()).ValueOrDie();
    Check(schema->GetFieldIndex("high") >= 0 && schema->field(0)->nullable(), "the schema has the forecast columns and nullable inputs");

    cyxwiz::ARIMAOperator arima;
    Check(arima.Configure({{"signal_col", "signal"}, {"p", "1"}, {"d", "1"}, {"q", "0"}, {"horizon", "2"}}, error), error);
    auto arima_result = arima.Apply(input);
    Check(arima_result.ok(), arima_result.status().ToString());
    Check(arima_result.ValueOrDie()->num_rows() == 34 && !IsNullAt(arima_result.ValueOrDie(), "forecast", 33), "ARIMA adds its future rows");
    Check(IsNullAt(arima_result.ValueOrDie(), "t", 33), "without time_col the time column stays empty ahead");
}

void TestDateColumnContinues() {
    cyxwiz::ExponentialSmoothingOperator op;
    std::string error;
    Check(op.Configure({{"signal_col", "signal"}, {"method", "holt"}, {"horizon", "2"}, {"time_col", "day"}}, error), error);
    auto result = op.Apply(MakeDailyTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output->num_rows() == 34, "daily table with two future rows");
    auto day = output->GetColumnByName("day");
    Check(day != nullptr && day->type()->id() == arrow::Type::DATE32, "the date column keeps its type");
    const auto& last_chunk = static_cast<const arrow::Date32Array&>(*day->chunk(day->num_chunks() - 1));
    Check(last_chunk.length() == 2 && last_chunk.Value(0) == 19723 + 32 && last_chunk.Value(1) == 19723 + 33,
          "the dates continue day by day: " + std::to_string(last_chunk.Value(0)));
}

void TestMissingValuesRefused() {
    arrow::DoubleBuilder values;
    for (int i = 0; i < 20; ++i) {
        if (i == 7) (void)values.AppendNull();
        else (void)values.Append(10.0 + i);
    }
    std::shared_ptr<arrow::Array> array;
    (void)values.Finish(&array);
    auto table = arrow::Table::Make(arrow::schema({arrow::field("signal", arrow::float64())}), {array});
    cyxwiz::ACFOperator op;
    std::string error;
    Check(op.Configure({{"signal_col", "signal"}, {"max_lag", "4"}}, error), error);
    auto result = op.Apply(table);
    Check(!result.ok() && result.status().ToString().find("1 missing value") != std::string::npos,
          "a missing value is refused with its count: " + result.status().ToString());
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
    CheckHasColumn(output, "ljung_box_q");
    CheckHasColumn(output, "ljung_box_p");
    Check(std::fabs(DoubleAt(output, "acf", 0) - 1.0) < 1e-12 && IsNullAt(output, "ljung_box_q", 0) && !IsNullAt(output, "ljung_box_q", 1), "acf(0) = 1, Ljung-Box from lag 1");
    Check(DoubleAt(output, "confidence_upper", 6) - DoubleAt(output, "acf", 6) > DoubleAt(output, "confidence_upper", 1) - DoubleAt(output, "acf", 1),
          "the Bartlett band (centred on the acf) widens with the lag");
    Check(!op.Configure({{"signal_col", "signal"}, {"confidence", "0.3"}}, error), "confidence 0.3 refused");
}

void TestPACF() {
    cyxwiz::PACFOperator op;
    std::string error;
    Check(op.Configure({
        {"signal_col", "signal"},
        {"max_lag", "6"},
        {"confidence", "0.99"},
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
    Check(std::fabs(DoubleAt(output, "confidence_upper", 3) - DoubleAt(output, "pacf", 3) - 2.5758 / std::sqrt(32.0)) < 1e-3, "the PACF band is pacf +- z / sqrt(n) at 99%");
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
    Check(output->num_rows() >= 2, "stationarity emits the series and its differences: " + std::to_string(output->num_rows()));
    for (const char* c : {"differencing", "adf_statistic", "adf_pvalue", "adf_lags", "kpss_statistic", "kpss_pvalue", "kpss_lags", "stationary",
                          "suggested_differencing", "analysis"})
        CheckHasColumn(output, c);
    const auto& d = static_cast<const arrow::Int32Array&>(*output->GetColumnByName("differencing")->chunk(0));
    Check(d.Value(0) == 0 && d.Value(1) == 1, "rows for d = 0, 1, ...");
    Check(!IsNullAt(output, "adf_lags", 0) && IsNullAt(output, "adf_lags", 1), "the lags are given for the series only");
    Check(op.Configure({{"signal_col", "signal"}, {"regression", "ct"}}, error), error);
    Check(op.Apply(MakeTimeSeriesTable()).ok(), "the trend regression runs");
    Check(!op.Configure({{"signal_col", "signal"}, {"regression", "ctt"}}, error), "regression ctt refused");
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
    for (const char* c : {"period", "share", "harmonic_of", "is_primary", "has_seasonality", "strength", "analysis"}) CheckHasColumn(output, c);
    const auto& period = static_cast<const arrow::Int32Array&>(*output->GetColumnByName("period")->chunk(0));
    const auto& primary = static_cast<const arrow::BooleanArray&>(*output->GetColumnByName("is_primary")->chunk(0));
    Check(period.Value(0) == 4 && primary.Value(0), "the saw's period 4 ranks first: " + std::to_string(period.Value(0)));
    Check(!IsNullAt(output, "strength", 0), "the primary row carries the strength");
}

// The routing test's shape: a small-int column (the CSV loader compacts
// integers) and a short lag count.
void TestSmallIntShortLags() {
    arrow::Int8Builder values;
    for (int i = 0; i < 32; ++i) (void)values.Append(static_cast<int8_t>(10 + i + (i % 4)));
    std::shared_ptr<arrow::Array> array;
    (void)values.Finish(&array);
    auto table = arrow::Table::Make(arrow::schema({arrow::field("signal", arrow::int8())}), {array});
    cyxwiz::ACFOperator op;
    std::string error;
    Check(op.Configure({{"signal_col", "signal"}, {"max_lag", "3"}}, error), error);
    auto result = op.Apply(table);
    Check(result.ok(), result.status().ToString());
    Check(result.ValueOrDie()->num_rows() == 4, "4 lag rows");
    cyxwiz::PACFOperator pacf;
    Check(pacf.Configure({{"signal_col", "signal"}, {"max_lag", "3"}}, error), error);
    Check(pacf.Apply(table).ok(), "PACF on the small-int column");
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
    TestForecastRows();
    TestDateColumnContinues();
    TestMissingValuesRefused();
    TestSmallIntShortLags();
    TestSeasonalNaiveLongFormForecast();
    std::cout << "Time-series analysis operators passed\n";
    return 0;
}
