#include "../src/core/node_executors/count_vectorizer_operator.h"
#include "../src/core/node_executors/pca_operator.h"
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

std::shared_ptr<arrow::Table> MakeRepeatedTextTable() {
    auto schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {
        FinishStringArray({
            "alpha alpha beta",
        }),
    }, 1);
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

void TestCountVectorizerPrefersNGramRangeWhenDefaultsArePresent() {
    const auto input = MakeTextTable();
    std::string error;

    cyxwiz::CountVectorizerOperator count_op;
    Check(count_op.Configure({
        {"text_col", "text"},
        {"max_features", "20"},
        {"norm", "none"},
        {"stop_words", "none"},
        {"ngram_range", "1,2"},
        {"ngram_min", "1"},
        {"ngram_max", "1"},
    }, error), error);
    auto count_result = count_op.Apply(input);
    Check(count_result.ok(), count_result.status().ToString());
    Check(count_result.ValueOrDie()->num_columns() == 8,
          "CountVectorizer should honor ngram_range=1,2 over stale ngram_max=1");
}

void TestCountVectorizerBinaryModeEmitsPresenceValues() {
    cyxwiz::CountVectorizerOperator op;
    const auto input = MakeRepeatedTextTable();
    std::string error;

    Check(op.Configure({
        {"text_col", "text"},
        {"max_features", "3"},
        {"norm", "none"},
        {"stop_words", "none"},
        {"binary", "true"},
    }, error), error);
    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    Check(!progress_events.empty(),
          "CountVectorizer should emit materialization progress events");
    Check(progress_events.front().stage == "CountVectorizer memory preflight",
          "CountVectorizer first progress event should be memory preflight");
    Check(progress_events.front().status == "running",
          "safe CountVectorizer preflight should stay in running status");
    Check(progress_events.front().memory_risk_level == "safe",
          "safe CountVectorizer preflight should report safe risk");
    Check(progress_events.front().estimated_memory_bytes >
              1ULL * 3ULL * static_cast<uint64_t>(sizeof(float)),
          "CountVectorizer preflight should include peak allocation overhead");
    Check(progress_events.front().message.find("Suggestion:") !=
              std::string::npos,
          "CountVectorizer preflight message should include mitigation guidance");
    auto table = result.ValueOrDie();
    Check(table->num_columns() == 2,
          "binary CountVectorizer should emit alpha and beta features");

    auto count_0 = std::static_pointer_cast<arrow::FloatArray>(
        table->GetColumnByName("count_0")->chunk(0));
    auto count_1 = std::static_pointer_cast<arrow::FloatArray>(
        table->GetColumnByName("count_1")->chunk(0));
    Check(count_0->Value(0) == 1.0f,
          "binary CountVectorizer should set repeated alpha to presence 1");
    Check(count_1->Value(0) == 1.0f,
          "binary CountVectorizer should set beta to presence 1");
}

void TestCountVectorizerRejectsSparseOutputFormat() {
    cyxwiz::CountVectorizerOperator op;
    std::string error;
    Check(!op.Configure({
        {"text_col", "text"},
        {"output_format", "sparse"},
    }, error), "CountVectorizer output_format=sparse should fail closed");
    Check(error.find("supports dense output only") != std::string::npos,
          "CountVectorizer sparse output error should be specific: " + error);
}

void TestPcaEmitsMemoryPreflight() {
    cyxwiz::PCAOperator op;
    const auto input = MakeTimeSeriesTable();
    std::string error;

    Check(op.Configure({
        {"feature_cols", "value,extra,time"},
        {"n_components", "2"},
        {"center", "true"},
        {"scale", "false"},
    }, error), error);
    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    Check(!progress_events.empty(),
          "PCA should emit materialization progress events");
    Check(progress_events.front().stage == "Resolving features",
          "PCA should resolve features before preflight");
    Check(progress_events.size() > 1,
          "PCA should emit memory preflight after feature resolution");
    Check(progress_events[1].stage == "Features resolved",
          "PCA should report resolved features before preflight");
    Check(progress_events.size() > 2,
          "PCA should emit memory preflight before reading feature columns");
    Check(progress_events[2].stage == "PCA memory preflight",
          "PCA third progress event should be memory preflight");
    Check(progress_events[2].status == "running",
          "safe PCA preflight should stay in running status");
    Check(progress_events[2].memory_risk_level == "safe",
          "safe PCA preflight should report safe risk");
    Check(progress_events[2].estimated_memory_bytes >
              6ULL * 5ULL * static_cast<uint64_t>(sizeof(double)),
          "PCA preflight should include peak allocation overhead");
    Check(progress_events[2].message.find("Suggestion:") !=
              std::string::npos,
          "PCA preflight message should include mitigation guidance");
    auto table = result.ValueOrDie();
    Check(table->num_rows() == 6, "PCA should preserve sample count");
    Check(table->num_columns() == 2, "PCA should emit requested components");
}

void TestTimeSeriesFeaturesClearsStaleFeatureLists() {
    cyxwiz::TimeSeriesFeaturesOperator op;
    const auto input = MakeTimeSeriesTable();
    std::string error;

    Check(op.Configure({
        {"value_col", "value"},
        {"lag_values", "2"},
    }, error), error);
    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });
    auto first = op.Apply(input);
    Check(first.ok(), first.status().ToString());
    Check(!progress_events.empty(),
          "TimeSeriesFeatures should emit materialization progress events");
    Check(progress_events.front().stage == "TimeSeriesFeatures memory preflight",
          "TimeSeriesFeatures first progress event should be memory preflight");
    Check(progress_events.front().status == "running",
          "safe TimeSeriesFeatures preflight should stay in running status");
    Check(progress_events.front().memory_risk_level == "safe",
          "safe TimeSeriesFeatures preflight should report safe risk");
    Check(progress_events.front().estimated_memory_bytes >
              4ULL * static_cast<uint64_t>(sizeof(float)),
          "TimeSeriesFeatures preflight should include peak allocation overhead");
    Check(progress_events.front().message.find("Suggestion:") !=
              std::string::npos,
          "TimeSeriesFeatures preflight message should include mitigation guidance");
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
    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });
    auto first = op.Apply(input);
    Check(first.ok(), first.status().ToString());
    Check(!progress_events.empty(),
          "TimeSeriesWindow should emit materialization progress events");
    Check(progress_events.front().stage == "TimeSeriesWindow memory preflight",
          "TimeSeriesWindow first progress event should be memory preflight");
    Check(progress_events.front().status == "running",
          "safe TimeSeriesWindow preflight should stay in running status");
    Check(progress_events.front().memory_risk_level == "safe",
          "safe TimeSeriesWindow preflight should report safe risk");
    Check(progress_events.front().estimated_memory_bytes >
              4ULL * 6ULL * static_cast<uint64_t>(sizeof(float)),
          "TimeSeriesWindow preflight should include peak allocation overhead");
    Check(progress_events.front().message.find("Suggestion:") !=
              std::string::npos,
          "TimeSeriesWindow preflight message should include mitigation guidance");
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
    TestCountVectorizerPrefersNGramRangeWhenDefaultsArePresent();
    TestCountVectorizerBinaryModeEmitsPresenceValues();
    TestCountVectorizerRejectsSparseOutputFormat();
    TestPcaEmitsMemoryPreflight();
    TestTimeSeriesFeaturesClearsStaleFeatureLists();
    TestTimeSeriesWindowClearsOptionalFeatureAndTimeColumns();
    std::cout << "Operator Configure reset regressions passed\n";
    return 0;
}
