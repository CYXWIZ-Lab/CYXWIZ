#include "../src/core/node_executors/clustering_operators.h"
#include "../src/core/node_executors/count_vectorizer_operator.h"
#include "../src/core/node_executors/pca_operator.h"
#include "../src/core/node_executors/time_series_features_operator.h"
#include "../src/core/node_executors/time_series_split_operator.h"
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

std::shared_ptr<arrow::Table> MakeLongTimeSeriesTable() {
    std::vector<float> values;
    values.reserve(15);
    for (int i = 0; i < 15; ++i) {
        values.push_back(static_cast<float>(i));
    }
    auto schema = arrow::schema({
        arrow::field("value", arrow::float32()),
    });
    return arrow::Table::Make(schema, {FinishFloatArray(values)}, 15);
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

void TestKMeansEmitsMemoryPreflight() {
    cyxwiz::KMeansOperator op;
    const auto input = MakeTimeSeriesTable();
    std::string error;

    Check(op.Configure({
        {"feature_cols", "value,extra"},
        {"n_clusters", "2"},
        {"max_iter", "10"},
        {"n_init", "1"},
        {"seed", "7"},
    }, error), error);
    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    Check(!progress_events.empty(),
          "KMeans should emit materialization progress events");
    Check(progress_events.front().stage == "KMeansCluster memory preflight",
          "KMeans first progress event should be memory preflight");
    Check(progress_events.front().status == "running",
          "safe KMeans preflight should stay in running status");
    Check(progress_events.front().memory_risk_level == "safe",
          "safe KMeans preflight should report safe risk");
    Check(progress_events.front().estimated_memory_bytes >
              6ULL * 3ULL * static_cast<uint64_t>(sizeof(double)),
          "KMeans preflight should include peak allocation overhead");
    Check(progress_events.front().total_items == 6ULL * 3ULL,
          "KMeans preflight should report planned matrix and output cells");
    Check(progress_events.front().message.find("Suggestion:") !=
              std::string::npos,
          "KMeans preflight message should include mitigation guidance");
    auto table = result.ValueOrDie();
    Check(table->num_rows() == 6, "KMeans should preserve input row count");
    Check(table->GetColumnByName("cluster_id") != nullptr,
          "KMeans should append cluster_id");
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
    Check(second_table->num_columns() == 5,
          "second time-series window configure should emit x_0, x_1, y and hidden target bounds");
}

void TestTimeSeriesWindowEmitsOrderedMultiStepTargets() {
    cyxwiz::TimeSeriesWindowOperator op;
    std::string error;
    Check(op.Configure({
        {"value_col", "value"},
        {"input_width", "2"},
        {"label_width", "3"},
        {"shift", "1"},
    }, error), error);

    auto result = op.Apply(MakeTimeSeriesTable());
    Check(result.ok(), result.status().ToString());
    auto table = result.ValueOrDie();
    Check(table->num_rows() == 2,
          "six values with lookback 2 and horizon 3 should produce two windows");
    Check(table->num_columns() == 7,
          "multi-step window should emit features, targets, and hidden target bounds");

    const std::vector<std::string> target_names = {"y", "y_1", "y_2"};
    const std::vector<std::vector<float>> expected = {
        {3.0f, 4.0f},
        {4.0f, 5.0f},
        {5.0f, 6.0f},
    };
    for (size_t target = 0; target < target_names.size(); ++target) {
        auto column = table->GetColumnByName(target_names[target]);
        Check(column && column->num_chunks() == 1,
              "ordered target column should exist: " + target_names[target]);
        auto values = std::static_pointer_cast<arrow::FloatArray>(
            column->chunk(0));
        for (int64_t row = 0; row < table->num_rows(); ++row) {
            Check(values->Value(row) == expected[target][static_cast<size_t>(row)],
                  "multi-step target values must preserve forecast order");
        }
    }

    auto starts = std::static_pointer_cast<arrow::Int64Array>(
        table->GetColumnByName("__target_start_index")->chunk(0));
    auto ends = std::static_pointer_cast<arrow::Int64Array>(
        table->GetColumnByName("__target_end_index")->chunk(0));
    Check(starts->Value(0) == 2 && ends->Value(0) == 4,
          "first multi-step window should preserve its exact source target bounds");
    Check(starts->Value(1) == 3 && ends->Value(1) == 5,
          "second multi-step window should preserve its exact source target bounds");
}

void TestTimeSeriesSplitPurgesCrossBoundaryTargets() {
    cyxwiz::TimeSeriesWindowOperator window;
    std::string error;
    Check(window.Configure({
        {"value_col", "value"},
        {"input_width", "4"},
        {"label_width", "3"},
        {"shift", "1"},
    }, error), error);
    auto windowed = window.Apply(MakeLongTimeSeriesTable());
    Check(windowed.ok(), windowed.status().ToString());

    cyxwiz::TimeSeriesSplitOperator split;
    Check(split.Configure({
        {"train_ratio", "0.5"},
        {"val_ratio", "0.25"},
        {"test_ratio", "0.25"},
        {"boundary_policy", "targets_within_partition"},
        {"train_end_source_row", "7"},
        {"val_end_source_row", "11"},
    }, error), error);
    auto result = split.Apply(windowed.ValueOrDie());
    Check(result.ok(), result.status().ToString());

    auto partitions = std::static_pointer_cast<arrow::Int8Array>(
        result.ValueOrDie()->GetColumnByName("__partition__")->chunk(0));
    const std::vector<int8_t> expected = {0, -1, -1, 1, 1, -1, -1, 2, 2};
    Check(partitions->length() == static_cast<int64_t>(expected.size()),
          "target-contained split should preserve every window row");
    for (int64_t row = 0; row < partitions->length(); ++row) {
        Check(partitions->Value(row) == expected[static_cast<size_t>(row)],
              "target-contained split should purge only boundary-crossing targets");
    }
}

} // namespace

int main() {
    TestCountVectorizerResetsOptionalLabelAndMaxFeatures();
    TestCountVectorizerPrefersNGramRangeWhenDefaultsArePresent();
    TestCountVectorizerBinaryModeEmitsPresenceValues();
    TestCountVectorizerRejectsSparseOutputFormat();
    TestKMeansEmitsMemoryPreflight();
    TestPcaEmitsMemoryPreflight();
    TestTimeSeriesFeaturesClearsStaleFeatureLists();
    TestTimeSeriesWindowClearsOptionalFeatureAndTimeColumns();
    TestTimeSeriesWindowEmitsOrderedMultiStepTargets();
    TestTimeSeriesSplitPurgesCrossBoundaryTargets();
    std::cout << "Operator Configure reset regressions passed\n";
    return 0;
}
