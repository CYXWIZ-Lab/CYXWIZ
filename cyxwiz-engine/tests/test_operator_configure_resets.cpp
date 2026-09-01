#include "../src/core/node_executors/clustering_operators.h"
#include "../src/core/node_executors/count_vectorizer_operator.h"
#include "../src/core/node_executors/pca_operator.h"
#include "../src/core/node_executors/text_vectorizer_contract.h"
#include "../src/core/node_executors/time_series_features_operator.h"
#include "../src/core/node_executors/time_series_segment_operator.h"
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

std::shared_ptr<arrow::Array> FinishInt64Array(
    const std::vector<int64_t>& values) {
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

std::shared_ptr<arrow::Table> MakeSegmentedTimeSeriesTable() {
    auto schema = arrow::schema({
        arrow::field("value", arrow::float32()),
        arrow::field("__segment_id", arrow::int64()),
    });
    return arrow::Table::Make(
        schema,
        {FinishFloatArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
         FinishInt64Array({0, 0, 0, 1, 1, 1})},
        6);
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

void TestTextVectorizerNGramContractRejectsMalformedRanges() {
    const std::vector<std::string> invalid_ranges = {
        "0,1", "1,0", "-1,2", "2,1", "1,4", "1,2junk", "1,2,3", "1"
    };
    for (const auto& range : invalid_ranges) {
        int ngram_min = 0;
        int ngram_max = 0;
        std::string error;
        Check(!cyxwiz::text_vectorizer_contract::ParseNGramRange(
                  {{"ngram_range", range}}, "TFIDFVectorizer",
                  ngram_min, ngram_max, error),
              "strict ngram_range parser should reject '" + range + "'");
        Check(!error.empty(),
              "invalid ngram_range should provide a specific error");
    }

    int ngram_min = 0;
    int ngram_max = 0;
    std::string error;
    Check(!cyxwiz::text_vectorizer_contract::ParseNGramRange(
              {{"ngram_min", "1junk"}, {"ngram_max", "2"}},
              "CountVectorizer", ngram_min, ngram_max, error),
          "legacy ngram aliases should reject trailing characters");

    error.clear();
    Check(cyxwiz::text_vectorizer_contract::ParseNGramRange(
              {{"ngram_range", " 1 ; 3 "},
               {"ngram_min", "9"}, {"ngram_max", "9"}},
              "CountVectorizer", ngram_min, ngram_max, error),
          "canonical ngram_range should accept the serialized semicolon alias: " +
              error);
    Check(ngram_min == 1 && ngram_max == 3,
          "canonical ngram_range should win over legacy aliases");

    const auto features =
        cyxwiz::text_vectorizer_contract::BuildNGramFeatures(
            {"not", "very", "good"}, 1, 3);
    Check(features == std::vector<std::string>({
              "not", "very", "good", "not very", "very good",
              "not very good"}),
          "shared n-gram builder should preserve deterministic range ordering");
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

void TestCountVectorizerBlocksBeforeAllocationHeavyWork() {
    cyxwiz::CountVectorizerOperator op;
    std::string error;
    Check(op.Configure({
        {"text_col", "text"},
        {"max_features", "4"},
    }, error), error);

    cyxwiz::MaterializationMemoryContext memory_context;
    memory_context.policy.hard_limit_bytes = 1;
    memory_context.snapshot_override = cyxwiz::MaterializationMemorySnapshot{
        1024, 1024, true};
    op.SetMaterializationMemoryContext(memory_context);

    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });

    const auto result = op.Apply(MakeTextTable());
    Check(!result.ok(),
          "CountVectorizer should reject a request above the hard limit");
    Check(result.status().IsCapacityError(),
          "blocked CountVectorizer should return a capacity error");
    Check(progress_events.size() == 1,
          "blocked CountVectorizer should stop after its preflight event");
    Check(progress_events.front().stage ==
              "CountVectorizer memory preflight",
          "blocked CountVectorizer should identify the preflight stage");
    Check(progress_events.front().status == "blocked",
          "blocked CountVectorizer should expose blocked status");
    Check(progress_events.front().memory_risk_level == "blocked",
          "blocked CountVectorizer should expose blocked memory risk");
    Check(progress_events.front().message.find("configured hard limit") !=
              std::string::npos,
          "blocked CountVectorizer should explain the configured hard limit");
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

void TestTimeSeriesSegmentDetectsGapsAndRejectsDuplicates() {
    auto schema = arrow::schema({
        arrow::field("timestamp", arrow::utf8()),
        arrow::field("value", arrow::float32()),
    });
    const auto input = arrow::Table::Make(
        schema,
        {FinishStringArray({
             "2020-01-01 00:00:00",
             "2020-01-01 00:00:10",
             "2020-01-01 00:00:20",
             "2020-01-01 00:00:50",
             "2020-01-01 00:01:00",
         }),
         FinishFloatArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f})},
        5);

    cyxwiz::TimeSeriesSegmentOperator op;
    std::string error;
    Check(op.Configure({
        {"timestamp_col", "timestamp"},
        {"gap_threshold_seconds", "30"},
    }, error), error);
    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    Check(!progress_events.empty() &&
              progress_events.front().stage ==
                  "TimeSeriesSegment memory preflight",
          "TimeSeriesSegment should report memory preflight first");
    auto segments = std::static_pointer_cast<arrow::Int64Array>(
        result.ValueOrDie()->GetColumnByName("__segment_id")->chunk(0));
    const std::vector<int64_t> expected = {0, 0, 0, 1, 1};
    for (int64_t row = 0; row < segments->length(); ++row) {
        Check(segments->Value(row) == expected[static_cast<size_t>(row)],
              "30-second gap should begin a new continuous segment");
    }
    auto deltas = std::static_pointer_cast<arrow::DoubleArray>(
        result.ValueOrDie()
            ->GetColumnByName("__time_delta_seconds")
            ->chunk(0));
    Check(deltas->IsNull(0), "first time delta should be null");
    Check(deltas->Value(3) == 30.0,
          "gap metadata should preserve exact elapsed seconds");

    const auto duplicate = arrow::Table::Make(
        schema,
        {FinishStringArray({
             "2020-01-01 00:00:00",
             "2020-01-01 00:00:00",
         }),
         FinishFloatArray({1.0f, 2.0f})},
        2);
    auto duplicate_result = op.Apply(duplicate);
    Check(!duplicate_result.ok(),
          "TimeSeriesSegment must fail closed on duplicate timestamps");
}

void TestTimeSeriesWindowSkipsCrossSegmentWindows() {
    cyxwiz::TimeSeriesWindowOperator op;
    std::string error;
    Check(op.Configure({
        {"value_col", "value"},
        {"segment_col", "__segment_id"},
        {"input_width", "2"},
        {"label_width", "1"},
        {"shift", "1"},
    }, error), error);

    auto result = op.Apply(MakeSegmentedTimeSeriesTable());
    Check(result.ok(), result.status().ToString());
    auto table = result.ValueOrDie();
    Check(table->num_rows() == 2,
          "gap-safe windowing should omit the two boundary-crossing samples");
    auto targets = std::static_pointer_cast<arrow::FloatArray>(
        table->GetColumnByName("y")->chunk(0));
    Check(targets->Value(0) == 3.0f && targets->Value(1) == 6.0f,
          "gap-safe windows should retain only targets within one segment");
    auto segments = std::static_pointer_cast<arrow::Int64Array>(
        table->GetColumnByName("__segment_id")->chunk(0));
    Check(segments->Value(0) == 0 && segments->Value(1) == 1,
          "window output should preserve source segment identity");
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
    TestTextVectorizerNGramContractRejectsMalformedRanges();
    TestCountVectorizerBinaryModeEmitsPresenceValues();
    TestCountVectorizerRejectsSparseOutputFormat();
    TestCountVectorizerBlocksBeforeAllocationHeavyWork();
    TestKMeansEmitsMemoryPreflight();
    TestPcaEmitsMemoryPreflight();
    TestTimeSeriesFeaturesClearsStaleFeatureLists();
    TestTimeSeriesWindowClearsOptionalFeatureAndTimeColumns();
    TestTimeSeriesWindowEmitsOrderedMultiStepTargets();
    TestTimeSeriesSegmentDetectsGapsAndRejectsDuplicates();
    TestTimeSeriesWindowSkipsCrossSegmentWindows();
    TestTimeSeriesSplitPurgesCrossBoundaryTargets();
    std::cout << "Operator Configure reset regressions passed\n";
    return 0;
}
