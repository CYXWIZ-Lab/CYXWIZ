#include "cross_validation_runner.h"

#include "arrow_dataset.h"
#include "arrow_number.h"
#include "label_vocabulary.h"
#include "test_dataset_selection.h"
#include "test_executor.h"
#include "training_batcher_setup.h"
#include "training_executor.h"

#include <cyxwiz/model_evaluation.h>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <random>

namespace cyxwiz {

namespace {

constexpr const char* kTrainName = "cross_validation_train";
constexpr const char* kValidationName = "cross_validation_validation";

std::shared_ptr<arrow::Table> TakeRows(const std::shared_ptr<arrow::Table>& table, const std::vector<int64_t>& rows,
                                       std::string& error) {
    arrow::Int64Builder builder;
    if (!builder.AppendValues(rows).ok()) {
        error = "cannot list the fold's rows";
        return nullptr;
    }
    std::shared_ptr<arrow::Array> indices;
    if (!builder.Finish(&indices).ok()) {
        error = "cannot list the fold's rows";
        return nullptr;
    }
    auto taken = arrow::compute::Take(arrow::Datum(table), arrow::Datum(indices));
    if (!taken.ok()) {
        error = "cannot take the fold's rows: " + taken.status().ToString();
        return nullptr;
    }
    return taken->table();
}

CrossValidationScores ScoresFrom(const TestingMetrics& m) {
    CrossValidationScores s;
    s.accuracy = m.test_accuracy;
    s.macro_f1 = m.macro_f1;
    s.loss = m.test_loss;
    s.mae = m.test_mae;
    s.rmse = m.test_rmse;
    s.r2 = RegressionRSquared(m);
    return s;
}

// The graph's test rows: what the Engine's own test batcher reads from the
// table under the graph's split (none when the graph has no test share or a
// separate Test input).
bool GraphTestRows(const TrainingConfiguration& config, const std::shared_ptr<ArrowDataset>& dataset,
                   const std::string& label_column, int batch_size, std::vector<char>& is_test, std::string& error) {
    if (config.dataset_roles.test.IsSupplied() || !config.has_data_split || config.test_ratio <= 0.0f) return true;
    TrainingConfiguration split = config;
    split.prefetch_factor = 0;
    TrainingBatcherSet batchers = BuildArrowTrainingBatchers(split, dataset, label_column, batch_size);
    if (!batchers.test || batchers.test->GetNumSamples() == 0) return true;
    batchers.test->Reset();
    while (!batchers.test->IsEpochComplete()) {
        const Batch batch = batchers.test->GetNextBatch();
        if (!batch.IsValid()) break;
        if (batch.row_ids.size() != batch.size) {
            error = "the test batcher does not say which rows it reads, so the graph's test rows cannot be left out";
            return false;
        }
        for (const int64_t row : batch.row_ids)
            if (row >= 0 && static_cast<size_t>(row) < is_test.size()) is_test[static_cast<size_t>(row)] = 1;
    }
    return true;
}

}  // namespace

CrossValidationOutcome RunCrossValidation(const TrainingConfiguration& config, const std::shared_ptr<ArrowDataset>& dataset,
                                         const std::string& label_column, int epochs, int batch_size,
                                         const CrossValidationSettings& settings, const CrossValidationCallbacks& callbacks) {
    const auto start = std::chrono::steady_clock::now();
    CrossValidationOutcome result;
    result.target = label_column;
    result.folds = settings.folds;
    const auto fail = [&result](std::string why) {
        result.error = std::move(why);
        result.ok = false;
        return result;
    };
    if (!dataset || !dataset->GetArrowTable()) return fail("the prepared table is not in memory");
    if (config.sequence_batch.enabled)
        return fail("cross-validation trains tables of rows; token and sequence graphs are not supported yet");
    if (settings.folds < 2 || settings.folds > 20) return fail("folds must be 2 to 20");
    for (const double share : settings.train_shares)
        if (!(share > 0.0 && share <= 1.0)) return fail("each share of the training rows must be above 0 and at most 1");
    epochs = std::max(1, epochs);
    batch_size = std::max(1, batch_size);
    result.regression = UsesContinuousTargetMetrics(config);

    std::shared_ptr<arrow::Table> table = dataset->GetArrowTable();
    if (!table->GetColumnByName(label_column)) return fail("the target column '" + label_column + "' is not in the prepared table");
    // Text labels: numbered once over the whole table (as training numbers them).
    if (const auto text = TextLabelColumn(table, label_column)) {
        const LabelVocabulary vocabulary = BuildLabelVocabulary({text});
        auto encoded = EncodeLabelColumn(table, label_column, vocabulary);
        if (!encoded.ok()) return fail("cannot number the classes of '" + label_column + "': " + encoded.status().ToString());
        table = *encoded;
        result.class_names = vocabulary.names;
    }
    const auto all = std::make_shared<ArrowDataset>(table, dataset->GetName());
    const size_t n = static_cast<size_t>(table->num_rows());

    // The rows used: the table less the graph's test rows.
    std::vector<char> is_test(n, 0);
    std::string error;
    if (!GraphTestRows(config, all, label_column, batch_size, is_test, error)) return fail(error);
    std::vector<int64_t> pool;
    for (size_t i = 0; i < n; ++i)
        if (!is_test[i]) pool.push_back(static_cast<int64_t>(i));
    result.rows = static_cast<int64_t>(pool.size());
    result.test_rows_left_out = static_cast<int64_t>(n - pool.size());
    if (pool.size() < static_cast<size_t>(settings.folds) * 2)
        return fail(std::to_string(pool.size()) + " rows are too few for " + std::to_string(settings.folds) + " folds");

    // Folds over the pool: given, stratified by class, or shuffled.
    std::vector<std::vector<int64_t>> fold_rows(static_cast<size_t>(settings.folds));
    if (!settings.fold_of_row.empty()) {
        if (settings.fold_of_row.size() != pool.size())
            return fail("the given folds name " + std::to_string(settings.fold_of_row.size()) + " rows; " +
                        std::to_string(pool.size()) + " rows are used");
        for (size_t i = 0; i < pool.size(); ++i) {
            const int f = settings.fold_of_row[i];
            if (f < 1 || f > settings.folds) return fail("a given fold is not 1 to " + std::to_string(settings.folds));
            fold_rows[static_cast<size_t>(f - 1)].push_back(pool[i]);
        }
    } else {
        std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
        if (!result.regression && settings.stratified) {
            const std::vector<double> values = ArrowNumbers(table->GetColumnByName(label_column));
            std::vector<int> labels;
            labels.reserve(pool.size());
            for (const int64_t row : pool) {
                const double v = static_cast<size_t>(row) < values.size() ? values[static_cast<size_t>(row)] : NAN;
                labels.push_back(std::isfinite(v) ? static_cast<int>(v) : -1);
            }
            splits = ModelEvaluation::StratifiedKFoldSplit(labels, settings.folds, true, settings.seed);
        } else {
            splits = ModelEvaluation::KFoldSplit(static_cast<int>(pool.size()), settings.folds, true, settings.seed);
        }
        if (splits.size() != fold_rows.size()) return fail("the folds could not be drawn");
        for (size_t k = 0; k < splits.size(); ++k) {
            for (const int i : splits[k].second) fold_rows[k].push_back(pool[static_cast<size_t>(i)]);
            std::sort(fold_rows[k].begin(), fold_rows[k].end());
        }
    }
    for (size_t k = 0; k < fold_rows.size(); ++k)
        if (fold_rows[k].empty()) return fail("fold " + std::to_string(k + 1) + " has no rows");

    const std::vector<double> shares = settings.train_shares.empty() ? std::vector<double>{1.0} : settings.train_shares;
    const bool curve = !settings.train_shares.empty();
    const int runs = static_cast<int>(shares.size()) * settings.folds;

    // Checkpoints of the folds go to a scratch folder that is removed after.
    static std::atomic<uint64_t> next{0};
    const std::filesystem::path scratch = std::filesystem::temp_directory_path() / "cyxwiz_cross_validation" / std::to_string(++next);
    struct RemoveAfter {
        std::filesystem::path path;
        ~RemoveAfter() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    } remove_after{scratch};

    int run_number = 0;
    for (const double share : shares) {
        for (int k = 0; k < settings.folds; ++k) {
            ++run_number;
            const auto run_start = std::chrono::steady_clock::now();
            const auto& validation = fold_rows[static_cast<size_t>(k)];
            std::vector<int64_t> training;
            training.reserve(pool.size());
            for (int j = 0; j < settings.folds; ++j)
                if (j != k) training.insert(training.end(), fold_rows[static_cast<size_t>(j)].begin(), fold_rows[static_cast<size_t>(j)].end());
            if (share < 1.0) {
                std::mt19937 rng(settings.seed + static_cast<uint32_t>(k) * 1009u + static_cast<uint32_t>(std::lround(share * 1000.0)));
                std::shuffle(training.begin(), training.end(), rng);
                training.resize(std::max<size_t>(1, static_cast<size_t>(std::lround(share * static_cast<double>(training.size())))));
            }
            std::sort(training.begin(), training.end());
            const auto train_table = TakeRows(table, training, error);
            const auto val_table = train_table ? TakeRows(table, validation, error) : nullptr;
            if (!train_table || !val_table) return fail(error);
            const auto train_data = std::make_shared<ArrowDataset>(train_table, kTrainName);
            const auto val_data = std::make_shared<ArrowDataset>(val_table, kValidationName);

            // The graph's configuration with this fold's rows as its training and
            // validation tables; trained for its epochs, scored at the last one
            // (early stopping and the best-epoch restore would pick the epoch by
            // the fold being scored).
            TrainingConfiguration fold = config;
            fold.dataset_roles.train.dataset_name = kTrainName;
            fold.dataset_roles.train.label_column = label_column;
            fold.dataset_roles.dev.dataset_name = kValidationName;
            fold.dataset_roles.dev.label_column = label_column;
            fold.dataset_roles.test = {};
            fold.dataset_roles.policy.train_ratio = 1.0f;
            fold.dataset_roles.policy.dev_ratio = 0.0f;
            fold.dataset_roles.policy.test_ratio = 0.0f;
            fold.early_stopping_patience = 0;
            fold.save_best_checkpoint = false;
            fold.checkpoint_dir = scratch.string();
            fold.epochs = epochs;
            ResolvedTabularDatasets data;
            data.train_arrow = train_data;
            data.dev_arrow = val_data;
            auto built = BuildResolvedTabularTrainingBatchers(fold, data, batch_size);
            if (!built.ok()) return fail("fold " + std::to_string(k + 1) + ": " + built.error_message);

            TrainingExecutor executor(fold, TakeResolvedExternalBatchers(std::move(built.batchers)));
            bool cancelled = false;
            CrossValidationProgress progress{run_number, runs, k + 1, settings.folds, share, 0, epochs, result.regression};
            if (callbacks.on_progress) callbacks.on_progress(progress);
            const BatchCallback on_batch = [&](int, int, int, float, float) {
                if (!cancelled && callbacks.should_cancel && callbacks.should_cancel()) {
                    cancelled = true;
                    executor.Stop();
                }
            };
            const EpochCallback on_epoch = [&](int epoch, float, float, float, float, float) {
                progress.epoch = epoch + 1;
                if (callbacks.on_progress) callbacks.on_progress(progress);
            };
            try {
                executor.Train(epochs, batch_size, on_batch, on_epoch);
            } catch (const std::exception& e) {
                return fail("fold " + std::to_string(k + 1) + ": training failed: " + e.what());
            }
            if (cancelled) {
                result.cancelled = true;
                return fail("cancelled");
            }
            const TrainingMetrics trained = executor.GetMetrics();
            if (trained.terminal_status == "failed")
                return fail("fold " + std::to_string(k + 1) + ": " +
                            (trained.terminal_reason.empty() ? std::string("training failed") : trained.terminal_reason));
            std::shared_ptr<IExecutableModel> model = executor.ReleaseModel();
            if (!model) return fail("fold " + std::to_string(k + 1) + ": training gave no model");

            const auto score = [&](const std::shared_ptr<ArrowDataset>& rows, CrossValidationScores& out) {
                TestExecutor tester(fold, rows, label_column, TestDatasetScope::EntireProvidedDataset);
                tester.SetModel(model);
                tester.Test(batch_size);
                const TestingMetrics m = tester.GetMetrics();
                if (!m.is_complete) {
                    error = "fold " + std::to_string(k + 1) + ": scoring failed: " + m.status_message;
                    return false;
                }
                out = ScoresFrom(m);
                return true;
            };
            CrossValidationRun run;
            run.fold = k + 1;
            run.share = share;
            run.train_rows = static_cast<int64_t>(training.size());
            run.val_rows = static_cast<int64_t>(validation.size());
            run.epochs = trained.current_epoch > 0 ? trained.current_epoch : epochs;
            if (!score(val_data, run.val) || (curve && !score(train_data, run.train))) return fail(error);
            run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
            spdlog::info("Cross-validation fold {} of {} (share {:.2f}): {} training rows, {} validation rows, {} {:.4f}",
                         run.fold, settings.folds, share, run.train_rows, run.val_rows, result.regression ? "MAE" : "accuracy",
                         result.regression ? run.val.mae : run.val.accuracy);
            if (callbacks.on_run) callbacks.on_run(run);
            result.runs.push_back(run);
        }
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.ok = true;
    return result;
}

std::shared_ptr<arrow::Table> CrossValidationTable(const CrossValidationOutcome& o) {
    const bool curve = std::any_of(o.runs.begin(), o.runs.end(), [](const CrossValidationRun& r) { return r.share < 1.0 - 1e-9; });
    arrow::Int64Builder fold, train_rows, val_rows, epochs;
    arrow::DoubleBuilder share, seconds;
    arrow::DoubleBuilder val[3], train[3];
    const auto three = [&o](const CrossValidationScores& s) {
        return o.regression ? std::array<double, 3>{s.mae, s.rmse, s.r2} : std::array<double, 3>{s.accuracy, s.macro_f1, s.loss};
    };
    for (const auto& r : o.runs) {
        if (!fold.Append(r.fold).ok() || !share.Append(r.share).ok() || !train_rows.Append(r.train_rows).ok() ||
            !val_rows.Append(r.val_rows).ok() || !epochs.Append(r.epochs).ok() || !seconds.Append(r.seconds).ok())
            return nullptr;
        const auto v = three(r.val), t = three(r.train);
        for (size_t i = 0; i < 3; ++i)
            if (!val[i].Append(v[i]).ok() || (curve && !train[i].Append(t[i]).ok())) return nullptr;
    }
    const char* names[3] = {o.regression ? "mae" : "accuracy", o.regression ? "rmse" : "macro_f1", o.regression ? "r2" : "loss"};
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    const auto add = [&](const std::string& name, const std::shared_ptr<arrow::DataType>& type, arrow::ArrayBuilder& builder) {
        std::shared_ptr<arrow::Array> array;
        if (!builder.Finish(&array).ok()) return false;
        fields.push_back(arrow::field(name, type));
        arrays.push_back(array);
        return true;
    };
    bool ok = add("fold", arrow::int64(), fold) && add("share", arrow::float64(), share) && add("train_rows", arrow::int64(), train_rows) &&
              add("val_rows", arrow::int64(), val_rows);
    for (size_t i = 0; i < 3 && ok; ++i) ok = add(std::string("val_") + names[i], arrow::float64(), val[i]);
    for (size_t i = 0; i < 3 && ok && curve; ++i) ok = add(std::string("train_") + names[i], arrow::float64(), train[i]);
    ok = ok && add("epochs", arrow::int64(), epochs) && add("seconds", arrow::float64(), seconds);
    return ok ? arrow::Table::Make(arrow::schema(fields), arrays) : nullptr;
}

ScoreSummary SummariseRuns(const std::vector<CrossValidationRun>& runs, double share,
                           double (*pick)(const CrossValidationRun&)) {
    ScoreSummary s;
    std::vector<double> values;
    for (const auto& run : runs)
        if (std::abs(run.share - share) < 1e-9) values.push_back(pick(run));
    s.count = static_cast<int>(values.size());
    if (values.empty()) return s;
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double squares = 0.0;
    for (const double v : values) squares += (v - s.mean) * (v - s.mean);
    s.sd = values.size() > 1 ? std::sqrt(squares / static_cast<double>(values.size() - 1)) : 0.0;
    s.low = *std::min_element(values.begin(), values.end());
    s.high = *std::max_element(values.begin(), values.end());
    return s;
}

}  // namespace cyxwiz
