#pragma once

// Data Studio > Results (TOFIX148 job 7, approved mockup v8): a model's
// results next to the data. Two sources the Engine already writes:
//
// - Language models: the generation previews training writes per epoch
//   (checkpoints/<graph>/<run>/generation_previews/epoch_<n>.json). Each
//   sample is shown as its question, the evidence reference, what the model
//   cited, whether book / chapter / verse match, how it stopped and whether
//   it loops; slices count them by stop reason.
// - Test runs: the latest test's per-row predictions (kTestPredictionsDataset).
//   Classifiers: accuracy, confusion per class, the classes' precision and
//   recall, and the worst rows (wrong with the highest confidence).
//   Regression: MAE, RMSE, bias and the largest errors. Both (TOFIX157 B2):
//   the error by the values of a column of the tested table, joined by the
//   rows' row_id.
//
// Pure data in and out, besides reading the preview files.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class ChunkedArray;
class Table;
}

namespace cyxwiz::studio {

// ---- Language-model previews ------------------------------------------------

struct PreviewRun {
    std::string run_id;   // "train-1791984271862"
    std::string graph;    // the checkpoint folder it belongs to ("sft_v2_768_d256_L6")
    std::string folder;   // .../generation_previews
    std::vector<int> epochs;  // ascending
    std::filesystem::file_time_type time{};  // the newest epoch file
};
// Every run under `checkpoints_root` that wrote previews, newest first.
std::vector<PreviewRun> FindPreviewRuns(const std::string& checkpoints_root);

struct PreviewSample {
    std::string question;      // the prompt's <question>, else its last "User:" text, else the prompt
    std::string evidence_ref;  // the prompt's first <reference> ("Psalm 34:13"), "" when none
    std::string generated;
    std::string cited_ref;     // the first "Book C:V" reference in the answer, "" when none
    std::string stop;          // "eos", "max_tokens", "cancelled", ...
    bool has_ref = false;      // the evidence has a Book C:V reference to match
    bool book = false, chapter = false, verse = false;
    bool loops = false;        // a three-word phrase repeats three times or more
};
// One sample from its prompt, answer and stop reason.
PreviewSample PreviewSampleFrom(const std::string& prompt, const std::string& generated, const std::string& stop);

struct PreviewEpoch {
    std::string run_id;
    int epoch = 0;
    std::string sampling;  // "greedy"
    int max_new_tokens = 0;
    bool cancelled = false;
    std::vector<PreviewSample> samples;
};
bool LoadPreviewEpoch(const std::filesystem::path& path, PreviewEpoch& epoch, std::string& error);

struct PreviewSummary {
    size_t samples = 0, with_refs = 0;
    size_t right_book = 0, right_chapter = 0, right_verse = 0;
    size_t stopped_itself = 0, hit_limit = 0, loops = 0;
};
PreviewSummary SummarizePreview(const PreviewEpoch& epoch);

struct PreviewSlice {
    std::string name;      // "stop = end", "stop = token limit", "loops"
    size_t rows = 0;
    size_t right_verse = 0;
};
std::vector<PreviewSlice> PreviewSlices(const PreviewEpoch& epoch);

// "end", "token limit", "cancelled", or the reason as written.
std::string StopWords(const std::string& stop);

// ---- Classifier test predictions ------------------------------------------

struct ClassRow {
    std::string name;
    int64_t support = 0;   // rows whose truth is this class
    double precision = 0.0, recall = 0.0;
};
struct WorstRow {
    int64_t row = 0;           // 1-based position in the test set
    int64_t dataset_row = -1;  // 0-based row of the tested table (row_id); -1 unknown
    std::string truth, predicted;
    double confidence = 0.0;
};
struct ClassifierResults {
    bool ok = false;
    std::string error;
    int64_t rows = 0;
    double accuracy = 0.0;
    std::vector<std::string> classes;
    std::vector<std::vector<int64_t>> confusion;  // [truth][predicted]
    std::vector<ClassRow> per_class;
    std::vector<WorstRow> worst;                  // wrong rows, highest confidence first, at most `worst_rows`
};
// From the test predictions table (true_class, predicted_class, confidence,
// score_<name> columns naming the classes, row_id when known).
ClassifierResults ClassifierFromPredictions(const std::shared_ptr<arrow::Table>& table, size_t worst_rows = 20);

// ---- Regression test predictions --------------------------------------------

struct RegressionRow {
    int64_t row = 0;           // 1-based position in the test set
    int64_t dataset_row = -1;  // as WorstRow
    double truth = 0.0, predicted = 0.0, error = 0.0;  // the first target; error = predicted - truth
};
struct RegressionResults {
    bool ok = false;
    std::string error;
    int64_t rows = 0;
    int targets = 0;
    double mae = 0.0, rmse = 0.0, bias = 0.0;  // over every target value; bias = mean error
    std::vector<RegressionRow> worst;          // largest |error| first, at most `worst_rows`
};
// From a regression test's table (true_value / predicted_value / error, or
// true_<i> / predicted_<i> / error_<i> per target).
RegressionResults RegressionFromPredictions(const std::shared_ptr<arrow::Table>& table, size_t worst_rows = 20);

// True when the table is a regression test's (it has an error column).
bool IsRegressionPredictions(const std::shared_ptr<arrow::Table>& table);

// ---- Errors by column value --------------------------------------------------
//
// Joins a test run's rows back to the tested table (row_id) and slices the
// error by the values of one of its columns.

// Per test row: its row in the tested table and its error (a classifier:
// 1 wrong, 0 right; regression: |error| averaged over the targets).
struct TestRowErrors {
    bool ok = false;
    std::string error;
    bool classifier = true;
    std::string source_dataset;     // the tested table's name (schema metadata)
    std::vector<int64_t> row_ids;
    std::vector<double> errors;
};
TestRowErrors RowErrorsFromPredictions(const std::shared_ptr<arrow::Table>& table);

// Reads one column of the tested table at the test rows, a piece at a time in
// table order (the whole column in memory, or one row group of a file).
struct GatheredColumn {
    bool numeric = false;
    std::vector<double> numbers;      // numeric: per test row
    std::vector<std::string> labels;  // otherwise: per test row
    std::vector<uint8_t> valid;       // 0 missing
};
class ColumnGatherer {
public:
    explicit ColumnGatherer(const std::vector<int64_t>& row_ids);
    // The next piece of the column; false (with `error`) on a type that cannot
    // be sliced or a row id past the table's end at Finish.
    bool Add(const std::shared_ptr<arrow::ChunkedArray>& piece, std::string& error);
    bool Finish(GatheredColumn& out, std::string& error);

private:
    std::vector<int64_t> row_ids_;
    std::vector<size_t> order_;  // test positions sorted by row id
    size_t cursor_ = 0;          // into order_
    int64_t offset_ = 0;         // rows seen
    GatheredColumn column_;
    bool typed_ = false;
};

struct ValueSlice {
    std::string value;  // the value, a range "a to b" for numbers, "(missing)", or "(other values)"
    int64_t rows = 0;
    double error = 0.0;  // a classifier: the share wrong; regression: mean |error|
};
struct ErrorSlices {
    bool binned = false;        // numbers grouped in equal-count ranges
    double overall = 0.0;       // the same measure over every test row
    std::vector<ValueSlice> slices;  // highest error first
};
// At most `max_values` distinct values (the most common; the rest as
// "(other values)"); a numeric column with more distinct values than that is
// cut into `bins` ranges of about equal rows.
ErrorSlices SliceErrors(const GatheredColumn& column, const std::vector<double>& errors, size_t max_values = 20,
                        size_t bins = 5);

}  // namespace cyxwiz::studio
