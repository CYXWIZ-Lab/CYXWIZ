#pragma once

// Data Studio > Results (TOFIX148 job 7, approved mockup v8): a model's
// results next to the data. Two sources the Engine already writes:
//
// - Language models: the generation previews training writes per epoch
//   (checkpoints/<graph>/<run>/generation_previews/epoch_<n>.json). Each
//   sample is shown as its question, the evidence reference, what the model
//   cited, whether book / chapter / verse match, how it stopped and whether
//   it loops; slices count them by stop reason.
// - Classifiers: the latest test run's per-row predictions
//   (kTestPredictionsDataset): accuracy, confusion per class, the classes'
//   precision and recall, and the worst rows (wrong with the highest
//   confidence).
//
// Pure data in and out, besides reading the preview files.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
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
    int64_t row = 0;       // 1-based position in the test set
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
// score_<name> columns naming the classes).
ClassifierResults ClassifierFromPredictions(const std::shared_ptr<arrow::Table>& table, size_t worst_rows = 20);

}  // namespace cyxwiz::studio
