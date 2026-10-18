#pragma once

#include "training_generation_preview_settings.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

// One presentation model for generation previews (TOFIX159 A1): the Training
// Dashboard's Generations view, Data Studio Results and the Console block all
// read the run's reports through it, so they agree on wording. Pure: no ImGui.

enum class GenerationPreviewCheckResult { NotSet, Passed, Failed };

struct GenerationPreviewSample {
    int prompt_index = 0;  // From 0.
    int sample_index = 0;  // From 0; above 0 only when sampling several per prompt.
    std::string prompt;            // As entered (the user message in chat mode).
    std::string rendered_prompt;   // As the model saw it.
    std::string text;              // Generated text, a matched stop text cut off.
    std::string stop_reason;       // max_tokens | eos | stop_text | user_cancel | error
    std::string stop_text;         // The stop text that matched.
    int prompt_tokens = 0;
    int new_tokens = 0;
    GenerationPreviewCheck check;
    GenerationPreviewCheckResult check_result = GenerationPreviewCheckResult::NotSet;
    std::string repeated_phrase;   // Empty when no phrase repeats.
};

// One generation pass: before training, after an epoch, or after a step.
struct GenerationPreviewReport {
    std::string run_id;
    std::string point;        // baseline | epoch | step | try (Generation tab's Try now) | chat (LM Generation)
    int epoch = 0;            // Completed epochs at this point.
    int total_epochs = 0;
    long long step = 0;       // Completed optimizer steps at this point.
    std::string weights;      // current | best
    std::string mode;         // continuation | chat
    std::string chat_template;
    std::string decoding;     // greedy | sampling
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    unsigned seed = 0;
    float repetition_penalty = 1.0f;
    int max_new_tokens = 0;
    int context = 0;
    std::vector<std::string> stop_texts;
    double elapsed_ms = 0.0;
    bool cancelled = false;
    std::optional<double> validation_loss;  // Of the epoch, when the run had validation.
    std::vector<GenerationPreviewSample> samples;
    std::filesystem::path file;
};

constexpr int kGenerationPreviewReportSchema = 2;

// Reads one report file. Returns nullopt with `error` set when the file is not a
// schema 2 report (for example one written before TOFIX159).
std::optional<GenerationPreviewReport> LoadGenerationPreviewReport(
    const std::filesystem::path& path, std::string& error);
// The run's reports, in order: baseline, then by epoch and step.
std::vector<GenerationPreviewReport> LoadGenerationPreviewReports(
    const std::filesystem::path& directory, std::vector<std::string>* skipped = nullptr);

// A training run that wrote reports: <checkpoints>/<graph>/<run>/generation_previews.
struct GenerationPreviewRun {
    std::string run_id;   // "train-1791984271862"
    std::string graph;    // the checkpoint folder it belongs to ("sft_v2_768_d256_L6")
    std::filesystem::path folder;
    size_t files = 0;     // report files (any schema)
    std::filesystem::file_time_type time{};  // the newest report
};
// Every run under `checkpoints_root` with report files, newest first.
std::vector<GenerationPreviewRun> FindGenerationPreviewRuns(const std::filesystem::path& checkpoints_root);

// Changes when a report is added or rewritten (for a live view to reload).
std::string GenerationPreviewFolderStamp(const std::filesystem::path& folder);

// Every sample of `reports` as one JSON object per line (point, epoch, step,
// prompt, output, stop reason, check). Returns false with `error` set.
bool ExportGenerationPreviewSamples(const std::vector<GenerationPreviewReport>& reports,
                                    const std::filesystem::path& path, std::string& error);

// "baseline.json", "epoch_4.json", "step_1200.json"; a second pass at the same
// point (a resumed run) gets "_2", "_3", ... before ".json".
std::filesystem::path GenerationPreviewReportPath(const std::filesystem::path& directory,
                                                  const std::string& point, int epoch, long long step);

GenerationPreviewCheckResult EvaluateGenerationPreviewCheck(const GenerationPreviewCheck& check,
                                                            const std::string& text,
                                                            const std::string& stop_reason);

// The longest phrase of 3 to 8 words that the text repeats (4+ words twice, or
// 3 words three times), lower case; "" when nothing repeats.
std::string DetectRepeatedPhrase(const std::string& text);

struct GenerationPreviewSummary {
    int samples = 0;
    int checks = 0;
    int checks_passed = 0;
    int stopped_by_itself = 0;  // eos or a stop text
    int repeating = 0;
};
GenerationPreviewSummary SummarizeGenerationPreview(const GenerationPreviewReport& report);

// Words for the views and the log.
std::string GenerationPreviewPointLabel(const GenerationPreviewReport& report);   // "Epoch 4 of 4"
std::string GenerationPreviewStopLabel(const GenerationPreviewSample& sample, int max_new_tokens);
std::string GenerationPreviewCheckLabel(const GenerationPreviewSample& sample);
std::string GenerationPreviewHowLine(const GenerationPreviewReport& report);

// The Console block at Info: a header, one line per sample, a summary.
std::vector<std::string> GenerationPreviewLogLines(const GenerationPreviewReport& report);

// The first `max_chars` characters of `text` on one line (UTF-8 safe), with
// "..." when cut.
std::string GenerationPreviewExcerpt(const std::string& text, size_t max_chars);

}  // namespace cyxwiz
