#pragma once

// Word Embeddings dialog (TOFIX152): the words the dialog shows, worked out
// from the vectors' facts. Pure data in and out, so it is tested on its own.

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

// The code points of UTF-8 text (an invalid byte gives U+FFFD).
std::vector<char32_t> Utf8CodePoints(const std::string& text);

// "U+1F60A U+2764" for a token the font cannot draw.
std::string CodePointNames(const std::string& text);

// 4824 -> "4,824".
std::string GroupedCount(size_t value);

// The vector file's format from its name and first line: "word2vec binary",
// "word2vec / fastText text" (a "count dim" header) or "GloVe text (no header)".
std::string VectorFileFormat(const std::string& path, const std::string& first_line);

// "GloVe text (no header) · 400,000 words · 100 values each · read 2.4 s".
std::string VectorFileSummary(const std::string& format, size_t words, int dim, double seconds);

// How a vocabulary was lined up with the vectors.
struct WordCoverageView {
    std::string found;         // "4,824 found as written"
    std::string lowercase;     // "0 found in lowercase"
    std::string missing;       // "172 missing"
    std::string pieces;        // fastText: "4 built from pieces" (empty otherwise)
    std::string special;       // "[PAD] zeros · [UNK] mean"
    float found_share = 0.0f;  // of the counted tokens (PAD and UNK left out), for the bar
    float lowercase_share = 0.0f;
    float pieces_share = 0.0f;
    std::string missing_preview;  // "Missing, first of 172:"
    std::vector<std::string> preview_words;
    std::string missing_kind;  // "mostly punctuation, digits and symbols", or empty
};

// `missing_words` in vocabulary order; `pad` / `unknown` the special tokens'
// words (empty when the vocabulary has none); `missing_rows` and
// `unknown_row` the alignment options ("zeros" / "random", "mean" / "missing").
WordCoverageView BuildWordCoverageView(size_t exact, size_t lowercase, size_t missing,
                                       const std::vector<std::string>& missing_words, const std::string& pad,
                                       const std::string& unknown, const std::string& missing_rows,
                                       const std::string& unknown_row, size_t preview = 13);

// fastText: tokens built from character n-grams join the counts and the bar.
void AddPiecesToCoverage(WordCoverageView& view, size_t exact, size_t lowercase, size_t pieces, size_t missing);

// "fastText model (.bin) · 16,334 words · 2,000,000 pieces · 100 values each ·
// 813 MB · read 3.1 s".
std::string FastTextFileSummary(size_t words, int bucket, int dim, unsigned long long bytes, double seconds);

// The note beside a word fastText builds from its n-grams: "not in the
// training words · from 26 pieces" (`trained`) or "... the model's words ...".
std::string UnseenWordNote(bool trained, size_t pieces);

// The share (0..1) a pipeline step reports in its status, "...: Training
// word2vec, epoch 3 (62%)"; -1 when the status has none.
float StepShareFromStatus(const std::string& status);

// The status without the step's name: "Training word2vec, epoch 3".
std::string StepMessageFromStatus(const std::string& status);

// A training outcome with a long per-epoch list ("cost per epoch 0.034, 0.018,
// ..., 0.0061)") cut to its first and last figure ("cost per epoch 0.034 to
// 0.0061)") when it has more than `keep`; any other text unchanged.
std::string ShortEpochFigures(const std::string& outcome, size_t keep = 5);

// How a run's outcome starts when it read the saved vectors instead of
// training (the text, settings and vocabulary had not changed).
inline constexpr const char* kReusedVectorsOutcome = "reused the saved vectors";

inline bool ReusedVectorsOutcome(const std::string& outcome) { return outcome.rfind(kReusedVectorsOutcome, 0) == 0; }

}  // namespace cyxwiz
