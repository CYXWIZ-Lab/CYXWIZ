#pragma once

// What a Word Embeddings node's saved vectors were trained from (TOFIX152):
// written beside the .vec after training (<vectors>.training.json), so a later
// run with the same text, settings and vocabulary reads the file instead of
// training again, and the dialog shows the saved training after the project is
// reopened.

#include <cstdint>
#include <optional>
#include <string>

namespace cyxwiz {

// Raised when a change to the trainers makes earlier vectors stale.
inline constexpr int kWordVectorsTrainingVersion = 1;

struct WordVectorsTrainingRecord {
    int version = kWordVectorsTrainingVersion;
    std::string settings;   // WordEmbeddingsOperator::TrainingSettings()
    std::string text_sha256;
    std::string vocab_sha256;  // empty without a vocabulary file
    std::string trained_at;    // local time, "2026-10-15 19:40"
    double seconds = 0.0;
    std::string summary;   // "trained 16,321 words x 100 (skip-gram, ...)"
    std::string coverage;  // "; vocabulary 5,000 tokens: ..." (empty without one)
    // The saved file as written; another size or time means it was replaced.
    uint64_t vectors_bytes = 0;
    int64_t vectors_time = 0;
};

std::string WordVectorsTrainingRecordPath(const std::string& vectors_path);

// Writes the record beside `vectors_path`, taking the file's size and time now.
bool WriteWordVectorsTrainingRecord(const std::string& vectors_path, WordVectorsTrainingRecord record,
                                    std::string* error = nullptr);

// The record of `vectors_path`; empty when there is none, it is unreadable or
// of another version, or the vectors file changed since it was written.
std::optional<WordVectorsTrainingRecord> ReadWordVectorsTrainingRecord(const std::string& vectors_path);

}  // namespace cyxwiz
