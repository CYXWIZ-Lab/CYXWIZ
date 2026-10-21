#pragma once

// The tokenizer a model was trained with, kept in every checkpoint of the run
// (TOFIX159; owner: an Engine behaviour that works for every model). A
// checkpoint then carries what turns text into its token ids and back, so
// the LM Generation panel, Run Test and Try now use exactly the tokenizer of
// the weights instead of rebuilding one from the training data (which may
// have changed, moved or be on another machine).
//
// Layout, as in a .cyxmodel package: <checkpoint>/tokenizer/config.json (the
// tokenizer's settings), vocab.txt (the vocabulary artifact; its sha256 is
// the tokenizer's identity, also in metadata.json) and sequence.json (what
// training read from the data: the sequence length, the vocabulary in id
// order, the output vocabulary size and the pad ids). No ImGui here.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct CheckpointTokenizer {
    std::string config_json;   // tokenizer settings (type, lowercase, BPE policy, ...)
    std::string vocabulary;    // the vocabulary artifact
    size_t sequence_length = 0;
    std::vector<std::string> labels;  // id -> label, as training saw them (tokens for a language model)
    size_t token_vocabulary_size = 0;
    size_t tag_vocabulary_size = 0;   // a tagger's output classes (0 for a language model)
    int64_t word_pad_id = 0;
    int64_t pos_pad_id = 0;

    // A run with no tokenizer (tabular, images) stores none.
    bool empty() const { return vocabulary.empty(); }
    // The identity: the sha256 of the vocabulary artifact.
    std::string Sha256() const;
};

// Writes <checkpoint>/tokenizer/; false with `error` when it cannot.
bool WriteCheckpointTokenizer(const std::filesystem::path& checkpoint, const CheckpointTokenizer& tokenizer,
                              std::string& error);

// The checkpoint's tokenizer; nullopt with `error` empty when it stores none
// (an older checkpoint, or a run without one), with `error` set when it is
// there but cannot be read.
std::optional<CheckpointTokenizer> ReadCheckpointTokenizer(const std::filesystem::path& checkpoint,
                                                           std::string& error);

}  // namespace cyxwiz
