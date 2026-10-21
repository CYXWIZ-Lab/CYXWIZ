#include "checkpoint_tokenizer.h"

#include <cyxwiz/utilities.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

bool WriteText(const fs::path& path, const std::string& text, std::string& error) {
    std::ofstream out(path, std::ios::binary);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out.good()) {
        error = "cannot write " + path.string();
        return false;
    }
    return true;
}

bool ReadText(const fs::path& path, std::string& text, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        error = "cannot read " + path.string();
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    text = buffer.str();
    return true;
}

}  // namespace

std::string CheckpointTokenizer::Sha256() const {
    return vocabulary.empty() ? std::string() : Utilities::HashText(vocabulary, "sha256").sha256_hash;
}

bool WriteCheckpointTokenizer(const fs::path& checkpoint, const CheckpointTokenizer& tokenizer, std::string& error) {
    const fs::path folder = checkpoint / "tokenizer";
    std::error_code ec;
    fs::create_directories(folder, ec);
    if (ec) {
        error = "cannot create " + folder.string() + ": " + ec.message();
        return false;
    }
    nlohmann::json sequence = {
        {"version", 1},
        {"sequence_length", tokenizer.sequence_length},
        {"token_vocabulary_size", tokenizer.token_vocabulary_size},
        {"tag_vocabulary_size", tokenizer.tag_vocabulary_size},
        {"word_pad_id", tokenizer.word_pad_id},
        {"pos_pad_id", tokenizer.pos_pad_id},
        {"labels", tokenizer.labels},
    };
    return WriteText(folder / "config.json", tokenizer.config_json, error) &&
           WriteText(folder / "vocab.txt", tokenizer.vocabulary, error) &&
           WriteText(folder / "sequence.json", sequence.dump(), error);
}

std::optional<CheckpointTokenizer> ReadCheckpointTokenizer(const fs::path& checkpoint, std::string& error) {
    error.clear();
    const fs::path folder = checkpoint / "tokenizer";
    std::error_code ec;
    if (!fs::exists(folder / "vocab.txt", ec)) return std::nullopt;
    CheckpointTokenizer tokenizer;
    std::string sequence_text;
    if (!ReadText(folder / "config.json", tokenizer.config_json, error) ||
        !ReadText(folder / "vocab.txt", tokenizer.vocabulary, error) ||
        !ReadText(folder / "sequence.json", sequence_text, error))
        return std::nullopt;
    try {
        const auto sequence = nlohmann::json::parse(sequence_text);
        if (sequence.value("version", 0) != 1) {
            error = "the checkpoint's tokenizer/sequence.json has an unknown version";
            return std::nullopt;
        }
        tokenizer.sequence_length = sequence.value("sequence_length", size_t{0});
        tokenizer.token_vocabulary_size = sequence.value("token_vocabulary_size", size_t{0});
        tokenizer.tag_vocabulary_size = sequence.value("tag_vocabulary_size", size_t{0});
        tokenizer.word_pad_id = sequence.value("word_pad_id", int64_t{0});
        tokenizer.pos_pad_id = sequence.value("pos_pad_id", int64_t{0});
        tokenizer.labels = sequence.value("labels", std::vector<std::string>{});
    } catch (const std::exception& exception) {
        error = std::string("the checkpoint's tokenizer/sequence.json cannot be read: ") + exception.what();
        return std::nullopt;
    }
    if (tokenizer.vocabulary.empty()) {
        error = "the checkpoint's tokenizer/vocab.txt is empty";
        return std::nullopt;
    }
    return tokenizer;
}

}  // namespace cyxwiz
