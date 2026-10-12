#include "cyxwiz/tokenizer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <string_view>
#include <sstream>
#include <unordered_set>

namespace cyxwiz {
namespace {
constexpr std::string_view kBPEHeader = "#cyxwiz-vocabulary-byte-bpe-v1";
constexpr std::string_view kBPEV2Header = "#cyxwiz-vocabulary-byte-bpe-v2";
constexpr std::string_view kCharacterBPEHeader = "#cyxwiz-vocabulary-character-bpe-v1";
constexpr std::string_view kEncodedHeader = "#cyxwiz-vocabulary-hex-v1";

bool ReadLine(std::istream& input, std::string& line) {
    if (!std::getline(input, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

int HexDigit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}
} // namespace

bool Vocabulary::SaveToStream(std::ostream& output) const {
    // Retain the legacy format for ordinary word vocabularies. Hex represents
    // arbitrary bytes (including individual UTF-8 bytes), not Unicode code points.
    const bool encoded = byte_bpe_ || std::any_of(idx_to_word_.begin(), idx_to_word_.end(),
        [](const std::string& token) {
            return token.empty() || std::any_of(token.begin(), token.end(),
                [](unsigned char c) { return c < 32 || c >= 127; });
        });
    if (encoded) {
        const bool leading_space = byte_bpe_ && bpe_piece_policy_ == ByteBPEPiecePolicy::LeadingSpaceV2;
        const bool characters = byte_bpe_ && bpe_initial_unit_ == BPEInitialUnit::UnicodeCharacter;
        output << (characters ? kCharacterBPEHeader :
            (byte_bpe_ ? (leading_space ? kBPEV2Header : kBPEHeader) : kEncodedHeader)) << '\n';
        if (characters) {
            output << (leading_space ? "leading_space_v2" : "whitespace_v1") << '\n';
            output << bpe_alphabet_size_ << '\n';
        } else if (leading_space) output << "leading_space_v2\n";
        output << idx_to_word_.size() << '\n';
    }
    constexpr char digits[] = "0123456789abcdef";
    for (const auto& token : idx_to_word_) {
        if (encoded) {
            for (unsigned char c : token) {
                output.put(digits[c >> 4]);
                output.put(digits[c & 15]);
            }
        } else {
            output.write(token.data(), static_cast<std::streamsize>(token.size()));
        }
        output.put('\n');
    }
    if (byte_bpe_) {
        output << bpe_merges_.size() << '\n';
        for (const auto& merge : bpe_merges_)
            output << merge[0] << ' ' << merge[1] << ' ' << merge[2] << '\n';
    }
    return output.good();
}

namespace {
bool ReadArtifact(std::istream& input, std::vector<std::string>& tokens,
                  std::vector<std::array<int, 3>>& merges, bool& bpe, ByteBPEPiecePolicy& policy, BPEInitialUnit& unit, size_t& alphabet_size) {
    // Parse into local storage; failures must not modify the caller's state.
    std::vector<std::string> parsed;
    std::string line;
    if (!ReadLine(input, line)) return false;
    const bool characters = line == kCharacterBPEHeader;
    bpe = line == kBPEHeader || line == kBPEV2Header || characters;
    unit = characters ? BPEInitialUnit::UnicodeCharacter : BPEInitialUnit::Byte;
    alphabet_size = 256;
    policy = ByteBPEPiecePolicy::WhitespaceV1;
    if (characters) {
        if (!ReadLine(input, line)) return false;
        if (line == "leading_space_v2") policy = ByteBPEPiecePolicy::LeadingSpaceV2;
        else if (line != "whitespace_v1") return false;
        if (!ReadLine(input, line)) return false;
        const auto count = std::from_chars(line.data(), line.data() + line.size(), alphabet_size);
        if (count.ec != std::errc{} || count.ptr != line.data() + line.size() ||
            alphabet_size > static_cast<size_t>(std::numeric_limits<int>::max()) - 4) return false;
    } else if (line == kBPEV2Header) {
        if (!ReadLine(input, line) || line != "leading_space_v2") return false;
        policy = ByteBPEPiecePolicy::LeadingSpaceV2;
    }
    if (line == kEncodedHeader || bpe) {
        if (!ReadLine(input, line)) return false;
        size_t count = 0;
        const auto result = std::from_chars(line.data(), line.data() + line.size(), count);
        if (result.ec != std::errc{} || result.ptr != line.data() + line.size() ||
            count == 0 || count > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
        std::unordered_set<std::string> seen;
        for (size_t i = 0; i < count; ++i) {
            if (!ReadLine(input, line) || line.size() % 2 != 0) return false;
            std::string token;
            token.reserve(line.size() / 2);
            for (size_t j = 0; j < line.size(); j += 2) {
                const int high = HexDigit(line[j]);
                const int low = HexDigit(line[j + 1]);
                if (high < 0 || low < 0) return false;
                token.push_back(static_cast<char>((high << 4) | low));
            }
            if (!seen.insert(token).second) return false;
            parsed.push_back(std::move(token));
        }
        if (bpe) {
            if (!ReadLine(input, line)) return false;
            size_t merge_count = 0;
            const auto parsed_count = std::from_chars(line.data(), line.data() + line.size(), merge_count);
            if (parsed_count.ec != std::errc{} || parsed_count.ptr != line.data() + line.size() ||
                merge_count > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
            for (size_t i = 0; i < merge_count; ++i) {
                if (!ReadLine(input, line)) return false;
                std::istringstream row(line);
                std::array<int, 3> merge{};
                std::string extra;
                if (!(row >> merge[0] >> merge[1] >> merge[2]) || row >> extra) return false;
                merges.push_back(merge);
            }
        }
        if (ReadLine(input, line) || input.bad()) return false;
    } else {
        if (line.starts_with("#cyxwiz-vocabulary-")) return false;
        do {
            if (!line.empty()) parsed.push_back(line);
            if (parsed.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
        } while (ReadLine(input, line));
        if (input.bad() || parsed.empty()) return false;
    }
    tokens = std::move(parsed);
    return true;
}

} // namespace

bool Vocabulary::ReadTokens(std::istream& input, std::vector<std::string>& tokens) {
    Vocabulary parsed;
    if (!parsed.LoadFromStream(input)) return false;
    tokens = parsed.GetWords();
    return true;
}

bool Vocabulary::SaveToFile(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open() || !SaveToStream(file)) {
        spdlog::error("Failed to write vocabulary: {}", filepath);
        return false;
    }
    file.close();
    if (file.fail()) {
        spdlog::error("Failed to finish vocabulary file: {}", filepath);
        return false;
    }
    spdlog::info("Vocabulary saved: {} words to {}", idx_to_word_.size(), filepath);
    return true;
}

bool Vocabulary::LoadFromFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open() || !LoadFromStream(file)) {
        spdlog::error("Failed to read vocabulary: {}", filepath);
        return false;
    }
    spdlog::info("Vocabulary loaded: {} words from {}", idx_to_word_.size(), filepath);
    return true;
}

bool Vocabulary::LoadFromStream(std::istream& input) {
    std::vector<std::string> words;
    std::vector<std::array<int, 3>> merges;
    bool bpe = false;
    ByteBPEPiecePolicy policy = ByteBPEPiecePolicy::WhitespaceV1;
    BPEInitialUnit unit = BPEInitialUnit::Byte;
    size_t alphabet_size = 256;
    if (!ReadArtifact(input, words, merges, bpe, policy, unit, alphabet_size)) return false;
    if (bpe) return SetBPE(words, merges, policy, unit, alphabet_size);
    std::unordered_map<std::string, int> indices;
    for (size_t i = 0; i < words.size(); ++i) {
        if (!indices.emplace(words[i], static_cast<int>(i)).second) {
            return false;
        }
    }
    // Keep legacy special-index lookup semantics and exact stored token order.
    const auto find_idx = [&](const char* token, int fallback) {
        const auto it = indices.find(token);
        return it == indices.end() ? fallback : it->second;
    };
    pad_idx_ = find_idx("[PAD]", 0);
    unk_idx_ = find_idx("[UNK]", 1);
    bos_idx_ = find_idx("[BOS]", 2);
    eos_idx_ = find_idx("[EOS]", 3);
    byte_bpe_ = false;
    bpe_initial_unit_ = BPEInitialUnit::Byte;
    bpe_alphabet_size_ = 256;
    bpe_piece_policy_ = ByteBPEPiecePolicy::WhitespaceV1;
    bpe_merges_.clear();
    bpe_ranks_.clear();
    idx_to_word_ = std::move(words);
    word_to_idx_ = std::move(indices);
    return true;
}
} // namespace cyxwiz
