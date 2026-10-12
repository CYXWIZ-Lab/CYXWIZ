#include "cyxwiz/tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <set>
#include <string_view>

namespace cyxwiz {
namespace {
// Artifact policy bounds pieces at 256 bytes; never merge across documents.
// v1 separates whitespace; v2 allows one ASCII space before a non-space run.
// This bounds the quadratic rank-selection encoder independently of input size.
bool IsSpace(unsigned char c) {
    return c == ' ' || (c >= '\t' && c <= '\r');
}

// Strict UTF-8 scalar boundary, excluding overlong encodings and surrogates.
// Zero denotes malformed/truncated input. Byte mode never calls this parser.
size_t ScalarSize(std::string_view text, size_t offset) {
    if (offset >= text.size()) return 0;
    const auto first = static_cast<unsigned char>(text[offset]);
    if (first < 0x80) return 1;
    size_t size = 0;
    uint32_t scalar = 0, minimum = 0;
    if (first >= 0xc2 && first <= 0xdf) { size = 2; scalar = first & 0x1f; minimum = 0x80; }
    else if (first >= 0xe0 && first <= 0xef) { size = 3; scalar = first & 0x0f; minimum = 0x800; }
    else if (first >= 0xf0 && first <= 0xf4) { size = 4; scalar = first & 0x07; minimum = 0x10000; }
    else return 0;
    if (size > text.size() - offset) return 0;
    for (size_t i = 1; i < size; ++i) {
        const auto next = static_cast<unsigned char>(text[offset + i]);
        if ((next & 0xc0) != 0x80) return 0;
        scalar = (scalar << 6) | (next & 0x3f);
    }
    if (scalar < minimum || scalar > 0x10ffff || (scalar >= 0xd800 && scalar <= 0xdfff)) return 0;
    return size;
}

size_t UnitSize(std::string_view text, size_t offset, BPEInitialUnit unit) {
    if (unit == BPEInitialUnit::Byte) return 1;
    const auto size = ScalarSize(text, offset);
    if (!size) throw std::invalid_argument("Character BPE requires valid UTF-8 text");
    return size;
}

std::vector<int> InitialIds(std::string_view text, const Vocabulary& vocab, BPEInitialUnit unit) {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const auto size = UnitSize(text, i, unit);
        ids.push_back(unit == BPEInitialUnit::Byte ? static_cast<unsigned char>(text[i]) + 4
            : vocab.WordToIndex(std::string(text.substr(i, size))));
        i += size;
    }
    return ids;
}

template<class Visitor>
void VisitPieces(const std::string& text, ByteBPEPiecePolicy policy, BPEInitialUnit unit, Visitor visit) {
    for (size_t start = 0; start < text.size();) {
        size_t end = start + UnitSize(text, start, unit);
        const bool leading_space = policy == ByteBPEPiecePolicy::LeadingSpaceV2 &&
            text[start] == ' ' && end < text.size() &&
            !IsSpace(static_cast<unsigned char>(text[end]));
        if (leading_space || !IsSpace(static_cast<unsigned char>(text[start]))) {
            while (end < text.size() && !IsSpace(static_cast<unsigned char>(text[end]))) {
                const auto size = UnitSize(text, end, unit);
                if (end - start + size > 256) break;
                end += size;
            }
        }
        visit(std::string_view(text).substr(start, end - start));
        start = end;
    }
}

void MergePair(std::vector<int>& ids, int left, int right, int result) {
    size_t out = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i + 1 < ids.size() && ids[i] == left && ids[i + 1] == right) {
            ids[out++] = result;
            ++i;
        } else {
            ids[out++] = ids[i];
        }
    }
    ids.resize(out);
}
} // namespace

bool Vocabulary::SetByteBPE(const std::vector<std::string>& words,
                           const std::vector<std::array<int, 3>>& merges) {
    return SetByteBPE(words, merges, ByteBPEPiecePolicy::WhitespaceV1);
}

bool Vocabulary::SetByteBPE(const std::vector<std::string>& words,
                           const std::vector<std::array<int, 3>>& merges,
                           ByteBPEPiecePolicy policy) {
    return SetBPE(words, merges, policy, BPEInitialUnit::Byte, 256);
}

bool Vocabulary::SetBPE(const std::vector<std::string>& words,
                        const std::vector<std::array<int, 3>>& merges, ByteBPEPiecePolicy policy,
                        BPEInitialUnit unit, size_t alphabet_size) {
    if (unit != BPEInitialUnit::Byte && unit != BPEInitialUnit::UnicodeCharacter) return false;
    if (words.size() < 4 || alphabet_size > words.size() - 4 ||
        words.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        merges.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    if (policy != ByteBPEPiecePolicy::WhitespaceV1 &&
        policy != ByteBPEPiecePolicy::LeadingSpaceV2) return false;
    Vocabulary candidate;
    for (int i = 0; i < 4; ++i) {
        if (words[i] != candidate.idx_to_word_[i]) return false;
    }
    if (unit == BPEInitialUnit::Byte && alphabet_size != 256) return false;
    for (size_t i = 0; i < alphabet_size; ++i) {
        const auto& token = words[i + 4];
        if (unit == BPEInitialUnit::Byte) {
            if (token != std::string(1, static_cast<char>(i))) return false;
        } else if (token.empty() || ScalarSize(token, 0) != token.size()) return false;
    }
    candidate.SetVocabulary(words);
    if (candidate.Size() != words.size()) return false;
    const size_t initial_size = 4 + alphabet_size;
    size_t available = initial_size;
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const auto [left, right, result] = merges[rank];
        if (left < 4 || right < 4 || result < 0 || static_cast<size_t>(result) < initial_size ||
            static_cast<size_t>(left) >= available || static_cast<size_t>(right) >= available ||
            static_cast<size_t>(result) > available || static_cast<size_t>(result) >= words.size()) return false;
        if (words[left].size() + words[right].size() > 256 ||
            words[result] != words[left] + words[right]) return false;
        size_t pieces = 0;
        try {
            VisitPieces(words[result], policy, unit, [&](std::string_view) { ++pieces; });
        } catch (const std::invalid_argument&) { return false; }
        if (pieces != 1) return false;
        if (!candidate.bpe_ranks_.emplace(std::make_pair(left, right),
                std::make_pair(static_cast<int>(rank), result)).second) return false;
        if (static_cast<size_t>(result) == available) ++available;
    }
    if (available != words.size()) return false;
    candidate.bpe_merges_ = merges;
    candidate.byte_bpe_ = true;
    candidate.bpe_initial_unit_ = unit;
    candidate.bpe_alphabet_size_ = alphabet_size;
    candidate.bpe_piece_policy_ = policy;
    *this = std::move(candidate);
    return true;
}

void Tokenizer::SetBPEFitPiecePolicy(ByteBPEPiecePolicy policy) {
    if (type_ != TokenizerType::ByteBPE ||
        (policy != ByteBPEPiecePolicy::WhitespaceV1 && policy != ByteBPEPiecePolicy::LeadingSpaceV2))
        throw std::invalid_argument("ByteBPE fit piece policy requires ByteBPE and a supported policy");
    bpe_fit_piece_policy_ = policy;
}

void Tokenizer::SetBPEFitInitialUnit(BPEInitialUnit unit) {
    if (type_ != TokenizerType::ByteBPE ||
        (unit != BPEInitialUnit::Byte && unit != BPEInitialUnit::UnicodeCharacter))
        throw std::invalid_argument("BPE initial unit requires BPE and Byte or UnicodeCharacter");
    bpe_fit_initial_unit_ = unit;
}

void Tokenizer::TrainByteBPE(const std::vector<std::string>& documents,
                            int min_freq, int max_vocab_size) {
    if (lowercase_) throw std::invalid_argument("Byte BPE requires lowercase=false");
    const bool bytes = bpe_fit_initial_unit_ == BPEInitialUnit::Byte;
    if (min_freq < 1 || max_vocab_size < (bytes ? 260 : 4))
        throw std::invalid_argument("BPE requires min_freq>=1 and max_vocab_size covering the initial alphabet plus 4 specials (bytes: 260 minimum)");
    CheckCancelled();
    Vocabulary candidate;
    if (bytes) for (int i = 0; i < 256; ++i) candidate.AddWord(std::string(1, static_cast<char>(i)));
    std::set<std::string> alphabet;

    // Deduplicate pieces before fitting; repeated words carry their corpus weight.
    std::map<std::string, uint64_t> frequencies;
    for (const auto& document : documents) {
        VisitPieces(document, bpe_fit_piece_policy_, bpe_fit_initial_unit_, [&](std::string_view piece) {
            CheckCancelled();
            ++frequencies[std::string(piece)];
            if (!bytes) for (size_t i = 0; i < piece.size();) {
                const auto size = UnitSize(piece, i, bpe_fit_initial_unit_);
                alphabet.emplace(piece.substr(i, size));
                if (alphabet.size() > static_cast<size_t>(max_vocab_size - 4))
                    throw std::invalid_argument("Character BPE max_vocab_size is smaller than the observed alphabet plus 4 special tokens");
                i += size;
            }
        });
    }
    for (const auto& character : alphabet) candidate.AddWord(character);
    const size_t alphabet_size = candidate.Size() - 4;
    struct Piece { std::vector<int> ids; uint64_t count; };
    std::vector<Piece> pieces;
    pieces.reserve(frequencies.size());
    for (const auto& [text, count] : frequencies) {
        Piece piece{{}, count};
        piece.ids = InitialIds(text, candidate, bpe_fit_initial_unit_);
        pieces.push_back(std::move(piece));
    }
    frequencies.clear();
    std::vector<std::array<int, 3>> merges;
    while (candidate.Size() < static_cast<size_t>(max_vocab_size)) {
        CheckCancelled();
        std::map<std::pair<int, int>, uint64_t> counts;
        for (const auto& piece : pieces) {
            CheckCancelled();
            for (size_t i = 1; i < piece.ids.size(); ++i)
                counts[{piece.ids[i - 1], piece.ids[i]}] += piece.count;
        }
        uint64_t best_count = 0;
        std::pair<int, int> best;
        std::string best_text;
        for (const auto& [pair, count] : counts) {
            if (count < static_cast<uint64_t>(min_freq) || count <= best_count) continue;
            const std::string text = candidate.IndexToWord(pair.first) + candidate.IndexToWord(pair.second);
            // Literal special spellings must stay ordinary text, never PAD/BOS/etc.
            if (candidate.HasWord(text) && candidate.WordToIndex(text) < 4) continue;
            best = pair;
            best_text = text;
            best_count = count;
        }
        if (!best_count) break;
        const int result = candidate.AddWord(best_text);
        merges.push_back({best.first, best.second, result});
        for (auto& piece : pieces) {
            CheckCancelled();
            MergePair(piece.ids, best.first, best.second, result);
        }
    }
    CheckCancelled();
    if (!candidate.SetBPE(candidate.GetWords(), merges, bpe_fit_piece_policy_, bpe_fit_initial_unit_, alphabet_size))
        throw std::logic_error("Byte BPE fitting produced an invalid merge artifact");
    vocab_ = std::move(candidate);
}

std::vector<std::string> Tokenizer::SplitByteBPE(const std::string& text) const {
    std::vector<std::string> tokens;
    const auto& ranks = vocab_.GetBPERanks();
    VisitPieces(text, vocab_.GetBPEPiecePolicy(), vocab_.GetBPEInitialUnit(), [&](std::string_view piece) {
        CheckCancelled();
        auto ids = InitialIds(piece, vocab_, vocab_.GetBPEInitialUnit());
        while (ids.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            int left = 0, right = 0, result = 0;
            for (size_t i = 1; i < ids.size(); ++i) {
                const auto it = ranks.find({ids[i - 1], ids[i]});
                if (it != ranks.end() && it->second.first < best_rank) {
                    best_rank = it->second.first;
                    left = ids[i - 1]; right = ids[i]; result = it->second.second;
                }
            }
            if (best_rank == std::numeric_limits<int>::max()) break;
            MergePair(ids, left, right, result);
        }
        for (int id : ids) tokens.push_back(vocab_.IndexToWord(id));
    });
    return tokens;
}
} // namespace cyxwiz
