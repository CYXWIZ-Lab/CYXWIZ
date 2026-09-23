#include "cyxwiz/tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace cyxwiz {
namespace {
// Artifact v1 contract: ASCII whitespace is kept as individual byte pieces;
// other runs are cut at 256 bytes. Never merge across a piece or document.
// This bounds the quadratic rank-selection encoder independently of input size.
bool IsSpace(unsigned char c) {
    return c == ' ' || (c >= '\t' && c <= '\r');
}

template<class Visitor>
void VisitPieces(const std::string& text, Visitor visit) {
    for (size_t start = 0; start < text.size();) {
        size_t end = start + 1;
        if (!IsSpace(static_cast<unsigned char>(text[start]))) {
            while (end < text.size() && end - start < 256 &&
                   !IsSpace(static_cast<unsigned char>(text[end]))) ++end;
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
    if (words.size() < 260 || words.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        merges.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    Vocabulary candidate;
    for (int i = 0; i < 4; ++i) {
        if (words[i] != candidate.idx_to_word_[i]) return false;
    }
    for (int i = 0; i < 256; ++i) {
        if (words[i + 4] != std::string(1, static_cast<char>(i))) return false;
    }
    candidate.SetVocabulary(words);
    if (candidate.Size() != words.size()) return false;
    size_t available = 260;
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const auto [left, right, result] = merges[rank];
        if (left < 4 || right < 4 || result < 260 ||
            static_cast<size_t>(left) >= available || static_cast<size_t>(right) >= available ||
            static_cast<size_t>(result) > available || static_cast<size_t>(result) >= words.size()) return false;
        if (words[left].size() + words[right].size() > 256 ||
            words[result] != words[left] + words[right] ||
            std::any_of(words[result].begin(), words[result].end(), IsSpace)) return false;
        if (!candidate.bpe_ranks_.emplace(std::make_pair(left, right),
                std::make_pair(static_cast<int>(rank), result)).second) return false;
        if (static_cast<size_t>(result) == available) ++available;
    }
    if (available != words.size()) return false;
    candidate.bpe_merges_ = merges;
    candidate.byte_bpe_ = true;
    *this = std::move(candidate);
    return true;
}

void Tokenizer::TrainByteBPE(const std::vector<std::string>& documents,
                            int min_freq, int max_vocab_size) {
    if (lowercase_) throw std::invalid_argument("Byte BPE requires lowercase=false");
    if (min_freq < 1 || max_vocab_size < 260)
        throw std::invalid_argument("Byte BPE requires min_freq>=1 and an explicit max_vocab_size>=260");
    CheckCancelled();
    Vocabulary candidate;
    for (int i = 0; i < 256; ++i) candidate.AddWord(std::string(1, static_cast<char>(i)));

    // Deduplicate pieces before fitting; repeated words carry their corpus weight.
    std::map<std::string, uint64_t> frequencies;
    for (const auto& document : documents) {
        VisitPieces(document, [&](std::string_view piece) {
            CheckCancelled();
            ++frequencies[std::string(piece)];
        });
    }
    struct Piece { std::vector<int> ids; uint64_t count; };
    std::vector<Piece> pieces;
    pieces.reserve(frequencies.size());
    for (const auto& [text, count] : frequencies) {
        Piece piece{{}, count};
        piece.ids.reserve(text.size());
        for (unsigned char c : text) piece.ids.push_back(c + 4);
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
    if (!candidate.SetByteBPE(candidate.GetWords(), merges))
        throw std::logic_error("Byte BPE fitting produced an invalid merge artifact");
    vocab_ = std::move(candidate);
}

std::vector<std::string> Tokenizer::SplitByteBPE(const std::string& text) const {
    std::vector<std::string> tokens;
    const auto& ranks = vocab_.GetBPERanks();
    VisitPieces(text, [&](std::string_view piece) {
        CheckCancelled();
        std::vector<int> ids;
        ids.reserve(piece.size());
        for (unsigned char c : piece) ids.push_back(c + 4);
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
