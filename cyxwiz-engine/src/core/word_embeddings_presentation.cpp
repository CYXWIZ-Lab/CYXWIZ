#include "word_embeddings_presentation.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>

namespace cyxwiz {

namespace {

const char* const kDot = " \xC2\xB7 ";  // " · "

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// Letters: ASCII ones, the alphabets below U+2000 (Latin, Greek, Cyrillic,
// Arabic, ...), kana, CJK and Hangul. Punctuation, symbols and emoji are not.
bool IsLetter(char32_t c) {
    if (c < 0x80) return std::isalpha(static_cast<unsigned char>(c)) != 0;
    if (c < 0xC0) return false;  // Latin-1 punctuation and signs
    if (c < 0x2000) return c != 0xD7 && c != 0xF7;
    return (c >= 0x3040 && c <= 0x9FFF) || (c >= 0xAC00 && c <= 0xD7AF);
}

bool HasLetter(const std::string& word) {
    for (const char32_t c : Utf8CodePoints(word)) {
        if (IsLetter(c)) return true;
    }
    return false;
}

}  // namespace

std::vector<char32_t> Utf8CodePoints(const std::string& text) {
    std::vector<char32_t> points;
    for (size_t i = 0; i < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        const size_t length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : (lead >> 3) == 0x1E ? 4 : 1;
        char32_t c = length == 1 ? lead : length == 2 ? (lead & 0x1F) : length == 3 ? (lead & 0x0F) : (lead & 0x07);
        size_t k = 1;
        for (; k < length && i + k < text.size(); ++k) {
            const unsigned char next = static_cast<unsigned char>(text[i + k]);
            if ((next >> 6) != 0x2) break;
            c = (c << 6) | (next & 0x3F);
        }
        points.push_back(k == length ? c : 0xFFFD);
        i += k;
    }
    return points;
}

std::string CodePointNames(const std::string& text) {
    std::string names;
    for (const char32_t c : Utf8CodePoints(text)) {
        char name[16];
        std::snprintf(name, sizeof(name), "U+%04X", static_cast<unsigned>(c));
        names += (names.empty() ? "" : " ") + std::string(name);
    }
    return names;
}

std::string GroupedCount(size_t value) {
    std::string digits = std::to_string(value);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) digits.insert(static_cast<size_t>(at), ",");
    return digits;
}

std::string VectorFileFormat(const std::string& path, const std::string& first_line) {
    const std::string lower = Lower(path);
    if (lower.size() >= 4 && lower.compare(lower.size() - 4, 4, ".bin") == 0) return "word2vec binary";
    std::istringstream in(first_line);
    long long count = 0, dim = 0;
    std::string rest;
    if (in >> count >> dim && !(in >> rest) && count > 0 && dim > 0) return "word2vec / fastText text";
    return "GloVe text (no header)";
}

std::string VectorFileSummary(const std::string& format, size_t words, int dim, double seconds) {
    char time[32];
    std::snprintf(time, sizeof(time), "%.1f s", seconds);
    return format + kDot + GroupedCount(words) + " words" + kDot + std::to_string(dim) + " values each" + kDot +
           "read " + time;
}

WordCoverageView BuildWordCoverageView(size_t exact, size_t lowercase, size_t missing,
                                       const std::vector<std::string>& missing_words, const std::string& pad,
                                       const std::string& unknown, const std::string& missing_rows,
                                       const std::string& unknown_row, size_t preview) {
    WordCoverageView view;
    view.found = GroupedCount(exact) + " found as written";
    view.lowercase = GroupedCount(lowercase) + " found in lowercase";
    view.missing = GroupedCount(missing) + " missing";
    const std::string missing_word = missing_rows == "random" ? "random" : "zeros";
    if (!pad.empty()) view.special = pad + " zeros";
    if (!unknown.empty()) {
        if (!view.special.empty()) view.special += kDot;
        view.special += unknown + " " + (unknown_row == "missing" ? missing_word : std::string("mean"));
    }
    const size_t counted = exact + lowercase + missing;
    if (counted > 0) {
        view.found_share = static_cast<float>(exact) / static_cast<float>(counted);
        view.lowercase_share = static_cast<float>(lowercase) / static_cast<float>(counted);
    }
    if (missing > 0) {
        view.missing_preview = "Missing, first of " + GroupedCount(missing) + ":";
        for (size_t i = 0; i < missing_words.size() && i < preview; ++i) view.preview_words.push_back(missing_words[i]);
        const size_t no_letters = static_cast<size_t>(
            std::count_if(missing_words.begin(), missing_words.end(), [](const std::string& w) { return !HasLetter(w); }));
        if (!missing_words.empty() && no_letters * 2 > missing_words.size())
            view.missing_kind = "mostly punctuation, digits and symbols";
    }
    return view;
}

float StepShareFromStatus(const std::string& status) {
    const auto close = status.rfind("%)");
    if (close == std::string::npos) return -1.0f;
    const auto open = status.rfind('(', close);
    if (open == std::string::npos) return -1.0f;
    try {
        const int percent = std::stoi(status.substr(open + 1, close - open - 1));
        return std::clamp(static_cast<float>(percent) / 100.0f, 0.0f, 1.0f);
    } catch (...) {
        return -1.0f;
    }
}

std::string StepMessageFromStatus(const std::string& status) {
    std::string message = status;
    if (const auto colon = message.find(": "); colon != std::string::npos) message = message.substr(colon + 2);
    if (const auto open = message.rfind(" ("); open != std::string::npos && message.size() >= 2 &&
                                               message.compare(message.size() - 2, 2, "%)") == 0)
        message = message.substr(0, open);
    return message;
}

}  // namespace cyxwiz
