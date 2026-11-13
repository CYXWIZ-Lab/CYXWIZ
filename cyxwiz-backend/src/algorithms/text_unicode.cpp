#include "text_unicode.h"

#include "text_utf8.h"

#include <algorithm>
#include <iterator>

namespace cyxwiz::text_unicode {
namespace {

struct CodePointRange {
    char32_t first;
    char32_t last;
};
struct LowerPair {
    char32_t from;
    char32_t to;
};

#include "text_unicode_tables.inc"

template <size_t N>
bool InRanges(const CodePointRange (&table)[N], char32_t cp) {
    const auto it = std::upper_bound(std::begin(table), std::end(table), cp,
                                     [](char32_t value, const CodePointRange& r) { return value < r.first; });
    return it != std::begin(table) && cp <= std::prev(it)->last;
}

}  // namespace

bool IsAlnum(char32_t cp) { return InRanges(kAlnum, cp); }
bool IsUpper(char32_t cp) { return InRanges(kUpper, cp); }
bool IsLowerOrTitle(char32_t cp) { return InRanges(kLowerOrTitle, cp); }
bool IsSpace(char32_t cp) { return InRanges(kSpace, cp); }

char32_t Lower(char32_t cp) {
    const auto it = std::lower_bound(std::begin(kLower), std::end(kLower), cp,
                                     [](const LowerPair& p, char32_t value) { return p.from < value; });
    return it != std::end(kLower) && it->from == cp ? it->to : cp;
}

std::u32string Decode(const std::string& text) {
    std::u32string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        char32_t cp = 0;
        size_t length = 1;
        if (!text_utf8::Next(text, i, cp, length)) cp = 0xFFFD;
        out.push_back(cp);
        i += length;
    }
    return out;
}

std::string Encode(const std::u32string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char32_t cp : text) text_utf8::Append(out, cp);
    return out;
}

std::u32string LowerString(const std::u32string& text) {
    std::u32string out(text);
    for (char32_t& cp : out) cp = Lower(cp);
    return out;
}

bool IsUpperString(const std::u32string& text) {
    bool cased = false;
    for (const char32_t cp : text) {
        if (IsLowerOrTitle(cp)) return false;
        if (IsUpper(cp)) cased = true;
    }
    return cased;
}

}  // namespace cyxwiz::text_unicode
