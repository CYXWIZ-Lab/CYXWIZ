#pragma once

// UTF-8 text helpers shared by the tokenizers, lowercasing and the word
// vectors: code points instead of bytes, and character classes that do not
// depend on the process locale (std::isalnum / std::tolower read the C
// runtime's locale, which the embedded Python sets to the user's code page,
// and cut a multi-byte character into single bytes).

#include <cstddef>
#include <string>

namespace cyxwiz::text_utf8 {

// The code point at `i`, and its length in bytes; false (length 1) for a byte
// that does not start a valid UTF-8 sequence.
inline bool Next(const std::string& s, size_t i, char32_t& cp, size_t& length) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    length = 1;
    if (c < 0x80) {
        cp = c;
        return true;
    }
    size_t n = 0;
    if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F;
        n = 2;
    } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F;
        n = 3;
    } else if ((c & 0xF8) == 0xF0) {
        cp = c & 0x07;
        n = 4;
    } else {
        return false;
    }
    if (i + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k) {
        const unsigned char next = static_cast<unsigned char>(s[i + k]);
        if ((next & 0xC0) != 0x80) return false;
        cp = (cp << 6) | (next & 0x3F);
    }
    length = n;
    return true;
}

inline void Append(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// ASCII letters and digits, and the letters of other scripts (Latin, Greek
// and Cyrillic to Georgian, CJK, Hangul); punctuation, symbols and emoji are
// not.
inline bool IsLetterOrDigit(char32_t cp) {
    return (cp >= '0' && cp <= '9') || (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
           (cp >= 0xC0 && cp <= 0x24F && cp != 0xD7 && cp != 0xF7) || (cp >= 0x370 && cp <= 0x1FFF) ||
           (cp >= 0x2C00 && cp <= 0x2DFF) || (cp >= 0x3040 && cp <= 0x9FFF) || (cp >= 0xAC00 && cp <= 0xD7AF) ||
           (cp >= 0xF900 && cp <= 0xFAFF);
}

// A letter, a digit or a combining mark (an accent written after its letter):
// what a word is made of.
inline bool IsWordCharacter(char32_t cp) {
    return IsLetterOrDigit(cp) || (cp >= 0x300 && cp <= 0x36F);
}

// Spaces, and the invisible joiners and selectors that only shape a
// neighbour: they separate words and make no token of their own.
inline bool IsSpace(char32_t cp) {
    return cp == ' ' || (cp >= '\t' && cp <= '\r') || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200D) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x2060 || cp == 0x3000 ||
           cp == 0xFE0E || cp == 0xFE0F || cp == 0xFEFF;
}

// Lowercase of ASCII, Latin-1, Latin Extended-A, Greek and Cyrillic capitals;
// other code points are returned unchanged.
inline char32_t Lower(char32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp < 0xC0) return cp;
    if (cp <= 0xDE) return cp == 0xD7 ? cp : cp + 32;
    if (cp >= 0x100 && cp <= 0x137) return cp | 1;
    if (cp >= 0x139 && cp <= 0x148) return (cp & 1) ? cp + 1 : cp;
    if (cp >= 0x14A && cp <= 0x177) return cp | 1;
    if (cp == 0x178) return 0xFF;
    if (cp >= 0x179 && cp <= 0x17E) return (cp & 1) ? cp + 1 : cp;
    if (cp >= 0x391 && cp <= 0x3A9 && cp != 0x3A2) return cp + 32;
    if (cp >= 0x410 && cp <= 0x42F) return cp + 32;
    if (cp >= 0x400 && cp <= 0x40F) return cp + 80;
    return cp;
}

}  // namespace cyxwiz::text_utf8
