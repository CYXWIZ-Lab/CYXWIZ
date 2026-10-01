#include "python_tokenizer.h"

#include <cctype>
#include <cstring>

namespace cyxwiz::pytokens {

namespace {
bool IsIdentStart(unsigned char c) { return std::isalpha(c) || c == '_' || c >= 0x80; }
bool IsIdentChar(unsigned char c) { return std::isalnum(c) || c == '_' || c >= 0x80; }

// String prefixes: r, b, f, u and their two-letter mixes, any case.
bool IsStringPrefix(const char* p, const char* q) {
    const size_t n = static_cast<size_t>(q - p);
    if (n == 0 || n > 2) return false;
    for (const char* s = p; s < q; ++s) {
        if (!std::strchr("rRbBfFuU", *s)) return false;
    }
    return true;
}

// From the opening quote at `p`: the end of the string, or `end` when it is
// not closed on this line (the rest of the line is the string).
const char* StringEnd(const char* p, const char* end) {
    const char q = *p;
    const bool triple = end - p >= 3 && p[1] == q && p[2] == q;
    p += triple ? 3 : 1;
    while (p < end) {
        if (*p == '\\') {
            p += 2;
            continue;
        }
        if (*p == q) {
            if (!triple) return p + 1;
            if (end - p >= 3 && p[1] == q && p[2] == q) return p + 3;
        }
        ++p;
    }
    return end;
}

const char* NumberEnd(const char* p, const char* end) {
    if (end - p >= 2 && p[0] == '0' && std::strchr("xXoObB", p[1])) {
        p += 2;
        while (p < end && (std::isxdigit(static_cast<unsigned char>(*p)) || *p == '_')) ++p;
        return p;
    }
    while (p < end && (std::isdigit(static_cast<unsigned char>(*p)) || *p == '_')) ++p;
    if (p < end && *p == '.') {
        ++p;
        while (p < end && (std::isdigit(static_cast<unsigned char>(*p)) || *p == '_')) ++p;
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
        const char* e = p + 1;
        if (e < end && (*e == '+' || *e == '-')) ++e;
        if (e < end && std::isdigit(static_cast<unsigned char>(*e))) {
            p = e;
            while (p < end && (std::isdigit(static_cast<unsigned char>(*p)) || *p == '_')) ++p;
        }
    }
    if (p < end && (*p == 'j' || *p == 'J')) ++p;
    return p;
}
}  // namespace

bool Next(const char* begin, const char* end, const char*& out_begin, const char*& out_end, Kind& kind) {
    const char* p = begin;
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
    if (p >= end) return false;
    const unsigned char c = static_cast<unsigned char>(*p);

    if (c == '"' || c == '\'') {
        out_begin = p;
        out_end = StringEnd(p, end);
        kind = Kind::String;
        return true;
    }
    if (std::isdigit(c) || (c == '.' && p + 1 < end && std::isdigit(static_cast<unsigned char>(p[1])))) {
        out_begin = p;
        out_end = NumberEnd(p, end);
        kind = Kind::Number;
        return true;
    }
    if (IsIdentStart(c)) {
        const char* q = p + 1;
        while (q < end && IsIdentChar(static_cast<unsigned char>(*q))) ++q;
        // rb"..." and f'...': the prefix belongs to the string.
        if (q < end && (*q == '"' || *q == '\'') && IsStringPrefix(p, q)) {
            out_begin = p;
            out_end = StringEnd(q, end);
            kind = Kind::String;
            return true;
        }
        out_begin = p;
        out_end = q;
        kind = Kind::Identifier;
        return true;
    }
    if (c == '@' && p + 1 < end && IsIdentStart(static_cast<unsigned char>(p[1]))) {
        const char* q = p + 1;
        while (q < end && (IsIdentChar(static_cast<unsigned char>(*q)) || *q == '.')) ++q;
        out_begin = p;
        out_end = q;
        kind = Kind::Decorator;
        return true;
    }
    if (std::strchr("()[]{}<>=!+-*/%&|^~.,:;", c) && c != 0) {
        out_begin = p;
        out_end = p + 1;
        kind = Kind::Punctuation;
        return true;
    }
    return false;
}

}  // namespace cyxwiz::pytokens
