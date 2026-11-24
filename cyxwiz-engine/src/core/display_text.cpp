#include "display_text.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cyxwiz::core {

namespace {

struct ArabicForms {
    char32_t letter, isolated, final, initial, medial;  // 0: no such form
};

// Generated from the Unicode compatibility decompositions of U+FB50-FDFF and
// U+FE70-FEFF (Python unicodedata): the letters of U+0600-06FF that have
// presentation forms, sorted by letter.
constexpr ArabicForms kArabicForms[] = {
    {0x0621, 0xFE80, 0x0000, 0x0000, 0x0000}, {0x0622, 0xFE81, 0xFE82, 0x0000, 0x0000},
    {0x0623, 0xFE83, 0xFE84, 0x0000, 0x0000}, {0x0624, 0xFE85, 0xFE86, 0x0000, 0x0000},
    {0x0625, 0xFE87, 0xFE88, 0x0000, 0x0000}, {0x0626, 0xFE89, 0xFE8A, 0xFE8B, 0xFE8C},
    {0x0627, 0xFE8D, 0xFE8E, 0x0000, 0x0000}, {0x0628, 0xFE8F, 0xFE90, 0xFE91, 0xFE92},
    {0x0629, 0xFE93, 0xFE94, 0x0000, 0x0000}, {0x062A, 0xFE95, 0xFE96, 0xFE97, 0xFE98},
    {0x062B, 0xFE99, 0xFE9A, 0xFE9B, 0xFE9C}, {0x062C, 0xFE9D, 0xFE9E, 0xFE9F, 0xFEA0},
    {0x062D, 0xFEA1, 0xFEA2, 0xFEA3, 0xFEA4}, {0x062E, 0xFEA5, 0xFEA6, 0xFEA7, 0xFEA8},
    {0x062F, 0xFEA9, 0xFEAA, 0x0000, 0x0000}, {0x0630, 0xFEAB, 0xFEAC, 0x0000, 0x0000},
    {0x0631, 0xFEAD, 0xFEAE, 0x0000, 0x0000}, {0x0632, 0xFEAF, 0xFEB0, 0x0000, 0x0000},
    {0x0633, 0xFEB1, 0xFEB2, 0xFEB3, 0xFEB4}, {0x0634, 0xFEB5, 0xFEB6, 0xFEB7, 0xFEB8},
    {0x0635, 0xFEB9, 0xFEBA, 0xFEBB, 0xFEBC}, {0x0636, 0xFEBD, 0xFEBE, 0xFEBF, 0xFEC0},
    {0x0637, 0xFEC1, 0xFEC2, 0xFEC3, 0xFEC4}, {0x0638, 0xFEC5, 0xFEC6, 0xFEC7, 0xFEC8},
    {0x0639, 0xFEC9, 0xFECA, 0xFECB, 0xFECC}, {0x063A, 0xFECD, 0xFECE, 0xFECF, 0xFED0},
    {0x0641, 0xFED1, 0xFED2, 0xFED3, 0xFED4}, {0x0642, 0xFED5, 0xFED6, 0xFED7, 0xFED8},
    {0x0643, 0xFED9, 0xFEDA, 0xFEDB, 0xFEDC}, {0x0644, 0xFEDD, 0xFEDE, 0xFEDF, 0xFEE0},
    {0x0645, 0xFEE1, 0xFEE2, 0xFEE3, 0xFEE4}, {0x0646, 0xFEE5, 0xFEE6, 0xFEE7, 0xFEE8},
    {0x0647, 0xFEE9, 0xFEEA, 0xFEEB, 0xFEEC}, {0x0648, 0xFEED, 0xFEEE, 0x0000, 0x0000},
    {0x0649, 0xFEEF, 0xFEF0, 0xFBE8, 0xFBE9}, {0x064A, 0xFEF1, 0xFEF2, 0xFEF3, 0xFEF4},
    {0x0671, 0xFB50, 0xFB51, 0x0000, 0x0000}, {0x0677, 0xFBDD, 0x0000, 0x0000, 0x0000},
    {0x0679, 0xFB66, 0xFB67, 0xFB68, 0xFB69}, {0x067A, 0xFB5E, 0xFB5F, 0xFB60, 0xFB61},
    {0x067B, 0xFB52, 0xFB53, 0xFB54, 0xFB55}, {0x067E, 0xFB56, 0xFB57, 0xFB58, 0xFB59},
    {0x067F, 0xFB62, 0xFB63, 0xFB64, 0xFB65}, {0x0680, 0xFB5A, 0xFB5B, 0xFB5C, 0xFB5D},
    {0x0683, 0xFB76, 0xFB77, 0xFB78, 0xFB79}, {0x0684, 0xFB72, 0xFB73, 0xFB74, 0xFB75},
    {0x0686, 0xFB7A, 0xFB7B, 0xFB7C, 0xFB7D}, {0x0687, 0xFB7E, 0xFB7F, 0xFB80, 0xFB81},
    {0x0688, 0xFB88, 0xFB89, 0x0000, 0x0000}, {0x068C, 0xFB84, 0xFB85, 0x0000, 0x0000},
    {0x068D, 0xFB82, 0xFB83, 0x0000, 0x0000}, {0x068E, 0xFB86, 0xFB87, 0x0000, 0x0000},
    {0x0691, 0xFB8C, 0xFB8D, 0x0000, 0x0000}, {0x0698, 0xFB8A, 0xFB8B, 0x0000, 0x0000},
    {0x06A4, 0xFB6A, 0xFB6B, 0xFB6C, 0xFB6D}, {0x06A6, 0xFB6E, 0xFB6F, 0xFB70, 0xFB71},
    {0x06A9, 0xFB8E, 0xFB8F, 0xFB90, 0xFB91}, {0x06AD, 0xFBD3, 0xFBD4, 0xFBD5, 0xFBD6},
    {0x06AF, 0xFB92, 0xFB93, 0xFB94, 0xFB95}, {0x06B1, 0xFB9A, 0xFB9B, 0xFB9C, 0xFB9D},
    {0x06B3, 0xFB96, 0xFB97, 0xFB98, 0xFB99}, {0x06BA, 0xFB9E, 0xFB9F, 0x0000, 0x0000},
    {0x06BB, 0xFBA0, 0xFBA1, 0xFBA2, 0xFBA3}, {0x06BE, 0xFBAA, 0xFBAB, 0xFBAC, 0xFBAD},
    {0x06C0, 0xFBA4, 0xFBA5, 0x0000, 0x0000}, {0x06C1, 0xFBA6, 0xFBA7, 0xFBA8, 0xFBA9},
    {0x06C5, 0xFBE0, 0xFBE1, 0x0000, 0x0000}, {0x06C6, 0xFBD9, 0xFBDA, 0x0000, 0x0000},
    {0x06C7, 0xFBD7, 0xFBD8, 0x0000, 0x0000}, {0x06C8, 0xFBDB, 0xFBDC, 0x0000, 0x0000},
    {0x06C9, 0xFBE2, 0xFBE3, 0x0000, 0x0000}, {0x06CB, 0xFBDE, 0xFBDF, 0x0000, 0x0000},
    {0x06CC, 0xFBFC, 0xFBFD, 0xFBFE, 0xFBFF}, {0x06D0, 0xFBE4, 0xFBE5, 0xFBE6, 0xFBE7},
    {0x06D2, 0xFBAE, 0xFBAF, 0x0000, 0x0000}, {0x06D3, 0xFBB0, 0xFBB1, 0x0000, 0x0000},
};

// Lam followed by one of these alefs is one ligature: the alef, then the
// isolated and final ligature.
constexpr char32_t kLamAlef[][3] = {
    {0x0622, 0xFEF5, 0xFEF6}, {0x0623, 0xFEF7, 0xFEF8}, {0x0625, 0xFEF9, 0xFEFA}, {0x0627, 0xFEFB, 0xFEFC}};
constexpr char32_t kLam = 0x0644;
constexpr char32_t kTatweel = 0x0640;
constexpr char32_t kZeroWidthJoiner = 0x200D;

const ArabicForms* FormsOf(char32_t c) {
    const auto* end = std::end(kArabicForms);
    const auto* it = std::lower_bound(std::begin(kArabicForms), end, c,
                                      [](const ArabicForms& f, char32_t v) { return f.letter < v; });
    return it != end && it->letter == c ? it : nullptr;
}

// Marks that sit on a letter (harakat, Quranic marks, Hebrew points,
// combining diacritics): no joining of their own, drawn with their letter.
bool IsMark(char32_t c) {
    return (c >= 0x0300 && c <= 0x036F) || (c >= 0x0591 && c <= 0x05BD) || c == 0x05BF || c == 0x05C1 ||
           c == 0x05C2 || c == 0x05C4 || c == 0x05C5 || c == 0x05C7 || (c >= 0x0610 && c <= 0x061A) ||
           (c >= 0x064B && c <= 0x065F) || c == 0x0670 || (c >= 0x06D6 && c <= 0x06DC) ||
           (c >= 0x06DF && c <= 0x06E4) || c == 0x06E7 || c == 0x06E8 || (c >= 0x06EA && c <= 0x06ED) ||
           (c >= 0x08D3 && c <= 0x08FF && c != 0x08E2) || (c >= 0xFE20 && c <= 0xFE2F);
}

// Joins the letter after it (dual-joining letters, tatweel, ZWJ).
bool JoinsNext(char32_t c) {
    if (c == kTatweel || c == kZeroWidthJoiner) return true;
    const ArabicForms* f = FormsOf(c);
    return f && (f->initial || f->medial);
}

// Joins the letter before it (dual- and right-joining letters, tatweel, ZWJ).
bool JoinsPrevious(char32_t c) {
    if (c == kTatweel || c == kZeroWidthJoiner) return true;
    const ArabicForms* f = FormsOf(c);
    return f && (f->final || f->medial);
}

// Arabic digits and the signs that go with numbers: the comma, the decimal
// and thousands separators and the percent sign.
bool IsArabicNumberSign(char32_t c) {
    return (c >= 0x0660 && c <= 0x066C) || (c >= 0x06F0 && c <= 0x06F9) || c == 0x060C;
}

bool IsRightToLeftLetter(char32_t c) {
    if (IsMark(c) || IsArabicNumberSign(c)) return false;
    return (c >= 0x0590 && c <= 0x08FF) || (c >= 0xFB1D && c <= 0xFDFF) || (c >= 0xFE70 && c <= 0xFEFF);
}

std::u32string Decode(std::string_view s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const auto b = static_cast<unsigned char>(s[i]);
        int extra = b < 0x80 ? 0 : (b >> 5) == 0x6 ? 1 : (b >> 4) == 0xE ? 2 : (b >> 3) == 0x1E ? 3 : -1;
        char32_t c = extra == 0 ? b : extra == 1 ? (b & 0x1F) : extra == 2 ? (b & 0x0F) : (b & 0x07);
        bool ok = extra >= 0 && i + static_cast<size_t>(std::max(extra, 0)) < s.size();
        for (int k = 1; ok && k <= extra; ++k) {
            const auto n = static_cast<unsigned char>(s[i + static_cast<size_t>(k)]);
            if ((n & 0xC0) != 0x80) ok = false;
            else c = (c << 6) | (n & 0x3F);
        }
        // Overlong forms, surrogates and values past U+10FFFF are not text
        static constexpr char32_t kLeast[] = {0, 0x80, 0x800, 0x10000};
        if (ok && (c < kLeast[extra] || (c >= 0xD800 && c <= 0xDFFF) || c > 0x10FFFF)) ok = false;
        if (!ok) {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        out.push_back(c);
        i += static_cast<size_t>(extra) + 1;
    }
    return out;
}

void Encode(char32_t c, std::string& out) {
    if (c < 0x80) {
        out.push_back(static_cast<char>(c));
    } else if (c < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (c >> 6)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (c >> 12)));
        out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (c >> 18)));
        out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
}

// Arabic letters in their joined forms, in logical order.
std::u32string Shape(const std::u32string& in) {
    std::u32string out;
    out.reserve(in.size());
    const auto neighbour = [&](size_t i, int step) -> long long {
        for (long long k = static_cast<long long>(i) + step; k >= 0 && k < static_cast<long long>(in.size()); k += step)
            if (!IsMark(in[static_cast<size_t>(k)])) return k;
        return -1;
    };
    for (size_t i = 0; i < in.size(); ++i) {
        const char32_t c = in[i];
        const ArabicForms* f = FormsOf(c);
        if (!f) {
            out.push_back(c);
            continue;
        }
        const long long prev = neighbour(i, -1);
        const long long next = neighbour(i, +1);
        const bool joins_prev = prev >= 0 && JoinsNext(in[static_cast<size_t>(prev)]) && JoinsPrevious(c);
        if (c == kLam && next >= 0) {
            const char32_t alef = in[static_cast<size_t>(next)];
            const auto* lig = std::find_if(std::begin(kLamAlef), std::end(kLamAlef),
                                           [alef](const char32_t* row) { return row[0] == alef; });
            if (lig != std::end(kLamAlef)) {
                out.push_back(joins_prev ? (*lig)[2] : (*lig)[1]);
                // Marks between the lam and the alef stay with the ligature
                for (size_t k = i + 1; k < static_cast<size_t>(next); ++k) out.push_back(in[k]);
                i = static_cast<size_t>(next);
                continue;
            }
        }
        const bool joins_next = next >= 0 && JoinsNext(c) && JoinsPrevious(in[static_cast<size_t>(next)]);
        char32_t form = joins_prev && joins_next ? f->medial : joins_prev ? f->final : joins_next ? f->initial : f->isolated;
        if (!form) form = joins_prev && f->final ? f->final : f->isolated;
        out.push_back(form);
    }
    return out;
}

enum class Dir { L, R, Number, Neutral, Mark };

Dir DirOf(char32_t c) {
    if (IsMark(c)) return Dir::Mark;
    if ((c >= '0' && c <= '9') || (c >= 0x0660 && c <= 0x0669) || (c >= 0x06F0 && c <= 0x06F9)) return Dir::Number;
    if (c == 0x066B || c == 0x066C) return Dir::Number;  // Arabic decimal and thousands separators
    if (c == 0x060C || c == 0x066A) return Dir::Neutral;  // Arabic comma and percent sign
    if (IsRightToLeftLetter(c)) return Dir::R;
    if (c < 0x80) return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ? Dir::L : Dir::Neutral;
    if ((c >= 0x0080 && c <= 0x00BF) || c == 0x00D7 || c == 0x00F7 || (c >= 0x2000 && c <= 0x2BFF) ||
        (c >= 0x3000 && c <= 0x303F) || (c >= 0xFE50 && c <= 0xFE6F) || (c >= 0xFF00 && c <= 0xFF20))
        return Dir::Neutral;
    return Dir::L;  // other letters (Latin, Greek, Cyrillic, CJK...)
}

char32_t Mirrored(char32_t c) {
    switch (c) {
        case '(': return ')';
        case ')': return '(';
        case '[': return ']';
        case ']': return '[';
        case '{': return '}';
        case '}': return '{';
        case '<': return '>';
        case '>': return '<';
        case 0x00AB: return 0x00BB;
        case 0x00BB: return 0x00AB;
        default: return c;
    }
}

}  // namespace

bool HasRightToLeft(std::string_view utf8) {
    // Every right-to-left letter is at U+0590 or above: lead byte 0xD6 or more
    if (std::none_of(utf8.begin(), utf8.end(), [](char ch) { return static_cast<unsigned char>(ch) >= 0xD6; }))
        return false;
    const std::u32string text = Decode(utf8);
    return std::any_of(text.begin(), text.end(), IsRightToLeftLetter);
}

std::string DisplayText(std::string_view utf8) {
    if (!HasRightToLeft(utf8)) return std::string(utf8);
    const std::u32string text = Shape(Decode(utf8));
    const size_t n = text.size();

    // Directions: numbers after a left-to-right letter (or at the start of a
    // left-to-right line) are left-to-right text; a separator between two
    // digits and a percent sign beside one belong to the number.
    std::vector<Dir> dir(n);
    for (size_t i = 0; i < n; ++i) dir[i] = DirOf(text[i]);
    Dir base = Dir::L;
    for (Dir d : dir)
        if (d == Dir::L || d == Dir::R) {
            base = d;
            break;
        }
    for (size_t i = 1; i + 1 < n; ++i)
        if (dir[i] == Dir::Neutral && dir[i - 1] == Dir::Number && dir[i + 1] == Dir::Number &&
            (text[i] == '.' || text[i] == ',' || text[i] == ':' || text[i] == '/' || text[i] == 0x060C))
            dir[i] = Dir::Number;
    for (size_t i = 0; i < n; ++i)
        if (text[i] == '%' || text[i] == 0x066A)
            if ((i > 0 && dir[i - 1] == Dir::Number) || (i + 1 < n && dir[i + 1] == Dir::Number)) dir[i] = Dir::Number;
    Dir last_strong = base;
    for (size_t i = 0; i < n; ++i) {
        if (dir[i] == Dir::L || dir[i] == Dir::R) last_strong = dir[i];
        else if (dir[i] == Dir::Number && last_strong == Dir::L) dir[i] = Dir::L;
    }

    // Levels: 0 left to right, 1 right to left, 2 numbers (and left-to-right
    // text) inside right-to-left text. A run of neutrals takes the direction
    // around it when both sides agree (numbers count as right-to-left), else
    // the line's.
    const int base_level = base == Dir::R ? 1 : 0;
    std::vector<int> level(n, base_level);
    const auto level_of = [base_level](Dir d) {
        if (d == Dir::R) return 1;
        if (d == Dir::Number) return 2;
        return base_level == 1 ? 2 : 0;  // L
    };
    for (size_t i = 0; i < n;) {
        if (dir[i] != Dir::Neutral && dir[i] != Dir::Mark) {
            level[i] = level_of(dir[i]);
            ++i;
            continue;
        }
        if (dir[i] == Dir::Mark) {  // a mark goes with the letter before it
            level[i] = i > 0 ? level[i - 1] : base_level;
            ++i;
            continue;
        }
        size_t end = i;
        while (end < n && (dir[end] == Dir::Neutral || (dir[end] == Dir::Mark && end > i))) ++end;
        const auto side = [](Dir d) { return d == Dir::Number ? Dir::R : d; };
        size_t prev = i;
        while (prev > 0 && dir[prev - 1] == Dir::Mark) --prev;  // past the marks to their letter
        const Dir before = prev > 0 ? side(dir[prev - 1]) : base;
        const Dir after = end < n ? side(dir[end]) : base;
        const int resolved = before == after ? (before == Dir::R ? 1 : base_level == 1 ? 2 : 0) : base_level;
        for (size_t k = i; k < end; ++k) level[k] = resolved;
        i = end;
    }

    // Clusters (a letter and the marks on it) keep their inner order.
    std::vector<std::pair<size_t, size_t>> clusters;  // [begin, end)
    for (size_t i = 0; i < n;) {
        size_t end = i + 1;
        while (end < n && dir[end] == Dir::Mark) ++end;
        clusters.emplace_back(i, end);
        i = end;
    }
    std::vector<int> cluster_level(clusters.size());
    for (size_t c = 0; c < clusters.size(); ++c) cluster_level[c] = level[clusters[c].first];
    const int top = *std::max_element(cluster_level.begin(), cluster_level.end());
    std::vector<size_t> order(clusters.size());
    for (size_t c = 0; c < order.size(); ++c) order[c] = c;
    for (int l = top; l >= 1; --l) {
        for (size_t c = 0; c < order.size();) {
            if (cluster_level[order[c]] < l) {
                ++c;
                continue;
            }
            size_t end = c;
            while (end < order.size() && cluster_level[order[end]] >= l) ++end;
            std::reverse(order.begin() + static_cast<long long>(c), order.begin() + static_cast<long long>(end));
            c = end;
        }
    }

    std::string out;
    out.reserve(utf8.size() + 8);
    for (size_t c : order) {
        const auto [b, e] = clusters[c];
        const bool rtl = cluster_level[c] % 2 == 1;
        for (size_t k = b; k < e; ++k) Encode(rtl ? Mirrored(text[k]) : text[k], out);
    }
    return out;
}

}  // namespace cyxwiz::core
