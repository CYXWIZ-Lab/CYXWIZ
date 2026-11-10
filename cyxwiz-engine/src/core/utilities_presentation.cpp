#include "utilities_presentation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace cyxwiz::utilities_view {

namespace {

// Characters (UTF-8 code points) in text[from, to).
int CharactersBetween(const std::string& text, size_t from, size_t to) {
    int n = 0;
    for (size_t i = from; i < to && i < text.size(); ++i)
        if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) ++n;
    return n;
}

std::string Plural(size_t n, const char* one, const char* many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

}  // namespace

const char* TabLabel(Tab tab) {
    switch (tab) {
        case Tab::Calculator: return "Calculator";
        case Tab::Units: return "Units";
        case Tab::Random: return "Random";
        case Tab::Hash: return "Hash";
        case Tab::Json: return "JSON";
        case Tab::Regex: return "Regex";
    }
    return "";
}

const char* TabId(Tab tab) {
    switch (tab) {
        case Tab::Calculator: return "calculator";
        case Tab::Units: return "units";
        case Tab::Random: return "random";
        case Tab::Hash: return "hash";
        case Tab::Json: return "json";
        case Tab::Regex: return "regex";
    }
    return "calculator";
}

Tab TabFromId(const std::string& id) {
    for (const Tab tab : kTabs)
        if (id == TabId(tab)) return tab;
    return Tab::Calculator;
}

std::string FormatNumber(double v, int digits) {
    if (std::isnan(v)) return "not a number";
    if (std::isinf(v)) return v > 0 ? "infinity" : "-infinity";
    digits = std::clamp(digits, 1, 17);
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*g", digits, v);
    std::string s = buffer;
    if (s == "-0") s = "0";
    return s;
}

std::string FullNumber(double v) {
    if (!std::isfinite(v)) return FormatNumber(v, 17);
    // Plain digits in the ordinary range ("10", not "1e+01"); an exponent
    // only for very large or very small values.
    const bool plain = v == 0.0 || (std::fabs(v) >= 1e-5 && std::fabs(v) < 1e15);
    for (int digits = 1; digits <= 17; ++digits) {
        const std::string s = FormatNumber(v, digits);
        if (std::stod(s) == v && (!plain || s.find('e') == std::string::npos)) return s;
    }
    return FormatNumber(v, 17);
}

std::string FormatCount(uint64_t n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
    return s;
}

// ---- Units

std::string CategoryLabel(const std::string& category) {
    std::string s = category;
    if (!s.empty()) s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    return s;
}

std::string UnitLabel(const UnitDefinition& unit) {
    return unit.label == unit.symbol ? unit.label : unit.label + " (" + unit.symbol + ")";
}

std::vector<const UnitDefinition*> UnitsOf(const std::string& category) {
    std::vector<const UnitDefinition*> units;
    for (const auto& u : Utilities::Units())
        if (u.category == category) units.push_back(&u);
    return units;
}

std::string CategoryNote(const std::string& category) {
    if (category == "length") return "Exact definitions: inch 0.0254 m, foot 0.3048 m, yard 0.9144 m, mile 1609.344 m, nautical mile 1852 m.";
    if (category == "mass") return "Pound 0.45359237 kg (exact); tonne 1000 kg, short ton 2000 lb, long ton 2240 lb.";
    if (category == "temperature") return "Through kelvin: 0 \xC2\xB0""C = 273.15 K, \xC2\xB0""F and \xC2\xB0""R are 5/9 K. Below absolute zero is refused.";
    if (category == "time") return "A year here is 365 days.";
    if (category == "data") return "SI prefixes are powers of 1000 (kB, MB, GB); binary ones powers of 1024 (KiB, MiB, GiB).";
    if (category == "area") return "From the exact inch, foot, yard and mile; acre 4840 square yards.";
    if (category == "volume") return "US gallon 231 cubic inches (3.785411784 L); imperial gallon 4.54609 L.";
    if (category == "speed") return "km/h is 1000/3600 m/s; a knot 1852/3600 m/s; mph 1609.344/3600 m/s.";
    return "";
}

// ---- Hash

HashCheck CheckHash(const std::string& expected, const std::string& computed) {
    std::string want;
    for (const unsigned char c : expected)
        if (!std::isspace(c)) want += static_cast<char>(std::tolower(c));
    HashCheck check;
    if (want.empty() || computed.empty()) return check;
    for (size_t i = 0; i < std::min(want.size(), computed.size()); ++i) {
        if (want[i] != computed[i]) {
            check.kind = HashCheck::Kind::Differs;
            check.text = "Differs at character " + std::to_string(i + 1);
            return check;
        }
    }
    if (want.size() != computed.size()) {
        check.kind = HashCheck::Kind::Differs;
        check.text = "Differs: " + std::to_string(want.size()) + " characters, the hash has " +
                     std::to_string(computed.size());
        return check;
    }
    check.kind = HashCheck::Kind::Match;
    check.text = "Match";
    return check;
}

std::string HashOf(const HashResult& result, const std::string& algorithm) {
    if (algorithm == "md5") return result.md5_hash;
    if (algorithm == "sha1") return result.sha1_hash;
    if (algorithm == "sha512") return result.sha512_hash;
    return result.sha256_hash;
}

std::string AlgorithmLabel(const std::string& algorithm) {
    if (algorithm == "md5") return "MD5";
    if (algorithm == "sha1") return "SHA-1";
    if (algorithm == "sha256") return "SHA-256";
    if (algorithm == "sha512") return "SHA-512";
    return algorithm;
}

std::string SizeText(uint64_t bytes) {
    if (bytes < 1000) return Plural(static_cast<size_t>(bytes), "byte", "bytes");
    const char* units[] = {"kB", "MB", "GB", "TB"};
    double v = static_cast<double>(bytes);
    int u = -1;
    while (v >= 1000.0 && u < 3) {
        v /= 1000.0;
        ++u;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(v < 10 ? 2 : 1) << v << " " << units[u];
    return oss.str();
}

// ---- JSON

std::string JsonSummary(const JSONResult& r) {
    const std::string dot = " \xC2\xB7 ";
    return Plural(static_cast<size_t>(r.object_count), "object", "objects") + dot +
           Plural(static_cast<size_t>(r.array_count), "array", "arrays") + dot +
           Plural(r.keys.size(), "top-level key", "top-level keys") + dot + "depth " + std::to_string(r.depth) + dot +
           FormatCount(r.formatted_json.size()) + " characters formatted" + dot + FormatCount(r.minified_json.size()) +
           " minified";
}

JsonErrorView JsonError(const JSONResult& result) {
    JsonErrorView view;
    // The parser's reason without its own (end-of-token) place.
    std::string reason = result.error_detail;
    if (const auto dash = reason.find(" - "); dash != std::string::npos) reason = reason.substr(dash + 3);
    else if (const auto colon = reason.rfind(": "); colon != std::string::npos) reason = reason.substr(colon + 2);
    if (result.error_line < 1) {
        view.sentence = reason.empty() ? "Not valid JSON." : reason;
        return view;
    }
    view.sentence = "Line " + std::to_string(result.error_line) + ", column " + std::to_string(result.error_column) +
                    ": " + reason;
    std::istringstream lines(result.input_json);
    std::string line;
    for (int n = 1; std::getline(lines, line); ++n) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (n == result.error_line - 1 || n == result.error_line) view.lines.push_back({n, line});
        if (n >= result.error_line) break;
    }
    view.caret_column = result.error_column;
    return view;
}

// ---- Regex

std::vector<RegexRow> RegexRows(const RegexResult& result, const std::string& text) {
    std::vector<RegexRow> rows;
    rows.reserve(result.matches.size());
    for (const auto& m : result.matches) {
        RegexRow row;
        row.text = m.match_text;
        row.groups = m.groups;
        const size_t start = std::min(static_cast<size_t>(std::max(0, m.start_pos)), text.size());
        const size_t end = std::min(static_cast<size_t>(std::max(0, m.end_pos)), text.size());
        const size_t newline = start == 0 ? std::string::npos : text.rfind('\n', start - 1);
        const size_t line_start = newline == std::string::npos ? 0 : newline + 1;
        row.line = 1 + static_cast<int>(std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(start), '\n'));
        row.column = 1 + CharactersBetween(text, line_start, start);
        row.start = CharactersBetween(text, 0, start);
        row.end = row.start + CharactersBetween(text, start, end);
        rows.push_back(std::move(row));
    }
    return rows;
}

std::string RegexSummary(const RegexResult& result) {
    if (!result.is_valid_pattern) return result.pattern_error.empty() ? result.error_message : result.pattern_error;
    if (result.matches.empty()) return "No match";
    return Plural(result.matches.size(), "match", "matches");
}

// ---- Random

std::string DistributionLabel(const std::string& distribution) {
    if (distribution == "uniform") return "Uniform";
    if (distribution == "normal") return "Normal";
    if (distribution == "exponential") return "Exponential";
    if (distribution == "poisson") return "Poisson";
    if (distribution == "integer") return "Integers";
    return distribution;
}

std::vector<Figure> RandomFigures(const RandomNumberResult& r, int digits) {
    const auto f = [digits](double v) { return FormatNumber(v, digits); };
    return {
        {"Mean", f(r.mean)},
        {"Median", f(r.median)},
        {"SD (sample, n \xE2\x88\x92 1)", f(r.std_dev_sample)},
        {"SD (population, n)", f(r.std_dev_population)},
        {"Q1 / Q3", f(r.q1) + " / " + f(r.q3)},
        {"Min / Max", f(r.min_value) + " / " + f(r.max_value)},
    };
}

std::string RandomTitle(const RandomNumberResult& r, const std::vector<double>& p) {
    const std::string dot = " \xC2\xB7 ";
    std::string params;
    for (size_t i = 0; i < p.size(); ++i) params += (i ? ", " : "") + FullNumber(p[i]);
    return FormatCount(static_cast<uint64_t>(r.values.size())) + " values" + dot + DistributionLabel(r.distribution) + "(" +
           params + ")" + dot + "seed " + std::to_string(r.seed_used);
}

std::string ValuesText(const std::vector<double>& values, const std::string& header) {
    std::string s;
    if (!header.empty()) s = header + "\n";
    for (const double v : values) s += FullNumber(v) + "\n";
    return s;
}

}  // namespace cyxwiz::utilities_view
