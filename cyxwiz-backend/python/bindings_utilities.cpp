#include "bindings_utilities.h"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "cyxwiz/utilities.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pycyxwiz_bindings {
namespace {

using cyxwiz::Utilities;

py::dict UnitDict(const cyxwiz::UnitDefinition& u) {
    return py::dict("id"_a = u.id, "label"_a = u.label, "symbol"_a = u.symbol, "category"_a = u.category,
                    "scale"_a = u.scale, "offset"_a = u.offset, "aliases"_a = u.aliases);
}

// Characters (UTF-8 code points) in text[from, to).
int Characters(const std::string& text, size_t from, size_t to) {
    int n = 0;
    for (size_t i = from; i < to && i < text.size(); ++i)
        if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) ++n;
    return n;
}

std::string Flags(bool ignore_case, bool multiline) {
    return std::string(ignore_case ? "i" : "") + (multiline ? "m" : "");
}

py::dict RandomDict(const cyxwiz::RandomNumberResult& r) {
    if (!r.success) throw py::value_error(r.error_message);
    return py::dict("values"_a = r.values, "seed"_a = r.seed_used, "mean"_a = r.mean,
                    "sd_sample"_a = r.std_dev_sample, "sd_population"_a = r.std_dev_population, "median"_a = r.median,
                    "q1"_a = r.q1, "q3"_a = r.q3, "min"_a = r.min_value, "max"_a = r.max_value,
                    "edges"_a = r.histogram_edges, "counts"_a = r.histogram_counts);
}

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

void BindUtilities(py::module_& m) {
    auto u = m.def_submodule(
        "utilities",
        "Tools > Utilities from Python: units, hashes, JSON, regex, seeded random values, UUIDs, the calculator");

    // ---- Calculator
    u.def(
        "evaluate",
        [](const std::string& expression, const std::map<std::string, double>& variables, bool degrees) {
            const auto r = Utilities::Evaluate(expression, variables, degrees ? "degrees" : "radians");
            if (!r.success) throw py::value_error(r.error_message);
            return r.result;
        },
        "expression"_a, "variables"_a = std::map<std::string, double>{}, "degrees"_a = false,
        "The value of an expression (the Calculator's parser, as calculus.Expression); variables by name.");

    // ---- Units
    u.def("unit_categories", &Utilities::GetUnitCategories, "length, mass, temperature, time, data, area, volume, speed");
    u.def(
        "units",
        [](const std::string& category) {
            py::list out;
            for (const auto& unit : Utilities::Units())
                if (category.empty() || unit.category == category) out.append(UnitDict(unit));
            return out;
        },
        "category"_a = "",
        "The unit table (id, label, symbol, category, scale, offset, aliases): base = value * scale + offset.");
    u.def(
        "find_unit",
        [](const std::string& name, const std::string& category) -> py::object {
            const auto* unit = Utilities::FindUnit(name, category);
            return unit ? py::object(UnitDict(*unit)) : py::object(py::none());
        },
        "name"_a, "category"_a = "", "A unit by id, label, symbol or alias, or None.");
    u.def(
        "convert",
        [](double value, const std::string& from_unit, const std::string& to_unit) {
            const auto r = Utilities::ConvertUnit(value, from_unit, to_unit);
            if (!r.success) throw py::value_error(r.error_message);
            return r.output_value;
        },
        "value"_a, "from_unit"_a, "to_unit"_a,
        "value converted between two units of one category (exact definitions; below absolute zero is refused).");
    u.def(
        "conversion_formula",
        [](const std::string& from_unit, const std::string& to_unit) {
            const auto r = Utilities::ConvertUnit(1.0, from_unit, to_unit);
            if (!r.success) throw py::value_error(r.error_message);
            return r.formula;
        },
        "from_unit"_a, "to_unit"_a, "The conversion as text, e.g. '\xC2\xB0""F = \xC2\xB0""C x 1.8 + 32'.");

    // ---- Hash
    u.def(
        "hash",
        [](const py::object& data, const std::string& algorithm) {
            const std::string bytes = py::isinstance<py::bytes>(data) ? data.cast<std::string>() : py::str(data).cast<std::string>();
            const auto r = Utilities::HashText(bytes, Lower(algorithm));
            if (!r.success) throw py::value_error(r.error_message);
            const std::string a = Lower(algorithm);
            return a == "md5" ? r.md5_hash : a == "sha1" ? r.sha1_hash : a == "sha512" ? r.sha512_hash : r.sha256_hash;
        },
        "data"_a, "algorithm"_a = "sha256",
        "The hex digest of text (UTF-8) or bytes: md5, sha1, sha256 or sha512 (OpenSSL).");
    u.def(
        "hash_file",
        [](const std::string& path, const std::string& algorithm,
           const std::function<bool(uint64_t, uint64_t)>& progress) {
            const std::string a = Lower(algorithm);
            const auto r = Utilities::HashFile(path, a, progress);
            if (!r.success) throw py::value_error(r.error_message);
            return a == "md5" ? r.md5_hash : a == "sha1" ? r.sha1_hash : a == "sha512" ? r.sha512_hash : r.sha256_hash;
        },
        "path"_a, "algorithm"_a = "sha256", "progress"_a = nullptr,
        "The hex digest of a file read in 1 MB blocks; progress(done, total) returning False stops (ValueError 'Stopped').");

    // ---- JSON
    u.def(
        "json_check",
        [](const std::string& text, int indent) {
            const auto r = Utilities::FormatJSON(text, indent);
            py::dict out("valid"_a = r.is_valid);
            if (!r.is_valid) {
                out["line"] = r.error_line;
                out["column"] = r.error_column;
                out["error"] = r.error_detail;
                return out;
            }
            out["formatted"] = r.formatted_json;
            out["minified"] = r.minified_json;
            out["keys"] = r.keys;
            out["depth"] = r.depth;
            out["objects"] = r.object_count;
            out["arrays"] = r.array_count;
            out["strings"] = r.string_count;
            out["numbers"] = r.number_count;
            out["bools"] = r.bool_count;
            out["nulls"] = r.null_count;
            return out;
        },
        "text"_a, "indent"_a = 2,
        "Whether text is JSON: formatted and minified text (keys in the document's order) and the counts, or the "
        "error's line and column (where Python's json puts it).");
    u.def(
        "json_get",
        [](const std::string& text, const std::string& path) {
            const auto found = Utilities::LookUpJSON(text, path);
            if (!found.found) throw py::key_error(found.error);
            return py::module_::import("json").attr("loads")(found.value_json);
        },
        "text"_a, "path"_a,
        "The Python value at a path such as 'config.max_workers' or 'items[0].name' (KeyError when it is not there).");

    // ---- Regex
    u.def(
        "regex_find",
        [](const std::string& pattern, const std::string& text, bool ignore_case, bool multiline) {
            const auto r = Utilities::TestRegex(pattern, text, Flags(ignore_case, multiline));
            if (!r.is_valid_pattern) throw py::value_error(r.pattern_error.empty() ? r.error_message : r.pattern_error);
            py::list out;
            for (const auto& match : r.matches) {
                const auto start = static_cast<size_t>(std::max(0, match.start_pos));
                const auto end = static_cast<size_t>(std::max(0, match.end_pos));
                const size_t newline = start == 0 ? std::string::npos : text.rfind('\n', start - 1);
                const size_t line_start = newline == std::string::npos ? 0 : newline + 1;
                const int chars = Characters(text, 0, start);
                out.append(py::dict("text"_a = match.match_text, "start"_a = chars,
                                    "end"_a = chars + Characters(text, start, end),
                                    "line"_a = 1 + static_cast<int>(std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(std::min(start, text.size())), '\n')),
                                    "column"_a = 1 + Characters(text, line_start, start), "groups"_a = match.groups));
            }
            return out;
        },
        "pattern"_a, "text"_a, "ignore_case"_a = false, "multiline"_a = false,
        "The matches (ECMAScript, C++ std::regex; \\w and \\b know ASCII letters): text, start / end in characters "
        "(as Python's re), line, column, groups.");
    u.def(
        "regex_replace",
        [](const std::string& pattern, const std::string& text, const std::string& replacement, bool ignore_case,
           bool multiline) {
            const auto r = Utilities::ReplaceRegex(pattern, text, replacement, Flags(ignore_case, multiline));
            if (!r.success) throw py::value_error(r.pattern_error.empty() ? r.error_message : r.pattern_error);
            return r.replaced_text;
        },
        "pattern"_a, "text"_a, "replacement"_a, "ignore_case"_a = false, "multiline"_a = false,
        "Every match replaced: $& the match, $1 .. $9 its groups (ECMAScript).");

    // ---- Random
    u.def(
        "random",
        [](const std::string& distribution, int count, double a, double b, int64_t seed) {
            const std::string d = Lower(distribution);
            if (d == "uniform") return RandomDict(Utilities::GenerateUniform(count, a, b, seed));
            if (d == "normal") return RandomDict(Utilities::GenerateNormal(count, a, b, seed));
            if (d == "exponential") return RandomDict(Utilities::GenerateExponential(count, a, seed));
            if (d == "poisson") return RandomDict(Utilities::GeneratePoisson(count, a, seed));
            if (d == "integers")
                return RandomDict(Utilities::GenerateIntegers(count, static_cast<int>(a), static_cast<int>(b), seed));
            throw py::value_error("distribution is uniform, normal, exponential, poisson or integers");
        },
        "distribution"_a, "count"_a, "a"_a = 0.0, "b"_a = 1.0, "seed"_a = -1,
        "Seeded values (the same on every platform): uniform(a=low, b=high), normal(a=mean, b=sd), exponential(a=rate), "
        "poisson(a=mean), integers(a=low, b=high). Returns the values, mean, both SDs, median, quartiles, range and "
        "histogram edges / counts; seed -1 draws one (returned as 'seed').");
    u.def("uuid4", &Utilities::GenerateUUIDs, "count"_a = 1,
          "Version 4 UUIDs, every random bit from the operating system's random source.");
}

}  // namespace pycyxwiz_bindings
