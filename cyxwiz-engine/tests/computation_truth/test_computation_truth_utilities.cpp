// cyxwiz::Utilities against Python (TOFIX129 D12): hashlib digests of texts and
// a file read in blocks, unit conversions against the exact definitions, JSON
// statistics and error places against Python's json, regex matches against
// Python's re, and the random generators' values, statistics and histograms
// against numpy. Fixture: reference/generate_utilities_fixtures.py.
#include <cyxwiz/utilities.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef CYXWIZ_UTILITIES_FIXTURE
#error "CYXWIZ_UTILITIES_FIXTURE must name the fixture file"
#endif

namespace {

using cyxwiz::Utilities;
using json = nlohmann::json;
namespace fs = std::filesystem;

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

bool Near(double got, double want, double rel, double abs_tol = 0.0) {
    return std::fabs(got - want) <= abs_tol + rel * std::max(1.0, std::fabs(want));
}

void CheckHashes(const json& fixture) {
    for (const auto& c : fixture.at("texts")) {
        const std::string text = c.at("text").get<std::string>();
        const auto r = Utilities::HashText(text, "all");
        Check(r.success, "hash: " + r.error_message);
        for (const char* a : {"md5", "sha1", "sha256", "sha512"}) {
            const std::string got = std::string(a) == "md5" ? r.md5_hash : std::string(a) == "sha1" ? r.sha1_hash
                                  : std::string(a) == "sha256" ? r.sha256_hash : r.sha512_hash;
            Check(got == c.at(a).get<std::string>(), std::string(a) + " of \"" + text.substr(0, 20) + "\"");
        }
        Check(Utilities::HashText(text, "sha256").md5_hash.empty(), "only the asked digest");
    }
    // A file over several 1 MiB blocks, with progress and a stop.
    const auto& f = fixture.at("file");
    const auto bytes = f.at("bytes").get<size_t>();
    const fs::path path = fs::temp_directory_path() / "cyxwiz_utilities_hash_truth.bin";
    {
        std::ofstream out(path, std::ios::binary);
        for (size_t i = 0; i < bytes; ++i) out.put(static_cast<char>((i * 31 + 7) % 256));
    }
    uint64_t last_done = 0, last_total = 0;
    int calls = 0;
    const auto r = Utilities::HashFile(path.string(), "all", [&](uint64_t done, uint64_t total) {
        ++calls;
        last_done = done;
        last_total = total;
        return true;
    });
    Check(r.success && r.input_size == bytes, "file hash: " + r.error_message);
    for (const char* a : {"md5", "sha1", "sha256", "sha512"}) {
        const std::string got = std::string(a) == "md5" ? r.md5_hash : std::string(a) == "sha1" ? r.sha1_hash
                              : std::string(a) == "sha256" ? r.sha256_hash : r.sha512_hash;
        Check(got == f.at(a).get<std::string>(), std::string("file ") + a);
    }
    Check(calls == 4 && last_done == bytes && last_total == bytes, "progress per 1 MiB block, ending at the size");
    const auto stopped = Utilities::HashFile(path.string(), "sha256", [](uint64_t, uint64_t) { return false; });
    Check(!stopped.success && stopped.error_message == "Stopped" && stopped.sha256_hash.empty(), "a stop gives no hash");
    fs::remove(path);
    const auto unknown = Utilities::HashText("abc", "sha3");
    Check(!unknown.success && unknown.sha256_hash.empty() && !unknown.error_message.empty(),
          "an unknown algorithm is refused, not shown as a hash");
    Check(!Utilities::HashFile((fs::temp_directory_path() / "no_such_cyxwiz_file.bin").string()).success,
          "a missing file is refused");
}

void CheckUnits(const json& fixture) {
    size_t cases = 0;
    for (const auto& c : fixture.at("cases")) {
        const double value = c.at("value").get<double>();
        const std::string from = c.at("from").get<std::string>(), to = c.at("to").get<std::string>();
        const auto r = Utilities::ConvertUnit(value, from, to);
        Check(r.success, "convert " + from + " -> " + to + ": " + r.error_message);
        Check(Near(r.output_value, c.at("expected").get<double>(), 1e-12, 1e-12),
              std::to_string(value) + " " + from + " -> " + to + ": " + std::to_string(r.output_value) + " vs " +
                  std::to_string(c.at("expected").get<double>()));
        ++cases;
    }
    Check(cases > 3000, "every unit pair in every category");
    for (const auto& c : fixture.at("refusals")) {
        const auto r = Utilities::ConvertUnit(c.at("value").get<double>(), c.at("from").get<std::string>(),
                                              c.at("to").get<std::string>());
        Check(!r.success && r.error_message.find(c.at("reason").get<std::string>()) != std::string::npos,
              "refused: " + c.at("reason").get<std::string>() + " (got: " + r.error_message + ")");
    }
    for (const auto& a : fixture.at("aliases")) {
        const auto* u = Utilities::FindUnit(a[0].get<std::string>(), a[1].get<std::string>());
        Check(u != nullptr && u->id == a[2].get<std::string>(), "alias " + a[0].get<std::string>());
    }
    // Every name finds its own unit and no other.
    for (const auto& u : Utilities::Units()) {
        for (const auto& name : std::vector<std::string>{u.id, u.label, u.symbol}) {
            const auto* found = Utilities::FindUnit(name, u.category);
            Check(found == &u, "the name '" + name + "' finds " + u.id);
        }
        for (const auto& alias : u.aliases)
            Check(Utilities::FindUnit(alias, u.category) == &u, "the alias '" + alias + "' finds " + u.id);
    }
    const auto all = Utilities::ConvertToAllUnits(1.0, "kilometer");
    Check(all.success && all.all_conversions.size() == Utilities::GetUnitsForCategory("length").size(),
          "all length units from one value");
}

void CheckJson(const json& fixture) {
    for (const auto& c : fixture.at("valid")) {
        const std::string text = c.at("text").get<std::string>();
        for (const auto& r : {Utilities::ValidateJSON(text), Utilities::FormatJSON(text, 4), Utilities::MinifyJSON(text)}) {
            Check(r.success && r.is_valid, "valid json: " + text.substr(0, 30));
            Check(r.object_count == c.at("objects").get<int>() && r.array_count == c.at("arrays").get<int>() &&
                      r.string_count == c.at("strings").get<int>() && r.number_count == c.at("numbers").get<int>() &&
                      r.bool_count == c.at("bools").get<int>() && r.null_count == c.at("nulls").get<int>() &&
                      r.depth == c.at("depth").get<int>(),
                  "json statistics (also after Format and Minify): " + text.substr(0, 30));
            Check(json::parse(r.minified_json) == json::parse(c.at("minified").get<std::string>()),
                  "minified text parses to the same value");
            Check(r.minified_json.find('\n') == std::string::npos, "minified has no line breaks");
        }
        Check(Utilities::ValidateJSON(text).keys == c.at("keys").get<std::vector<std::string>>(), "top-level keys");
    }
    for (const auto& c : fixture.at("invalid")) {
        const std::string text = c.at("text").get<std::string>();
        const auto r = Utilities::ValidateJSON(text);
        Check(!r.is_valid && !r.success, "invalid json refused");
        Check(r.error_line == c.at("line").get<int>() && r.error_column == c.at("column").get<int>(),
              "error place of " + json(text).dump() + ": line " + std::to_string(r.error_line) + ", column " +
                  std::to_string(r.error_column) + " (Python: line " + std::to_string(c.at("line").get<int>()) +
                  ", column " + std::to_string(c.at("column").get<int>()) + ")");
        Check(Utilities::MinifyJSON(text).error_line == r.error_line, "Minify names the same place");
    }
}

void CheckRegex(const json& fixture) {
    const std::string text = fixture.at("text").get<std::string>();
    for (const auto& c : fixture.at("cases")) {
        const std::string pattern = c.at("pattern").get<std::string>(), flags = c.at("flags").get<std::string>();
        const auto r = Utilities::TestRegex(pattern, text, flags);
        const auto& want = c.at("matches");
        Check(r.success && r.is_valid_pattern, "regex " + pattern + ": " + r.error_message + r.pattern_error);
        Check(r.matches.size() == want.size(), "regex " + pattern + " /" + flags + ": " +
                                                   std::to_string(r.matches.size()) + " matches, Python " +
                                                   std::to_string(want.size()));
        for (size_t i = 0; i < want.size(); ++i) {
            const auto& m = r.matches[i];
            Check(m.match_text == want[i].at("text").get<std::string>() && m.start_pos == want[i].at("start").get<int>() &&
                      m.end_pos == want[i].at("end").get<int>(),
                  "regex " + pattern + " match " + std::to_string(i));
            const auto groups = want[i].at("groups").get<std::vector<std::string>>();
            Check(m.groups == groups, "regex " + pattern + " groups of match " + std::to_string(i));
        }
    }
    const auto& rep = fixture.at("replace");
    const auto r = Utilities::ReplaceRegex(rep.at("pattern").get<std::string>(), text,
                                           rep.at("replacement").get<std::string>(), "");
    Check(r.success && r.replaced_text == rep.at("expected").get<std::string>(), "replace with groups");
}

void CheckRandom(const json& cases) {
    for (const auto& c : cases) {
        const std::string kind = c.at("kind").get<std::string>();
        const int count = c.at("count").get<int>();
        const double a = c.at("a").get<double>(), b = c.at("b").get<double>();
        const int64_t seed = c.at("seed").get<int64_t>();
        cyxwiz::RandomNumberResult r;
        if (kind == "uniform") r = Utilities::GenerateUniform(count, a, b, seed);
        else if (kind == "normal") r = Utilities::GenerateNormal(count, a, b, seed);
        else if (kind == "exponential") r = Utilities::GenerateExponential(count, a, seed);
        else if (kind == "poisson") r = Utilities::GeneratePoisson(count, a, seed);
        else r = Utilities::GenerateIntegers(count, static_cast<int>(a), static_cast<int>(b), seed);
        const std::string name = kind + " seed " + std::to_string(seed);
        Check(r.success && r.values.size() == static_cast<size_t>(count), name + ": " + r.error_message);
        const auto first = c.at("first").get<std::vector<double>>();
        for (size_t i = 0; i < first.size(); ++i)
            Check(Near(r.values[i], first[i], 1e-13), name + ": value " + std::to_string(i));
        Check(Near(r.mean, c.at("mean").get<double>(), 1e-12), name + ": mean");
        Check(Near(r.std_dev_population, c.at("std_population").get<double>(), 1e-10), name + ": SD (n)");
        Check(Near(r.std_dev_sample, c.at("std_sample").get<double>(), 1e-10), name + ": SD (n - 1)");
        Check(Near(r.q1, c.at("q1").get<double>(), 1e-12) && Near(r.median, c.at("median").get<double>(), 1e-12) &&
                  Near(r.q3, c.at("q3").get<double>(), 1e-12),
              name + ": quartiles as numpy.percentile");
        Check(r.min_value == c.at("min").get<double>() && r.max_value == c.at("max").get<double>(), name + ": range");
        const auto edges = c.at("edges").get<std::vector<double>>();
        Check(r.histogram_edges.size() == edges.size(), name + ": bins");
        for (size_t i = 0; i < edges.size(); ++i) Check(Near(r.histogram_edges[i], edges[i], 1e-12), name + ": edge");
        Check(r.histogram_counts == c.at("counts").get<std::vector<int>>(), name + ": counts as numpy.histogram");
    }
    const auto ids = Utilities::GenerateUUIDs(3);
    Check(ids.size() == 3 && ids[0] != ids[1], "UUIDs");
}

}  // namespace

int main() {
    std::ifstream in(CYXWIZ_UTILITIES_FIXTURE);
    Check(static_cast<bool>(in), "cannot open the utilities fixture");
    const json fixture = json::parse(in);
    CheckHashes(fixture.at("hash"));
    CheckUnits(fixture.at("units"));
    CheckJson(fixture.at("json"));
    CheckRegex(fixture.at("regex"));
    CheckRandom(fixture.at("random"));
    std::cout << "utilities match hashlib, json, re, numpy and the unit definitions: " << checks << " checks\n";
    return 0;
}
