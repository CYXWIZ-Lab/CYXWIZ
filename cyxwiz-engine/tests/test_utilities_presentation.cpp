// What the Utilities window shows (TOFIX129 D12), from real cyxwiz::Utilities
// results: numbers, unit labels and notes, the hash check, the JSON error
// view, regex rows in characters (UTF-8 text), and the random figures.
#include "core/utilities_presentation.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {

namespace view = cyxwiz::utilities_view;
using cyxwiz::Utilities;

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    // Tabs.
    for (const view::Tab tab : view::kTabs) Check(view::TabFromId(view::TabId(tab)) == tab, "tab id round trip");
    Check(view::TabFromId("nonsense") == view::Tab::Calculator, "an unknown id opens the Calculator");

    // Numbers.
    Check(view::FormatNumber(10.283185307179586, 6) == "10.2832", "six significant digits");
    Check(view::FormatNumber(-0.0, 6) == "0", "no -0");
    Check(view::FullNumber(0.1) == "0.1" && view::FullNumber(1.0 / 3.0) == "0.3333333333333333", "full numbers");
    Check(view::FormatNumber(std::numeric_limits<double>::infinity(), 6) == "infinity", "infinity in words");

    // Units.
    const auto speed = view::UnitsOf("speed");
    Check(speed.size() == 5 && speed[1]->id == "kilometer_per_hour", "speed units in table order");
    Check(view::UnitLabel(*speed[1]) == "kilometre per hour (km/h)", "a label with its symbol");
    Check(view::UnitLabel(*Utilities::FindUnit("bit")) == "bit", "a label that is its symbol");
    Check(view::CategoryLabel("temperature") == "Temperature", "category label");
    for (const auto& c : Utilities::GetUnitCategories()) Check(!view::CategoryNote(c).empty(), "a note for " + c);

    // Hash check.
    const auto hello = Utilities::HashText("Hello, World!", "sha256");
    const std::string sha = view::HashOf(hello, "sha256");
    Check(view::CheckHash(" DFFD6021BB2BD5B0AF676290809EC3A53191DD81C7F70A4B28688A362182986F ", sha).kind ==
              view::HashCheck::Kind::Match,
          "upper case and spaces still match");
    const auto differs = view::CheckHash("dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986e", sha);
    Check(differs.kind == view::HashCheck::Kind::Differs && differs.text == "Differs at character 64", "where it differs");
    Check(view::CheckHash("dffd", sha).text == "Differs: 4 characters, the hash has 64", "a short hash");
    Check(view::CheckHash("", sha).kind == view::HashCheck::Kind::Nothing, "nothing to check");
    Check(view::AlgorithmLabel("sha512") == "SHA-512" && view::SizeText(13) == "13 bytes" &&
              view::SizeText(143800000) == "143.8 MB" && view::SizeText(2500) == "2.50 kB",
          "labels and sizes");

    // JSON.
    const auto good = Utilities::ValidateJSON(R"({"b": 1, "a": [true, null]})");
    Check(view::JsonSummary(good) ==
              "1 object \xC2\xB7 1 array \xC2\xB7 2 top-level keys \xC2\xB7 depth 2 \xC2\xB7 45 characters formatted \xC2\xB7 23 minified",
          "summary: " + view::JsonSummary(good));
    const auto bad = Utilities::ValidateJSON("{\n  \"name\": \"CyxWiz\",\n  \"version\": \"1.0.0\"\n  \"gpu\": true\n}");
    const auto err = view::JsonError(bad);
    Check(err.sentence.rfind("Line 4, column 3: ", 0) == 0, "error sentence: " + err.sentence);
    Check(err.lines.size() == 2 && err.lines[0].first == 3 && err.lines[1].second == "  \"gpu\": true" &&
              err.caret_column == 3,
          "the line before, the error's line and the caret");

    // Regex rows in characters. (std::regex reads bytes: \w and \b know ASCII
    // letters only, so "caf\xC3\xA9" would give "caf"; the pattern here does not
    // depend on them.)
    const std::string text = "caf\xC3\xA9 one\nna\xC3\xAFve two";
    const auto r = Utilities::TestRegex("one|two", text, "");
    const auto rows = view::RegexRows(r, text);
    Check(rows.size() == 2 && rows[0].text == "one" && rows[0].line == 1 && rows[0].column == 6 &&
              rows[0].start == 5 && rows[0].end == 8,
          "a match after a two-byte character counts characters");
    Check(rows[1].text == "two" && rows[1].line == 2 && rows[1].column == 7 && rows[1].start == 15,
          "second line");
    Check(view::RegexSummary(r) == "2 matches", "regex summary");
    Check(view::RegexSummary(Utilities::TestRegex("(", text, "")) != "No match", "a bad pattern says why");
    Check(view::RegexSummary(Utilities::TestRegex("zzz", text, "")) == "No match", "no match");
    const auto m = Utilities::TestRegex("^\\w+", "alpha\nbeta", "m");
    Check(view::RegexRows(m, "alpha\nbeta").size() == 2, "multiline anchors");

    // Random.
    const auto n = Utilities::GenerateNormal(1000, 10.0, 2.0, 42);
    const auto figures = view::RandomFigures(n, 6);
    Check(figures.size() == 6 && figures[2].label.rfind("SD (sample", 0) == 0 && figures[3].label == "SD (population, n)",
          "both SDs, named");
    Check(view::RandomTitle(n, {10, 2}) == "1,000 values \xC2\xB7 Normal(10, 2) \xC2\xB7 seed 42",
          "title: " + view::RandomTitle(n, {10, 2}));
    Check(view::ValuesText({1.5, 2.0}, "value") == "value\n1.5\n2\n", "values text");
    Check(view::DistributionLabel("integer") == "Integers", "distribution label");

    std::cout << "utilities presentation: " << checks << " checks passed\n";
    return 0;
}
