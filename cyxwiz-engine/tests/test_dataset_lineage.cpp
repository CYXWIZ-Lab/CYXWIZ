// Dataset lineage (TOFIX148 Publish): the file written beside an export, on
// the Berean tier1 recipe (Data Input -> SQL step -> 4 checks -> Export).

#include "../src/core/dataset_lineage.h"
#include "../src/core/sha256_digest.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Same(const std::string& got, const std::string& want, const std::string& what) {
    Check(got == want, what + ": got \"" + got + "\", want \"" + want + "\"");
}

LineageStep Step(int id, std::string type, std::string name, std::string recipe, std::map<std::string, std::string> p,
                 std::vector<int> inputs, std::string outcome = {}) {
    LineageStep s;
    s.id = id;
    s.type = std::move(type);
    s.name = std::move(name);
    s.recipe = std::move(recipe);
    s.parameters = std::move(p);
    s.inputs = std::move(inputs);
    s.outcome = std::move(outcome);
    return s;
}
}  // namespace

int main() {
    std::vector<LineageStep> steps{
        Step(50, "DataInput", "Tier 1 books (ZIP text)", "", {{"file_path", "datasets/tier1_books.zip"}}, {}),
        Step(101, "SQLQuery", "tier1_paragraphs", "tier1_paragraphs", {{"query", "SELECT * FROM input"}}, {50}),
        Step(102, "RowCountCheck", "Check: paragraphs", "tier1_paragraphs", {{"expected_rows", "15349"}}, {101},
             "passed: 15349 rows (expected exactly 15349)"),
        Step(103, "RowCountCheck", "Check: Law", "tier1_paragraphs", {{"expected_rows", "1942"}}, {102},
             "passed: 1942 rows where source_id = 'ccel_law_serious_call' (expected exactly 1942)"),
    };

    // The recipe's revision follows its steps' settings, not their names or outcomes.
    {
        const std::string rev = RecipeRevision(steps, "tier1_paragraphs");
        Check(rev.size() == 12, "a 12-digit revision");
        auto renamed = steps;
        renamed[2].name = "Check: rows";
        renamed[2].outcome = "something else";
        Same(RecipeRevision(renamed, "tier1_paragraphs"), rev, "names and outcomes do not change the revision");
        auto edited = steps;
        edited[1].parameters["query"] = "SELECT * FROM input WHERE 1 = 1";
        Check(RecipeRevision(edited, "tier1_paragraphs") != rev, "a changed query changes the revision");
        Same(RecipeRevision(steps, ""), "", "no recipe, no revision");
        Same(RecipeRevision(steps, "other"), "", "a recipe without steps here");
    }

    // Checks: passed by their outcome.
    {
        auto checks = LineageChecks(steps);
        Check(checks.size() == 2 && checks[0].passed && checks[1].passed, "two passed checks");
        auto failed = steps;
        failed[3].outcome = "";
        checks = LineageChecks(failed);
        Check(!checks[1].passed, "a check with no pass outcome did not pass");
    }

    // Written, read back, summarized.
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_test_dataset_lineage";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path out = dir / "tier1_paragraphs.parquet";
    {
        std::ofstream f(out, std::ios::binary);
        f << "PAR1-test-bytes";
    }
    DatasetLineage l;
    l.written_at = "2026-10-15 14:28:41";
    l.engine = "CyxWiz Engine";
    l.graph = "D:/Project Berean/cyxgraph/preprocessing_v3/berean_v3_tier1_paragraphs.cyxgraph";
    l.output.path = out.string();
    l.output.bytes = fs::file_size(out);
    std::string error;
    Check(Sha256File(out, l.output.sha256, error), "hash the output: " + error);
    l.format = "parquet";
    l.rows = 15349;
    l.columns = 13;
    l.schema = {{"source_id", "string"}, {"word_count", "int64"}};
    l.inputs = {{"D:/Project Berean/datasets/tier1_books.zip", 2202009, std::string(64, 'a')}};
    l.steps = steps;
    l.recipe = "tier1_paragraphs";
    l.recipe_revision = RecipeRevision(steps, l.recipe);
    l.checks = LineageChecks(steps);
    Check(WriteLineage(l, error), "write: " + error);
    Check(fs::exists(LineagePathFor(out.string())), "the file is beside the output");
    Check(!fs::exists(LineagePathFor(out.string()) + ".tmp"), "no temporary file left");

    DatasetLineage back;
    Check(ReadLineage(out.string(), back, error), "read: " + error);
    Same(back.recipe_revision, l.recipe_revision, "revision round trip");
    Check(back.steps.size() == 4 && back.steps[1].parameters.at("query") == "SELECT * FROM input", "steps round trip");
    Check(back.schema.size() == 2 && back.schema[1].second == "int64", "schema round trip");
    Same(back.output.sha256, l.output.sha256, "output digest round trip");

    const LineageSummary s = SummarizeLineage(back);
    Same(s.made_by, "recipe tier1_paragraphs, revision " + l.recipe_revision + " (4 steps)", "made by");
    Same(s.inputs, "tier1_books.zip \xC2\xB7 2.1 MB \xC2\xB7 sha256 aaaa\xE2\x80\xA6" "aa", "inputs");
    Check(s.result.rfind("15,349 rows \xC2\xB7 13 columns \xC2\xB7 15 B \xC2\xB7 sha256 ", 0) == 0, "result: " + s.result);
    Same(s.checks, "2 of 2 pass", "checks");
    Check(s.checks_ok, "checks ok");
    Same(s.written, "2026-10-15 14:28:41 from berean_v3_tier1_paragraphs.cyxgraph", "written");

    // A large input not read to hash it (TOFIX157): its note and write time
    // round trip, and the summary says so.
    {
        DatasetLineage large = l;
        large.inputs = {{"D:/data/big.parquet", 6000000000ull, "", "2026-10-16 23:00:00",
                         "not hashed: larger than the data budget's result threshold (1.00 GB); size and write time identify it"}};
        Check(WriteLineage(large, error), "write the large input: " + error);
        DatasetLineage read;
        Check(ReadLineage(out.string(), read, error), "read the large input: " + error);
        Check(read.inputs.size() == 1 && read.inputs[0].sha256.empty() && read.inputs[0].modified == "2026-10-16 23:00:00" &&
                  read.inputs[0].note == large.inputs[0].note,
              "note and write time round trip");
        Same(SummarizeLineage(read).inputs, "big.parquet \xC2\xB7 5.59 GB \xC2\xB7 not hashed (large file; size and time identify it)",
             "summary of an input not hashed");
        Check(WriteLineage(l, error), "write back: " + error);
    }

    back.checks[1].passed = false;
    const LineageSummary f = SummarizeLineage(back);
    Same(f.checks, "1 of 2 pass: 'Check: Law' did not", "a failed check named");
    Check(!f.checks_ok, "not ok");

    // A file that is not lineage is refused.
    {
        std::ofstream bad(LineagePathFor(out.string()), std::ios::binary | std::ios::trunc);
        bad << "{\"something\": 1}";
    }
    DatasetLineage none;
    Check(!ReadLineage(out.string(), none, error) && error.find("lineage_version") != std::string::npos, "refused: " + error);
    Check(!ReadLineage((dir / "missing.parquet").string(), none, error), "no file");
    fs::remove_all(dir);

    std::cout << "test_dataset_lineage: all checks passed\n";
    return 0;
}
