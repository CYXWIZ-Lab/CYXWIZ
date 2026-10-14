// Data Studio's shell model (TOFIX148 step 3): jobs, rail, node strip and the
// last run, on the Berean tier1 graph as it runs in the Engine (Data Input ->
// recipe tier1_paragraphs [SQL step + 4 Row Count Checks] -> Export Parquet).

#include "../src/core/data_studio_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::studio;
using gui::NodeType;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

GraphNode Node(int id, NodeType type, std::string name) {
    GraphNode n;
    n.id = id;
    n.type = type;
    n.name = std::move(name);
    return n;
}

GraphNode Ran(GraphNode n, int64_t rows, int64_t cols, double seconds, std::string at, std::string dataset,
              std::string outcome = "") {
    n.ran = true;
    n.rows = rows;
    n.columns = cols;
    n.seconds = seconds;
    n.finished_at = std::move(at);
    n.output_dataset = std::move(dataset);
    n.outcome = std::move(outcome);
    return n;
}

const JobView& JobOfView(const View& v, Job job) { return v.jobs[static_cast<size_t>(job)]; }
}  // namespace

int main() {
    Input in;
    in.graph_name = "berean_v3_tier1_paragraphs";

    GraphNode input = Node(157, NodeType::DataInput, "Tier 1 books (ZIP text)");
    input.file_path = "datasets/staging/base_expansion_v3/archives/tier1_books.zip";
    GraphNode recipe = Node(119, NodeType::Subgraph, "tier1_paragraphs");
    recipe.recipe = true;
    recipe.recipe_steps = 5;
    GraphNode out = Node(198, NodeType::ExportParquet, "Export tier1_paragraphs.parquet");
    out.file_path = "datasets/staging/base_expansion_v3/tier1_paragraphs.parquet";

    // Before any run: nothing loaded, checks not run.
    {
        Input fresh = in;
        fresh.nodes = {input, recipe, Node(162, NodeType::SQLQuery, "tier1_paragraphs"),
                       Node(347, NodeType::RowCountCheck, "Check: paragraphs"), out};
        const View v = Build(fresh);
        Check(v.graph_label == "berean_v3_tier1_paragraphs \xC2\xB7 recipe tier1_paragraphs", "graph line: " + v.graph_label);
        Check(JobOfView(v, Job::Validate).state == "1 check, not run", "checks not run: " + JobOfView(v, Job::Validate).state);
        Check(v.rail[0].items.size() == 1 && v.rail[0].items[0].meta == "not loaded" && v.rail[0].items[0].tone == Tone::Warn,
              "a Data Input not loaded says so");
        Check(v.rail[2].items[0].label == "tier1_paragraphs.parquet" && v.rail[2].items[0].meta == "not written",
              "an export not written yet");
        Check(v.last_run.empty(), "no last run yet");
    }

    // After the run (the Engine's numbers, 2026-10-14).
    in.nodes = {
        Ran(input, 7, 6, 0.7, "12:39:01", "ds_datainput_157"),
        recipe,
        Ran(Node(162, NodeType::SQLQuery, "tier1_paragraphs"), 15349, 13, 3.1, "12:39:04", "ds_sql_162"),
        Ran(Node(347, NodeType::RowCountCheck, "Check: paragraphs"), 15349, 13, 0.1, "12:39:05", "ds_check_347", "passed: 15349 rows"),
        Ran(Node(348, NodeType::RowCountCheck, "Check: Baxter Part 1"), 15349, 13, 0.1, "12:39:05", "ds_check_348", "passed: 3712 rows"),
        Ran(Node(349, NodeType::RowCountCheck, "Check: Law"), 15349, 13, 0.1, "12:39:05", "ds_check_349", "passed: 1942 rows"),
        Ran(Node(350, NodeType::RowCountCheck, "Check: nothing admitted"), 15349, 13, 0.1, "12:39:05", "ds_check_350", "passed: 0 rows"),
        Ran(out, 15349, 13, 0.4, "12:39:06", ""),
        Node(400, NodeType::Dense, "a layer no job shows"),
    };
    in.datasets = {{"ds_datainput_157", "Tier 1 books (ZIP text)", 7, 6, true},
                   {"ds_sql_162", "", 15349, 13, true},
                   {"query_result", "", 4, 3, true},
                   {"images_1", "", 10, 0, false}};
    const View v = Build(in);

    Check(JobOfView(v, Job::BringIn).state == "1 Data Input", "Bring in: " + JobOfView(v, Job::BringIn).state);
    Check(JobOfView(v, Job::Prepare).state == "1 recipe", "Prepare: " + JobOfView(v, Job::Prepare).state);
    Check(JobOfView(v, Job::Validate).state == "4 of 4 pass", "Validate: " + JobOfView(v, Job::Validate).state);
    Check(JobOfView(v, Job::Ready).state == "no split", "ML readiness: " + JobOfView(v, Job::Ready).state);
    Check(JobOfView(v, Job::Publish).state == "1 output", "Publish: " + JobOfView(v, Job::Publish).state);
    Check(JobOfView(v, Job::Results).state == "nothing yet", "Results: " + JobOfView(v, Job::Results).state);

    const auto& prep = JobOfView(v, Job::Prepare).chips;
    Check(prep.size() == 2 && prep[0].name == "Preparation Recipe" && prep[0].note == "tier1_paragraphs" &&
              prep[1].name == "SQL Query" && prep[1].note == "tier1_paragraphs",
          "Prepare chips: the recipe and its SQL step");
    const auto& val = JobOfView(v, Job::Validate);
    Check(val.chips.size() == 1 && val.chips[0].name == "Row Count Check" && val.chips[0].note == "\xC3\x97 4" &&
              val.node_ids.size() == 4,
          "Validate chip: Row Count Check x 4");
    Check(JobOfView(v, Job::Explore).chips.empty(), "Explore: no kept views yet");

    Check(v.rail.size() == 4 && v.rail[0].title == "Data Inputs" && v.rail[1].title == "Node results" &&
              v.rail[2].title == "Published" && v.rail[3].title == "Other datasets",
          "rail sections");
    Check(v.rail[0].items[0].dataset == "ds_datainput_157" && v.rail[0].items[0].meta == "7", "a loaded Data Input");
    Check(v.rail[1].items.size() == 5 && v.rail[1].items[0].meta == "15,349" && v.rail[1].items[0].dataset == "ds_sql_162",
          "node results: the step and its checks");
    Check(v.rail[1].items[1].meta == "pass" && v.rail[1].items[1].tone == Tone::Ok, "a passing check");
    Check(v.rail[2].items[0].meta == "15,349" && v.rail[2].items[0].tone == Tone::Ok, "a written export");
    Check(v.rail[3].items.size() == 1 && v.rail[3].items[0].dataset == "query_result", "other tabular datasets only");

    Check(v.last_run == "tier1_paragraphs \xC2\xB7 15,349 rows \xC2\xB7 4/4 checks \xC2\xB7 3.1 s \xC2\xB7 at 12:39:04",
          "last run: " + v.last_run);
    Check(v.last_run_tone == Tone::Ok, "last run fine");

    // A failing check and a failing step.
    in.nodes[3].outcome = "";
    in.nodes[3].error = "expected 15,349 rows, found 15,348";
    const View failed = Build(in);
    Check(JobOfView(failed, Job::Validate).state == "3 of 4 pass", "a failed check counts");
    Check(failed.rail[1].items[1].meta == "failed" && failed.rail[1].items[1].tone == Tone::Error, "a failed check is red");
    Check(failed.last_run_tone == Tone::Warn, "the last run warns when a check failed");
    in.nodes[2].error = "Binder Error: column \"txt\" not found";
    const View broken = Build(in);
    Check(broken.last_run == "tier1_paragraphs failed: Binder Error: column \"txt\" not found" &&
              broken.last_run_tone == Tone::Error,
          "a failed step: " + broken.last_run);

    Check(Thousands(15349) == "15,349" && Thousands(7) == "7" && Thousands(1234567) == "1,234,567", "thousands");
    std::cout << "data studio presentation: jobs, chips, rail, last run, failures (tier1 graph). OK\n";
    return 0;
}
