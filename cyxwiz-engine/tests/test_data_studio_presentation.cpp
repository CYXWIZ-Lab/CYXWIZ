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
        Check(v.sources.size() == 1 && v.sources[0].state == "not loaded" && v.sources[0].size.empty() &&
                  v.sources[0].file == "datasets/staging/base_expansion_v3/archives/tier1_books.zip",
              "Bring in: a Data Input not loaded");
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
    Check(v.sources.size() == 1 && v.sources[0].state == "loaded" && v.sources[0].size == "7 \xC3\x97 6" &&
              v.sources[0].dataset == "ds_datainput_157" && v.sources[0].tone == Tone::Ok,
          "Bring in: loaded, 7 x 6");
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

    in.nodes[0].error = "cannot open tier1_books.zip";
    const View input_failed = Build(in);
    Check(input_failed.sources[0].state == "failed" && input_failed.sources[0].detail == "cannot open tier1_books.zip" &&
              input_failed.sources[0].tone == Tone::Error,
          "Bring in: a Data Input that failed");

    // ML readiness: the Berean base-corpus training graph (two Data Inputs into
    // the Data Split's train and validation pins, a Data Loader without balance).
    {
        Input t;
        GraphNode train = Node(1, NodeType::DataInput, "Base corpus v2 train: 100,235 windows");
        GraphNode validation = Node(15, NodeType::DataInput, "Base corpus v2 validation: 2,310 windows");
        GraphNode split = Node(2, NodeType::DataSplit, "Frozen train and validation families");
        split.input_from = {1, 15, -1};
        GraphNode loader = Node(3, NodeType::DataLoader, "3 passes over the schedule");
        loader.balance = "none";
        t.nodes = {train, validation, split, loader};
        t.datasets = {{"train_schedule_windows_v2", "Base corpus v2 train: 100,235 windows", 100235, 19, true},
                      {"validation_windows_v2", "Base corpus v2 validation: 2,310 windows", 2310, 17, true}};
        const View rv = Build(t);
        Check(rv.split_sets.size() == 1 && rv.split_sets[0].node_id == 2 && rv.split_sets[0].parts.size() == 2 &&
                  rv.split_sets[0].parts[0].split == "train" && rv.split_sets[0].parts[0].dataset == "train_schedule_windows_v2" &&
                  rv.split_sets[0].parts[1].split == "validation" && rv.split_sets[0].note.empty(),
              "the Data Split's inputs are the splits");
        Check(JobOfView(rv, Job::Ready).state == "2 splits" && rv.balance == "none", "ready state: " + JobOfView(rv, Job::Ready).state);

        // Validation not loaded: named in the note.
        t.datasets.pop_back();
        const View half = Build(t);
        Check(half.split_sets[0].parts.size() == 1 &&
                  half.split_sets[0].note == "validation: \"Base corpus v2 validation: 2,310 windows\" is not loaded",
              "a split not loaded: " + half.split_sets[0].note);

        // No Data Split: kjv80_base_v1's files, grouped by name.
        Input k;
        k.datasets = {{"documents_test", "", 66, 20, true},
                      {"documents_train", "", 1228, 20, true},
                      {"documents_validation", "", 68, 20, true},
                      {"interval_val", "", 3, 1, true},
                      {"query_result", "", 3, 2, true}};
        const View kv = Build(k);
        Check(kv.split_sets.size() == 1 && kv.split_sets[0].name == "documents" && kv.split_sets[0].node_id == -1 &&
                  kv.split_sets[0].parts.size() == 3 && kv.split_sets[0].parts[0].dataset == "documents_train" &&
                  kv.split_sets[0].parts[2].split == "test",
              "catalog names make a split set (train first; a set needs train)");
        Check(JobOfView(kv, Job::Ready).state == "3 splits", "kjv80 ready state");

        // Three Data Inputs all named "Data Input" (added from the Asset Browser): each finds its own
        // dataset by dataset_name, not the shared name.
        k.datasets = {{"documents_train", "Data Input #106", 1228, 20, true},
                      {"documents_validation", "Data Input #107", 68, 20, true},
                      {"documents_test", "Data Input #108", 66, 20, true}};
        for (int id : {106, 107, 108}) k.nodes.push_back(Node(id, NodeType::DataInput, "Data Input"));
        k.nodes[0].dataset_name = "documents_train";
        k.nodes[1].dataset_name = "documents_validation";
        k.nodes[2].dataset_name = "documents_test";
        const View same = Build(k);
        Check(same.rail[0].items.size() == 3 && same.rail[0].items[0].meta == "1,228" && same.rail[0].items[1].meta == "68" &&
                  same.rail[0].items[2].meta == "66" && same.rail.size() == 3,
              "same-named Data Inputs show their own rows, none in Other datasets");
        k.nodes[0].dataset_name.clear();
        k.datasets[0].label = "Data Input";
        k.datasets[1].label = "Data Input";
        Check(Build(k).rail[0].items[0].meta == "not loaded", "a shared label matches no node");
        std::string stem;
        Check(SplitOfName("documents_train", &stem) == "train" && stem == "documents" && SplitOfName("x-dev") == "validation" &&
                  SplitOfName("interval").empty() && SplitOfName("train").empty(),
              "split names");
    }

    // Prepare's step list: the tier1 recipe after its run.
    {
        const auto rows = StepRows({in.nodes[2], in.nodes[3], Node(500, NodeType::FilterRows, "Keep admitted")});
        Check(rows.size() == 3 && rows[0].sql && rows[0].kind == "SQL Query" && rows[0].meta == "failed" &&
                  rows[0].tone == Tone::Error && rows[0].tip.find("Binder Error") != std::string::npos,
              "a failed SQL step");
        Check(rows[1].kind == "Row Count Check" && rows[1].meta == "failed" && !rows[1].sql, "a failed check");
        Check(rows[2].kind == "step node" && rows[2].meta == "not run", "a step node not run");
        const auto ran = StepRows({Ran(Node(162, NodeType::SQLQuery, "tier1_paragraphs"), 15349, 13, 3.1, "12:39:04", "ds_sql_162"),
                                   Ran(Node(347, NodeType::RowCountCheck, "Check: paragraphs"), 15349, 13, 0.1, "12:39:05",
                                       "ds_check_347", "passed: 15349 rows")});
        Check(ran[0].meta == "15,349 rows \xC2\xB7 3.1 s" && ran[1].meta == "pass" && ran[1].tone == Tone::Ok &&
                  ran[1].tip == "passed: 15349 rows",
              "a run step and a passing check: " + ran[0].meta);
    }

    Check(Thousands(15349) == "15,349" && Thousands(7) == "7" && Thousands(1234567) == "1,234,567", "thousands");
    std::cout << "data studio presentation: jobs, chips, rail, last run, failures (tier1 graph), split sets (Data Split inputs, catalog names). OK\n";
    return 0;
}
