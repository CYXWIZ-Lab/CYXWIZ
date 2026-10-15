// Data Studio > Publish's model (TOFIX148): the outputs table on the Berean
// tier1 recipe (SQL step -> 4 checks -> Export Parquet).

#include "../src/core/data_studio_publish.h"

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
void Same(const std::string& got, const std::string& want, const std::string& what) {
    Check(got == want, what + ": got \"" + got + "\", want \"" + want + "\"");
}
GraphNode Node(int id, NodeType type, std::string name, std::vector<int> from = {}) {
    GraphNode n;
    n.id = id;
    n.type = type;
    n.name = std::move(name);
    n.input_from = std::move(from);
    return n;
}
}  // namespace

int main() {
    const std::string project = "D:/Project Berean";
    const std::string exports = "D:/Project Berean/exports";

    // Where an export writes, as the executor resolves it.
    Same(OutputPath(NodeType::ExportParquet, {{"file_path", "datasets/staging/tier1_paragraphs.parquet"}, {"path_base", "project"}},
                    project, exports),
         "D:/Project Berean/datasets/staging/tier1_paragraphs.parquet", "project-relative");
    Same(OutputPath(NodeType::ExportParquet, {{"file_path", "out/tier1"}, {"path_base", "exports"}}, project, exports),
         "D:/Project Berean/exports/out/tier1.parquet", "exports-relative, extension added");
    Same(OutputPath(NodeType::ExportCSV, {{"file_path", "C:/data/x.csv"}}, project, exports), "C:/data/x.csv", "absolute kept");
    Same(OutputPath(NodeType::DataOutput, {{"path", "a/b"}, {"format", "parquet"}, {"path_base", "project"}}, project, exports),
         "D:/Project Berean/a/b.parquet", "Data Output reads `path` and its format");
    Same(OutputPath(NodeType::ExportJSON, {}, project, exports), "", "no path: named at run time");

    // The tier1 outputs table.
    GraphNode step = Node(101, NodeType::SQLQuery, "tier1_paragraphs", {50});
    GraphNode c1 = Node(102, NodeType::RowCountCheck, "Check: paragraphs", {101});
    GraphNode c2 = Node(105, NodeType::RowCountCheck, "Check: nothing admitted", {102});
    GraphNode out = Node(198, NodeType::ExportParquet, "Export tier1_paragraphs.parquet", {105});
    out.parameters = {{"file_path", "datasets/staging/tier1_paragraphs.parquet"}, {"path_base", "project"}};
    std::vector<GraphNode> nodes{step, c1, c2, out};

    auto rows = OutputRows(nodes, project, exports);
    Check(rows.size() == 1, "one output");
    Same(rows[0].kind, "Export Parquet", "kind");
    Same(rows[0].reads, "tier1_paragraphs", "reads past the checks");
    Check(rows[0].reads_id == 101, "reads the SQL step");
    Same(rows[0].writes, "D:/Project Berean/datasets/staging/tier1_paragraphs.parquet", "will write");
    Same(rows[0].last, "not written yet", "not run");
    Check(!rows[0].written, "not written");

    nodes[3].ran = true;
    nodes[3].rows = 15349;
    nodes[3].finished_at = "14:28:41";
    nodes[3].written_path = R"(D:\Project Berean\datasets\staging\tier1_paragraphs.parquet)";
    rows = OutputRows(nodes, project, exports);
    Same(rows[0].last, "15,349 rows \xC2\xB7 14:28:41", "written");
    Same(rows[0].writes, "D:/Project Berean/datasets/staging/tier1_paragraphs.parquet", "the written path, forward slashes");
    Check(rows[0].written && rows[0].tone == Tone::Ok, "written ok");

    nodes[3].error = "ExportParquet: Export failed";
    rows = OutputRows(nodes, project, exports);
    Same(rows[0].last, "failed", "failed");
    Same(rows[0].detail, "ExportParquet: Export failed", "the error");

    Check(OutputRows({step, c1}, project, exports).empty(), "no outputs");
    Same(SuggestedExportPath("tier1_paragraphs"), "datasets/published/tier1_paragraphs.parquet", "suggested path");
    Same(SuggestedExportPath("Tier 1: books!"), "datasets/published/Tier_1__books.parquet", "cleaned stem");
    Same(SuggestedExportPath("??"), "datasets/published/dataset.parquet", "empty stem");

    std::cout << "test_data_studio_publish: all checks passed\n";
    return 0;
}
