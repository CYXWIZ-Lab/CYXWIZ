#pragma once

// Data Studio > Publish (TOFIX148, approved mockup v8): the graph's outputs
// (Export CSV / JSON / Parquet, Data Output) as a table: the node, what it
// reads, the file it writes, when it last wrote. The lineage beside a written
// file is core/dataset_lineage. Pure data in and out.

#include "data_studio_presentation.h"

#include <map>
#include <string>
#include <vector>

namespace cyxwiz::studio {

bool IsOutputNode(gui::NodeType type);

// The file an export node writes, as the executor resolves it
// (ResolveDatasetExportPath): a relative path against the project folder
// (path_base "project") or the exports folder, the format's extension added
// when missing. "" when the path is empty (the executor names it after the
// table it reads, known only once it runs).
std::string OutputPath(gui::NodeType type, const std::map<std::string, std::string>& parameters,
                       const std::string& project_root, const std::string& exports_root);

struct OutputRow {
    int id = -1;
    std::string name;
    std::string kind;        // "Export Parquet", "Data Output"
    std::string reads;       // the step it reads, past any checks ("" when nothing is linked)
    int reads_id = -1;
    std::string writes;      // the file written in the last run, else the file it will write
    bool written = false;    // `writes` was written in this session's last run
    std::string last;        // "15,349 rows · 14:28:41", "not written yet", "failed"
    std::string detail;      // the error when it failed
    Tone tone = Tone::Plain;
};
// The graph's outputs in graph order. Nodes carry their parameters,
// input_from and written_path.
std::vector<OutputRow> OutputRows(const std::vector<GraphNode>& nodes, const std::string& project_root,
                                  const std::string& exports_root);

// "datasets/published/tier1_paragraphs.parquet" for a new export of a step
// named "tier1_paragraphs" (the stem cleaned to letters, digits, - and _).
std::string SuggestedExportPath(const std::string& step_name);

}  // namespace cyxwiz::studio
