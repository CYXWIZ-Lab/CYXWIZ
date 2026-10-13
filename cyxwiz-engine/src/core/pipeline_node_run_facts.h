#pragma once

#include <cstdint>
#include <string>

namespace cyxwiz {

// What one node produced in its latest run (TOFIX146: SQL step Properties,
// Workbench). Kept per node across runs; a cached node keeps its earlier facts.
struct PipelineNodeRunFacts {
    std::string output_dataset;   // registry name, e.g. "ds_sql_101"
    int64_t rows = -1;            // -1 when the output is not a registered table
    int64_t columns = -1;
    double seconds = 0.0;         // from start to completion of the node
    std::string outcome;          // operator summary, e.g. a Row Count Check's "passed: ..."
    std::string error;            // set when the node failed in its latest run
    std::string finished_at;      // local time "HH:MM:SS"
};

}  // namespace cyxwiz
