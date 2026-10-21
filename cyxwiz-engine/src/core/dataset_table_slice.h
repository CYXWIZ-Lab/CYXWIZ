#pragma once

// A dataset's first rows as a Table Viewer table (TOFIX140), read off the UI
// thread (TOFIX159): straight from Arrow, so cells keep their types, a chunk
// at a time with progress and a way to stop. A dataset the Data Input keeps
// on disk (Parquet) is read row group by row group, each group once; the
// dataset itself stays where the Data Input put it. No ImGui here.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace arrow {
class Table;
}

namespace cyxwiz {

class DataTable;

// Called as a read goes: the share done (0..1) and what it is doing; false
// stops the read.
using SliceProgress = std::function<bool(float fraction, const std::string& message)>;

struct DatasetSlice {
    std::shared_ptr<DataTable> table;
    std::shared_ptr<arrow::Table> source;  // the rows read, as Arrow (the Stats profile reads them)
    int64_t total_rows = 0;                // rows in the dataset
    std::string error;
    bool stopped = false;                  // `progress` said stop
};

// The first `max_rows` rows of `source` as a table named `name`, converted
// `chunk_rows` at a time; `total_rows` is the dataset's own count.
DatasetSlice DatasetSliceFromArrow(const std::shared_ptr<arrow::Table>& source, const std::string& name,
                                   int64_t max_rows, int64_t total_rows, const SliceProgress& progress = {},
                                   int64_t chunk_rows = 20000);

// The first `max_rows` rows of `groups` row groups read in order by
// `read_group`, each once, joined; null with `error` when a group cannot be
// read or `progress` says stop (`error` empty then).
using RowGroupReader = std::function<std::shared_ptr<arrow::Table>(int group)>;
std::shared_ptr<arrow::Table> FirstRowsOfRowGroups(int groups, const RowGroupReader& read_group, int64_t max_rows,
                                                   const SliceProgress& progress, std::string& error);

}  // namespace cyxwiz
