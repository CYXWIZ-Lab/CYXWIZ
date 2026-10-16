#pragma once

// A loaded dataset's first rows as a Table Viewer table (TOFIX140). The rows
// come through the data service page by page, so a dataset the Data Input
// keeps in memory or on disk (Parquet) both work and only the slice is held;
// the dataset itself stays where the Data Input put it. No ImGui here.

#include "data_preview_service.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace cyxwiz {

class DataTable;

// One page's rows as typed cells: integer columns stay integers, floating
// ones doubles, empty cells missing, the rest text. Appends to `table` when
// given, else starts a new one named `name` with the page's columns.
std::shared_ptr<DataTable> DataTableFromPreview(const DataPreviewPage& page, const std::string& name,
                                                std::shared_ptr<DataTable> table = nullptr);

struct DatasetSlice {
    std::shared_ptr<DataTable> table;
    int64_t total_rows = 0;  // rows in the dataset
    std::string error;
};

// The first `max_rows` rows of a registered tabular dataset, page by page
// from `read_page` (the Engine passes DataPreviewService).
using PreviewPageReader = std::function<DataPreviewPage(const DataPreviewRequest&)>;
DatasetSlice ReadDatasetSlice(const PreviewPageReader& read_page, const std::string& dataset_name, int64_t max_rows);

}  // namespace cyxwiz
