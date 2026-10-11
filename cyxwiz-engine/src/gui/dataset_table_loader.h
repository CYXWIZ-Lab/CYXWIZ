#pragma once

// Reads a catalog dataset for a tool (TOFIX140 A): every row, through the
// session query service, so in-memory and disk-backed datasets read the same
// way, in the background (Task View). The tools keep working on a DataTable.

#include <functional>
#include <memory>
#include <string>

namespace cyxwiz {

class DataTable;

// `done` runs on the UI thread with the table, or null and the reason; it does
// not run once `owner` is gone (a closed panel).
using DatasetTableDone = std::function<void(std::shared_ptr<DataTable> table, const std::string& error)>;
void LoadDatasetTable(const std::string& dataset_name, std::weak_ptr<const void> owner, DatasetTableDone done);

}  // namespace cyxwiz
