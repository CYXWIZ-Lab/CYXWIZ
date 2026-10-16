#pragma once

// What the dataset picker says about each catalog entry (Data Studio and the
// analysis tools share it; TOFIX140 A). Pure: no ImGui, no registry calls.
// Data enters only through Data Input nodes; the picker also lists the tables
// graph nodes make from them (owner decision 2026-10-11).

#include "dataset_catalog.h"

#include <string>
#include <vector>

namespace cyxwiz {

// "8,582 rows · 15 cols", "2,000 files · 2 classes".
std::string DatasetSizeText(const DatasetEntry& entry);

// "Data Input", "node result" or "older dataset".
const char* DatasetOriginText(const DatasetEntry& entry);

// A table the analysis tools can read: in memory (Arrow) or on disk (Parquet).
bool IsTabularDataset(const DatasetEntry& entry);

// The closed picker's text for `selected`: "Spotify · Data Input · 8,582 rows
// · 15 cols", "Select a dataset..." when nothing is picked, or "<name>  (no
// longer loaded)" when the picked dataset is gone.
std::string DatasetPickerPreview(const std::vector<DatasetEntry>& entries, const std::string& selected);

}  // namespace cyxwiz
