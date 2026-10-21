#include "dataset_picker_presentation.h"

namespace cyxwiz {

namespace {

std::string Thousands(size_t n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
    return s;
}

constexpr const char* kDot = "  \xC2\xB7  ";

}  // namespace

std::string DatasetSizeText(const DatasetEntry& entry) {
    std::string unit = "rows";
    if (entry.modality == DatasetModality::Image) unit = "files";
    else if (entry.modality == DatasetModality::Audio) unit = "clips";
    else if (entry.modality == DatasetModality::Text && !entry.Has(kBackingArrow)) unit = "texts";
    std::string s = Thousands(entry.rows) + " " + unit;
    if (entry.columns) s += " \xC2\xB7 " + Thousands(entry.columns) + " cols";
    if (entry.classes) s += " \xC2\xB7 " + Thousands(entry.classes) + " classes";
    return s;
}

const char* DatasetOriginText(const DatasetEntry& entry) {
    // A graph node's output (pipeline run "ds_operator_<Type>_<id>" or a
    // materialized result) is made from Data Input data, not loaded itself.
    if (entry.materialized || entry.name.rfind("ds_operator_", 0) == 0) return "node result";
    if (entry.Has(kBackingLegacy) && entry.backings == kBackingLegacy) return "older dataset";
    return "Data Input";
}

bool IsTabularDataset(const DatasetEntry& entry) {
    return entry.Has(kBackingArrow) || entry.Has(kBackingParquet);
}

std::string DatasetPickerPreview(const std::vector<DatasetEntry>& entries, const std::string& selected) {
    if (selected.empty()) return "Select a dataset...";
    for (const auto& entry : entries) {
        if (entry.name == selected) return entry.Shown() + kDot + DatasetOriginText(entry) + kDot + DatasetSizeText(entry);
    }
    return selected + "  (no longer loaded)";
}

}  // namespace cyxwiz
