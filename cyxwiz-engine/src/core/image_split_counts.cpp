#include "image_split_counts.h"

#include "arrow_dataset.h"
#include "data_registry.h"
#include "graph_compiler_dataset_hooks.h"
#include "sample_split.h"

#include <arrow/api.h>

#include <unordered_set>

namespace cyxwiz {

ImageSplitCounts CountImageSplit(const std::string& image_dataset, const SampleSplitSettings& split,
                                 const std::vector<std::string>& excluded_files) {
    ImageSplitCounts counts;
    auto& registry = DataRegistry::Instance();
    const auto* entry = registry.GetImageDatasetEntry(image_dataset);
    if (!entry) {
        counts.reason = "'" + image_dataset + "' is not loaded";
        return counts;
    }
    const auto index = registry.GetArrowDataset(DataRegistry::ImageIndexName(image_dataset));
    const auto table = index ? index->GetArrowTable() : nullptr;
    if (!table) {
        counts.reason = "the images of '" + image_dataset + "' are still being indexed";
        return counts;
    }
    const auto path_column = table->GetColumnByName("path");
    const auto label_column = table->GetColumnByName("label");
    const auto readable_column = table->GetColumnByName("readable");
    if (!path_column || !label_column || !readable_column) {
        counts.reason = "the index of '" + image_dataset + "' has no path / label / readable columns";
        return counts;
    }

    const std::unordered_set<std::string> left_out(excluded_files.begin(), excluded_files.end());
    const auto rows = static_cast<size_t>(table->num_rows());
    std::vector<int> labels(rows, -1);
    std::vector<size_t> usable;
    usable.reserve(rows);
    size_t row = 0;
    for (int chunk = 0; chunk < path_column->num_chunks(); ++chunk) {
        const auto paths = std::static_pointer_cast<arrow::StringArray>(path_column->chunk(chunk));
        for (int64_t i = 0; i < paths->length(); ++i, ++row) {
            if (left_out.count(paths->GetString(i))) {
                ++counts.left_out;
                labels[row] = -2;  // marked; skipped below
            }
        }
    }
    row = 0;
    for (int chunk = 0; chunk < label_column->num_chunks(); ++chunk) {
        const auto values = std::static_pointer_cast<arrow::Int32Array>(label_column->chunk(chunk));
        for (int64_t i = 0; i < values->length(); ++i, ++row)
            if (labels[row] != -2) labels[row] = values->IsNull(i) ? -1 : values->Value(i);
    }
    row = 0;
    for (int chunk = 0; chunk < readable_column->num_chunks(); ++chunk) {
        const auto readable = std::static_pointer_cast<arrow::BooleanArray>(readable_column->chunk(chunk));
        for (int64_t i = 0; i < readable->length(); ++i, ++row) {
            if (labels[row] == -2) continue;
            if (readable->IsNull(i) || !readable->Value(i)) {
                ++counts.unreadable;
                continue;
            }
            usable.push_back(row);
        }
    }
    for (int& label : labels)
        if (label == -2) label = -1;

    counts.class_names = entry->class_names;
    counts.shown_names = entry->ShownClassNames();
    const size_t classes = counts.class_names.size();
    counts.per_class.assign(classes, {0, 0, 0});
    const auto parts = SplitSamples(usable, labels, split);
    const std::vector<size_t>* by_part[3] = {&parts.train, &parts.val, &parts.test};
    for (size_t part = 0; part < 3; ++part) {
        counts.parts[part] = by_part[part]->size();
        for (const size_t image : *by_part[part]) {
            const int label = labels[image];
            const size_t c = label >= 0 && static_cast<size_t>(label) < classes ? static_cast<size_t>(label) : classes;
            if (c >= counts.per_class.size()) counts.per_class.resize(c + 1, {0, 0, 0});
            ++counts.per_class[c][part];
        }
    }
    counts.images = usable.size();
    counts.known = true;
    return counts;
}

namespace {

// Hosts that compile this file (they link the registry) give the compiler the
// image split counts; the other catalog lookups keep what was installed.
const bool kImageSplitHookInstalled = [] {
    GraphDatasetCatalog catalog = GetGraphDatasetCatalog();
    catalog.image_split = CountImageSplit;
    SetGraphDatasetCatalog(std::move(catalog));
    return true;
}();

}  // namespace

}  // namespace cyxwiz
